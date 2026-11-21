#!/usr/bin/env python3
"""
Generate a synthetic PCAP using the VAE+GMM+LSTM models trained by train_fullstack.py.
"""

import argparse
import ctypes.util
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pathlib
import sys
import time
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from autoregressive_inference import autoregressive_flow_features

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "utils"))
from random_seed import derive_seed, environment_seed, seed_rngs

_orig_find_library = ctypes.util.find_library


def _prefer_glibc(name: str):
    if name == "libc":
        return "/lib/x86_64-linux-gnu/libc.so.6"
    return _orig_find_library(name)


ctypes.util.find_library = _prefer_glibc

from scapy.all import Ether, IP, UDP, TCP, Raw, wrpcap

try:
    from tqdm.auto import tqdm
except ImportError:  # Fallback if tqdm is unavailable
    def tqdm(iterable=None, total=None, desc="", unit="", leave=True, **kwargs):
        if iterable is None:
            return []
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        for idx, item in enumerate(iterable, 1):
            prefix = f"{desc} " if desc else ""
            suffix = f"{idx}/{total}" if total else f"{idx}"
            print(f"{prefix}{suffix} {unit}".strip(), end="\r" if leave else "\n")
            yield item
        print()


def sample_with_probs(values: np.ndarray, probs: np.ndarray, size: int = 1):
    probs = probs / probs.sum()
    return np.random.choice(values, size=size, p=probs)


def load_models(models_dir: pathlib.Path):
    scaler_pkt = joblib.load(models_dir / "scaler_packet.joblib")
    scaler_flow = joblib.load(models_dir / "scaler_flow.joblib")
    gmm = joblib.load(models_dir / "packet_gmm.joblib")
    meta = joblib.load(models_dir / "meta_fullstack.joblib")
    encoder = tf.keras.models.load_model(models_dir / "vae_encoder")
    decoder = tf.keras.models.load_model(models_dir / "vae_decoder")
    lstm = tf.keras.models.load_model(models_dir / "lstm_flow")
    cfg = json.loads((models_dir / "config.json").read_text())
    return scaler_pkt, scaler_flow, gmm, meta, encoder, decoder, lstm, cfg


def decode_packets(decoder, scaler_pkt, gmm, num_packets: int) -> pd.DataFrame:
    # sample latent from GMM, decode to scaled features, inverse-scale
    z, _ = gmm.sample(num_packets)
    feats_scaled = decoder.predict(z, verbose=0)
    feats = scaler_pkt.inverse_transform(feats_scaled)
    return feats


_fullstack_ctx = {}


def _init_fullstack_ctx(models_dir: str, seed: int = 0):
    """Load models once per worker process."""
    global _fullstack_ctx
    if _fullstack_ctx.get("models_dir") == models_dir:
        return
    scaler_pkt, scaler_flow, gmm, meta, encoder, decoder, lstm, cfg = load_models(pathlib.Path(models_dir))
    _fullstack_ctx = {
        "seed": seed,
        "models_dir": models_dir,
        "scaler_pkt": scaler_pkt,
        "scaler_flow": scaler_flow,
        "gmm": gmm,
        "meta": meta,
        "encoder": encoder,
        "decoder": decoder,
        "lstm": lstm,
    }


def _build_flow_fullstack(idx: int) -> Tuple[int, List]:
    """Worker entrypoint: build a single flow starting at t=0, using global ctx."""
    ctx = _fullstack_ctx
    flow_seed = derive_seed(ctx["seed"], "flow", idx)
    seed_rngs(flow_seed, tensorflow=True)
    ctx["gmm"].random_state = np.random.RandomState(flow_seed)
    meta = ctx["meta"]
    flow_len = int(sample_with_probs(meta["flow_size_values"], meta["flow_size_probs"], 1)[0])
    endpoints = sample_with_probs(meta["endpoints_values"], meta["endpoints_probs"], 1)[0]
    ports = sample_with_probs(meta["ports_values"], meta["ports_probs"], 1)[0]
    app = sample_with_probs(meta["apps_values"], meta["apps_probs"], 1)[0]

    pkts = build_packets(
        flow_len=flow_len,
        endpoints=endpoints,
        ports=ports,
        app=app,
        lstm=ctx["lstm"],
        scaler_flow=ctx["scaler_flow"],
        decoder=ctx["decoder"],
        scaler_pkt=ctx["scaler_pkt"],
        gmm=ctx["gmm"],
        base_time=0.0,
    )
    return idx, pkts


def build_packets(
    flow_len: int,
    endpoints: Tuple[str, str],
    ports: Tuple[int, int],
    app: str,
    lstm,
    scaler_flow,
    decoder,
    scaler_pkt,
    gmm,
    base_time: float,
):
    ip_src, ip_dst = endpoints
    sport, dport = int(ports[0]), int(ports[1])
    use_tcp = str(app).upper() in {"HTTP", "SSH", "FTP", "SMTP", "TELNET"}

    # Generate temporal/size dynamics with LSTM
    preds = autoregressive_flow_features(lstm, flow_len)
    preds = preds.reshape(flow_len, -1)
    flow_feats = scaler_flow.inverse_transform(preds)
    # Ensure positive time_diff/payload
    flow_feats[:, 1] = np.clip(flow_feats[:, 1], 0.001, None)

    # Decode packet feature vectors
    pkt_feats = decode_packets(decoder, scaler_pkt, gmm, num_packets=flow_len)

    timestamps = []
    t = base_time
    for td in flow_feats[:, 1]:
        t += float(td)
        timestamps.append(t)

    packets = []
    for i in range(flow_len):
        payload_len = max(int(pkt_feats[i, 2]), 0)
        payload = Raw(b"\x00" * payload_len)
        l4 = TCP(sport=sport, dport=dport, flags="PA") if use_tcp else UDP(sport=sport, dport=dport)
        pkt = Ether() / IP(src=ip_src, dst=ip_dst) / l4 / payload
        pkt.time = timestamps[i]
        packets.append(pkt)
    return packets


def generate_pcap(
    models_dir: pathlib.Path,
    output_pcap: pathlib.Path,
    num_flows: int,
    start_time: float,
    workers: int = 1,
    seed: int = 0,
):
    seed_rngs(seed, tensorflow=True)
    flow_packets: List[List] = [None] * num_flows  # type: ignore
    workers_local = max(1, workers)

    if workers_local == 1:
        scaler_pkt, scaler_flow, gmm, meta, encoder, decoder, lstm, cfg = load_models(models_dir)
        for i in range(num_flows):
            flow_seed = derive_seed(seed, "flow", i)
            seed_rngs(flow_seed, tensorflow=True)
            gmm.random_state = np.random.RandomState(flow_seed)
            flow_len = int(sample_with_probs(meta["flow_size_values"], meta["flow_size_probs"], 1)[0])
            endpoints = sample_with_probs(meta["endpoints_values"], meta["endpoints_probs"], 1)[0]
            ports = sample_with_probs(meta["ports_values"], meta["ports_probs"], 1)[0]
            app = sample_with_probs(meta["apps_values"], meta["apps_probs"], 1)[0]

            flow_started = time.monotonic()
            print(f"Flow {i} starting with {flow_len} packets", flush=True)
            pkts = build_packets(
                flow_len=flow_len,
                endpoints=endpoints,
                ports=ports,
                app=app,
                lstm=lstm,
                scaler_flow=scaler_flow,
                decoder=decoder,
                scaler_pkt=scaler_pkt,
                gmm=gmm,
                base_time=0.0,
            )
            flow_packets[i] = pkts
            print(f"Flow {i} created with {len(pkts)} packets in {time.monotonic()-flow_started:.3f}s", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers_local,
            initializer=_init_fullstack_ctx,
            initargs=(str(models_dir), seed),
        ) as pool:
            futures = [pool.submit(_build_flow_fullstack, i) for i in range(num_flows)]
            for fut in as_completed(futures):
                idx, pkts = fut.result()
                flow_packets[idx] = pkts
                print(f"Flow {idx} created with {len(pkts)} packets")

    # Normalize timestamps to a single sequential timeline
    packets: List = []
    base_time = start_time
    for pkts in flow_packets:
        if not pkts:
            continue
        first_ts = pkts[0].time
        shift = base_time - first_ts
        for pkt in pkts:
            pkt.time += shift
        duration = pkts[-1].time - pkts[0].time
        base_time += duration
        packets.extend(pkts)

    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(output_pcap), packets)
    print(f"Wrote synthetic PCAP: {output_pcap} ({len(packets)} packets)")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate PCAP from VAE+GMM+LSTM stack")
    parser.add_argument("--models-dir", required=True, help="Directory produced by train_fullstack.py")
    parser.add_argument("--output-pcap", required=True, help="Where to write the generated PCAP")
    parser.add_argument("--num-flows", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for flow generation")
    parser.add_argument("--start-time", type=float, default=None, help="Unix start time")
    parser.add_argument("--seed", type=int, default=environment_seed(generation=True))
    args = parser.parse_args(argv)

    start_time = args.start_time if args.start_time is not None else time.time()
    generate_pcap(
        models_dir=pathlib.Path(args.models_dir),
        output_pcap=pathlib.Path(args.output_pcap),
        num_flows=args.num_flows,
        start_time=start_time,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
