#!/usr/bin/env python3
"""
Generate a synthetic PCAP using the VAE+GMM+LSTM models trained by train_fullstack.py.
"""

import argparse
import json
import pathlib
import time
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from scapy.all import Ether, IP, UDP, TCP, Raw, wrpcap


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
    prev = np.zeros((1, 1, len(scaler_flow.mean_)))
    preds = []
    for _ in range(flow_len):
        next_step = lstm.predict(prev, verbose=0)[:, -1, :]
        preds.append(next_step[0])
        prev = np.concatenate([prev, next_step[:, None, :]], axis=1)

    preds = np.array(preds)
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


def generate_pcap(models_dir: pathlib.Path, output_pcap: pathlib.Path, num_flows: int, start_time: float):
    scaler_pkt, scaler_flow, gmm, meta, encoder, decoder, lstm, cfg = load_models(models_dir)

    packets: List = []
    base_time = start_time

    for _ in range(num_flows):
        flow_len = int(sample_with_probs(meta["flow_size_values"], meta["flow_size_probs"], 1)[0])
        endpoints = sample_with_probs(meta["endpoints_values"], meta["endpoints_probs"], 1)[0]
        ports = sample_with_probs(meta["ports_values"], meta["ports_probs"], 1)[0]
        app = sample_with_probs(meta["apps_values"], meta["apps_probs"], 1)[0]

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
            base_time=base_time,
        )
        if pkts:
            base_time = pkts[-1].time
        packets.extend(pkts)

    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(output_pcap), packets)
    print(f"Wrote synthetic PCAP: {output_pcap} ({len(packets)} packets)")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate PCAP from VAE+GMM+LSTM stack")
    parser.add_argument("--models-dir", required=True, help="Directory produced by train_fullstack.py")
    parser.add_argument("--output-pcap", required=True, help="Where to write the generated PCAP")
    parser.add_argument("--num-flows", type=int, default=10)
    parser.add_argument("--start-time", type=float, default=None, help="Unix start time")
    args = parser.parse_args(argv)

    start_time = args.start_time if args.start_time is not None else time.time()
    generate_pcap(
        models_dir=pathlib.Path(args.models_dir),
        output_pcap=pathlib.Path(args.output_pcap),
        num_flows=args.num_flows,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
