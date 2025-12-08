#!/usr/bin/env python3
"""
Generate a synthetic PCAP from the models trained by train_models.py.
"""

import argparse
import pathlib
import time
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, UDP, TCP, Raw, wrpcap, Ether

NUMERIC_COLS = [
    "length_total",
    "header_length",
    "payload_length",
    "rate",
    "time_diff",
    "rolling_rate_byte_sec",
    "rolling_rate_byte_min",
    "rolling_rate_packet_sec",
    "rolling_rate_packet_min",
]


def sample_with_probs(values: np.ndarray, probs: np.ndarray, size: int = 1):
    return np.random.choice(values, size=size, p=probs / probs.sum())


def build_packet(
    ip_src: str,
    ip_dst: str,
    sport: int,
    dport: int,
    payload_len: int,
    timestamp: float,
    use_tcp: bool,
):
    payload = Raw(b"\x00" * max(int(payload_len), 0))
    l4 = TCP(sport=sport, dport=dport, flags="PA") if use_tcp else UDP(sport=sport, dport=dport)
    pkt = Ether() / IP(src=ip_src, dst=ip_dst) / l4 / payload
    pkt.time = timestamp
    return pkt


def generate_packets(models_path: pathlib.Path, num_flows: int, base_time: float) -> List:
    artifacts = joblib.load(models_path)
    scaler = artifacts["scaler"]
    gmm = artifacts["packet_gmm"]

    flow_size_values = artifacts["flow_size_values"]
    flow_size_probs = artifacts["flow_size_probs"]
    endpoints_values = artifacts["endpoints_values"]
    endpoints_probs = artifacts["endpoints_probs"]
    ports_values = artifacts["ports_values"]
    ports_probs = artifacts["ports_probs"]
    apps_values = artifacts["apps_values"]
    apps_probs = artifacts["apps_probs"]

    packets = []
    current_time = base_time

    for _ in range(num_flows):
        flow_len = int(sample_with_probs(flow_size_values, flow_size_probs, size=1)[0])
        ip_src, ip_dst = sample_with_probs(endpoints_values, endpoints_probs, size=1)[0]
        sport, dport = sample_with_probs(ports_values, ports_probs, size=1)[0]
        app = sample_with_probs(apps_values, apps_probs, size=1)[0]
        use_tcp = str(app).upper() in {"HTTP", "SSH", "FTP", "SMTP", "TELNET"}

        samples, _ = gmm.sample(flow_len)
        features = pd.DataFrame(scaler.inverse_transform(samples), columns=NUMERIC_COLS)
        features["time_diff"] = features["time_diff"].clip(lower=0.0001)
        features["payload_length"] = features["payload_length"].clip(lower=0).astype(int)

        timestamps = []
        for td in features["time_diff"]:
            current_time += float(td)
            timestamps.append(current_time)

        for ts, payload_len in zip(timestamps, features["payload_length"]):
            pkt = build_packet(
                ip_src=ip_src,
                ip_dst=ip_dst,
                sport=int(sport),
                dport=int(dport),
                payload_len=int(payload_len),
                timestamp=ts,
                use_tcp=use_tcp,
            )
            packets.append(pkt)

    return packets


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate PCAP from trained models")
    parser.add_argument("--models", required=True, help="Path to traffic_models.joblib")
    parser.add_argument("--output-pcap", required=True, help="Output PCAP path")
    parser.add_argument("--num-flows", type=int, default=10, help="How many flows to synthesize")
    parser.add_argument(
        "--start-time", type=float, default=None, help="Unix start time for generated packets"
    )
    args = parser.parse_args(argv)

    base_time = args.start_time if args.start_time is not None else time.time()
    packets = generate_packets(
        models_path=pathlib.Path(args.models),
        num_flows=args.num_flows,
        base_time=base_time,
    )
    out_path = pathlib.Path(args.output_pcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(out_path), packets)
    print(f"Wrote synthetic PCAP: {out_path}")


if __name__ == "__main__":
    main()
