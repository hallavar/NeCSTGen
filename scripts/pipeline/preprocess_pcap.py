#!/usr/bin/env python3
"""
Lightweight end-to-end preprocessing for NeCSTGen.

Given a PCAP, this script:
1) extracts per-packet features with Scapy,
2) builds flow identifiers,
3) computes basic time/rate features,
4) writes a single processed CSV ready for model training.

It is intentionally self-contained so you don't have to edit the
original research scripts. The produced CSVs are compatible with
the training/generation helpers in scripts/pipeline.
"""

import argparse
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scapy.all import PcapReader


# Basic app heuristics (mirrors the original scripts)
APP_LIST = [
    "ARP",
    "LLC",
    "LOOP",
    "SNAP",
    "TELNET",
    "HTTP",
    "SSH",
    "SNMP",
    "SMTP",
    "DNS",
    "NTP",
    "FTP",
    "RIP",
    "IRC",
    "POP",
    "ICMP",
    "FINGER",
    "TIME",
    "NETBIOS",
]


def get_layers(pkt) -> List[str]:
    layers = []
    for lyr in pkt.layers():
        name = str(lyr).split(".")[-1].rstrip("'>").split("'")[0]
        layers.append(name.replace("Ethernet", "Ether"))  # keep naming consistent
    return layers


def classify_application(row: pd.Series) -> str:
    sport = int(row.get("sport", 0) or 0)
    dport = int(row.get("dport", 0) or 0)
    layers = [row.get(f"layers_{i}", "") for i in range(6)]

    conds = [
        (sport == 0 and dport == 0 and layers[1] == "ARP"),
        (sport == 0 and dport == 0 and layers[1] == "LLC" and layers[2] == "Raw"),
        (sport == 0 and dport == 0 and layers[0] == "Ether" and layers[1] == "Raw"),
        (sport == 0 and dport == 0 and layers[1] == "LLC" and layers[2] == "SNAP"),
        (sport == 23 or dport == 23 or layers[3] == "TELNET"),
        (sport == 80 or dport == 80),
        (layers[3] == "SSH" or sport == 22 or dport == 22),
        (layers[3] == "SNMP"),
        (layers[3] in ("SMTPRequest", "SMTPResponse") or sport == 25 or dport == 25),
        (sport == 53 or dport == 53 or layers[3] == "DNS"),
        (layers[3] == "NTPHeader"),
        (
            layers[3] in ("FTPRequest", "FTPResponse")
            or sport in (20, 21)
            or dport in (20, 21)
        ),
        (layers[3] == "RIP"),
        (
            layers[3] in ("IRCRes", "IRCReq")
            or sport in (113, 6667)
            or dport in (113, 6667)
        ),
        (layers[3] == "POP" or sport == 110 or dport == 110),
        (layers[2] == "ICMP"),
        (sport == 79 or dport == 79),
        (sport == 37 or dport == 37),
        (sport == 137 or dport == 137),
    ]

    for cond, app in zip(conds, APP_LIST):
        if cond:
            return app
    return "OTHER"


def extract_packet_features(pkt, idx: int, filename: str) -> Dict[str, object]:
    layers = get_layers(pkt)
    feat: Dict[str, object] = {}

    for i in range(6):
        if i < len(layers):
            feat[f"layers_{i}"] = layers[i]
            try:
                feat[f"length_{i}"] = int(len(pkt[layers[i]]))
            except Exception:
                feat[f"length_{i}"] = 0
        else:
            feat[f"layers_{i}"] = "None"
            feat[f"length_{i}"] = 0

    feat["timestamps"] = float(pkt.time)
    feat["length_total"] = int(len(pkt))
    feat["filename"] = filename
    feat["num_packet"] = idx

    # MAC
    feat["mac_src"] = getattr(pkt, "src", None) if "Ether" in layers else None
    feat["mac_dst"] = getattr(pkt, "dst", None) if "Ether" in layers else None

    # IP / ports / flags
    try:
        feat["ip_src"] = pkt["IP"].src
        feat["ip_dst"] = pkt["IP"].dst
    except Exception:
        feat["ip_src"] = None
        feat["ip_dst"] = None

    proto = None
    try:
        feat["sport"] = int(pkt["TCP"].sport)
        feat["dport"] = int(pkt["TCP"].dport)
        feat["flags"] = int(pkt["TCP"].flags.value)
        proto = "TCP"
    except Exception:
        try:
            feat["sport"] = int(pkt["UDP"].sport)
            feat["dport"] = int(pkt["UDP"].dport)
            feat["flags"] = 0
            proto = "UDP"
        except Exception:
            feat["sport"] = 0
            feat["dport"] = 0
            feat["flags"] = 0

    feat["protocol"] = proto or (layers[1] if len(layers) > 1 else layers[0])
    return feat


def assign_flows(df: pd.DataFrame) -> pd.DataFrame:
    flow_map: Dict[Tuple, int] = {}
    next_id = 0

    flow_ids: List[int] = []
    for _, row in df.iterrows():
        has_ports = (row.get("sport", 0) or 0) != 0 or (row.get("dport", 0) or 0) != 0
        if has_ports and row.get("ip_src") not in (None, "None") and row.get(
            "ip_dst"
        ) not in (None, "None"):
            key = (
                row["ip_src"],
                row["ip_dst"],
                int(row["sport"]),
                int(row["dport"]),
                row["applications"],
            )
            rev_key = (
                row["ip_dst"],
                row["ip_src"],
                int(row["dport"]),
                int(row["sport"]),
                row["applications"],
            )
        else:
            key = (
                row.get("mac_src"),
                row.get("mac_dst"),
                row.get("layers_1"),
                row.get("applications"),
            )
            rev_key = (
                row.get("mac_dst"),
                row.get("mac_src"),
                row.get("layers_1"),
                row.get("applications"),
            )

        if key in flow_map:
            fid = flow_map[key]
        elif rev_key in flow_map:
            fid = flow_map[rev_key]
        else:
            fid = next_id
            flow_map[key] = fid
            next_id += 1
        flow_ids.append(fid)

    df = df.copy()
    df["flow_id"] = flow_ids
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamps"], unit="s")

    def _per_flow(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamps")
        group["time_diff"] = group["timestamps"].diff().fillna(0)
        group["rate"] = 0.0
        valid = group["time_diff"] > 0
        group.loc[valid, "rate"] = (
            group.loc[valid, "length_total"]
            / group.loc[valid, "time_diff"].replace(0, np.nan)
        ).fillna(0)

        idx_dt = group.set_index("datetime")
        group["rolling_rate_byte_sec"] = (
            idx_dt["length_total"].rolling("1s").sum().values
        )
        group["rolling_rate_byte_min"] = (
            idx_dt["length_total"].rolling("60s").sum().values
        )
        group["rolling_rate_packet_sec"] = (
            idx_dt["length_total"].rolling("1s").count().values
        )
        group["rolling_rate_packet_min"] = (
            idx_dt["length_total"].rolling("60s").count().values
        )
        return group

    df = df.groupby("flow_id", group_keys=False).apply(_per_flow)

    # Rough header/payload split
    header_cols = [f"length_{i}" for i in range(6)]
    df["header_length"] = df[header_cols].sum(axis=1)
    df["payload_length"] = (df["length_total"] - df["header_length"]).clip(lower=0)
    return df


def preprocess(pcap_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    rows: List[Dict[str, object]] = []

    with PcapReader(str(pcap_path)) as reader:
        for idx, pkt in enumerate(reader):
            rows.append(extract_packet_features(pkt, idx, filename=pcap_path.name))

    df = pd.DataFrame(rows)
    df.fillna(
        {"ip_src": "None", "ip_dst": "None", "mac_src": "None", "mac_dst": "None"},
        inplace=True,
    )
    for col in ["sport", "dport", "flags"]:
        df[col] = df[col].fillna(0).astype(int)
    df["applications"] = df.apply(classify_application, axis=1)
    df = assign_flows(df)
    df = compute_time_features(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    packets_path = output_dir / "packets.csv"
    processed_path = output_dir / "packets_processed.csv"

    df.to_csv(packets_path, index=False)
    # Keep only the model-friendly columns for downstream steps
    keep_cols = [
        "flow_id",
        "timestamps",
        "length_total",
        "header_length",
        "payload_length",
        "rate",
        "time_diff",
        "rolling_rate_byte_sec",
        "rolling_rate_byte_min",
        "rolling_rate_packet_sec",
        "rolling_rate_packet_min",
        "ip_src",
        "ip_dst",
        "sport",
        "dport",
        "protocol",
        "applications",
    ]
    df[keep_cols].to_csv(processed_path, index=False)
    print(f"Wrote packet-level CSV: {packets_path}")
    print(f"Wrote processed CSV   : {processed_path}")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="PCAP -> processed CSV preprocessing")
    parser.add_argument("--pcap", required=True, help="Path to input PCAP")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write packet and processed CSVs",
    )
    args = parser.parse_args(argv)

    preprocess(pcap_path=pathlib.Path(args.pcap), output_dir=pathlib.Path(args.output_dir))


if __name__ == "__main__":
    main()
