#!/usr/bin/env python3
"""
Train lightweight generative models on the processed CSV produced by preprocess_pcap.py.

We keep the modeling deliberately simple to avoid the heavy dependencies from the
research code:
- Numeric packet features are modeled with a Gaussian Mixture Model.
- Flow lengths, endpoint pairs, ports, and applications are modeled with
  empirical frequency distributions.
"""

import argparse
import pathlib
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


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


def fit_freq(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    counts = values.value_counts()
    return counts.index.values, counts.values / counts.values.sum()


def train_models(df: pd.DataFrame, output_dir: pathlib.Path, n_components: int = 8) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Numeric packet model
    scaler = StandardScaler()
    X = scaler.fit_transform(df[NUMERIC_COLS].astype(float))
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0)
    gmm.fit(X)

    # Empirical distributions for categorical/sequence aspects
    flow_len_values, flow_len_probs = fit_freq(df.groupby("flow_id").size())
    endpoints = df.groupby(["ip_src", "ip_dst"]).size()
    ports = df.groupby(["sport", "dport"]).size()
    apps = df["applications"]
    app_values, app_probs = fit_freq(apps)

    model_artifacts: Dict[str, object] = {
        "scaler": scaler,
        "packet_gmm": gmm,
        "flow_size_values": flow_len_values,
        "flow_size_probs": flow_len_probs,
        "endpoints_values": endpoints.index.values,
        "endpoints_probs": endpoints.values / endpoints.values.sum(),
        "ports_values": ports.index.values,
        "ports_probs": ports.values / ports.values.sum(),
        "apps_values": app_values,
        "apps_probs": app_probs,
    }

    joblib.dump(model_artifacts, output_dir / "traffic_models.joblib")
    print(f"Saved models to {output_dir / 'traffic_models.joblib'}")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Train simple generative models from processed CSV")
    parser.add_argument("--data", required=True, help="Path to packets_processed.csv from preprocess_pcap.py")
    parser.add_argument("--output-dir", required=True, help="Where to store trained models")
    parser.add_argument("--gmm-components", type=int, default=8, help="Number of components for packet GMM")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    train_models(df, pathlib.Path(args.output_dir), n_components=args.gmm_components)


if __name__ == "__main__":
    main()
