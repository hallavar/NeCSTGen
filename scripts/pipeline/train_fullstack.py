#!/usr/bin/env python3
"""
Train a pared-down version of the original NeCSTGen stack:
- Packet-level VAE + GMM on latent space
- Flow-level LSTM on packet length/time dynamics

Inputs:
  packets_processed.csv from preprocess_pcap.py (must contain flow_id, length_total, time_diff, etc.)

Outputs (in --output-dir):
  - scaler_packet.joblib : StandardScaler for VAE features
  - vae_encoder/, vae_decoder/ : SavedModel Keras models
  - packet_gmm.joblib : GaussianMixture on latent space
  - scaler_flow.joblib : StandardScaler for flow features (length_total, time_diff)
  - lstm_flow/ : SavedModel flow LSTM
  - meta_fullstack.joblib : dict with empirical distributions (flows, endpoints, ports, apps)
  - config.json : hyperparameters used
"""

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Columns used for packet-level VAE
VAE_FEATURES = [
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

# Flow-level features (sequence modeling)
FLOW_FEATURES = ["length_total", "time_diff"]


def build_vae(input_dim: int, latent_dim: int = 8) -> Tuple[keras.Model, keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_m, z_lv = args
        eps = tf.random.normal(shape=(tf.shape(z_m)[0], tf.shape(z_m)[1]))
        return z_m + tf.exp(0.5 * z_lv) * eps

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation=None)(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    outputs = decoder(z)

    # VAE loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(inputs - outputs), axis=1)
    )
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    vae_loss = reconstruction_loss + kl_loss

    vae = keras.Model(inputs, outputs, name="vae")
    vae.add_loss(vae_loss)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    return encoder, decoder, vae


def build_lstm(input_dim: int, hidden: int = 64) -> keras.Model:
    inputs = keras.Input(shape=(None, input_dim))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(hidden, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)
    model = keras.Model(inputs, outputs, name="flow_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def pad_sequences(seqs: List[np.ndarray], max_len: int) -> np.ndarray:
    padded = np.zeros((len(seqs), max_len, seqs[0].shape[1]))
    for i, s in enumerate(seqs):
        length = min(len(s), max_len)
        padded[i, :length, :] = s[:length]
    return padded


def fit_freq(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    counts = values.value_counts()
    return counts.index.values, counts.values / counts.values.sum()


def train_fullstack(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    latent_dim: int = 8,
    gmm_components: int = 8,
    max_flow_len: int = 64,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Packet VAE
    scaler_pkt = StandardScaler()
    X_pkt = scaler_pkt.fit_transform(df[VAE_FEATURES].astype(float))

    encoder, decoder, vae = build_vae(input_dim=X_pkt.shape[1], latent_dim=latent_dim)
    vae.fit(X_pkt, epochs=15, batch_size=256, shuffle=True, verbose=1)
    z_mean, z_log_var, z = encoder.predict(X_pkt, batch_size=512, verbose=0)
    gmm = GaussianMixture(n_components=gmm_components, covariance_type="full", random_state=0)
    gmm.fit(z_mean)

    # Flow LSTM
    flow_seqs: List[np.ndarray] = []
    for _, g in df.groupby("flow_id"):
        if len(g) < 2:
            continue
        arr = g[FLOW_FEATURES].astype(float).values
        flow_seqs.append(arr)

    if not flow_seqs:
        raise RuntimeError("No flows with length >=2 found for LSTM training.")

    # Fit scaler on concatenated flow data
    all_flow = np.concatenate(flow_seqs, axis=0)
    scaler_flow = StandardScaler().fit(all_flow)
    flow_seqs_scaled = [scaler_flow.transform(seq) for seq in flow_seqs]
    padded = pad_sequences(flow_seqs_scaled, max_len=max_flow_len)
    X_lstm = padded[:, :-1, :]
    y_lstm = padded[:, 1:, :]

    lstm = build_lstm(input_dim=len(FLOW_FEATURES))
    lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=64, verbose=1)

    # Save artifacts
    joblib.dump(scaler_pkt, output_dir / "scaler_packet.joblib")
    joblib.dump(gmm, output_dir / "packet_gmm.joblib")
    joblib.dump(scaler_flow, output_dir / "scaler_flow.joblib")
    encoder.save(output_dir / "vae_encoder")
    decoder.save(output_dir / "vae_decoder")
    lstm.save(output_dir / "lstm_flow")

    # Empirical distributions
    flow_len_vals, flow_len_probs = fit_freq(df.groupby("flow_id").size())
    endpoints = df.groupby(["ip_src", "ip_dst"]).size()
    ports = df.groupby(["sport", "dport"]).size()
    apps_vals, apps_probs = fit_freq(df["applications"])

    meta: Dict[str, object] = {
        "flow_size_values": flow_len_vals,
        "flow_size_probs": flow_len_probs,
        "endpoints_values": endpoints.index.values,
        "endpoints_probs": endpoints.values / endpoints.values.sum(),
        "ports_values": ports.index.values,
        "ports_probs": ports.values / ports.values.sum(),
        "apps_values": apps_vals,
        "apps_probs": apps_probs,
    }
    joblib.dump(meta, output_dir / "meta_fullstack.joblib")

    cfg = {
        "latent_dim": latent_dim,
        "gmm_components": gmm_components,
        "max_flow_len": max_flow_len,
        "vae_features": VAE_FEATURES,
        "flow_features": FLOW_FEATURES,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved models and metadata to {output_dir}")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Train VAE+GMM+LSTM stack on processed packets")
    parser.add_argument("--data", required=True, help="Path to packets_processed.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to store trained models")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--max-flow-len", type=int, default=64)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    train_fullstack(
        df=df,
        output_dir=pathlib.Path(args.output_dir),
        latent_dim=args.latent_dim,
        gmm_components=args.gmm_components,
        max_flow_len=args.max_flow_len,
    )


if __name__ == "__main__":
    main()
