#!/usr/bin/env python3
"""
Incremental training / fine-tuning for the simplified NeCSTGen stack.

Given a packets_processed.csv and a directory holding per-application models
produced by train_fullstack.py, this script will:
  * If an application already has models, warm start the VAE, LSTM, and scalers
    from the existing weights and continue training on the new data.
  * If an application is not covered, train new VAE+GMM+LSTM models for it.

Output is written to a new directory (per application) so the original
models remain untouched.
"""

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DEFAULT_VAE_FEATURES = [
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

DEFAULT_FLOW_FEATURES = ["length_total", "time_diff"]


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
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=1))
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


def load_existing_models(app_dir: pathlib.Path):
    """Load existing artifacts if present; return None if missing."""
    cfg_path = app_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        scaler_pkt = joblib.load(app_dir / "scaler_packet.joblib")
        scaler_flow = joblib.load(app_dir / "scaler_flow.joblib")
        gmm = joblib.load(app_dir / "packet_gmm.joblib")
        encoder = tf.keras.models.load_model(app_dir / "vae_encoder")
        decoder = tf.keras.models.load_model(app_dir / "vae_decoder")
        lstm = tf.keras.models.load_model(app_dir / "lstm_flow")
        meta = joblib.load(app_dir / "meta_fullstack.joblib")
    except Exception:
        return None

    return {
        "cfg": cfg,
        "scaler_pkt": scaler_pkt,
        "scaler_flow": scaler_flow,
        "gmm": gmm,
        "encoder": encoder,
        "decoder": decoder,
        "lstm": lstm,
        "meta": meta,
    }


def train_or_finetune_for_app(
    df_app: pd.DataFrame,
    base_app_dir: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    latent_dim: int,
    gmm_components: int,
    max_flow_len: int,
    vae_features: List[str],
    flow_features: List[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_models(base_app_dir) if base_app_dir and base_app_dir.exists() else None

    # Scalers
    if existing:
        scaler_pkt = existing["scaler_pkt"]
        scaler_flow = existing["scaler_flow"]
    else:
        scaler_pkt = StandardScaler()
        scaler_flow = StandardScaler()

    X_pkt_raw = df_app[vae_features].astype(float).values
    X_pkt = scaler_pkt.fit_transform(X_pkt_raw) if not existing else scaler_pkt.transform(X_pkt_raw)

    # VAE
    encoder, decoder, vae = build_vae(input_dim=X_pkt.shape[1], latent_dim=latent_dim)
    if existing:
        try:
            encoder.set_weights(existing["encoder"].get_weights())
            decoder.set_weights(existing["decoder"].get_weights())
        except Exception:
            pass
    vae.fit(X_pkt, epochs=10, batch_size=256, shuffle=True, verbose=1)
    z_mean, _, _ = encoder.predict(X_pkt, batch_size=512, verbose=0)

    # GMM (re-fit on new latent; sklearn has no warm start)
    gmm = GaussianMixture(n_components=gmm_components, covariance_type="full", random_state=0)
    gmm.fit(z_mean)

    # Flow sequences
    flow_seqs = []
    for _, g in df_app.groupby("flow_id"):
        if len(g) < 2:
            continue
        flow_seqs.append(g[flow_features].astype(float).values)
    if not flow_seqs:
        raise RuntimeError(f"No flows with >=2 packets for app {df_app['applications'].iloc[0]}")

    all_flow = np.concatenate(flow_seqs, axis=0)
    if existing:
        flow_seqs_scaled = [scaler_flow.transform(seq) for seq in flow_seqs]
    else:
        scaler_flow.fit(all_flow)
        flow_seqs_scaled = [scaler_flow.transform(seq) for seq in flow_seqs]

    padded = pad_sequences(flow_seqs_scaled, max_len=max_flow_len)
    X_lstm = padded[:, :-1, :]
    y_lstm = padded[:, 1:, :]

    lstm = build_lstm(input_dim=len(flow_features))
    if existing:
        try:
            lstm.set_weights(existing["lstm"].get_weights())
        except Exception:
            pass
    lstm.fit(X_lstm, y_lstm, epochs=8, batch_size=64, verbose=1)

    # Empirical distributions
    flow_len_vals, flow_len_probs = fit_freq(df_app.groupby("flow_id").size())
    endpoints = df_app.groupby(["ip_src", "ip_dst"]).size()
    ports = df_app.groupby(["sport", "dport"]).size()
    apps_vals, apps_probs = fit_freq(df_app["applications"])

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

    # Save
    joblib.dump(scaler_pkt, output_dir / "scaler_packet.joblib")
    joblib.dump(gmm, output_dir / "packet_gmm.joblib")
    joblib.dump(scaler_flow, output_dir / "scaler_flow.joblib")
    encoder.save(output_dir / "vae_encoder")
    decoder.save(output_dir / "vae_decoder")
    lstm.save(output_dir / "lstm_flow")
    joblib.dump(meta, output_dir / "meta_fullstack.joblib")

    cfg = {
        "latent_dim": latent_dim,
        "gmm_components": gmm_components,
        "max_flow_len": max_flow_len,
        "vae_features": vae_features,
        "flow_features": flow_features,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved fine-tuned/trained models for app to {output_dir}")


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune or train VAE+GMM+LSTM per application.")
    parser.add_argument("--data", required=True, help="packets_processed.csv")
    parser.add_argument("--base-models-dir", required=True, help="Directory containing existing per-app models")
    parser.add_argument("--output-root", required=True, help="Where to write updated models")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--max-flow-len", type=int, default=64)
    parser.add_argument("--app-column", default="applications")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.app_column not in df.columns:
        raise ValueError(f"Column {args.app_column} not found in data.")

    base_root = pathlib.Path(args.base_models_dir)
    out_root = pathlib.Path(args.output_root)

    for app in df[args.app_column].astype(str).unique():
        df_app = df[df[args.app_column].astype(str) == app].reset_index(drop=True)
        app_sanitized = str(app).replace(" ", "_")
        base_app_dir = base_root / app_sanitized
        out_app_dir = out_root / app_sanitized

        # Determine feature lists (fallback to defaults)
        if base_app_dir.exists() and (base_app_dir / "config.json").exists():
            cfg = json.loads((base_app_dir / "config.json").read_text())
            vae_features = cfg.get("vae_features", DEFAULT_VAE_FEATURES)
            flow_features = cfg.get("flow_features", DEFAULT_FLOW_FEATURES)
            latent_dim = cfg.get("latent_dim", args.latent_dim)
            gmm_components = cfg.get("gmm_components", args.gmm_components)
            max_flow_len = cfg.get("max_flow_len", args.max_flow_len)
        else:
            vae_features = DEFAULT_VAE_FEATURES
            flow_features = DEFAULT_FLOW_FEATURES
            latent_dim = args.latent_dim
            gmm_components = args.gmm_components
            max_flow_len = args.max_flow_len

        print(f"Processing app '{app}' -> base models at {base_app_dir if base_app_dir.exists() else 'new'}")
        train_or_finetune_for_app(
            df_app=df_app,
            base_app_dir=base_app_dir if base_app_dir.exists() else None,
            output_dir=out_app_dir,
            latent_dim=latent_dim,
            gmm_components=gmm_components,
            max_flow_len=max_flow_len,
            vae_features=vae_features,
            flow_features=flow_features,
        )


if __name__ == "__main__":
    main()
