#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NECSTGEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${NECSTGEN_DIR}/../.." && pwd)"

PCAP_PATH="${REPO_ROOT}/datasets/train.pcap"
RUN_NAME="${RUN_NAME:-train_pcap}"
DATA_DIR="${NECSTGEN_DIR}/data/${RUN_NAME}"
PROC_CSV="${DATA_DIR}/packets_processed.csv"
MODELS_DIR="${NECSTGEN_DIR}/models/${RUN_NAME}_fullstack"
OUTPUT_DIR="${NECSTGEN_DIR}/output/${RUN_NAME}"
OUTPUT_PCAP="${OUTPUT_DIR}/synthetic.pcap"
LATENT_DIM="${LATENT_DIM:-8}"
GMM_COMPONENTS="${GMM_COMPONENTS:-8}"
MAX_FLOW_LEN="${MAX_FLOW_LEN:-64}"
NUM_FLOWS="${NUM_FLOWS:-50}"

if [[ ! -f "${PCAP_PATH}" ]]; then
  echo "Missing ${PCAP_PATH}. Copy datasets/train.pcap first." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${OUTPUT_DIR}"

echo "[NeCSTGen] Preprocessing ${PCAP_PATH}"
python3 "${NECSTGEN_DIR}/scripts/pipeline/preprocess_pcap.py" \
  --pcap "${PCAP_PATH}" \
  --output-dir "${DATA_DIR}"

echo "[NeCSTGen] Training models into ${MODELS_DIR}"
python3 "${NECSTGEN_DIR}/scripts/pipeline/train_fullstack.py" \
  --data "${PROC_CSV}" \
  --output-dir "${MODELS_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --gmm-components "${GMM_COMPONENTS}" \
  --max-flow-len "${MAX_FLOW_LEN}"

echo "[NeCSTGen] Generating synthetic PCAP (${OUTPUT_PCAP})"
python3 "${NECSTGEN_DIR}/scripts/pipeline/generate_fullstack.py" \
  --models-dir "${MODELS_DIR}" \
  --output-pcap "${OUTPUT_PCAP}" \
  --num-flows "${NUM_FLOWS}"

echo "Synthetic PCAP available at ${OUTPUT_PCAP}"
