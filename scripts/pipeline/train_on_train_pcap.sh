#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NECSTGEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${NECSTGEN_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
NECSTGEN_VENV="${NECSTGEN_VENV:-${REPO_ROOT}/benchmark_programs/envs/necstgen}"
if [[ -x "${NECSTGEN_VENV}/bin/python" ]] && "${NECSTGEN_VENV}/bin/python" -c "pass" >/dev/null 2>&1; then
  PYTHON_BIN="${NECSTGEN_VENV}/bin/python"
  # Conda's legacy CUDA/cuDNN libraries are isolated to this model process.
  export LD_LIBRARY_PATH="${NECSTGEN_VENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "[NeCSTGen] Skipping unusable venv: ${NECSTGEN_VENV}" >&2
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "${REPO_ROOT}"

DATASETS_DIR="${DATASETS_DIR:-/media/datasets/aschoen_datasets/dataset_campus}"
PCAP_PATH="${TRAIN_PCAP:-${DATASETS_DIR}/train.pcap}"
REFERENCE_PCAP="${REFERENCE_PCAP:-${TEST_PCAP:-${DATASETS_DIR}/reference.pcap}}"
RUN_NAME="${RUN_NAME:-campus_reference_size}"
DATA_DIR="${NECSTGEN_DATA_DIR:-${NECSTGEN_DIR}/data/${RUN_NAME}}"
PROC_CSV="${DATA_DIR}/packets_processed.csv"
MODELS_DIR="${NECSTGEN_MODELS_DIR:-${NECSTGEN_DIR}/models/${RUN_NAME}_fullstack}"
OUTPUT_DIR="${NECSTGEN_OUTPUT_DIR:-${NECSTGEN_DIR}/output/${RUN_NAME}}"
FINAL_PCAP="${FINAL_PCAP:-${DATASETS_DIR}/necstgen.pcap}"
LATENT_DIM="${LATENT_DIM:-8}"
GMM_COMPONENTS="${GMM_COMPONENTS:-8}"
MAX_FLOW_LEN="${MAX_FLOW_LEN:-64}"
TARGET_BYTES="${TARGET_BYTES:-0}"
WORKERS="${WORKERS:-1}"

if [[ "${1:-}" == "--generate-batch" ]]; then
  "${PYTHON_BIN}" "${NECSTGEN_DIR}/scripts/pipeline/generate_fullstack.py" \
    --models-dir "${MODELS_DIR}" --output-pcap "$4" --num-flows "$2" --workers "${WORKERS}"
  exit 0
fi

if [[ ! -f "${PCAP_PATH}" ]]; then
  echo "Missing training PCAP: ${PCAP_PATH}" >&2
  exit 1
fi

TARGET_BYTES="$("${PYTHON_BIN}" "${REPO_ROOT}/benchmark_programs/utils/pcap_size.py" target \
  --reference "${REFERENCE_PCAP}" --target-bytes "${TARGET_BYTES}")"
echo "[NeCSTGen] Target PCAP size: ${TARGET_BYTES} bytes"

mkdir -p "${DATA_DIR}" "${MODELS_DIR}" "${OUTPUT_DIR}"

echo "[NeCSTGen] Preprocessing ${PCAP_PATH}"
"${PYTHON_BIN}" "${NECSTGEN_DIR}/scripts/pipeline/preprocess_pcap.py" \
  --pcap "${PCAP_PATH}" \
  --output-dir "${DATA_DIR}"

echo "[NeCSTGen] Training models into ${MODELS_DIR}"
"${PYTHON_BIN}" "${NECSTGEN_DIR}/scripts/pipeline/train_fullstack.py" \
  --data "${PROC_CSV}" \
  --output-dir "${MODELS_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --gmm-components "${GMM_COMPONENTS}" \
  --max-flow-len "${MAX_FLOW_LEN}" \
  --vae-batch-size "${NECSTGEN_VAE_BATCH_SIZE:-256}" \
  --encoder-batch-size "${NECSTGEN_ENCODER_BATCH_SIZE:-512}" \
  --lstm-batch-size "${NECSTGEN_LSTM_BATCH_SIZE:-64}"

export PYTHON_BIN RUN_NAME WORKERS
export NECSTGEN_DATA_DIR="${DATA_DIR}" NECSTGEN_MODELS_DIR="${MODELS_DIR}"
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark_programs/utils/pcap_size.py" generate \
  --reference "${REFERENCE_PCAP}" --output "${FINAL_PCAP}" --target-bytes "${TARGET_BYTES}" \
  --batch-units "${NUM_FLOWS:-512}" --max-batches "${MAX_GENERATION_BATCHES:-1000}" \
  -- bash "${SCRIPT_DIR}/train_on_train_pcap.sh" --generate-batch '{count}' '{batch_dir}' '{output}'

echo "Synthetic PCAP available at ${FINAL_PCAP}"
