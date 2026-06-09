#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NECSTGEN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${NECSTGEN_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
NECSTGEN_VENV="${NECSTGEN_VENV:-${REPO_ROOT}/benchmark_programs/envs/necstgen}"
if [[ -x "${NECSTGEN_VENV}/bin/python" ]] && "${NECSTGEN_VENV}/bin/python" -c "pass" >/dev/null 2>&1; then
  PYTHON_BIN="${NECSTGEN_VENV}/bin/python"
else
  echo "[NeCSTGen] Skipping unusable venv: ${NECSTGEN_VENV}" >&2
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "${REPO_ROOT}"

DATASETS_DIR="${DATASETS_DIR:-/media/datasets/campus_traffic/dataset_april_imc_paper}"
PCAP_PATH="${TRAIN_PCAP:-${DATASETS_DIR}/train.pcap}"
TEST_PCAP="${TEST_PCAP:-${DATASETS_DIR}/test.pcap}"
RUN_NAME="${RUN_NAME:-train_pcap}"
DATA_DIR="${NECSTGEN_DIR}/data/${RUN_NAME}"
PROC_CSV="${DATA_DIR}/packets_processed.csv"
MODELS_DIR="${NECSTGEN_DIR}/models/${RUN_NAME}_fullstack"
OUTPUT_DIR="${NECSTGEN_DIR}/output/${RUN_NAME}"
OUTPUT_PCAP="${OUTPUT_DIR}/synthetic.pcap"
FINAL_PCAP="${FINAL_PCAP:-${DATASETS_DIR}/necstgen.pcap}"
LATENT_DIM="${LATENT_DIM:-8}"
GMM_COMPONENTS="${GMM_COMPONENTS:-8}"
MAX_FLOW_LEN="${MAX_FLOW_LEN:-64}"
TRAFFIC_DURATION_SECONDS="${TRAFFIC_DURATION_SECONDS:-600}"
TARGET_FLOWS="${TARGET_FLOWS:-}"
WORKERS="${WORKERS:-1}"

if [[ ! -f "${PCAP_PATH}" ]]; then
  echo "Missing training PCAP: ${PCAP_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TEST_PCAP}" ]]; then
  echo "Missing test/reference PCAP: ${TEST_PCAP}" >&2
  exit 1
fi

if [[ -z "${TARGET_FLOWS:-}" ]]; then
  echo "[NeCSTGen] Counting target flows in first ${TRAFFIC_DURATION_SECONDS}s of ${TEST_PCAP}"
  TARGET_FLOWS="$("${PYTHON_BIN}" "${REPO_ROOT}/benchmark_programs/utils/pcap_flow_count.py" \
    --pcap "${TEST_PCAP}" --duration "${TRAFFIC_DURATION_SECONDS}")"
  echo "[NeCSTGen] Target flow count: ${TARGET_FLOWS}"
else
  echo "[NeCSTGen] Using TARGET_FLOWS=${TARGET_FLOWS}"
fi

NUM_FLOWS="${NUM_FLOWS:-${TARGET_FLOWS}}"
echo "[NeCSTGen] Using NUM_FLOWS=${NUM_FLOWS}"

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
  --max-flow-len "${MAX_FLOW_LEN}"

echo "[NeCSTGen] Generating synthetic PCAP (${OUTPUT_PCAP})"
"${PYTHON_BIN}" "${NECSTGEN_DIR}/scripts/pipeline/generate_fullstack.py" \
  --models-dir "${MODELS_DIR}" \
  --output-pcap "${OUTPUT_PCAP}" \
  --num-flows "${NUM_FLOWS}" \
  --workers "${WORKERS}"

echo "[NeCSTGen] Trimming to ${TRAFFIC_DURATION_SECONDS}s and rebasing to test.pcap"
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark_programs/utils/trim_pcap.py" \
  --input "${OUTPUT_PCAP}" \
  --output "${FINAL_PCAP}" \
  --duration "${TRAFFIC_DURATION_SECONDS}" \
  --reference-pcap "${TEST_PCAP}"

echo "Synthetic PCAP available at ${FINAL_PCAP}"
