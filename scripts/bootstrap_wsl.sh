#!/usr/bin/env bash
set -euo pipefail

# Recreate the verified Ubuntu 22.04 / Python 3.10 experiment environment.
# Override these variables only when the workstation layout requires it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DS340W_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${DS340W_VENV_DIR:-${HOME}/.venvs/ds340w}"
PYTHON_BIN="${DS340W_PYTHON_BIN:-python3.10}"
TORCH_INDEX_URL="${DS340W_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

readonly EXPECTED_PYTHON_MINOR="3.10"
readonly EXPECTED_TORCH="2.1.2+cu121"
readonly EXPECTED_OPENDELTA_COMMIT="20e426f716ea53618657cd34bbfc4b347b98cd74"

fail() {
  printf 'bootstrap error: %s\n' "$*" >&2
  exit 1
}

install_system_packages() {
  local -a required=(python3.10 python3.10-venv python3-tk git)
  local -a missing=()
  local package

  command -v apt-get >/dev/null 2>&1 || fail "apt-get is required (Ubuntu 22.04 expected)"
  for package in "${required[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null \
      | grep -q '^install ok installed$'; then
      missing+=("${package}")
    fi
  done
  ((${#missing[@]} == 0)) && return

  local -a apt_command=(apt-get)
  if ((EUID != 0)); then
    command -v sudo >/dev/null 2>&1 \
      || fail "sudo is required to install: ${missing[*]}"
    apt_command=(sudo apt-get)
  fi
  "${apt_command[@]}" update
  "${apt_command[@]}" install -y "${missing[@]}"
}

[[ -f "${PROJECT_ROOT}/requirements.lock" ]] \
  || fail "requirements.lock not found under ${PROJECT_ROOT}"
[[ -f "${PROJECT_ROOT}/pyproject.toml" ]] \
  || fail "pyproject.toml not found under ${PROJECT_ROOT}"

install_system_packages
[[ -f "${PROJECT_ROOT}/.gitmodules" ]] \
  || fail ".gitmodules not found under ${PROJECT_ROOT}"
git -C "${PROJECT_ROOT}" submodule update --init --recursive -- vendor/OpenDelta
UPSTREAM_PATH="${PROJECT_ROOT}/vendor/OpenDelta"
actual_opendelta_commit="$(git -C "${UPSTREAM_PATH}" rev-parse HEAD)"
[[ "${actual_opendelta_commit}" == "${EXPECTED_OPENDELTA_COMMIT}" ]] \
  || fail "expected OpenDelta ${EXPECTED_OPENDELTA_COMMIT}, found ${actual_opendelta_commit}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || fail "${PYTHON_BIN} is unavailable after system package installation"
command -v nvidia-smi >/dev/null 2>&1 \
  || fail "nvidia-smi is unavailable inside WSL"
nvidia-smi >/dev/null \
  || fail "the NVIDIA driver/GPU is not usable inside WSL"

if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  fail "${VENV_DIR} exists but is not a usable virtual environment"
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  mkdir -p -- "$(dirname -- "${VENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
actual_python_minor="$("${VENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${actual_python_minor}" == "${EXPECTED_PYTHON_MINOR}" ]] \
  || fail "expected Python ${EXPECTED_PYTHON_MINOR}, found ${actual_python_minor}"

# Fetch the CUDA wheel from PyTorch's official CUDA 12.1 index, then install
# the complete environment lock. Re-running these commands is intentionally a
# no-op when the virtual environment already matches the lock.
"${VENV_PYTHON}" -m pip --disable-pip-version-check install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${EXPECTED_TORCH}"
"${VENV_PYTHON}" -m pip --disable-pip-version-check install \
  -r "${PROJECT_ROOT}/requirements.lock"
"${VENV_PYTHON}" -m pip --disable-pip-version-check install \
  --no-deps -e "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m pip check
PYTHONPATH="${PROJECT_ROOT}/src" \
  "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/fetch_pinned_inputs.py" --sync
PYTHONPATH="${PROJECT_ROOT}/src" \
  "${VENV_PYTHON}" -m ds340w_repro.data.materialize_sst2

LEGACY_PATH="${UPSTREAM_PATH}/examples/legacies"
PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_PATH}:${LEGACY_PATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV_PYTHON}" -c \
  'from examples_seq2seq.run_seq2seq import main; assert callable(main)'
"${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/probe_environment.py" >/dev/null

printf 'Environment ready: %s\n' "${VENV_DIR}"
printf 'Activate with: source %q\n' "${VENV_DIR}/bin/activate"
