#!/usr/bin/env bash
set -euo pipefail

: "${REPO_URL:?You must set REPO_URL}"
REPO_DIR="${REPO_DIR:-/content/vision-uss-research}"

echo "==> REPO_URL: ${REPO_URL}"
echo "==> REPO_DIR: ${REPO_DIR}"

if [ -d "${REPO_DIR}/.git" ]; then
  echo "==> Repo already exists, refreshing"
  cd "${REPO_DIR}"
  git fetch --all
  git reset --hard origin/main
else
  echo "==> Cloning repo"
  git clone "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

echo "==> Installing uv"
python -m pip install -q uv

echo "==> Syncing environment"
uv sync

echo "==> Verifying colab_drive profile"
uv run python scripts/print_paths.py --profile colab_drive

echo "==> Bootstrap complete"