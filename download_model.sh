#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/tiny-aya-earth-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/CohereLabs/tiny-aya-earth-GGUF/resolve/main/tiny-aya-earth-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"
if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi
echo "downloading $MODEL_URL → $MODEL_FILE (~2.14 GB)…"
if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi
mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE"
