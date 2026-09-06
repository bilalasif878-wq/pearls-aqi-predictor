#!/usr/bin/env bash
# One-shot local setup — creates .venv, installs, seeds .env.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env — please edit it and add your real tokens."
fi

echo ""
echo "Done. To activate later: source .venv/bin/activate"
