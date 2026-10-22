#!/usr/bin/env bash
# Build the documentation site and deploy it to agentorchestrationprocess.com.
# Reads CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN from .env, which Git ignores.
set -euo pipefail
cd "$(dirname "$0")"

set -a
. ./.env
set +a

./ci/install-zola.sh >/dev/null
python3 ci/build_docs.py
(cd docs-site && ../.cache/zola/0.23.4/zola build)
npx --yes wrangler@4.69.0 deploy
