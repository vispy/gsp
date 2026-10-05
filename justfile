docs_host := env("GSP_DOCS_HOST", "")
docs_port := env("GSP_DOCS_PORT", "8296")
review_python := env("VISPY2_REVIEW_PYTHON", ".venv/bin/python")

# Review this repo together with its sibling VisPy2 producer.
[positional-arguments]
review *args:
    @"{{review_python}}" ../vispy2/tools/review.py "$@"

review-setup:
    @uv pip install --python "{{review_python}}" 'PySide6>=6.8,<7'

lint:
    @uvx --from 'ruff==0.16.1' ruff check packages conformance tools
    @uvx --from 'ruff==0.16.1' ruff format --check packages conformance tools

format:
    @uvx --from 'ruff==0.16.1' ruff check --fix packages conformance tools
    @uvx --from 'ruff==0.16.1' ruff format packages conformance tools

spec-check:
    @uv run --no-project python tools/spec_traceability.py --check

pre-commit-check: lint
    @git diff --check
    @git diff --cached --check

_docs-command command:
    @uv run --no-project --with 'mkdocs-material==9.7.7' mkdocs {{command}}

docs-build-check:
    @just _docs-command "build --strict"

serve:
    #!/usr/bin/env bash
    set -euo pipefail
    host="{{docs_host}}"
    tailnet_host=""
    if command -v tailscale >/dev/null 2>&1; then
        tailnet_ip=$(tailscale ip -4 2>/dev/null || true)
        if [ -z "$host" ]; then
            host="$tailnet_ip"
        fi
        if [ -n "$tailnet_ip" ] && [ "$host" = "$tailnet_ip" ]; then
            tailnet_host=$(tailscale status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
        fi
    fi
    host="${host:-127.0.0.1}"
    display_host="${tailnet_host:-$host}"
    echo "GSP documentation: http://${display_host}:{{docs_port}}/"
    uv run --no-project --with 'mkdocs-material==9.7.7' mkdocs serve -a "${host}:{{docs_port}}"

docs-serve:
    @just serve
