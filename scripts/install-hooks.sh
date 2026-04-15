#!/bin/bash
# Install git hooks for the oMLX Hypercar project.
# Run once after cloning: ./scripts/install-hooks.sh

set -e
HOOKS_DIR="$(git rev-parse --show-toplevel)/.git/hooks"

cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/bin/bash
# Pre-commit hook: run fast tests (no GPU required, ~1.5s)
# Skip with: git commit --no-verify (not recommended)
set -e
if ! .venv/bin/python -m pytest --version >/dev/null 2>&1; then
    echo "pytest not found, skipping pre-commit tests"
    exit 0
fi
echo "Running pre-commit tests..."
.venv/bin/python -m pytest tests/ -x -q --tb=line 2>&1
if [ $? -ne 0 ]; then
    echo ""
    echo "Pre-commit tests FAILED. Fix the failures before committing."
    echo "To bypass: git commit --no-verify"
    exit 1
fi
HOOK

chmod +x "$HOOKS_DIR/pre-commit"
echo "Pre-commit hook installed: 106 tests run before every commit (~1.5s)"
