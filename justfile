default: help

# ===== Git =====

commit msg="update":
    git add -A && git commit -m "msg"

push:
    git push

cp msg="update": (commit msg) push

# ===== Dependencies =====

# Full sync (downloads dependencies)
install:
    uv sync

install-qt:
    uv pip install -r requirements_qt.txt

# ===== Run =====

run:
    python pyct_app.py

run-cli:
    python ct_cli.py

calibrate path="/home/foods/pro/data/20260327-jz-1/":
    python -m algorithm.calibration.cal --proj-path {{path}}

# ===== Build (Windows only) =====

# First build: create venv + install deps + package + zip
[windows]
build:
    powershell -ExecutionPolicy RemoteSigned -File scripts/build_win.ps1

# Fast rebuild: skip dependency download, reuse existing venv
[windows]
rebuild:
    powershell -ExecutionPolicy RemoteSigned -File scripts/build_win.ps1 -SkipVenv

# Clean build artifacts
[windows]
build-clean:
    powershell -ExecutionPolicy RemoteSigned -File scripts/build_win.ps1 -Clean

# ===== Check & Test =====

check:
    python -m py_compile pyct_app.py
    python -m py_compile qt_gui/gui.py
    python -m py_compile qt_gui/reconstruction.py
    python -m py_compile algorithm/astra/conebeam.py
    python -m py_compile algorithm/calibration/cal.py

test:
    python ct_cli_test.py
    python thread_test.py
    python detector_test.py

clean:
    rm -rf build_output dist build *.egg-info
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ===== Help =====

help:
    @echo "PyCT Commands:"
    @echo ""
    @echo "  Run:"
    @echo "    just run                  - Start GUI"
    @echo "    just run-cli              - CLI reconstruction"
    @echo "    just calibrate            - Run calibration"
    @echo ""
    @echo "  Build (Windows):"
    @echo "    just build                - Full build (first time)"
    @echo "    just rebuild              - Fast rebuild (no download)"
    @echo "    just build-clean          - Clean build artifacts"
    @echo ""
    @echo "  Dev:"
    @echo "    just install              - Sync dependencies"
    @echo "    just check                - Syntax check"
    @echo "    just test                 - Run tests"
    @echo "    just clean                - Clean all artifacts"
    @echo ""
    @echo "  Git:"
    @echo "    just commit msg='...'     - Commit"
    @echo "    just push                 - Push"
    @echo "    just cp msg='...'         - Commit + Push"
