#!/bin/bash
set -e
echo "=================================================================="
echo "  KHOI TAO MOI TRUONG DU AN: INDUSTRIAL ANOMALY DETECTION (IAD)"
echo "           SOTA Performance | Knowledge Distillation | XAI"
echo "=================================================================="
echo ""

echo "[1/5] Kiem tra thu muc ao (Virtual Environment)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Da tao .venv."
fi
source .venv/bin/activate
echo "[OK] Da kich hoat .venv."

echo ""
echo "[2/5] Cai dat thu vien tu requirements.txt..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[3/5] Tao cac thu muc can thiet..."
mkdir -p checkpoints/backbone
mkdir -p checkpoints/patchcore
mkdir -p checkpoints/autoencoder
mkdir -p checkpoints/ocsvm
mkdir -p datasets/mvtec
mkdir -p results
mkdir -p experiments/logs
echo "[OK] Thu muc da san sang."

echo ""
echo "[4/5] Kiem tra he thong kiem thu (Unit Tests)..."
python3 -m pytest tests/test_all.py -v --tb=short

echo ""
echo "[5/5] Kiem tra bien moi truong (Environment Variables)..."
if [ ! -f ".env" ]; then
    echo "GEMINI_API_KEY=your_key_here" > .env
    echo "[!] Da tao file .env. Vui long dien GEMINI_API_KEY truoc khi chay app."
else
    echo "[OK] File .env da ton tai."
fi

echo ""
echo "=================================================================="
echo "  CAI DAT THANH CONG!"
echo ""
echo "  BUOC TIEP THEO:"
echo "  1. [BAT BUOC] Cap nhat GEMINI_API_KEY trong file .env"
echo "  2. Tai du lieu: python3 tools/download_mvtec.py"
echo "  3. Huan luyen va danh gia: python3 experiments/run_full_benchmark.py --category all"
echo "  4. Khoi dong Web App: python3 app.py"
echo "     Truy cap: http://localhost:5000"
echo "=================================================================="

