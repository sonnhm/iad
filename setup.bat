@echo off
echo ==================================================================
echo   KHOI TAO MOI TRUONG DU AN: INDUSTRIAL ANOMALY DETECTION (IAD)
echo            SOTA Performance ^| Knowledge Distillation ^| XAI
echo ==================================================================
echo.

echo [1/5] Cap nhat cong cu quan ly goi (pip)...
python -m pip install --upgrade pip

echo.
echo [2/5] Cai dat thu vien tu requirements.txt...
pip install -r requirements.txt

echo.
echo [3/5] Tao cac thu muc can thiet...
if not exist "checkpoints\backbone"   mkdir "checkpoints\backbone"
if not exist "checkpoints\patchcore"  mkdir "checkpoints\patchcore"
if not exist "checkpoints\autoencoder" mkdir "checkpoints\autoencoder"
if not exist "checkpoints\ocsvm"      mkdir "checkpoints\ocsvm"
if not exist "datasets\mvtec"         mkdir "datasets\mvtec"
if not exist "results"                mkdir "results"
if not exist "experiments\logs"       mkdir "experiments\logs"
echo [OK] Thu muc da san sang.

echo.
echo [4/5] Kiem tra he thong kiem thu (Unit Tests)...
python -m pytest tests/test_all.py -v --tb=short

echo.
echo [5/5] Kiem tra bien moi truong (Environment Variables)...
IF NOT EXIST .env (
    echo GEMINI_API_KEY=your_key_here > .env
    echo [!] Da tao file .env. Vui long mo file va dien API Key truoc khi chay app.
) ELSE (
    echo [OK] File .env da ton tai.
)

echo.
echo ==================================================================
echo   CAI DAT THANH CONG!
echo.
echo   BUOC TIEP THEO:
echo   1. [BAT BUOC] Cap nhat GEMINI_API_KEY trong file .env
echo   2. Tai du lieu: python tools/download_mvtec.py
echo   3. Huan luyen va danh gia: python experiments/run_full_benchmark.py --category all
echo   4. Khoi dong Web App: python app.py
echo      Truy cap: http://localhost:5000
echo ==================================================================
pause

