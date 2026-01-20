#!/bin/bash
set -e  # Stopp hvis noe feiler

echo "============================================="
echo "   STARTER QWEN-2.5-OMNI FINETUNE PIPELINE   "
echo "============================================="

echo "[1/2] Kjører data-forberedelse (data.py)..."
python data.py

# Sjekk om datafilen faktisk ble laget
if [ -f "./norsk_data/train.jsonl" ]; then
    echo "Data OK! Fant train.jsonl."
else
    echo "FEIL: train.jsonl ble ikke funnet. Stopper."
    exit 1
fi

echo "============================================="
echo "[2/2] Starter trening (train.py)..."
# Vi sender eventuelle argumenter videre til train.py
python train.py "$@"

echo "============================================="
echo "   JOBB FULLFØRT! Sjekk /workspace/output    "
echo "============================================="
