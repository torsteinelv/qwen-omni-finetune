#!/bin/bash
set -e 

echo "============================================="
echo "    STARTER QWEN-2.5-OMNI FINETUNE PIPELINE   "
echo "============================================="

echo "[1/2] Kjører data-forberedelse (data.py)..."
#python data.py
python data_npsc.py

# Lagt til python foran her:
echo "[1.5/2] Koder lyd til tokens (prepare_talker.py)..."
#python prepare_talker.py
python prepare_talker_instruct.py
# Sjekk talker_data siden det er den vi skal trene på nå
if [ -f "./norsk_data/talker_data.jsonl" ]; then
    echo "Data OK! Fant talker_data.jsonl."
else
    echo "FEIL: talker_data.jsonl ble ikke funnet. Stopper."
    exit 1
fi

echo "============================================="
echo "[2/2] Starter trening (train_talker.py)..."
python train_talker_only.py "$@"
#python train_talker_only_patch.py
echo "============================================="
echo "    JOBB FULLFØRT! Sjekk /workspace/output_talker    "
echo "============================================="
