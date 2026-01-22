#!/bin/bash
set -e 

echo "============================================="
echo "    STARTER QWEN-2.5-OMNI FINETUNE PIPELINE   "
echo "    (Nå med Instruction Tuning!)             "
echo "============================================="

echo "[1/3] Kjører data-forberedelse (data_npsc.py)..."
# Laster ned rådata fra Hugging Face hvis de ikke finnes
python data_npsc.py

echo "============================================="
echo "[2/3] Genererer smarte instruksjoner (prepare_talker_instruct.py)..."

# Kjører scriptet som lager instruksjons-dataene
python prepare_talker_instruct.py

# VIKTIG FIX: Flytt og døp om filen så trenings-scriptet finner den!
# Vi overskriver den gamle talker_data.jsonl med den nye "smarte" versjonen.
if [ -f "talker_data_instruct.jsonl" ]; then
    echo "Flytter data til ./norsk_data/talker_data.jsonl..."
    mkdir -p ./norsk_data
    mv talker_data_instruct.jsonl ./norsk_data/talker_data.jsonl
fi

# Sjekk at filen faktisk ligger der treningsscriptet forventer den
if [ -f "./norsk_data/talker_data.jsonl" ]; then
    echo "✅ Data OK! Fant talker_data.jsonl (Instruction Tuned Edition)."
else
    echo "❌ FEIL: talker_data.jsonl mangler i ./norsk_data/. Stopper."
    exit 1
fi

echo "============================================="
echo "[3/3] Starter trening (train_talker_only.py)..."

# Tips: Sjekk at train_talker_only.py peker på riktig jsonl-fil, 
# eller at den laster fra ./norsk_data/talker_data.jsonl som default.
python train_talker_only.py "$@"

echo "============================================="
echo "    JOBB FULLFØRT! Sjekk /workspace/output_talker     "
echo "============================================="
