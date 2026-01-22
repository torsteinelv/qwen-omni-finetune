# Project Plan: Norwegian Voice Fine-tuning for Qwen-Omni

The primary goal of this project is to enable **Qwen2.5-Omni-3B** to generate natural Norwegian speech by training the model's "Thinker" component to predict the correct **Mimi** audio tokens.

---

## 📋 Execution Plan

### Background & Previous Attempt
* **Initial Training**: We previously conducted a full training run using the **Google FLEURS** dataset.
* **Outcome**: The results were unsatisfactory, with poor audio quality and unnatural speech patterns.
* **The Pivot**: Based on these results, we decided to restart the training using a larger, higher-quality dataset (NPSC) with a focus on better audio fidelity (24kHz) and natural prosody.

### Phase 1: Data Acquisition & Refinement
* **Primary Source**: Utilizing the **NPSC (National Parliamentary Speech Corpus)** to ensure high-fidelity Norwegian audio data.
* **Processing**: Resampling all audio to 24kHz to meet the requirements of the Mimi encoder.
* **Selection**: Targeting a dataset of approximately 15,000 clips to provide the model with enough variety for high-quality synthesis.

### Phase 2: Audio Tokenization (Talker Preparation)
* **Encoding**: Using the Mimi model to convert raw Norwegian waveforms into discrete audio tokens.
* **Dataset Formatting**: Generating `talker_data.jsonl` where the speech is represented as tokens between `<|audio_bos|>` and `<|audio_eos|>` tags.

### Phase 3: Fine-tuning with LoRA & Quantization
* **Efficiency**: Implementing 8-bit quantization and LoRA (Low-Rank Adaptation) to train the model effectively on available hardware.
* **Architecture**: Utilizing the `QwenOmniWrapper` to manage multimodal inputs and ensure stable checkpoint saving.

### Phase 4: Validation & Testing
* **Inference**: Running `test_speak.py` to generate Norwegian speech from text and verify the quality of the new adapter.
* **Benchmarking**: Comparing the output against the original base model to document improvements.

---

## 🚦 Status Report

### ✅ Completed
* **Infrastructure**: Docker environment is fully configured with `transformers`, `peft`, and `bitsandbytes`.
* **Automation**: The `entrypoint.sh` pipeline is finalized, connecting data processing, tokenization, and training.
* **Data Scripts**: Refined scripts for NPSC data extraction and Mimi tokenization are operational.
* **CI/CD**: GitHub Actions are set up for automated Docker builds.

### ⏳ Ongoing / Next Steps
* **Full-scale Training**: Currently executing the new training run on the 15,000 NPSC samples.
* **Hyperparameter Tuning**: Monitoring the training loss and adjusting the learning rate for the "Talker-only" phase.
* **Quality Assurance**: Evaluating generated audio samples to ensure the model correctly captures Norwegian phonemes and intonation.
