# Project Plan: Norwegian Voice Fine-tuning for Qwen-Omni

The primary goal of this project is to enable **Qwen2.5-Omni-3B** to generate natural Norwegian speech.

**Current Focus:** Transitioning from "Raw Audio Training" to **"Instruction Tuning"**. The previous model achieved excellent voice quality but suffered from behavioral issues (mimicking a politician too closely and ignoring user commands). We are now retraining to fix these logic issues while retaining the high-fidelity voice.

---

## 📋 Execution Plan

### 🔄 Iteration History
* **Attempt 1 (FLEURS):** Resulted in poor audio quality and unnatural speech patterns.
* **Attempt 2 (NPSC Raw):** **Technical Success on Voice.** The model learned perfect Norwegian prosody, dialect, and intonation.
    * *Issue Identified:* **"Catastrophic Forgetting" / Domain Overfitting.** The model learned the *style* of the Storting (Parliament) too well. It ignored instructions and instead generated endless, hallucinated political speeches (e.g., starting sentences with "Høre president..."), eventually leading to infinite loops and OOM crashes.
* **Current Attempt (NPSC Instruct):** Retraining using an **Instruction Tuning** approach to restore the model's ability to follow commands (e.g., "Si dette på norsk") while retaining the high-quality voice.

### Phase 1: Data Acquisition & Filtering
* **Source:** **NPSC (National Parliamentary Speech Corpus)**.
* **Refinement:**
    * **Filtering:** Removing all clips shorter than **1.5 seconds** to prevent "repetition loops" (the "Pai... Pai..." audio glitch).
    * **Instruction Wrapping:** Instead of mapping `Text -> Audio` directly, we now map `User Instruction` -> `Audio`.
    * **Format:** `User: "Les opp denne setningen: [Text]"` -> `Assistant: [Audio Tokens]`

### Phase 2: Audio Tokenization & Formatting
* **Script:** `prepare_talker_instruct.py`
* **Method:** Generating `talker_data.jsonl` where every audio clip is paired with a unique command prompt (e.g., "Les opp denne setningen:", "Uttal dette på norsk:") to teach the model **obedience**.

### Phase 3: Fine-tuning (The "Lobotomy Fix")
* **Strategy:** Reducing the number of epochs to prevent overfitting on the "political style" of the dataset.
* **Goal:** To balance the high-fidelity voice generation with the logical capability to stop speaking when the sentence is finished (EOS token learning).

### Phase 4: Validation & Debugging
* **Inference Strategy:** Using `debug_norsk.py` with strict `max_new_tokens` limits (e.g., 50 tokens) to prevent VRAM crashes during testing.
* **Metrics:** Evaluating both **Voice Quality** (MOS) and **Instruction Adherence** (Does it say exactly what is written, or does it start a debate?).

---

## 🚦 Status Report

### ✅ Completed & Verified
* **Infrastructure:** Full pipeline (Docker, PEFT, BitsAndBytes) is stable.
* **Voice Quality:** Verified high-quality Norwegian acoustics from the NPSC dataset (Attempt 2).
* **Diagnosis:** Successfully identified why the model was "babbling" (Lack of EOS token training / Overfitting on long speeches).
* **New Pipeline:** Created `prepare_talker_instruct.py` to handle data filtering (>1.5s) and instruction formatting.

### ⚠️ Known Issues (Being Fixed)
* **"Politician Mode":** The previous model would start sentences with "President..." regardless of the input text.
* **Infinite Loops:** Without instruction tuning, the model often fails to generate an `<|im_end|>` token, leading to "echo" glitches (ASCII garbage) and memory crashes.

### ⏳ Ongoing / Next Steps
* **Run 3:** Executing the **Instruction Tuned** training run now.
* **Verification:** Testing if the new model can say "Hei, dette er en test" without adding a 5-minute speech about climate change.
