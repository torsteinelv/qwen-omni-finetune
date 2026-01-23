# Project Plan: Norwegian Voice Fine-tuning for Qwen-Omni

The primary goal of this project is to enable **Qwen2.5-Omni-3B** to generate natural Norwegian speech.

**Current Focus:** **Run 4: "Official Prompt Alignment".**
We discovered that Qwen-Omni has a hardcoded dependency on its specific system prompt to activate the audio module. We are now retraining with the official prompt to fix the "Model Collapse" issue experienced in Run 3.

---

## 🧪 Experiment Log & Post-Mortems (What NOT to do)

This log documents failed attempts to ensure we don't repeat mistakes.

| Attempt | Strategy | Outcome / Error | Root Cause & Lesson Learned |
| :--- | :--- | :--- | :--- |
| **1. FLEURS** | Train on `google/fleurs` (nb_no). | **Failure:** Robotic, unstable voice. | **Lesson:** Dataset quality is too low/varied. Need single-speaker, studio-quality data (NPSC). |
| **2. NPSC Raw** | Train directly on `Text -> Audio` from Stortinget. | **Partial Success:** Perfect voice, but "Lobotomized". Model ignored instructions and hallucinated political speeches. **Glitch:** "Pai... Pai..." loops. | **Lesson 1:** Model overfitted on the *role* of a politician. Needs **Instruction Tuning**.<br>**Lesson 2:** Short clips (<1.5s) cause infinite repetition loops. Filter them out. |
| **3. Instruct v1** | Instruction Tuning ("Si dette..."), but used standard "You are a helpful assistant" system prompt. | **Critical Failure:** Model Collapse during inference (`<ee> 1015 1015...`). Audio module failed to activate, leading to text-babling loops. | **Lesson:** **CRITICAL:** Qwen-Omni's `Talker` module is hardcoded to only activate if the System Prompt is exactly: *"You are Qwen, a virtual human..."*. We must train with this exact string. |

---

## 📋 Execution Plan (Current Run)

### Phase 1: Data Acquisition & Filtering (Revised)
* **Source:** **NPSC (National Parliamentary Speech Corpus)**.
* **Filtering:** Strict filter: `1.5s < duration < 15.0s`. Removing short clips prevents the "Pai..." loop.
* **Instruction Format:**
    * **User:** `"Si dette på norsk: [Text]"`
    * **System Prompt:** *MUST BE:* `"You are Qwen, a virtual human developed by the Qwen Team..."` (Official string).
    * **Assistant:** `[Mimi Audio Tokens]`

### Phase 2: Fine-tuning Strategy
* **Method:** LoRA (Low-Rank Adaptation) on the `Thinker` module.
* **Hyperparameters:**
    * **Epochs:** `1` (Reduced from 2-3 to prevent "Politician Overfitting").
    * **Learning Rate:** `2e-5` (Reduced from 1e-4 to prevent "Model Collapse" / Brain damage).

### Phase 3: Validation
* **Inference Strategy:**
    * Use **Text Streaming** to detect babbling/loops early.
    * Use **Hybrid Memory Offloading** (CPU + GPU) to prevent OOM crashes caused by potential loops or large audio buffers.
    * Use **Official System Prompt** during inference to ensure the audio module unlocks.

---

## 🚦 Status Report

### ✅ Completed & Verified
* **Infrastructure:** Docker pipeline, PEFT, BitsAndBytes, and Accelerate (CPU offload) are stable.
* **Voice Acoustics:** Verified that the model *can* produce Norwegian sounds (proven by the "Pai..." glitch having a Norwegian accent).
* **Diagnosis:** Identified the "System Prompt Lock" mechanism in Qwen-Omni source code.

### ⚠️ Known Issues (Being Fixed in Run 4)
* **Identity Crisis:** The model must learn to accept the "You are Qwen" prompt while still speaking Norwegian.
* **Memory Constraints:** Inference on consumer hardware (<40GB VRAM) requires aggressive offloading.

### ⏳ Next Steps
* **Run 4 Training:** Execute training with `2e-5` LR and Official Prompt.
* **Verification:** Test if the glitch is gone and if it obeys the "Si dette på norsk" command.
