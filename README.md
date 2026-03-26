<p align="center">
  <strong style="font-size: 2.2rem;">Meeting Summarizer</strong><br>
  <sub style="font-size: 1.05rem;">Summarize meetings, compare models, ship insights.</sub>
</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg">
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.x-ffbf00.svg">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-E64A19.svg">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-SAMSum-0F766E.svg">
</p>

End‑to‑end workflow for meeting‑style dialogue summarization. The repo includes data prep for SAMSum, training and evaluation pipelines for multiple encoder‑decoder models (BART, PEGASUS, FLAN‑T5, T5 Small, LoRA variants), and a Streamlit UI to compare models and generate summaries.

## Features
- 📦 Data lifecycle: download SAMSum, preprocess into tokenizer‑ready splits, save reproducibly to disk.
- 🏋️ Training recipes: full fine‑tuning and LoRA for BART, PEGASUS, FLAN‑T5, T5 Small with ready scripts.
- 📊 Evaluation + plots: ROUGE scoring and loss/ROUGE visualizations in `outputs/plots`.
- 🗂️ Model registry: `configs/models.json` lists trained checkpoints with ROUGE scores and is used by the UI.
- 🎛️ Streamlit app: pick a model, paste a conversation, and generate summaries; view model insights and comparison dashboard.

## Setup
Use Conda (Python 3.9.6):
```bash
# If Conda is missing: install Miniconda
echo "(macOS) curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
echo "(Linux) curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
echo "bash miniconda.sh -b -p \"$HOME/miniconda\""
echo "source \"$HOME/miniconda/bin/activate\""
```

## 🚀 Quick Start (just run the UI)
Ensure Git LFS is installed (once per machine):
```bash
git lfs install
```
If Git LFS is missing, install via Homebrew (macOS):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git-lfs
git lfs install
```
On Linux/Windows, install Git LFS from https://git-lfs.com

```bash
# 1) Clone
git clone <your-repo-url> meeting-summarizer
cd meeting-summarizer

# 2) Create env (Conda, Python 3.9.6)
# If Conda is missing: install Miniconda (see above)
conda create -n meeting-summarizer python=3.9.6
conda activate meeting-summarizer
python --version   # expect Python 3.9.6

# 3) Install deps
pip install -r requirements.txt

# 4) Pull LFS-tracked model files (if not auto-downloaded)
git lfs pull

# 5) Launch the UI
streamlit run app/streamlit_app.py
```

Notes:
- The app loads models from `experiments/*` listed in `configs/models.json`; inference doesn’t require `data/`.
- If you pulled checkpoints, you can stop here—no data download/preprocess needed for UI-only use.
- Large folders `data/`, `outputs/`, `venv/` are gitignored; regenerate locally as needed.

## 🔁 Retraining (when you want to fine-tune again)
```bash
# Download SAMSum
python src/data_loader.py

# Preprocess for your target model(s)
python src/preprocess.py                  # FLAN/T5
python src/preprocess_bart.py             # BART
python src/preprocess_pegasus_speaker.py  # Pegasus speaker-aware

# Train (example)
python src/train_bart_base_lora.py

# Evaluate / plot
python src/evaluation_bart_lora.py
python src/generate_plots.py
```

## 🧪 Advanced Analysis Tasks (Task-01 to Task-05)
The project now includes a full analysis track under `src/Analysis/` with outputs written to `outputs/analysis/`.
These tasks are designed to answer *why* the model behaves the way it does, not just produce a single ROUGE score.
All training/evaluation hyperparameters are defined in the corresponding script defaults; the commands below keep CLI args minimal.

### Task-01 — Attention Patterns for Speaker Attribution & Key Moment Detection
**What was implemented**
- Attention extraction for 100 SAMSum test dialogues (`outputs/analysis/attention/attention_tensors/`).
- Speaker-aware attribution (`outputs/analysis/attention/speaker_distribution/`).
- Key moment detection and token-to-turn alignment (`outputs/analysis/attention/key_moments/`).
- Per-sample reports with top-3 contributing turns (`outputs/analysis/attention/reports/test_*.json`).
- Heatmap visualizations (`outputs/analysis/attention/heatmaps/`).

**Observed results (100/100 samples processed, 0 failures)**
- Mean speaker entropy: **0.7163** (normalized entropy: **0.8911**).
- Mean dominant-speaker attention share: **0.6267**.
- Dialogues with dominant speaker share > 0.70: **32%** (>0.80: **7%**).
- Top-1 turn contribution inside top-3 turns: **57.94%** on average.
- Summary-token grounding:
  - top attended token mapped to a dialogue turn in **92.81%** of summary tokens,
  - and to one of the sample’s top-3 turns in **82.37%** of summary tokens.

**Why these outputs look this way**
- Summarization compresses long dialogues into a few salient moments, so attention naturally concentrates on a small set of turns.
- Entropy is still high overall (0.891 normalized), which indicates the model does not collapse to a single speaker in most conversations.
- The remaining non-turn alignment comes from control/special tokens and generic connective words used during generation.

**Code reproducibility (Task-01)**
```bash
# End-to-end Task-01 pipeline
python src/Analysis/attention/extract_attention.py
```

### Task-02 — Quantization for Real-time Summarization at Scale
**What was implemented**
- Quantized T5-small LoRA with llama.cpp into `Q4_K_M`, `Q5_K_M`, `Q8_0`.
- Benchmarked variable conversation lengths (10, 50, 100, 200 turns).
- Compared streaming vs batch inference (latency, memory, ROUGE-L).
- Benchmarked parallel inference on Mac M-series process counts: **1, 2, 4**.
- Auto-generated deployment recommendation (`outputs/analysis/quantization/reports/deployment_guide.json`).

**Core results**
- Quantized model size:
  - `Q4_K_M`: **40.18 MB**
  - `Q5_K_M`: **44.15 MB**
  - `Q8_0`: **62.34 MB**
- ROUGE-L (batch): `Q4_K_M` **0.2981**, `Q5_K_M` **0.2983**, `Q8_0` **0.2910**.
- Length=50 latency (sec): `Q4_K_M` **1.0236**, `Q5_K_M` **0.9356**, `Q8_0` **0.7332**.
- Parallel throughput at 4 processes (samples/sec):
  - `Q4_K_M`: **6.7349**
  - `Q5_K_M`: **7.3373**
  - `Q8_0`: **10.0262**
- Streaming vs batch:
  - ROUGE-L delta: **0.0** for all three quantization levels,
  - mean peak memory in streaming is lower by ~**6.8–7.7 MB**,
  - total latency is higher in streaming because each new chunk triggers re-generation.
- Final recommendation:
  - Throughput-first: `Q8_0` + 4 processes
  - Quality-first realtime default: `Q5_K_M` + 4 processes

**Why these outputs look this way**
- Increasing process count gives near-linear throughput gains up to 4 workers, with some overhead at higher contention.
- Streaming preserves quality because it reuses the same model and decoding setup; it mainly changes *when* generation is triggered.
- Memory drops in streaming are expected because each decode step sees a shorter partial context than full-batch summarization.

**Code reproducibility (Task-02)**
```bash
# End-to-end Task-02 pipeline
python src/Analysis/quantization/benchmark_inference.py \
  --parallel-processes 1 2 4

# Rebuild comparison report and validate outputs
python src/Analysis/quantization/compare_results.py
python src/Analysis/quantization/validate_quantization_results.py \
  --expected-processes 1 2 4 \
  --allow-partial-num-samples
```

### Task-03 — Steering for Focus Control (Topic vs Action Items)
**What was implemented**
- Extracted decoder middle-layer activations (100 samples).
- Computed a steering direction and injected it at inference time.
- Evaluated scales: `0.0`, `0.5`, `1.0`, `1.5`, `2.0`, `3.0`.
- Performed layer ablation and selected best steering layer(s).

**Core results (50-sample evaluation set)**
- Best scale: **1.5**
- Best layer: **6**
- Baseline (`scale=0.0`) ROUGE-L: **0.2740**
- Best (`scale=1.5`) ROUGE-L: **0.2837** (no ROUGE drop under 2% constraint)
- Action score improved from **0.1112** to **0.1817**.
- Higher scales (`2.0`, `3.0`) raised action score further but violated quality guardrail (ROUGE drop **3.38%** and **5.46%**).

**Why these outputs look this way**
- Mild steering amplifies action-related dimensions without disrupting core semantic content.
- Over-steering pushes generation off the base summary manifold, increasing action-biased words but hurting faithfulness/quality.
- Layer 6 likely captures the most controllable abstraction between content planning and lexical realization for this setup.

**Code reproducibility (Task-03)**
```bash
# End-to-end Task-03 pipeline
python src/Analysis/steering/steering_generate.py

# To exactly match the reported extended scale sweep, rerun with:
# python src/Analysis/steering/steering_generate.py --scales 0.0 0.5 1.0 1.5 2.0 3.0
```
This single command reproduces the Task-03 artifacts in `outputs/analysis/steering/`.

### Task-04 — Adversarial Transcripts & Robustness Testing
**What was implemented**
- Generated adversarial variants (overlap/noise/off-topic/length) and evaluated **150 original + 150 adversarial** samples.
- Compared pre- vs post-adversarial retraining.
- Produced failure mode breakdown (`outputs/analysis/robustness/failure_analysis/failures.json`).

**Current results**
- Adversarial ROUGE-L gain after retraining: **-0.1111** (drop).
- Coherence gain: **+0.0033** (almost unchanged).
- Action completeness gain: **-0.2533** (drop).
- Clean-set ROUGE shift: **-0.1366**.
- Failure analysis on adversarial split:
  - failures: **79 / 150** (**52.7%**),
  - most common failure types: `missing_key_info` (**61**), `wrong_entities` (**50**),
  - strongest perturbation impact: `off_topic` (79 failures), then `overlap`/`length` (74 each), `noise` (73).

**Why these outputs look this way**
- Hard perturbations inject distractor content that competes with true salient turns, so entity tracking and key-info retention degrade first.
- Coherence remains relatively stable because fluent surface form can stay intact even when factual content is wrong.
- Negative post-training shift suggests the current adversarial fine-tuning setup still over-regularizes/overfits noisy patterns relative to clean summarization.

**Code reproducibility (Task-04)**
```bash
# 1) Build adversarial dataset
python src/Analysis/robustness/create_adversarial_data.py

# 2) Pre-training predictions (base LoRA model)
python src/Analysis/robustness/generate_predictions.py

# 3) Failure mode analysis on pre-training outputs
python src/Analysis/robustness/failure_analysis.py

# 4) Adversarial retraining (current robust setup)
python src/Analysis/robustness/train_adversarial.py

# 5) Post-training predictions (robust model)
python src/Analysis/robustness/generate_predictions.py \
  --model-dir experiments/t5_small_lora_robust \
  --output-prefix post

# 6) Pre/Post evaluation and final comparison report
python src/Analysis/robustness/evaluate_robustness.py
python src/Analysis/robustness/compare_results.py
```

### Task-05 — LoRA Rank Ablation & Structured Output Constraints
**What was implemented**
- Trained LoRA ranks: `2, 4, 8, 16, 32`.
- Measured ROUGE-L, inference latency, and model size.
- Compared free-form vs structured prompt generation.
- Added a schema repair layer for strict JSON outputs.

**Core results**
- ROUGE-L by rank increased from **0.3361 (r2)** to **0.3490 (r32)**.
- Model size increased from **17.03 MB (r2)** to **125.30 MB (r32)**.
- Latency stayed in a narrow range (~**0.91s–1.14s** average/sample).
- Raw structured JSON validity from the model: **0%** across all ranks.
- Post-repair JSON validity: **100%** across all ranks (`validity_repaired.json`).
- Structured vs free-form ROUGE delta remained negative (about **-0.0046** to **-0.0101**), with slightly lower output-length variance under structured prompts.
- Final repaired report selects **rank 32** as best quality point.

**Why these outputs look this way**
- Higher LoRA rank increases adaptation capacity, but gains are gradual relative to size growth (diminishing returns).
- Prompt-only schema enforcement is weak for T5-small in this setup; unconstrained decoding often violates strict JSON syntax.
- A deterministic post-processing layer can guarantee production-safe JSON without retraining, at the cost of separating “model validity” from “system validity”.

**GitHub checkpoint note (Task-05)**
- We intentionally keep only `experiments/t5_small_lora_r32/` (best model; referred to as `t5_small_lora_32`) in GitHub because rank-32 is the selected best model in this study.
- Other Task-05 rank checkpoints (`r2`, `r4`, `r8`, `r16`) are treated as experiment artifacts and are not kept as primary GitHub model assets.

**Code reproducibility (Task-05)**
```bash
# A) Full rank-ablation reproduction (requires training ranks 2/4/8/16/32)
python src/Analysis/rank_ablation/train_lora_ranks.py
python src/Analysis/rank_ablation/evaluate_ranks.py
python src/Analysis/rank_ablation/measure_latency.py
python src/Analysis/rank_ablation/measure_model_size.py
python src/Analysis/rank_ablation/structured_inference.py
python src/Analysis/rank_ablation/evaluate_json_validity.py
python src/Analysis/rank_ablation/compare_modes.py

# B) Fast report regeneration from existing outputs (no retraining)
python src/Analysis/rank_ablation/repair_structured_outputs.py
python src/Analysis/rank_ablation/evaluate_json_validity.py \
  --structured-dir outputs/analysis/rank_ablation/structured_outputs_repaired \
  --output-path outputs/analysis/rank_ablation/validity/validity_repaired.json
```

---
All generated analysis artifacts are versioned under `outputs/analysis/*` so every claim above can be traced to exact files and rerun scripts in `src/Analysis/*`.

**GitHub artifact notes for the other tasks**
- **Task-01:** code and structured reports are kept; large raw attention tensors/heatmaps are reproducible outputs.
- **Task-02:** code, deployment reports, and benchmark summaries are kept; heavy model binaries are managed separately (LFS/local artifacts).
- **Task-03:** code and final reports are kept; activation tensors are reproducible and can be regenerated via the command above.
- **Task-04:** code, failure analysis, and robustness reports are kept; regenerated predictions can be produced from scripts in `src/Analysis/robustness/`.

## 🧠 Final Insights & Key Takeaways
### 1) Model Behavior Understanding (Task-01)
- The model does not uniformly attend to all speakers.
- A small subset of dialogue turns contributes disproportionately to summaries.
- Despite this, high entropy indicates the model still considers multiple speakers in most cases.

👉 **Conclusion:**  
The model is selective but not biased toward a single speaker, which is desirable for summarization tasks.

### 2) Real-Time Deployment Feasibility (Task-02)
- Quantization significantly improves inference speed with minimal quality drop.
- `Q5_K_M` provides the best balance between quality and latency.
- Parallel inference (4 processes) scales throughput effectively.

👉 **Conclusion:**  
Real-time summarization is achievable on consumer hardware (Mac M-series) using quantization + parallelization.

### 3) Controllability of Summaries (Task-03)
- Steering successfully increases action-oriented content.
- Best trade-off is achieved at `scale = 1.5`, while staying within <2% ROUGE drop.
- Over-steering degrades summary quality.

👉 **Conclusion:**  
Controlled generation is possible without retraining, but requires careful tuning to avoid quality degradation.

### 4) Robustness to Real-world Noise (Task-04)
- Model performance drops significantly on adversarial inputs.
- Adversarial training did not improve robustness and degraded clean performance.

👉 **Conclusion:**  
T5-small has limited capacity to handle both clean and noisy data simultaneously. More advanced training strategies or larger models are needed.

### 5) Efficiency vs Quality Trade-off (Task-05)
- Increasing LoRA rank improves ROUGE but increases model size.
- Structured generation alone fails (0% JSON validity).
- Post-processing ensures 100% JSON validity without retraining.

👉 **Conclusion:**  
A hybrid approach (model + repair layer) is necessary for production-grade structured outputs.


## 🗂️ Repository Structure
```
├── app/
│   └── streamlit_app.py           # Streamlit UI
├── configs/
│   └── models.json                # Model registry with scores/paths
├── data/
│   ├── raw/                       # SAMSum saved_to_disk splits (gitignored)
│   ├── processed/                 # Tokenized generic (FLAN/T5, gitignored)
│   ├── processed_bart/            # Tokenized for BART (gitignored)
│   └── processed_pegasus_speaker/ # Tokenized for Pegasus speaker-aware (gitignored)
├── experiments/
│   ├── bart_base_full/            # (local) checkpoints/metrics
│   ├── bart_base_lora/            # (local) checkpoints/metrics
│   ├── flan_t5_base/              # (local) checkpoints/metrics
│   ├── pegasus_lora/              # (local) checkpoints/metrics
│   └── t5_small_lora/             # (local) checkpoints/metrics
├── outputs/
│   ├── plots/                     # Loss/ROUGE charts 
│   └── history/history.json       # Streamlit generation history 
├── notebooks/                     # Exploration notebooks
├── src/
│   ├── data_loader.py             # Download SAMSum
│   ├── preprocess.py              # Generic preprocessing
│   ├── preprocess_bart.py         # BART-specific preprocessing
│   ├── preprocess_pegasus_speaker.py
│   ├── train_bart_base.py         # Training scripts (full/LoRA variants)
│   ├── train_bart_base_lora.py
│   ├── train_flan_t5_base.py
│   ├── train_t5_small.py
│   ├── train_lora_t5_small.py
│   ├── train_pegasus_lora.py
│   ├── evaluation*.py             # Evaluation scripts
│   ├── generate_plots.py          # Plot creation
│   ├── inference.py               # Generation helper
│   └── model_loader.py            # Registry-driven loading + device select
├── venv/                          # Virtual env (gitignored)
├── requirements.txt
└── README.md
```

### 🧩 Additional Analysis Structure (new)
```
├── experiments/
│   ├── t5_small_lora_r2/          # LoRA rank-2 checkpoint(s)
│   ├── t5_small_lora_r4/          # LoRA rank-4 checkpoint(s)
│   ├── t5_small_lora_r8/          # LoRA rank-8 checkpoint(s)
│   ├── t5_small_lora_r16/         # LoRA rank-16 checkpoint(s)
│   ├── t5_small_lora_r32/         # LoRA rank-32 checkpoint(s)
│   └── t5_small_lora_robust/      # Adversarially trained checkpoint(s)
├── src/
│   └── Analysis/
│       ├── attention/             # Task-01 (attention extraction + attribution)
│       ├── quantization/          # Task-02 (gguf quantize + bench + reports)
│       ├── steering/              # Task-03 (activation steering + evaluation)
│       ├── robustness/            # Task-04 (adversarial data + robustness eval)
│       └── rank_ablation/         # Task-05 (LoRA rank sweep + JSON constraints)
└── outputs/
    └── analysis/
        ├── attention/             # Attention tensors, heatmaps, per-sample reports
        ├── quantization/          # Batch/stream/parallel metrics + deployment reports
        ├── steering/              # Activations, generated outputs, steering reports
        ├── robustness/            # Predictions, failure analysis, final robustness report
        └── rank_ablation/         # Rank metrics, latency/size, validity, final report
```

## 📥 Data
1) Download SAMSum locally:
```bash
python src/data_loader.py
```
Saves splits to `data/raw/`.

2) Preprocess for a model family (examples):
```bash
# General FLAN/T5-style
python src/preprocess.py
# BART-specific
python src/preprocess_bart.py
# PEGASUS with speaker-aware variant
python src/preprocess_pegasus_speaker.py
```
Outputs go to the corresponding `data/processed*` directory.

## 🏋️ Training
Run one of the ready recipes (edit hyperparams inside each script if needed):
```bash
python src/train_bart_base.py          # full FT
python src/train_bart_base_lora.py     # LoRA
python src/train_pegasus_lora.py       # LoRA
python src/train_flan_t5_base.py       # full FT
python src/train_t5_small.py           # full FT
python src/train_lora_t5_small.py      # LoRA
```
Checkpoints and metrics are written under `experiments/<model_name>/`.

## ✅ Evaluation
ROUGE evaluation per model:
```bash
python src/evaluation_bart.py
python src/evaluation_bart_lora.py
python src/evaluation_pegasus_lora.py
python src/evaluation_t5_small.py
python src/evaluation.py              # generic helper
```

## 📊 Plotting
Generate loss/ROUGE comparison charts (saved to `outputs/plots/`):
```bash
python src/generate_plots.py
```

## 🖥️ Streamlit App
```bash
streamlit run app/streamlit_app.py
```
- Uses `configs/models.json` to list models; the “best” model is picked by highest ROUGE.
- Runs on MPS if available (Apple Silicon) else CPU per `model_loader.py`.




## 🗂️ Model Registry
`configs/models.json` example entry:
```json
{
  "BART-FULL": {
    "path": "experiments/bart_base_full",
    "rougeL": 41.33,
    "type": "Full Fine-Tuning",
    "architecture": "BART"
  }
}
```
Add new trained runs here to surface them in the UI and dashboards.
Note: checkpoints referenced in `path` should exist locally in `experiments/`; they are not committed to git.

## 💾 Datasets & Storage
- `data/raw/` — HuggingFace `save_to_disk` output.
- `data/processed*/` — tokenized datasets matched to tokenizer/model.
- `outputs/history/history.json` — Streamlit generation history (last summaries shown in UI).

## 💡 Tips
- Keep conversations in the UI short, speaker-labeled lines: `Speaker: text`.
- For T5/FLAN models the code injects an instruction prefix; for others it uses raw text.
- Adjust generation params in `src/inference.py` (`num_beams`, `max_new_tokens`, etc.) if you need faster or shorter outputs.

## 📜 License / Credits
Built on Hugging Face Transformers, Datasets, and Streamlit; datasets: SAMSum (`knkarthick/samsum`). Add your license/credit note here as needed.

## 🔮 Future Improvements
- Use larger models (FLAN-T5-large / PEGASUS-large)
- Improve adversarial training strategy (curriculum learning)
- Replace post-processing with constrained decoding
- Add human evaluation for summary quality
