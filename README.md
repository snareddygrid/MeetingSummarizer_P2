<p align="center">
  <strong>🧠 Meeting Summarizer</strong><br>
  <sub>Summarize meetings, compare models, ship insights.</sub>
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

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
