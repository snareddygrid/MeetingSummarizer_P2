import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ===============================
# CONFIG
# ===============================

EXPERIMENTS = {
    "PEGASUS-LoRA": "experiments/pegasus_lora",
    "BART-LoRA": "experiments/bart_base_lora",
    "BART-FFT": "experiments/bart_base_full",
    "FLAN-T5-BASE": "experiments/flan_t5_base",
    "T5-SMALL": "experiments/t5_small_optimized"
}

FINAL_METRICS = {
    "PEGASUS-LoRA": 42.95,
    "BART-LoRA": 41.49,
    "BART-FFT": 41.33,
    "FLAN-T5-BASE": 40.61,
    "T5-SMALL": 38.14
}

OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# Find Latest Checkpoint
# ===============================

def get_latest_checkpoint(exp_path):
    checkpoints = [
        d for d in os.listdir(exp_path)
        if d.startswith("checkpoint")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(exp_path, checkpoints[-1])


# ===============================
# Extract Loss
# ===============================

def extract_loss(exp_path):
    checkpoint_path = get_latest_checkpoint(exp_path)
    if checkpoint_path is None:
        return [], []

    trainer_state = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state):
        return [], []

    with open(trainer_state) as f:
        state = json.load(f)

    steps = []
    losses = []

    for log in state["log_history"]:
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])

    return np.array(steps), np.array(losses)


# ===============================
# Improved Loss Plot
# ===============================

def plot_loss_curve(model_name, exp_path):

    steps, losses = extract_loss(exp_path)

    if len(losses) == 0:
        print(f"No loss data for {model_name}")
        return

    # Smooth using moving average
    window = 20
    smooth_losses = np.convolve(losses, np.ones(window)/window, mode="valid")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps[:len(smooth_losses)],
        y=smooth_losses,
        mode="lines",
        line=dict(width=3),
        name="Smoothed Loss"
    ))

    fig.update_layout(
        title=f"Training Loss Curve — {model_name}",
        xaxis_title="Training Steps",
        yaxis_title="Loss (Log Scale)",
        template="plotly_dark",
        width=1100,
        height=650
    )

    fig.update_yaxes(type="log")  # LOG SCALE

    html_path = os.path.join(OUTPUT_DIR, f"loss_{model_name}.html")
    png_path = os.path.join(OUTPUT_DIR, f"loss_{model_name}.png")

    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)

    print(f"Saved HTML + PNG for {model_name}")

    print(f"Saved improved loss plot for {model_name}")


# ===============================
# Improved ROUGE Bar Chart
# ===============================

def plot_rouge_comparison():

    models = list(FINAL_METRICS.keys())
    scores = list(FINAL_METRICS.values())

    colors = ["#1f77b4" if s < max(scores) else "#ff4b4b" for s in scores]

    fig = go.Figure(go.Bar(
        x=models,
        y=scores,
        text=[f"{s:.2f}%" for s in scores],
        textposition="outside",
        marker_color=colors
    ))

    fig.update_layout(
        title="Final Model Comparison (ROUGE-L Test Set)",
        yaxis_title="ROUGE-L (%)",
        template="plotly_white",
        width=1100,
        height=650
    )

    # Start from 30%
    fig.update_yaxes(range=[30, max(scores)+2])

    html_path = os.path.join(OUTPUT_DIR, "final_rouge_comparison.html")
    png_path = os.path.join(OUTPUT_DIR, "final_rouge_comparison.png")

    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)

    print("Saved HTML + PNG for ROUGE comparison")

    print("Saved improved ROUGE comparison plot")


# ===============================
# RUN
# ===============================

if __name__ == "__main__":

    print("\nGenerating improved plots...\n")

    for model, path in EXPERIMENTS.items():
        plot_loss_curve(model, path)

    plot_rouge_comparison()

    print("\nAll improved plots saved in outputs/plots/")