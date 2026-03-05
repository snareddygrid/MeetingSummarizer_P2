import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CONFIG_PATH = "configs/models.json"


def load_model_registry():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_best_model():
    models = load_model_registry()
    best_model = max(models.items(), key=lambda x: x[1]["rougeL"])[0]
    return best_model


def load_selected_model(model_key):
    models = load_model_registry()

    model_path = models[model_key]["path"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device