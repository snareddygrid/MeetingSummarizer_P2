import html
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import generate_summary
from src.model_loader import get_best_model, load_model_registry, load_selected_model

PLOT_FOLDER = Path("outputs/plots")
HISTORY_FILE = Path("outputs/history/history.json")
NAV_OPTIONS = {"generate": "Generate Summary", "dashboard": "Analytics Dashboard"}
DEFAULT_MODEL_KEY = "BART-FULL"
ARCHITECTURE_COLORS = {
    "BART": "#1D4ED8",
    "PEGASUS": "#EA580C",
    "T5": "#15803D",
}
TRAINING_COLORS = {
    "Full Fine-Tuning": "#0F766E",
    "LoRA": "#B45309",
}
MODEL_BRIEFS = {
    "BART-FULL": "Produces balanced and faithful summaries with strong context coverage.",
    "BART-LoRA": "Generates concise summaries with efficient LoRA fine-tuning.",
    "PEGASUS-LoRA": "Generates abstract-style summaries focused on key takeaways.",
    "FLAN-T5-BASE": "Creates instruction-friendly summaries with clear wording.",
    "T5-SMALL": "Builds simple, lightweight summaries for fast turnaround.",
}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_round(value, digits=2):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(29, 78, 216, {alpha})"
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def get_architecture_color(architecture: str) -> str:
    return ARCHITECTURE_COLORS.get((architecture or "").upper(), "#0F766E")


def get_training_color(training_type: str) -> str:
    return TRAINING_COLORS.get(training_type or "", "#6B7280")


def get_model_brief(model_name: str, architecture: str, training_type: str) -> str:
    if model_name in MODEL_BRIEFS:
        return MODEL_BRIEFS[model_name]

    architecture_label = (architecture or "Model").upper()
    if training_type == "LoRA":
        return f"{architecture_label} with LoRA tuning for concise summaries."
    return f"{architecture_label} model tuned for meeting summarization."


def load_models_safely():
    try:
        loaded_models = load_model_registry()
    except Exception as exc:
        st.error(f"Failed to load model registry: {exc}")
        st.stop()

    if not isinstance(loaded_models, dict) or not loaded_models:
        st.error("Model registry is empty or invalid: configs/models.json")
        st.stop()

    return loaded_models


def load_history(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def append_history(model_name: str, summary: str):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(HISTORY_FILE)
    history.append({"model": model_name, "summary": summary})
    HISTORY_FILE.write_text(json.dumps(history, indent=4), encoding="utf-8")


def render_plot_html(file_path: Path, height=620):
    try:
        html_content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        st.warning(f"Could not read {file_path.name}: {exc}")
        return False
    # Make Plotly HTML fit inside the Streamlit iframe without requiring scroll.
    html_content = html_content.replace(
        "<body>",
        '<body style="margin:0; overflow:hidden; background:transparent;">',
        1,
    )
    html_content = html_content.replace("width:1100px;", "width:100%;")
    components.html(html_content, height=height, scrolling=False)
    return True


def render_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #f8fafc;
            --ink-main: #0f172a;
            --ink-soft: #475569;
            --primary: #0f766e;
            --primary-strong: #0e7490;
            --accent: #f59e0b;
            --card: #ffffff;
            --border: #e2e8f0;
        }

        .stApp {
            background:
                radial-gradient(1200px 400px at 80% -50%, rgba(245, 158, 11, 0.18), transparent 70%),
                radial-gradient(900px 350px at 10% -40%, rgba(14, 116, 144, 0.18), transparent 70%),
                var(--bg-main);
        }

        /* Hide Streamlit's default top chrome (Deploy/menu bar). */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none;
        }

        /* Remove extra top spacer left by Streamlit chrome and retune vertical rhythm. */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            top: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0.7rem;
        }

        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1220px;
            padding-top: 0.25rem;
            padding-bottom: 2rem;
        }

        html, body, [class*="css"] {
            font-family: "Sora", "Avenir Next", "Segoe UI", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fefb 0%, #eef4ff 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        .sidebar-brand-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            background: linear-gradient(140deg, rgba(255, 255, 255, 0.98), rgba(236, 253, 245, 0.94));
            padding: 0.75rem 0.8rem 0.7rem 0.8rem;
            margin-bottom: 0.65rem;
        }

        .sidebar-brand-kicker {
            margin: 0;
            color: #0f766e;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }

        .sidebar-brand-title {
            margin: 0.2rem 0 0 0;
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }

        .sidebar-brand-subtitle {
            margin: 0.28rem 0 0 0;
            color: #475569;
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .sidebar-section-label {
            margin: 0.1rem 0 0.42rem 0;
            color: #64748b;
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }

        [data-testid="stSegmentedControl"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.3rem;
            margin-bottom: 0.5rem;
        }

        [data-testid="stSegmentedControl"] button {
            border-radius: 10px;
            min-height: 2.2rem;
            font-weight: 600;
        }

        [data-testid="stSegmentedControl"] button[aria-selected="true"],
        [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background: linear-gradient(120deg, #0f766e 0%, #0e7490 100%);
            color: white;
        }

        .hero-banner {
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: linear-gradient(120deg, #ffffff 0%, #f0fdfa 55%, #eef2ff 100%);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            margin-top: 0;
            margin-bottom: 1rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .hero-title {
            margin: 0;
            color: #0f172a !important;
            font-size: 1.7rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            text-align: center;
            width: 100%;
        }

        .hero-subtitle {
            margin: 0.45rem auto 0 auto;
            color: #0f172a !important;
            font-size: 0.95rem;
            line-height: 1.4;
            width: fit-content;
            max-width: 90%;
            text-align: center !important;
        }

        .chip {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            margin-right: 0.35rem;
            margin-top: 0.35rem;
        }

        .sidebar-model-card {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 5px solid #0f766e;
            border-radius: 14px;
            background: #ffffff;
            padding: 0.8rem 0.9rem;
            margin-top: 0.55rem;
            margin-bottom: 0.5rem;
        }

        .sidebar-model-title {
            margin: 0 0 0.35rem 0;
            font-size: 0.88rem;
            font-weight: 700;
            color: #0f172a;
        }

        .sidebar-model-line {
            margin: 0.16rem 0;
            color: #334155;
            font-size: 0.83rem;
        }

        .sidebar-tip-card {
            border: 1px dashed rgba(148, 163, 184, 0.6);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.72);
            padding: 0.68rem 0.75rem;
            margin-top: 0.2rem;
        }

        .sidebar-tip-title {
            margin: 0 0 0.35rem 0;
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .sidebar-tip-line {
            margin: 0.2rem 0;
            color: #334155;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .sidebar-tip-line code {
            background: rgba(15, 118, 110, 0.1);
            color: #0f766e;
            border-radius: 6px;
            padding: 0.08rem 0.28rem;
            font-size: 0.74rem;
        }

        .insight-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            background: #ffffff;
            padding: 0.9rem 1rem;
            min-height: 104px;
        }

        .insight-label {
            margin: 0;
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }

        .insight-value {
            margin: 0.28rem 0 0.2rem 0;
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 800;
        }

        .summary-output {
            border: 1px solid rgba(45, 212, 191, 0.35);
            border-left: 6px solid #14b8a6;
            border-radius: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
            padding: 1rem 1.1rem;
            margin-top: 0.8rem;
        }

        .summary-output h4 {
            margin: 0 0 0.45rem 0;
            color: #0f172a;
        }

        .summary-output p {
            margin: 0;
            color: #1e293b;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .comparison-table-wrap {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            overflow: hidden;
            margin-top: 0.35rem;
            margin-bottom: 0.9rem;
            background: #ffffff;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.89rem;
        }

        .comparison-table th {
            background: #0f172a;
            color: #f8fafc;
            text-align: left;
            padding: 0.65rem 0.75rem;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .comparison-table td {
            padding: 0.58rem 0.75rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.85);
            color: #0f172a;
            vertical-align: middle;
        }

        .comparison-table tr:last-child td {
            border-bottom: none;
        }

        .badge-pill {
            display: inline-block;
            padding: 0.16rem 0.5rem;
            border-radius: 999px;
            font-size: 0.71rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            border: 1px solid transparent;
        }

        .recent-item {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            background: #ffffff;
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.55rem;
        }

        .example-panel {
            margin-top: 0.8rem;
            border: 1px solid rgba(148, 163, 184, 0.26);
            border-radius: 14px;
            background: #ffffff;
            padding: 0.8rem 0.9rem;
        }

        .example-title {
            margin: 0 0 0.55rem 0;
            color: #0f172a !important;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }

        .example-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .example-card {
            border: 1px solid rgba(203, 213, 225, 0.8);
            border-radius: 10px;
            background: #f8fafc;
            padding: 0.65rem 0.7rem;
        }

        .example-label {
            margin: 0 0 0.35rem 0;
            color: #0f172a !important;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .example-card pre {
            margin: 0;
            color: #0f172a !important;
            font-size: 0.82rem;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: "JetBrains Mono", "SFMono-Regular", "Consolas", monospace;
        }

        @media (max-width: 980px) {
            .example-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_comparison_rows(models: dict, best_model: str):
    sorted_models = sorted(
        models.items(),
        key=lambda x: safe_float(x[1].get("rougeL"), 0.0),
        reverse=True,
    )
    rows = []
    for rank, (name, info) in enumerate(sorted_models, start=1):
        rows.append(
            {
                "Rank": rank,
                "Model": name,
                "Architecture": info.get("architecture", "Unknown"),
                "Training": info.get("type", "Unknown"),
                "ROUGE-L (%)": safe_float(info.get("rougeL"), 0.0),
                "Brief": get_model_brief(
                    name,
                    info.get("architecture", "Unknown"),
                    info.get("type", "Unknown"),
                ),
                "Is Best": name == best_model,
            }
        )
    return rows


def render_comparison_table(rows):
    body_rows = []
    for row in rows:
        architecture = row["Architecture"]
        training_type = row["Training"]
        arch_color = get_architecture_color(architecture)
        train_color = get_training_color(training_type)
        row_bg = hex_to_rgba(arch_color, 0.06)
        status_badge = "⭐" if row["Is Best"] else ""

        body_rows.append(
            (
                f'<tr style="background:{row_bg};">'
                f'<td>{row["Rank"]}</td>'
                f'<td><strong>{html.escape(row["Model"])}</strong> {status_badge}</td>'
                f'<td><span class="badge-pill" '
                f'style="color:{arch_color}; background:{hex_to_rgba(arch_color, 0.14)}; '
                f'border-color:{hex_to_rgba(arch_color, 0.35)};">'
                f'{html.escape(architecture)}</span></td>'
                f'<td><span class="badge-pill" '
                f'style="color:{train_color}; background:{hex_to_rgba(train_color, 0.14)}; '
                f'border-color:{hex_to_rgba(train_color, 0.35)};">'
                f'{html.escape(training_type)}</span></td>'
                f'<td><strong>{row["ROUGE-L (%)"]:.2f}%</strong></td>'
                f'<td>{html.escape(row["Brief"])}</td>'
                "</tr>"
            )
        )

    table_html = (
        '<div class="comparison-table-wrap">'
        '<table class="comparison-table">'
        "<thead><tr>"
        "<th>Rank</th>"
        "<th>Model</th>"
        "<th>Architecture</th>"
        "<th>Training</th>"
        "<th>ROUGE-L</th>"
        "<th>Brief</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def sync_conversation_text():
    st.session_state.conversation_text = st.session_state.get("conversation_text_widget", "")


st.set_page_config(page_title="Meeting Summarizer", page_icon="🧠", layout="wide")
render_theme()

if "latest_summary" not in st.session_state:
    st.session_state.latest_summary = ""
if "latest_model" not in st.session_state:
    st.session_state.latest_model = ""
if "latest_elapsed" not in st.session_state:
    st.session_state.latest_elapsed = None
if "conversation_text" not in st.session_state:
    st.session_state.conversation_text = ""
if "last_generated_input" not in st.session_state:
    st.session_state.last_generated_input = ""

# Persist the latest draft before Streamlit cleans hidden widget state.
if "conversation_text_widget" in st.session_state:
    st.session_state.conversation_text = st.session_state.conversation_text_widget

models = load_models_safely()
try:
    best_model = get_best_model()
except Exception:
    best_model = next(iter(models))
if best_model not in models:
    best_model = next(iter(models))

model_names = list(models.keys())
if DEFAULT_MODEL_KEY in model_names:
    selected_index = model_names.index(DEFAULT_MODEL_KEY)
elif best_model in model_names:
    selected_index = model_names.index(best_model)
else:
    selected_index = 0

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-card">
            <p class="sidebar-brand-kicker">Conversation Insights</p>
            <p class="sidebar-brand-title">Meeting Notes Generator</p>
            <p class="sidebar-brand-subtitle">Input transcripts and generate clean summaries with model-driven quality.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-section-label">Workspace</p>', unsafe_allow_html=True)
    navigation = st.segmented_control(
        "Workspace",
        list(NAV_OPTIONS.keys()),
        default="generate",
        selection_mode="single",
        format_func=lambda x: "Generate Summary" if x == "generate" else "Dashboard",
        label_visibility="collapsed",
        width="stretch",
    )
    if navigation is None:
        navigation = "generate"

    st.markdown('<p class="sidebar-section-label">Model Selection</p>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Select Model",
        model_names,
        index=selected_index,
        help="Choose the model for generation and analytics context.",
        accept_new_options=False,
    )

model_info = models.get(selected_model, {})
architecture = model_info.get("architecture", "Unknown")
training_type = model_info.get("type", "Unknown")
rouge_value = safe_round(model_info.get("rougeL"), 2)
rouge_text = f"{rouge_value}%" if rouge_value is not None else "N/A"
arch_color = get_architecture_color(architecture)
train_color = get_training_color(training_type)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-model-card">
            <p class="sidebar-model-title">{html.escape(selected_model)}</p>
            <p class="sidebar-model-line"><strong>ROUGE-L:</strong> {rouge_text}</p>
            <p class="sidebar-model-line">
                <span class="chip" style="background:{hex_to_rgba(arch_color, 0.13)}; color:{arch_color};">
                    {html.escape(architecture)}
                </span>
                <span class="chip" style="background:{hex_to_rgba(train_color, 0.13)}; color:{train_color};">
                    {html.escape(training_type)}
                </span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-tip-card">
            <p class="sidebar-tip-title">Quick Tips</p>
            <p class="sidebar-tip-line">Write dialogue as <code>Speaker: text</code> for best summaries.</p>
            <p class="sidebar-tip-line">Keep conversations focused and short for faster generation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_cached_model(model_key):
    return load_selected_model(model_key)


st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">Meeting Summarizer Studio</h1>
        <p class="hero-subtitle">
            Beautiful, fast summarization with model intelligence at your fingertips.
            Active model: <strong>{html.escape(selected_model)}</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if navigation == "generate":
    left_col, right_col = st.columns([1.65, 1.0], gap="large")

    with left_col:
        st.subheader("Conversation Input")
        if "conversation_text_widget" not in st.session_state:
            st.session_state.conversation_text_widget = st.session_state.conversation_text

        user_input = st.text_area(
            "Paste your meeting conversation",
            height=320,
            placeholder="Alex: Let's finalize the roadmap...\nPriya: Q2 should focus on launch readiness...",
            label_visibility="collapsed",
            key="conversation_text_widget",
            on_change=sync_conversation_text,
        )
        st.session_state.conversation_text = user_input
        generate_clicked = st.button("Generate Summary", type="primary", use_container_width=True)

        if generate_clicked:
            if not user_input.strip():
                st.warning("Please enter conversation text before generating a summary.")
            else:
                with st.spinner("Loading model and creating summary..."):
                    try:
                        start_time = time.time()
                        model, tokenizer, device = load_cached_model(selected_model)
                        summary = generate_summary(
                            model,
                            tokenizer,
                            device,
                            user_input,
                            architecture=architecture,
                        )
                        elapsed = round(time.time() - start_time, 2)
                        st.session_state.latest_summary = summary
                        st.session_state.latest_model = selected_model
                        st.session_state.latest_elapsed = elapsed
                        st.session_state.last_generated_input = user_input.strip()
                        append_history(selected_model, summary)
                    except Exception as exc:
                        st.error(f"Failed to generate summary: {exc}")

        current_input = st.session_state.conversation_text.strip()
        generated_for_current_input = (
            bool(current_input)
            and current_input == st.session_state.last_generated_input
        )

        if st.session_state.latest_summary and generated_for_current_input:
            safe_summary = html.escape(st.session_state.latest_summary).replace("\n", "<br>")
            runtime = st.session_state.latest_elapsed
            runtime_text = f"{runtime} sec" if runtime is not None else "N/A"
            st.markdown(
                f"""
                <div class="summary-output">
                    <h4>Generated Summary</h4>
                    <p>{safe_summary}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Generated with {st.session_state.latest_model} in {runtime_text}"
            )

    with right_col:
        st.subheader("Model Insights")
        info_col1, info_col2 = st.columns(2, gap="small")
        with info_col1:
            st.markdown(
                f"""
                <div class="insight-card">
                    <p class="insight-label">Architecture</p>
                    <p class="insight-value">{html.escape(architecture)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_col2:
            st.markdown(
                f"""
                <div class="insight-card">
                    <p class="insight-label">ROUGE-L</p>
                    <p class="insight-value">{rouge_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Example Conversations")
        st.markdown(
            """
            <div class="example-panel">
                <p class="example-title">Try These Inputs</p>
                <div class="example-grid" style="grid-template-columns:1fr;">
                    <div class="example-card">
                        <p class="example-label">Example 1</p>
                        <pre>Alex: Can we finalize the launch date?
Priya: Yes, let's target April 15.
Alex: Great, I will share the updated plan today.</pre>
                    </div>
                    <div class="example-card">
                        <p class="example-label">Example 2</p>
                        <pre>Sam: Who will send the weekly report?
Nina: I will send it every Friday.
Sam: Perfect, please include action items.</pre>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif navigation == "dashboard":
    st.subheader("Model Comparison Dashboard")

    comparison_rows = build_comparison_rows(models, best_model)
    comparison_df = pd.DataFrame(comparison_rows)
    avg_rouge = comparison_df["ROUGE-L (%)"].mean() if not comparison_df.empty else 0.0
    top_score = comparison_df["ROUGE-L (%)"].max() if not comparison_df.empty else 0.0

    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="small")
    with metric_col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">Best Model</p>
                <p class="insight-value">{html.escape(best_model)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">Top ROUGE-L</p>
                <p class="insight-value">{top_score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_col3:
        st.markdown(
            f"""
            <div class="insight-card">
                <p class="insight-label">Average ROUGE-L</p>
                <p class="insight-value">{avg_rouge:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Model Comparison Table")
    render_comparison_table(comparison_rows)

    st.markdown("#### Interactive Loss Curves")
    if PLOT_FOLDER.exists():
        loss_html_files = sorted(PLOT_FOLDER.glob("loss_*.html"))
        loss_png_files = sorted(PLOT_FOLDER.glob("loss_*.png"))

        if loss_html_files:
            tabs = st.tabs([file.stem.replace("loss_", "") for file in loss_html_files])
            for tab, html_file in zip(tabs, loss_html_files):
                with tab:
                    rendered = render_plot_html(html_file, height=720)
                    if not rendered:
                        fallback_png = PLOT_FOLDER / f"{html_file.stem}.png"
                        if fallback_png.exists():
                            st.image(str(fallback_png), use_container_width=True)
                        else:
                            st.info(f"No fallback image found for {html_file.name}.")
        elif loss_png_files:
            for file in loss_png_files:
                st.image(str(file), use_container_width=True)
        else:
            st.info("No loss plots found.")
    else:
        st.info("No loss plots found.")

    st.markdown("#### ROUGE Comparison Chart")
    rouge_html = PLOT_FOLDER / "final_rouge_comparison.html"
    rouge_png = PLOT_FOLDER / "final_rouge_comparison.png"
    legacy_rouge_png = PLOT_FOLDER / "rouge_comparison.png"

    if rouge_html.exists():
        render_plot_html(rouge_html, height=620)
    elif rouge_png.exists():
        st.image(str(rouge_png), use_container_width=True)
    elif legacy_rouge_png.exists():
        st.image(str(legacy_rouge_png), use_container_width=True)
    else:
        st.info("ROUGE comparison plot not found.")
