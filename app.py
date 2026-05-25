import os
import re
import json
import tempfile
import unicodedata
import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from transformers import AutoConfig, AutoModel, AutoTokenizer

# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Sentiment Analysis Multitask",
    page_icon="📊",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_ORDER = ["Negatif", "Netral", "Positif"]


# =========================================================
# MODEL DEFINITION
# =========================================================
class MultiTaskIndoBERT(nn.Module):
    def __init__(self, base_model: str, num_labels: int = 3, task_names=("news", "sosmed"), dropout=0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model, config=self.config)
        hidden_size = self.config.hidden_size
        self.num_labels = num_labels

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels)
            )
            for task in task_names
        })
        self.task_id_to_name = {i: task for i, task in enumerate(task_names)}

    def forward(self, input_ids, attention_mask, task_ids):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        cls_pooled = hidden[:, 0]

        pooled = 0.5 * cls_pooled + 0.5 * mean_pooled
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        logits = torch.empty(
            pooled.size(0),
            self.num_labels,
            dtype=torch.float32,
            device=pooled.device
        )

        for task_id, task_name in self.task_id_to_name.items():
            task_mask = (task_ids == task_id)
            if task_mask.any():
                logits[task_mask] = self.heads[task_name](pooled[task_mask]).float()

        return logits


# =========================================================
# PREPROCESSING
# =========================================================
class TrainingAlignedPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.zero_width_pattern = re.compile(r'[\u200b-\u200f\uFEFF]')
        self.multispace_pattern = re.compile(r'\s+')
        self.repeat_char_pattern = re.compile(r'(.)\1{3,}', re.IGNORECASE)
        self.repeat_punct_pattern = re.compile(r'([!?.,])\1{2,}')
        self.space_before_punct = re.compile(r'\s+([.,!?;:])')
        self.bad_escape_pattern = re.compile(r'\\[nrt]')
        self.orphan_noise_pattern = re.compile(r'\b[nrt]\b', flags=re.IGNORECASE)

        self.slang_map = {
            "gk": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak",
            "tp": "tapi", "tpi": "tapi", "karna": "karena", "krn": "karena",
            "bgt": "banget", "bngt": "banget", "yg": "yang", "dgn": "dengan",
            "utk": "untuk", "dr": "dari", "dll": "dan lain lain",
        }

        self.boilerplate_patterns = [
            r'subscribe\s+channel',
            r'klik\s+link\s+di\s+bio',
            r'hubungi\s+kami',
            r'contact\s+us',
            r'wa\s*:\s*\d+',
            r'whatsapp\s*:\s*\d+',
            r'promo\s+terbatas',
        ]

    def normalize_informal_words(self, text: str) -> str:
        tokens = text.split()
        normalized = [self.slang_map.get(tok, tok) for tok in tokens]
        return " ".join(normalized)

    def strip_social_boilerplate(self, text: str) -> str:
        lowered = text.lower()
        for pattern in self.boilerplate_patterns:
            lowered = re.sub(pattern, " ", lowered, flags=re.IGNORECASE)
        return lowered

    def clean_text_basic(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = self.zero_width_pattern.sub(" ", text)
        text = self.html_pattern.sub(" ", text)
        text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        text = self.bad_escape_pattern.sub(" ", text)
        text = self.url_pattern.sub(" [url] ", text)
        text = self.mention_pattern.sub(" [user] ", text)
        text = self.hashtag_pattern.sub(r" \1 ", text)
        text = self.repeat_char_pattern.sub(lambda m: m.group(1) * 2, text)
        text = self.repeat_punct_pattern.sub(lambda m: m.group(1), text)
        text = self.space_before_punct.sub(r"\1", text)
        text = self.orphan_noise_pattern.sub(" ", text)
        text = self.multispace_pattern.sub(" ", text).strip()
        return text

    def compress_salient_text(self, text: str, max_chars: int = 1000) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def transform(self, text: str, task_key: str) -> str:
        text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = self.normalize_informal_words(text)
        text = self.strip_social_boilerplate(text)
        text = self.compress_salient_text(text)
        text = self.clean_text_basic(text)
        return text.strip()


# =========================================================
# ARTIFACT / MODEL LOADING
# =========================================================
def list_all_artifacts_recursive(run_id: str, tracking_uri: Optional[str] = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    all_paths = []

    def walk(path=""):
        items = client.list_artifacts(run_id, path)
        for item in items:
            all_paths.append(item.path)
            if item.is_dir:
                walk(item.path)

    walk("")
    return all_paths


def find_artifact_path(all_paths, target_name: str):
    matches = [p for p in all_paths if p.endswith("/" + target_name) or p == target_name]
    return matches[0] if matches else None


def download_mlflow_artifacts_smart(run_id: str, dst_dir: str, tracking_uri: Optional[str] = None) -> Path:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    all_paths = list_all_artifacts_recursive(run_id, tracking_uri)
    targets = {
        "config_runtime.json": find_artifact_path(all_paths, "config_runtime.json"),
        "multitask_model.pt": find_artifact_path(all_paths, "multitask_model.pt"),
        "tokenizer": find_artifact_path(all_paths, "tokenizer"),
        "encoder": find_artifact_path(all_paths, "encoder"),
    }

    for required_name in ["config_runtime.json", "multitask_model.pt"]:
        artifact_path = targets[required_name]
        if artifact_path is None:
            raise FileNotFoundError(f"Artifact '{required_name}' tidak ditemukan dalam run {run_id}.")
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(dst)
        )

    for optional_name in ["tokenizer", "encoder"]:
        artifact_path = targets[optional_name]
        if artifact_path is not None:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=str(dst)
            )

    return dst


def find_bundle_root(base_dir: str) -> Path:
    base_dir = Path(base_dir)
    if (base_dir / "config_runtime.json").exists() and (base_dir / "multitask_model.pt").exists():
        return base_dir

    for cfg in base_dir.rglob("config_runtime.json"):
        parent = cfg.parent
        if (parent / "multitask_model.pt").exists():
            return parent

    raise FileNotFoundError(
        f"Tidak ditemukan folder artifact yang berisi config_runtime.json + multitask_model.pt di {base_dir}"
    )


def load_runtime_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "config_runtime.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_runtime.json tidak ditemukan di {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label_mapping(raw_label2id: Dict):
    raw_label2id = raw_label2id or {"Negatif": 0, "Netral": 1, "Positif": 2}
    pretty = {}
    for k, v in raw_label2id.items():
        pretty[int(v)] = str(k).strip()
    return pretty


@st.cache_resource(show_spinner=False)
def load_model_bundle_from_secrets():
    secrets = st.secrets

    tracking_uri = secrets.get("MLFLOW_TRACKING_URI", "")
    run_id = secrets.get("MLFLOW_RUN_ID", "")
    local_model_dir = secrets.get("LOCAL_MODEL_DIR", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dagshub_token = secrets.get("DAGSHUB_TOKEN", "")
    dagshub_username = secrets.get("DAGSHUB_USERNAME", "")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    if run_id and tracking_uri:
        tmp_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")
        downloaded_base = download_mlflow_artifacts_smart(
            run_id=run_id,
            dst_dir=tmp_dir,
            tracking_uri=tracking_uri,
        )
        model_dir = find_bundle_root(downloaded_base)
    elif local_model_dir:
        model_dir = find_bundle_root(local_model_dir)
    else:
        raise ValueError(
            "Isi st.secrets dengan MLFLOW_TRACKING_URI + MLFLOW_RUN_ID, "
            "atau LOCAL_MODEL_DIR."
        )

    runtime_cfg = load_runtime_config(model_dir)
    base_model_name = (
        runtime_cfg.get("base_model_name")
        or runtime_cfg.get("base_model")
        or "indobenchmark/indobert-base-p1"
    )
    max_length = int(runtime_cfg.get("max_length", 256))
    dropout = float(runtime_cfg.get("dropout", 0.2))

    raw_label2id = runtime_cfg.get("label2id", {"Negatif": 0, "Netral": 1, "Positif": 2})
    raw_task2id = runtime_cfg.get("task2id", {"news": 0, "sosmed": 1})

    id2label = normalize_label_mapping(raw_label2id)
    task2id = {str(k).strip().lower(): int(v) for k, v in raw_task2id.items()}
    task_names = [task for task, _ in sorted(task2id.items(), key=lambda x: x[1])]

    tokenizer_dir = model_dir / "tokenizer"
    encoder_dir = model_dir / "encoder"

    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer_source = "artifact tokenizer"
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer_source = "base model tokenizer"

    encoder_source = str(encoder_dir) if encoder_dir.exists() else base_model_name

    model = MultiTaskIndoBERT(
        base_model=encoder_source,
        num_labels=len(id2label),
        task_names=tuple(task_names),
        dropout=dropout
    )

    state_path = model_dir / "multitask_model.pt"
    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_length": max_length,
        "task2id": task2id,
        "id2label": id2label,
        "runtime_cfg": runtime_cfg,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "model_dir": str(model_dir),
        "tokenizer_source": tokenizer_source,
        "base_model_name": base_model_name,
    }


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def predict_deterministic(df: pd.DataFrame, model_bundle: Dict, batch_size: int = 16) -> pd.DataFrame:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    max_length = model_bundle["max_length"]
    id2label = model_bundle["id2label"]
    dev = model_bundle["device"]

    rows = []
    model.eval()

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size].copy()
        texts = batch_df["text"].astype(str).tolist()
        task_ids = torch.tensor(batch_df["task_id"].tolist(), dtype=torch.long, device=dev)

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            task_ids=task_ids
        )
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)

        out = batch_df.copy()
        out["pred_id"] = pred_ids
        out["pred_label"] = [id2label[int(i)] for i in pred_ids]
        out["max_prob"] = probs.max(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        out["top2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

        for idx, label_name in sorted(id2label.items()):
            out[f"prob_{str(label_name).lower()}"] = probs[:, idx]

        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def compute_sentiment_score(row: pd.Series) -> float:
    prob_pos = float(row.get("prob_positif", 0.0))
    prob_net = float(row.get("prob_netral", 0.0))
    prob_neg = float(row.get("prob_negatif", 0.0))
    return prob_pos * 5.0 + prob_net * 0.0 + prob_neg * (-5.0)


def build_single_input_df(text: str, task_key: str, task2id: Dict[str, int], preprocessor) -> pd.DataFrame:
    raw_text = "" if text is None else str(text)
    cleaned = preprocessor.transform(raw_text, task_key)
    return pd.DataFrame([{
        "row_id": f"{task_key}_single_0",
        "task_key": task_key,
        "task_id": int(task2id[task_key]),
        "raw_text": raw_text,
        "text": cleaned,
    }])


def build_batch_input_df(df: pd.DataFrame, text_col: str, task_col: Optional[str], default_task: Optional[str], task2id: Dict[str, int], preprocessor) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    if text_col not in data.columns:
        raise ValueError(f"Kolom text '{text_col}' tidak ditemukan. Kolom tersedia: {list(data.columns)}")

    if task_col and task_col in data.columns:
        data["task_key"] = data[task_col].astype(str).str.strip().str.lower()
    elif default_task:
        data["task_key"] = default_task
    else:
        raise ValueError("Pilih kolom task atau pilih default task untuk seluruh file.")

    invalid_tasks = sorted(set(data["task_key"]) - set(task2id.keys()))
    if invalid_tasks:
        raise ValueError(f"Task tidak valid: {invalid_tasks}. Pilihan valid: {list(task2id.keys())}")

    data = data.reset_index(drop=True)
    data["task_id"] = data["task_key"].map(task2id).astype(int)
    data["raw_text"] = data[text_col].fillna("").astype(str).str.strip()
    data["text"] = data.apply(lambda r: preprocessor.transform(r["raw_text"], r["task_key"]), axis=1)
    data["row_id"] = [f"{task}_{i}" for i, task in enumerate(data["task_key"].tolist())]
    return data


# =========================================================
# UI HELPERS
# =========================================================
def render_header():
    st.title("🎯 Sentiment Analysis Multitask")
    st.markdown(
        "Analisis sentimen **news** dan **sosmed** menggunakan model multitask IndoBERT "
        "yang dimuat dari **MLflow / DagsHub** atau artifact lokal."
    )


def render_sidebar(bundle: Dict):
    st.sidebar.header("🔧 Model Information")
    st.sidebar.write(f"**Device:** {bundle['device']}")
    st.sidebar.write(f"**Base model:** {bundle['base_model_name']}")
    st.sidebar.write(f"**Max length:** {bundle['max_length']}")
    st.sidebar.write(f"**Task mapping:** `{bundle['task2id']}`")
    st.sidebar.write(f"**Tokenizer source:** {bundle['tokenizer_source']}")
    st.sidebar.write(f"**Missing keys:** {len(bundle['missing_keys'])}")
    st.sidebar.write(f"**Unexpected keys:** {len(bundle['unexpected_keys'])}")
    with st.sidebar.expander("Runtime config"):
        st.json(bundle["runtime_cfg"])


def render_probability_chart(result_row: pd.Series):
    probs = pd.DataFrame({
        "label": LABEL_ORDER,
        "probability": [
            float(result_row.get("prob_negatif", 0.0)),
            float(result_row.get("prob_netral", 0.0)),
            float(result_row.get("prob_positif", 0.0)),
        ]
    })
    fig = px.bar(probs, x="label", y="probability", text="probability")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=350, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_batch_summary(results_df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", len(results_df))
    with col2:
        st.metric("Avg Confidence", f"{results_df['max_prob'].mean() * 100:.2f}%")
    with col3:
        st.metric("Avg Sentiment Score", f"{results_df['sentiment_score'].mean():.3f}")
    with col4:
        dominant = results_df["pred_label"].mode().iloc[0] if not results_df.empty else "-"
        st.metric("Dominant Label", dominant)

    label_counts = results_df["pred_label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    fig = px.pie(label_counts, names="label", values="count", title="Distribusi Label Prediksi")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# MAIN APP
# =========================================================
def main():
    render_header()

    with st.spinner("Loading model..."):
        bundle = load_model_bundle_from_secrets()

    render_sidebar(bundle)
    preprocessor = TrainingAlignedPreprocessor()

    tab1, tab2 = st.tabs(["📝 Single Text", "📊 Batch Processing"])

    with tab1:
        st.header("Single Text Analysis")
        task_key = st.selectbox("Pilih task/source", options=list(bundle["task2id"].keys()), key="single_task")
        input_text = st.text_area(
            "Masukkan teks untuk dianalisis:",
            placeholder="Tulis teks di sini...",
            height=140,
        )

        if st.button("🔍 Analyze", type="primary"):
            if input_text.strip():
                single_df = build_single_input_df(
                    text=input_text,
                    task_key=task_key,
                    task2id=bundle["task2id"],
                    preprocessor=preprocessor,
                )
                result_df = predict_deterministic(single_df, bundle, batch_size=1)
                row = result_df.iloc[0].copy()
                row["sentiment_score"] = compute_sentiment_score(row)

                st.subheader("🔸 Hasil Prediksi")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Label", row["pred_label"])
                with col2:
                    st.metric("Confidence", f"{float(row['max_prob']) * 100:.2f}%")
                with col3:
                    st.metric("Sentiment Score", f"{row['sentiment_score']:.3f}")
                with col4:
                    st.metric("Token-ready Text Length", len(str(row["text"]).split()))

                st.markdown("**Teks setelah preprocessing**")
                st.code(str(row["text"]), language="text")

                render_probability_chart(row)
            else:
                st.warning("Masukkan teks terlebih dahulu.")

    with tab2:
        st.header("Batch Processing")
        uploaded_file = st.file_uploader(
            "Upload file CSV atau Excel",
            type=["csv", "xlsx", "xls"],
            help="File minimal harus memiliki satu kolom teks.",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ File uploaded successfully! {len(df)} rows found.")

                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)

                text_column = st.selectbox("Pilih kolom teks:", df.columns, key="batch_text_col")

                task_mode = st.radio(
                    "Task mode",
                    options=["Satu task untuk semua baris", "Pakai kolom task per baris"],
                    horizontal=True,
                )

                task_col = None
                default_task = None
                if task_mode == "Pakai kolom task per baris":
                    task_col = st.selectbox("Pilih kolom task:", df.columns, key="task_col")
                else:
                    default_task = st.selectbox("Pilih default task:", list(bundle["task2id"].keys()), key="default_task")

                max_rows = st.slider("Maximum rows to process:", 1, min(len(df), 500), min(len(df), 100))

                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner(f"Processing {max_rows} rows..."):
                        working_df = df.head(max_rows).copy()
                        prepared_df = build_batch_input_df(
                            df=working_df,
                            text_col=text_column,
                            task_col=task_col,
                            default_task=default_task,
                            task2id=bundle["task2id"],
                            preprocessor=preprocessor,
                        )

                        results_df = predict_deterministic(prepared_df, bundle, batch_size=16)
                        results_df["sentiment_score"] = results_df.apply(compute_sentiment_score, axis=1)

                        st.session_state.batch_results = results_df
                        st.success("✅ Batch processing completed!")

                if "batch_results" in st.session_state and not st.session_state.batch_results.empty:
                    results_df = st.session_state.batch_results.copy()

                    render_batch_summary(results_df)

                    display_cols = [c for c in [
                        "row_id", "task_key", "raw_text", "pred_label", "max_prob",
                        "sentiment_score", "prob_negatif", "prob_netral", "prob_positif"
                    ] if c in results_df.columns]

                    st.dataframe(results_df[display_cols], use_container_width=True)

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv_bytes,
                        file_name="sentiment_multitask_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error reading or processing file: {e}")


if __name__ == "__main__":
    main()
