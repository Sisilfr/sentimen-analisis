import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

st.set_page_config(page_title="Sentiment Analysis Multitask", page_icon="📊", layout="wide")

LABEL_ORDER = ["Negatif", "Netral", "Positif"]


def _import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        return mlflow, MlflowClient
    except Exception as e:
        raise ImportError(
            "MLflow gagal diimport. Cek pin dependency (mlflow/protobuf/opentelemetry) "
            "atau gunakan LOCAL_MODEL_DIR."
        ) from e


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
            task: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))
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

        logits = torch.empty(pooled.size(0), self.num_labels, dtype=torch.float32, device=pooled.device)
        for task_id, task_name in self.task_id_to_name.items():
            task_mask = (task_ids == task_id)
            if task_mask.any():
                logits[task_mask] = self.heads[task_name](pooled[task_mask]).float()
        return logits


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

    def normalize_informal_words(self, text: str) -> str:
        return " ".join(self.slang_map.get(tok, tok) for tok in text.split())

    def strip_social_boilerplate(self, text: str) -> str:
        for pattern in [
            r'subscribe\s+channel', r'klik\s+link\s+di\s+bio', r'hubungi\s+kami',
            r'contact\s+us', r'wa\s*:\s*\d+', r'whatsapp\s*:\s*\d+', r'promo\s+terbatas'
        ]:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return text

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
        return text[:max_chars].strip() if len(text) > max_chars else text.strip()

    def transform(self, text: str, task_key: str) -> str:
        text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = self.normalize_informal_words(text)
        text = self.strip_social_boilerplate(text)
        text = self.compress_salient_text(text)
        text = self.clean_text_basic(text)
        return text.strip()


def _normalize_label_mapping(raw_label2id: Dict):
    raw_label2id = raw_label2id or {"Negatif": 0, "Netral": 1, "Positif": 2}
    return {int(v): str(k).strip() for k, v in raw_label2id.items()}


def _find_bundle_root(base_dir: str) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Path model tidak ditemukan: {base_dir}")
    if (base_dir / "config_runtime.json").exists() and (base_dir / "multitask_model.pt").exists():
        return base_dir
    for cfg in base_dir.rglob("config_runtime.json"):
        parent = cfg.parent
        if (parent / "multitask_model.pt").exists():
            return parent
    raise FileNotFoundError(f"Tidak ditemukan config_runtime.json + multitask_model.pt di {base_dir}")


def _load_runtime_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "config_runtime.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_runtime.json tidak ditemukan di {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _download_mlflow_artifacts(run_id: str, dst_dir: str, tracking_uri: Optional[str] = None) -> Path:
    mlflow, MlflowClient = _import_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    all_paths = []

    def walk(path=""):
        for item in client.list_artifacts(run_id, path):
            all_paths.append(item.path)
            if item.is_dir:
                walk(item.path)

    walk("")
    targets = {
        "config_runtime.json": next((p for p in all_paths if p.endswith("/config_runtime.json") or p == "config_runtime.json"), None),
        "multitask_model.pt": next((p for p in all_paths if p.endswith("/multitask_model.pt") or p == "multitask_model.pt"), None),
        "tokenizer": next((p for p in all_paths if p.endswith("/tokenizer") or p == "tokenizer"), None),
        "encoder": next((p for p in all_paths if p.endswith("/encoder") or p == "encoder"), None),
    }

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for required in ("config_runtime.json", "multitask_model.pt"):
        p = targets[required]
        if p is None:
            raise FileNotFoundError(f"Artifact '{required}' tidak ditemukan pada run {run_id}")
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=p, dst_path=str(dst))

    for optional in ("tokenizer", "encoder"):
        p = targets[optional]
        if p is not None:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=p, dst_path=str(dst))

    return dst


@st.cache_resource(show_spinner=False)
def load_model_bundle_from_secrets():
    secrets = st.secrets
    local_model_dir = secrets.get("LOCAL_MODEL_DIR", "")
    run_id = secrets.get("MLFLOW_RUN_ID", "")
    tracking_uri = secrets.get("MLFLOW_TRACKING_URI", "")

    if run_id and tracking_uri:
        tmp_dir = Path("/tmp") / "mlflow_artifacts_streamlit"
        downloaded = _download_mlflow_artifacts(run_id, str(tmp_dir), tracking_uri)
        model_dir = _find_bundle_root(downloaded)
    elif local_model_dir:
        model_dir = _find_bundle_root(local_model_dir)
    else:
        raise ValueError("Isi MLFLOW_TRACKING_URI + MLFLOW_RUN_ID, atau LOCAL_MODEL_DIR.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_cfg = _load_runtime_config(model_dir)
    base_model_name = runtime_cfg.get("base_model_name") or runtime_cfg.get("base_model") or "indobenchmark/indobert-base-p1"
    max_length = int(runtime_cfg.get("max_length", 256))
    dropout = float(runtime_cfg.get("dropout", 0.2))
    raw_task2id = runtime_cfg.get("task2id", {"news": 0, "sosmed": 1})
    task2id = {str(k).strip().lower(): int(v) for k, v in raw_task2id.items()}
    id2label = _normalize_label_mapping(runtime_cfg.get("label2id", {"Negatif": 0, "Netral": 1, "Positif": 2}))
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
    model = MultiTaskIndoBERT(encoder_source, num_labels=len(id2label), task_names=tuple(task_names), dropout=dropout)

    state = torch.load(model_dir / "multitask_model.pt", map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    return {
        "model": model, "tokenizer": tokenizer, "device": device, "max_length": max_length,
        "task2id": task2id, "id2label": id2label, "runtime_cfg": runtime_cfg,
        "missing_keys": missing, "unexpected_keys": unexpected,
        "model_dir": str(model_dir), "tokenizer_source": tokenizer_source, "base_model_name": base_model_name,
    }


@torch.no_grad()
def predict_deterministic(df: pd.DataFrame, model_bundle: Dict, batch_size: int = 16) -> pd.DataFrame:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    max_length = model_bundle["max_length"]
    id2label = model_bundle["id2label"]
    dev = model_bundle["device"]

    rows = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size].copy()
        enc = tokenizer(
            batch_df["text"].astype(str).tolist(),
            truncation=True, max_length=max_length, padding=True,
            return_attention_mask=True, return_tensors="pt"
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        task_ids = torch.tensor(batch_df["task_id"].tolist(), dtype=torch.long, device=dev)

        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], task_ids=task_ids)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)

        out = batch_df.copy()
        out["pred_id"] = pred_ids
        out["pred_label"] = [id2label[int(i)] for i in pred_ids]
        out["max_prob"] = probs.max(axis=1)
        out["top2_margin"] = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
        for idx, label_name in sorted(id2label.items()):
            out[f"prob_{str(label_name).lower()}"] = probs[:, idx]
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def compute_sentiment_score(row: pd.Series) -> float:
    return float(row.get("prob_positif", 0.0)) * 5.0 + float(row.get("prob_negatif", 0.0)) * (-5.0)


def build_input_df(df: pd.DataFrame, text_col: str, task_col: Optional[str], default_task: Optional[str], task2id: Dict[str, int], preprocessor):
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
    invalid = sorted(set(data["task_key"]) - set(task2id.keys()))
    if invalid:
        raise ValueError(f"Task tidak valid: {invalid}. Pilihan valid: {list(task2id.keys())}")
    data = data.reset_index(drop=True)
    data["task_id"] = data["task_key"].map(task2id).astype(int)
    data["raw_text"] = data[text_col].fillna("").astype(str).str.strip()
    data["text"] = data.apply(lambda r: preprocessor.transform(r["raw_text"], r["task_key"]), axis=1)
    data["row_id"] = [f"{task}_{i}" for i, task in enumerate(data["task_key"].tolist())]
    return data


def render_header():
    st.title("🎯 Sentiment Analysis Multitask")


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
        st.metric("Dominant Label", results_df["pred_label"].mode().iloc[0] if not results_df.empty else "-")
    label_counts = results_df["pred_label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    st.plotly_chart(px.pie(label_counts, names="label", values="count", title="Distribusi Label Prediksi"), use_container_width=True)


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
        input_text = st.text_area("Masukkan teks untuk dianalisis:", placeholder="Tulis teks di sini...", height=140)
        if st.button("🔍 Analyze", type="primary"):
            if input_text.strip():
                single_df = pd.DataFrame([{
                    "row_id": f"{task_key}_single_0",
                    "task_key": task_key,
                    "task_id": int(bundle["task2id"][task_key]),
                    "raw_text": input_text,
                    "text": preprocessor.transform(input_text, task_key),
                }])
                result_df = predict_deterministic(single_df, bundle, batch_size=1)
                row = result_df.iloc[0].copy()
                row["sentiment_score"] = compute_sentiment_score(row)

                st.subheader("🔸 Hasil Prediksi")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Label", row["pred_label"])
                c2.metric("Confidence", f"{float(row['max_prob']) * 100:.2f}%")
                c3.metric("Sentiment Score", f"{row['sentiment_score']:.3f}")
                c4.metric("Token-ready Text Length", len(str(row["text"]).split()))
                st.markdown("**Teks setelah preprocessing**")
                st.code(str(row["text"]), language="text")
                render_probability_chart(row)
            else:
                st.warning("Masukkan teks terlebih dahulu.")

    with tab2:
        st.header("Batch Processing")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success(f"✅ File uploaded successfully! {len(df)} rows found.")
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                text_column = st.selectbox("Pilih kolom teks:", df.columns, key="batch_text_col")
                task_mode = st.radio("Task mode", ["Satu task untuk semua baris", "Pakai kolom task per baris"], horizontal=True)
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
                        prepared_df = build_input_df(working_df, text_column, task_col, default_task, bundle["task2id"], preprocessor)
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
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_multitask_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error reading or processing file: {e}")


if __name__ == "__main__":
    main()
