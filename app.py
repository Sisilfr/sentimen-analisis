
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

st.set_page_config(
    page_title="Sentiment Analysis Perbankan Indonesia",
    page_icon="🏦",
    layout="wide",
)

# =========================
# CONSTANTS
# =========================
DEFAULT_BASE_MODEL = "indobenchmark/indobert-base-p1"
DEFAULT_MAX_LENGTH = 256
DEFAULT_DROPOUT = 0.20

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
TASK2ID = {"news": 0, "sosmed": 1}
ID2TASK = {v: k for k, v in TASK2ID.items()}

DEFAULT_MODEL_CANDIDATES = [
    Path("model"),
    Path("best_model"),
    Path("artifacts/best_model"),
    Path("/mount/src/model"),
]

# =========================
# PREPROCESSOR
# Match inference notebook:
# helper-based, no slang dictionary
# =========================
class SourceAwarePreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.html_pattern = re.compile(r"<[^>]+>")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.zero_width_pattern = re.compile(r"[\u200b-\u200f\uFEFF]")
        self.multispace_pattern = re.compile(r"\s+")
        self.repeat_char_pattern = re.compile(r"(.)\1{3,}", re.IGNORECASE)
        self.repeat_punct_pattern = re.compile(r"([!?.,])\1{2,}")
        self.space_before_punct = re.compile(r"\s+([.,!?;:])")
        self.bad_escape_pattern = re.compile(r"\\[nrt]")
        self.orphan_noise_pattern = re.compile(r"\b[nrt]\b", flags=re.IGNORECASE)

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def remove_zero_width(self, text: str) -> str:
        return self.zero_width_pattern.sub(" ", text)

    def strip_html(self, text: str) -> str:
        return self.html_pattern.sub(" ", text)

    def normalize_url(self, text: str) -> str:
        return self.url_pattern.sub(" [url] ", text)

    def normalize_mentions(self, text: str) -> str:
        return self.mention_pattern.sub(" [user] ", text)

    def normalize_hashtags(self, text: str) -> str:
        return self.hashtag_pattern.sub(r" \1 ", text)

    def normalize_escaped_chars(self, text: str) -> str:
        text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        return self.bad_escape_pattern.sub(" ", text)

    def normalize_repeated_chars(self, text: str) -> str:
        def repl(match):
            ch = match.group(1)
            return ch * 2
        return self.repeat_char_pattern.sub(repl, text)

    def normalize_repeated_punct(self, text: str) -> str:
        def repl(match):
            return match.group(1) * 2
        return self.repeat_punct_pattern.sub(repl, text)

    def remove_orphan_noise(self, text: str) -> str:
        return self.orphan_noise_pattern.sub(" ", text)

    def clean_basic(self, text: str) -> str:
        text = self.normalize_unicode(text)
        text = self.remove_zero_width(text)
        text = self.strip_html(text)
        text = self.normalize_escaped_chars(text)
        text = self.normalize_url(text)
        text = self.normalize_mentions(text)
        text = self.normalize_hashtags(text)
        text = self.normalize_repeated_chars(text)
        text = self.normalize_repeated_punct(text)
        text = self.space_before_punct.sub(r"\1", text)
        text = self.remove_orphan_noise(text)
        text = self.multispace_pattern.sub(" ", text).strip()
        return text

    def preprocess_news(self, text: str) -> str:
        return self.clean_basic(text).lower().strip()

    def preprocess_sosmed(self, text: str) -> str:
        return self.clean_basic(text).lower().strip()

    def transform(self, text: str, task_key: str) -> str:
        text = "" if pd.isna(text) else str(text)
        return self.preprocess_news(text) if task_key == "news" else self.preprocess_sosmed(text)


# =========================
# MODEL ARCHITECTURE
# Match training/inference notebook
# shared encoder + pooled(0.5 cls + 0.5 mean)
# layernorm + dropout + task heads
# =========================
class MultiTaskIndoBERT(nn.Module):
    def __init__(self, base_model: str, num_labels: int = 3, task_names=("news", "sosmed"), dropout=0.20):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model, config=self.config)
        hidden_size = self.config.hidden_size
        self.num_labels = num_labels
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_labels),
                )
                for task in task_names
            }
        )
        self.task_id_to_name = {0: "news", 1: "sosmed"}

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
            device=pooled.device,
        )

        for task_id, task_name in self.task_id_to_name.items():
            task_mask = task_ids == task_id
            if task_mask.any():
                logits[task_mask] = self.heads[task_name](pooled[task_mask]).float()

        return logits


def resolve_model_dir(input_path: str) -> Path:
    if input_path and input_path.strip():
        path = Path(input_path.strip())
        if path.exists():
            return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Folder model tidak ditemukan. Pastikan folder berisi multitask_model.pt "
        "dan idealnya config_runtime.json."
    )


def load_runtime_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "config_runtime.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=True)
def load_inference_assets(model_dir_str: str):
    model_dir = Path(model_dir_str)
    runtime_cfg = load_runtime_config(model_dir)

    base_model_name = runtime_cfg.get("base_model_name", runtime_cfg.get("base_model", DEFAULT_BASE_MODEL))
    max_length = int(runtime_cfg.get("max_length", DEFAULT_MAX_LENGTH))
    dropout = float(runtime_cfg.get("dropout", DEFAULT_DROPOUT))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = MultiTaskIndoBERT(
        base_model=base_model_name,
        num_labels=len(LABEL2ID),
        task_names=("news", "sosmed"),
        dropout=dropout,
    )

    state_path = model_dir / "multitask_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"State dict tidak ditemukan: {state_path}")

    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "runtime_cfg": runtime_cfg,
        "max_length": max_length,
        "dropout": dropout,
        "model_dir": str(model_dir.resolve()),
        "base_model_name": base_model_name,
    }


def pick_text_column(df: pd.DataFrame) -> str:
    priority = ["text", "content", "full_text", "tweet", "caption", "body", "raw_text"]
    for c in priority:
        if c in df.columns:
            return c
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("Tidak menemukan kolom teks yang valid.")
    return max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())


def predict_batch(
    df: pd.DataFrame,
    task_key: str,
    assets: Dict,
    preprocessor: SourceAwarePreprocessor,
    batch_size: int = 16,
    positive_gate_enabled: bool = False,
    positive_gate_threshold: float = 0.85,
):
    model = assets["model"]
    tokenizer = assets["tokenizer"]
    device = assets["device"]
    max_length = assets["max_length"]

    work_df = df.copy().reset_index(drop=True)
    work_df["task_key"] = task_key
    work_df["task_id"] = TASK2ID[task_key]
    work_df["text_clean"] = work_df["text_input"].astype(str).apply(lambda x: preprocessor.transform(x, task_key))

    rows = []
    for start in range(0, len(work_df), batch_size):
        batch_df = work_df.iloc[start:start + batch_size]
        texts = batch_df["text_clean"].tolist()
        task_ids = torch.tensor(batch_df["task_id"].tolist(), dtype=torch.long, device=device)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                task_ids=task_ids,
            )
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        pred_ids = probs.argmax(axis=1)

        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            prob_neg = float(probs[local_idx, LABEL2ID["Negatif"]])
            prob_net = float(probs[local_idx, LABEL2ID["Netral"]])
            prob_pos = float(probs[local_idx, LABEL2ID["Positif"]])

            pred_id = int(pred_ids[local_idx])
            pred_label = ID2LABEL[pred_id]

            if positive_gate_enabled and pred_label == "Positif" and prob_pos < positive_gate_threshold:
                pred_label_final = "Netral"
            else:
                pred_label_final = pred_label

            rows.append(
                {
                    "row_id": row["row_id"],
                    "task_key": task_key,
                    "text_input": row["text_input"],
                    "text_clean": row["text_clean"],
                    "pred_label": pred_label,
                    "pred_label_final": pred_label_final,
                    "prob_negatif": prob_neg,
                    "prob_netral": prob_net,
                    "prob_positif": prob_pos,
                    "max_prob": max(prob_neg, prob_net, prob_pos),
                    "char_len": len(str(row["text_input"])),
                    "word_len": len(str(row["text_clean"]).split()),
                }
            )

    out = pd.DataFrame(rows)
    out["confidence_pct"] = (out["max_prob"] * 100).round(2)
    return out


def single_text_card(result_row: pd.Series):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediksi Final", result_row["pred_label_final"])
    c2.metric("Confidence", f'{result_row["confidence_pct"]:.2f}%')
    c3.metric("Task", result_row["task_key"])
    c4.metric("Word Count", int(result_row["word_len"]))

    prob_df = pd.DataFrame(
        {
            "label": ["Negatif", "Netral", "Positif"],
            "probabilitas": [
                result_row["prob_negatif"],
                result_row["prob_netral"],
                result_row["prob_positif"],
            ],
        }
    )
    st.bar_chart(prob_df.set_index("label"))


def main():
    st.title("🏦 Deployment Streamlit — IndoBERT Multitask Perbankan")
    st.caption(
        "Inference setelah training multitask multisource (news + sosmed) "
        "dengan preprocessing helper-based dan arsitektur yang match ke notebook training/inference."
    )

    with st.sidebar:
        st.header("Konfigurasi Model")
        model_dir_input = st.text_input(
            "Folder model",
            value="model",
            help="Folder berisi multitask_model.pt dan, bila ada, config_runtime.json",
        )
        batch_size = st.slider("Batch size inference", min_value=1, max_value=64, value=16, step=1)
        positive_gate_enabled = st.checkbox(
            "Aktifkan positive gate",
            value=False,
            help="Opsional: prediksi Positif dengan confidence rendah dipaksa menjadi Netral.",
        )
        positive_gate_threshold = st.slider(
            "Threshold positive gate",
            min_value=0.50,
            max_value=0.99,
            value=0.85,
            step=0.01,
            disabled=not positive_gate_enabled,
        )

    try:
        resolved_model_dir = resolve_model_dir(model_dir_input)
        assets = load_inference_assets(str(resolved_model_dir))
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

    preprocessor = SourceAwarePreprocessor()

    with st.expander("Detail model", expanded=False):
        st.write(
            {
                "model_dir": assets["model_dir"],
                "base_model_name": assets["base_model_name"],
                "max_length": assets["max_length"],
                "dropout": assets["dropout"],
                "device": assets["device"],
            }
        )
        st.json(assets["runtime_cfg"] if assets["runtime_cfg"] else {"note": "config_runtime.json tidak ditemukan"})

    tab1, tab2 = st.tabs(["Single Inference", "Batch Inference"])

    with tab1:
        st.subheader("Prediksi teks tunggal")
        task_key = st.radio("Pilih sumber teks", options=["news", "sosmed"], horizontal=True)
        user_text = st.text_area("Masukkan teks", height=180, placeholder="Masukkan teks berita atau sosmed di sini...")

        if st.button("Prediksi teks", type="primary"):
            if not user_text.strip():
                st.warning("Teks tidak boleh kosong.")
            else:
                single_df = pd.DataFrame(
                    [{"row_id": "single_0", "text_input": user_text}]
                )
                result_df = predict_batch(
                    single_df,
                    task_key=task_key,
                    assets=assets,
                    preprocessor=preprocessor,
                    batch_size=1,
                    positive_gate_enabled=positive_gate_enabled,
                    positive_gate_threshold=positive_gate_threshold,
                )
                result_row = result_df.iloc[0]
                single_text_card(result_row)

                with st.expander("Teks setelah preprocessing"):
                    st.code(result_row["text_clean"])

    with tab2:
        st.subheader("Prediksi batch file CSV/XLSX")
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
                st.stop()

            st.write("Preview data:")
            st.dataframe(df.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Kolom teks", options=df.columns, index=df.columns.get_loc(pick_text_column(df)))
            with col2:
                task_mode = st.radio(
                    "Mode task",
                    options=["manual", "dari kolom task_key"],
                    horizontal=True,
                    help="Gunakan manual jika seluruh file hanya news atau hanya sosmed.",
                )

            if task_mode == "manual":
                chosen_task = st.selectbox("Task untuk semua baris", options=["news", "sosmed"])
                df_work = pd.DataFrame(
                    {
                        "row_id": [f"row_{i}" for i in range(len(df))],
                        "text_input": df[text_col].astype(str),
                        "task_key": chosen_task,
                    }
                )
            else:
                task_col = st.selectbox("Kolom task", options=df.columns)
                df_work = pd.DataFrame(
                    {
                        "row_id": [f"row_{i}" for i in range(len(df))],
                        "text_input": df[text_col].astype(str),
                        "task_key": df[task_col].astype(str).str.lower().str.strip(),
                    }
                )
                invalid_mask = ~df_work["task_key"].isin(["news", "sosmed"])
                if invalid_mask.any():
                    st.error("Kolom task mengandung nilai di luar ['news', 'sosmed'].")
                    st.dataframe(df_work.loc[invalid_mask].head(20), use_container_width=True)
                    st.stop()

            if st.button("Proses batch", type="primary"):
                all_parts = []
                progress = st.progress(0.0)

                unique_tasks = df_work["task_key"].unique().tolist()
                processed = 0

                for task in unique_tasks:
                    part = df_work[df_work["task_key"] == task][["row_id", "text_input"]].copy()
                    pred_part = predict_batch(
                        part,
                        task_key=task,
                        assets=assets,
                        preprocessor=preprocessor,
                        batch_size=batch_size,
                        positive_gate_enabled=positive_gate_enabled,
                        positive_gate_threshold=positive_gate_threshold,
                    )
                    all_parts.append(pred_part)
                    processed += len(part)
                    progress.progress(min(processed / max(len(df_work), 1), 1.0))

                result_df = pd.concat(all_parts, ignore_index=True)

                st.success(f"Berhasil memproses {len(result_df)} baris.")
                st.dataframe(result_df, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total baris", len(result_df))
                c2.metric("Positif", int((result_df["pred_label_final"] == "Positif").sum()))
                c3.metric("Negatif", int((result_df["pred_label_final"] == "Negatif").sum()))

                summary = (
                    result_df.groupby(["task_key", "pred_label_final"])
                    .size()
                    .rename("jumlah")
                    .reset_index()
                )
                st.subheader("Ringkasan prediksi")
                st.dataframe(summary, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download hasil CSV",
                    data=csv_bytes,
                    file_name="predictions_streamlit_multitask_perbankan.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
