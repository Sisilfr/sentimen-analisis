import os
import re
import json
import time
import logging
import unicodedata
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Sentiment Analysis Perbankan Indonesia",
    page_icon="📊",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16

# Mapping final yang dipakai app
# Sesuaikan kalau nanti label mapping asli dari training notebook berbeda
LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL = {0: "negatif", 1: "netral", 2: "positif"}

SUPPORTED_TEXT_COLUMNS = [
    "text",
    "raw_text",
    "content",
    "full_text",
    "clean_text",
    "tweet",
    "caption",
    "news",
    "body",
    "title",
]

CACHE_ROOT = Path("/tmp/streamlit_mlflow_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# SECRETS
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def get_required_config() -> Dict[str, str]:
    cfg = {
        "tracking_uri": get_secret("MLFLOW_TRACKING_URI"),
        "dagshub_token": get_secret("DAGSHUB_TOKEN"),
        "dagshub_username": get_secret("DAGSHUB_USERNAME", "token"),
        "run_id": get_secret("MLFLOW_RUN_ID"),
        "artifact_path": get_secret("MLFLOW_ARTIFACT_PATH", "model"),
        "base_model_name": get_secret("BASE_MODEL_NAME", "indobenchmark/indobert-base-p1"),
    }

    missing = [
        k for k, v in cfg.items()
        if k in {"tracking_uri", "dagshub_token", "run_id"} and not v
    ]
    if missing:
        raise ValueError(f"Secrets belum lengkap: {missing}")

    return cfg


# =========================================================
# DAGSHUB / MLFLOW REST HELPERS
# =========================================================
def build_session(username: str, token: str) -> requests.Session:
    s = requests.Session()
    s.auth = (username, token)
    s.headers.update({"User-Agent": "streamlit-mlflow-rest-client/1.0"})
    return s


@st.cache_data(show_spinner=False, ttl=3600)
def get_run_artifact_base(
    tracking_uri: str,
    username: str,
    token: str,
    run_id: str,
) -> str:
    session = build_session(username, token)

    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/get"
    resp = session.get(url, params={"run_id": run_id}, timeout=30)
    resp.raise_for_status()

    payload = resp.json()
    if "run" not in payload:
        raise RuntimeError(f"Response MLflow invalid: {payload}")

    artifact_uri = payload["run"]["info"]["artifact_uri"]
    base = tracking_uri.rstrip("/")

    if artifact_uri.startswith("mlflow-artifacts:/"):
        suffix = artifact_uri.replace("mlflow-artifacts:/", "", 1).lstrip("/")
        return f"{base}/api/2.0/mlflow-artifacts/artifacts/{suffix}"

    if artifact_uri.startswith("runs:/"):
        suffix = artifact_uri.replace("runs:/", "", 1).lstrip("/")
        return f"{base}/api/2.0/mlflow-artifacts/artifacts/{suffix}"

    if artifact_uri.startswith("http://") or artifact_uri.startswith("https://"):
        return artifact_uri.rstrip("/")

    raise ValueError(f"Format artifact_uri belum didukung: {artifact_uri}")


def download_file_if_exists(
    session: requests.Session,
    url: str,
    dst_path: Path,
    timeout: int = 120,
) -> bool:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and dst_path.stat().st_size > 0:
        return True

    resp = session.get(url, stream=True, timeout=timeout)

    if resp.status_code == 404:
        return False

    # DagsHub kadang balas 500 untuk file opsional
    if resp.status_code >= 500:
        return False

    resp.raise_for_status()

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return False

    tmp_path.replace(dst_path)
    return True


def prepare_model_from_mlflow_rest(force_refresh: bool = False) -> Path:
    cfg = get_required_config()

    cache_dir = CACHE_ROOT / cfg["run_id"] / cfg["artifact_path"].strip("/")
    marker = cache_dir / ".ready"

    if marker.exists() and not force_refresh:
        return cache_dir

    if force_refresh and cache_dir.exists():
        for p in cache_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

    artifact_root_http = get_run_artifact_base(
        tracking_uri=cfg["tracking_uri"],
        username=cfg["dagshub_username"],
        token=cfg["dagshub_token"],
        run_id=cfg["run_id"],
    )

    session = build_session(cfg["dagshub_username"], cfg["dagshub_token"])
    artifact_path = cfg["artifact_path"].strip("/")
    cache_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "multitask_model.pt",
        "config_runtime.json",
    ]

    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
        "config.json",
    ]

    for fname in required_files:
        remote_url = f"{artifact_root_http}/{quote(artifact_path)}/{quote(fname)}"
        local_file = cache_dir / fname
        ok = download_file_if_exists(session, remote_url, local_file)
        if not ok:
            raise FileNotFoundError(
                f"File wajib tidak ditemukan di artifact MLflow: {fname}. "
                f"Pastikan file ada di path artifact '{artifact_path}'."
            )

    for fname in optional_files:
        try:
            remote_url = f"{artifact_root_http}/{quote(artifact_path)}/{quote(fname)}"
            local_file = cache_dir / fname
            download_file_if_exists(session, remote_url, local_file)
        except Exception:
            pass

    marker.write_text("ok", encoding="utf-8")
    return cache_dir


# =========================================================
# PREPROCESSING
# =========================================================
class TextPreprocessor:
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    def remove_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def remove_mentions_hashtags(self, text: str) -> str:
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#\w+", " ", text)
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = self.normalize_unicode(text)
        text = text.lower()
        text = self.remove_urls(text)
        text = self.remove_html(text)
        text = self.remove_mentions_hashtags(text)
        text = re.sub(r"[^\w\s\.\,\!\?\-\%\&\/\:;()]", " ", text)
        text = self.remove_extra_whitespace(text)
        return text


# =========================================================
# MODEL
# =========================================================
class MultiTaskIndoBERT(nn.Module):
    """
    Versi deploy sederhana:
    shared encoder + 2 classification heads
    """

    def __init__(self, base_model_name: str, num_labels: int = 3, dropout_prob: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_news = nn.Linear(hidden_size, num_labels)
        self.classifier_sosmed = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, task_name: str):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        if task_name == "news":
            logits = self.classifier_news(pooled)
        elif task_name == "sosmed":
            logits = self.classifier_sosmed(pooled)
        else:
            raise ValueError(f"task_name tidak valid: {task_name}")

        return logits


def safe_read_json(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Gagal membaca json %s: %s", path, e)
    return {}


class ModelBundle:
    def __init__(self, model, tokenizer, device, runtime_config, model_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.runtime_config = runtime_config
        self.model_dir = model_dir


def load_model_bundle(model_dir: Path) -> ModelBundle:
    cfg = get_required_config()

    runtime_config = safe_read_json(model_dir / "config_runtime.json")
    ckpt_path = model_dir / "multitask_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError("multitask_model.pt tidak ditemukan.")

    base_model_name = (
        runtime_config.get("base_model_name")
        or runtime_config.get("model_name")
        or runtime_config.get("pretrained_model_name")
        or cfg["base_model_name"]
    )

    dropout_prob = float(runtime_config.get("dropout_prob", runtime_config.get("dropout", 0.2)))
    max_length = int(runtime_config.get("max_length", DEFAULT_MAX_LENGTH))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        logger.info("Tokenizer loaded from local artifact folder.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        logger.info("Tokenizer local tidak ada. Fallback ke base model HF: %s", base_model_name)

    model = MultiTaskIndoBERT(
        base_model_name=base_model_name,
        num_labels=3,
        dropout_prob=dropout_prob,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    runtime_config["resolved_base_model_name"] = base_model_name
    runtime_config["resolved_dropout"] = dropout_prob
    runtime_config["resolved_max_length"] = max_length

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        runtime_config=runtime_config,
        model_dir=model_dir,
    )


# =========================================================
# CONFIDENCE / POST RULES
# =========================================================
def compute_confidence_tier(conf_percent: float, margin: float) -> str:
    if conf_percent >= 80 and margin >= 0.20:
        return "sangat_kuat"
    if conf_percent >= 65 and margin >= 0.12:
        return "kuat"
    if conf_percent >= 55 and margin >= 0.08:
        return "menengah"
    if conf_percent >= 45 and margin >= 0.05:
        return "lemah"
    return "sangat_lemah"


def apply_task_rules(
    task_name: str,
    pred_label_raw: str,
    confidence: float,
    margin: float,
    neg_prob: float,
    net_prob: float,
    pos_prob: float,
    clean_text: str,
) -> str:
    final_label = pred_label_raw
    text = clean_text.lower()

    negative_cues = [
        "anjlok", "melemah", "turun", "krisis", "macet", "mencekik",
        "mahal", "error", "lemot", "keluhan", "ngeluh", "mengeluhkan",
        "gangguan", "phishing", "penipuan", "scam", "rugi", "beban",
        "likuiditas", "tertekan", "depresiasi", "biaya naik", "saldo hilang",
        "menurun", "terjerat", "kebobolan", "bangkrut"
    ]

    neutral_cues = [
        "mengumumkan", "menginformasikan", "informasi", "penyesuaian",
        "rapat umum", "rups", "jadwal", "standar", "kebijakan",
        "meluncurkan", "merilis", "edukasi", "imbauan", "sosialisasi",
        "menjelaskan", "menyampaikan"
    ]

    has_negative_cue = any(c in text for c in negative_cues)
    has_neutral_cue = any(c in text for c in neutral_cues)

    if task_name == "sosmed":
        if confidence < 45:
            return "netral"
        if margin < 0.08:
            return "netral"

        if pred_label_raw == "positif":
            if confidence < 58:
                return "netral"
            if pos_prob < 0.52:
                return "netral"
            if (pos_prob - net_prob) < 0.08:
                return "netral"
            if has_negative_cue:
                return "netral"

        if pred_label_raw == "negatif":
            if neg_prob < 0.50 and (neg_prob - net_prob) < 0.07 and not has_negative_cue:
                return "netral"

        if has_neutral_cue and net_prob >= 0.32:
            return "netral"

    if task_name == "news":
        if has_neutral_cue and margin < 0.10 and net_prob >= 0.30:
            return "netral"

        if pred_label_raw == "positif":
            if confidence < 60:
                return "netral"
            if pos_prob < 0.55:
                return "netral"
            if (pos_prob - net_prob) < 0.10:
                return "netral"

        if has_negative_cue and neg_prob >= 0.42:
            return "negatif"

        if net_prob >= 0.34 and margin < 0.07:
            return "netral"

    return final_label


# =========================================================
# ENGINE
# =========================================================
class SentimentEngine:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.preprocessor = TextPreprocessor()
        self.id2label = ID2LABEL.copy()
        self.max_length = int(bundle.runtime_config.get("resolved_max_length", DEFAULT_MAX_LENGTH))

    def predict_one(self, text: str, task_name: str) -> Dict:
        clean_text = self.preprocessor.clean_text(text)

        encoded = self.bundle.tokenizer(
            clean_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.bundle.device)
        attention_mask = encoded["attention_mask"].to(self.bundle.device)

        with torch.no_grad():
            logits = self.bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_name=task_name,
            )

        probs = F.softmax(logits, dim=-1).squeeze(0)
        prob_values = probs.detach().cpu().numpy().tolist()

        pred_idx = int(torch.argmax(probs).item())
        pred_label_raw = self.id2label.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx].item() * 100.0)

        sorted_indices = torch.argsort(probs, descending=True)
        top1_idx = int(sorted_indices[0].item())
        top2_idx = int(sorted_indices[1].item())
        top1_prob = float(probs[top1_idx].item())
        top2_prob = float(probs[top2_idx].item())
        margin = top1_prob - top2_prob

        neg_prob = prob_values[0]
        net_prob = prob_values[1]
        pos_prob = prob_values[2]

        pred_label_final = apply_task_rules(
            task_name=task_name,
            pred_label_raw=pred_label_raw,
            confidence=confidence,
            margin=margin,
            neg_prob=neg_prob,
            net_prob=net_prob,
            pos_prob=pos_prob,
            clean_text=clean_text,
        )

        confidence_tier = compute_confidence_tier(confidence, margin)

        # referensi Anda memakai sentiment score [5, 0, -5] untuk positive-neutral-negative.
        # Karena mapping app ini negatif-netral-positif, bobot dibalik jadi [-5, 0, 5].
        score_tensor = torch.tensor([-5.0, 0.0, 5.0], device=self.bundle.device)
        sentiment_score = float(torch.dot(probs, score_tensor).item())

        return {
            "task_key": task_name,
            "raw_text": text,
            "clean_text": clean_text,
            "pred_label": pred_label_final,
            "pred_label_raw": pred_label_raw,
            "confidence": round(confidence, 4),
            "confidence_tier": confidence_tier,
            "sentiment_score": round(sentiment_score, 4),
            "token_length": int(input_ids.shape[1]),
            "margin": round(margin, 6),
            "prob_negatif": round(float(neg_prob), 6),
            "prob_netral": round(float(net_prob), 6),
            "prob_positif": round(float(pos_prob), 6),
        }

    def predict_batch(self, df: pd.DataFrame, text_col: str, task_col: str) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            text = row.get(text_col, "")
            if pd.isna(text):
                continue

            task_name = str(row.get(task_col, "news")).strip().lower()
            if task_name not in {"news", "sosmed"}:
                task_name = "news"

            pred = self.predict_one(str(text), task_name)
            item = row.to_dict()
            item.update(pred)
            rows.append(item)

        return pd.DataFrame(rows)


# =========================================================
# UI HELPERS
# =========================================================
def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in SUPPORTED_TEXT_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def ensure_task_column(df: pd.DataFrame, selected_task: str) -> pd.DataFrame:
    out = df.copy()
    if "task_key" not in out.columns:
        out["task_key"] = selected_task
    else:
        out["task_key"] = out["task_key"].astype(str).str.strip().str.lower()
        out.loc[~out["task_key"].isin(["news", "sosmed"]), "task_key"] = selected_task
    return out


@st.cache_resource(show_spinner=False)
def load_engine_cached(model_dir_str: str):
    model_dir = Path(model_dir_str)
    bundle = load_model_bundle(model_dir)
    return SentimentEngine(bundle)


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.title("🎯 Sentiment Analysis Perbankan Indonesia")

    st.sidebar.header("🔧 Model Configuration")

    if st.sidebar.button("🔄 Load / Reload Model"):
        try:
            t0 = time.time()
            with st.spinner("Menyiapkan model dari MLflow..."):
                model_dir = prepare_model_from_mlflow_rest(force_refresh=True)
                st.session_state["model_dir"] = str(model_dir)
                st.session_state["models_loaded"] = True
                load_engine_cached.clear()

            st.sidebar.success(f"✅ Model loaded in {time.time() - t0:.1f}s")

        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
            st.session_state["models_loaded"] = False

    if "models_loaded" not in st.session_state or not st.session_state["models_loaded"]:
        st.warning("⚠️ Klik **Load / Reload Model** di sidebar dulu.")
        st.info("Model hanya akan diunduh sekali lalu disimpan di cache lokal agar penggunaan berikutnya lebih cepat.")
        return

    try:
        engine = load_engine_cached(st.session_state["model_dir"])
    except Exception as e:
        st.error(f"Gagal inisialisasi engine: {e}")
        return

    tab1, tab2 = st.tabs(["📝 Single Text", "📊 Batch Processing"])

    with tab1:
        st.header("Single Text Analysis")

        task_name = st.radio("Pilih task", ["news", "sosmed"], horizontal=True)
        input_text = st.text_area(
            "Masukkan teks untuk analisis sentimen:",
            placeholder="Tempel teks berita atau teks sosmed di sini...",
            height=140
        )

        if st.button("🔍 Analyze", type="primary"):
            if input_text.strip():
                with st.spinner("Processing..."):
                    result = engine.predict_one(input_text, task_name)

                st.subheader("🔸 Model Results")
                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Final Label", result["pred_label"])
                    st.metric("Raw Label", result["pred_label_raw"])
                    st.metric("Confidence", f"{result['confidence']:.2f}%")

                with c2:
                    st.metric("Confidence Tier", result["confidence_tier"])
                    st.metric("Margin", f"{result['margin']:.4f}")
                    st.metric("Token Length", result["token_length"])

                st.write("### Probability")
                prob_df = pd.DataFrame({
                    "label": ["negatif", "netral", "positif"],
                    "probability": [
                        result["prob_negatif"],
                        result["prob_netral"],
                        result["prob_positif"],
                    ],
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                st.write("### Detail")
                st.json({
                    "sentiment_score": result["sentiment_score"],
                    "clean_text": result["clean_text"],
                })
            else:
                st.warning("Masukkan teks dulu.")

    with tab2:
        st.header("Batch Processing")

        uploaded_file = st.file_uploader(
            "Upload CSV / Excel file",
            type=["csv", "xlsx", "xls"],
            help="File harus punya minimal satu kolom teks."
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

                detected_text_col = detect_text_column(df)
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns,
                    index=list(df.columns).index(detected_text_col) if detected_text_col in df.columns else 0
                )

                default_task = st.radio("Default task", ["news", "sosmed"], horizontal=True, key="batch_task")
                max_rows = st.slider("Maximum rows to process:", 1, min(len(df), 500), min(len(df), 50))

                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner(f"Processing {max_rows} rows..."):
                        df_proc = df.head(max_rows).copy()
                        df_proc = ensure_task_column(df_proc, default_task)
                        results_df = engine.predict_batch(
                            df_proc,
                            text_col=text_column,
                            task_col="task_key",
                        )
                        st.session_state["batch_results"] = results_df

                    st.success("✅ Batch processing completed!")

                if "batch_results" in st.session_state and not st.session_state["batch_results"].empty:
                    results_df = st.session_state["batch_results"]

                    st.write("### Distribusi Label")
                    st.dataframe(
                        results_df["pred_label"].value_counts(dropna=False).rename("count").to_frame(),
                        use_container_width=True
                    )

                    st.write("### Distribusi Confidence Tier")
                    st.dataframe(
                        results_df["confidence_tier"].value_counts(dropna=False).rename("count").to_frame(),
                        use_container_width=True
                    )

                    st.write("### Hasil Batch")
                    display_cols = [c for c in [
                        text_column,
                        "clean_text",
                        "task_key",
                        "pred_label",
                        "pred_label_raw",
                        "confidence",
                        "confidence_tier",
                        "margin",
                        "token_length",
                        "prob_negatif",
                        "prob_netral",
                        "prob_positif",
                        "sentiment_score",
                    ] if c in results_df.columns]

                    st.dataframe(results_df[display_cols], use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv_data,
                        file_name="sentiment_multitask_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
