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

# Mapping label deploy
# Sesuaikan kalau config runtime Anda nanti berbeda
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
    Versi deploy stabil:
    shared encoder + 2 classification heads
    """
    def __init__(self, base_model_name: str, num_labels: int = 3, dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_news = nn.Linear(hidden_size, num_labels)
        self.classifier_sosmed = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, task_name: str):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (
            outputs.pooler_output
            if getattr(outputs, "pooler_output", None) is not None
            else outputs.last_hidden_state[:, 0]
        )
        pooled = self.dropout(pooled)

        if task_name == "news":
            return self.classifier_news(pooled)
        elif task_name == "sosmed":
            return self.classifier_sosmed(pooled)
        else:
            raise ValueError(f"task_name tidak valid: {task_name}")


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
    dropout_prob = float(runtime_config.get("dropout_prob", runtime_config.get("dropout", 0.1)))
    max_length = int(runtime_config.get("max_length", DEFAULT_MAX_LENGTH))

    # tokenizer lokal opsional; fallback ke base model
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
# ENGINE
# =========================================================
class SentimentEngine:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.preprocessor = TextPreprocessor()

        label_map = bundle.runtime_config.get("label_map")
        if isinstance(label_map, dict):
            self.id2label = {int(k): v for k, v in label_map.items()}
        else:
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
        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.id2label.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx].item() * 100.0)

        prob_values = probs.detach().cpu().numpy().tolist()

        # Mapping app ini: negatif, netral, positif
        score_tensor = torch.tensor([-5.0, 0.0, 5.0], device=self.bundle.device)
        sentiment_score = float(torch.dot(probs, score_tensor).item())

        result = {
            "task_key": task_name,
            "raw_text": text,
            "clean_text": clean_text,
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "token_length": int(input_ids.shape[1]),
            "sentiment_score": round(sentiment_score, 4),
        }

        for idx, val in enumerate(prob_values):
            label_name = self.id2label.get(idx, str(idx))
            result[f"prob_{label_name}"] = round(float(val), 6)

        return result

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
# MAIN UI
# =========================================================
def main():
    st.title("📊 Sentiment Analysis Perbankan Indonesia")

    st.sidebar.header("🔧 Model Configuration")

    if st.sidebar.button("🔄 Load / Reload Model"):
        try:
            t0 = time.time()
            with st.spinner("Mengunduh model dari MLflow/DagsHub..."):
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

    tab1, tab2 = st.tabs(["📝 Single Inference", "📊 Batch Inference"])

    with tab1:
        st.subheader("Single Inference")

        task_name = st.radio("Pilih task", ["news", "sosmed"], horizontal=True)
        input_text = st.text_area("Masukkan teks", height=180)

        if st.button("🔍 Predict"):
            if not input_text.strip():
                st.warning("Teks belum diisi.")
            else:
                result = engine.predict_one(input_text, task_name)
                st.write("### Hasil Prediksi")
                st.write(f"**Label:** {result['pred_label']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}%")
                st.write(f"**Token Length:** {result['token_length']}")
                st.write(f"**Sentiment Score:** {result['sentiment_score']:.4f}")

                prob_df = pd.DataFrame({
                    "label": [k.replace("prob_", "") for k in result if k.startswith("prob_")],
                    "probability": [result[k] for k in result if k.startswith("prob_")]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Batch Inference")

        uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
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
                text_col = st.selectbox(
                    "Pilih kolom teks",
                    df.columns.tolist(),
                    index=df.columns.tolist().index(detected_text_col) if detected_text_col in df.columns else 0
                )

                default_task = st.radio("Default task", ["news", "sosmed"], horizontal=True, key="default_task")
                max_rows = st.number_input("Max rows", min_value=1, max_value=len(df), value=min(100, len(df)))

                if st.button("🚀 Process Batch"):
                    df_proc = df.head(int(max_rows)).copy()
                    df_proc = ensure_task_column(df_proc, default_task)

                    with st.spinner("Processing batch..."):
                        result_df = engine.predict_batch(df_proc, text_col=text_col, task_col="task_key")

                    st.session_state["batch_results"] = result_df
                    st.success(f"Selesai memproses {len(result_df)} row.")

                if "batch_results" in st.session_state and not st.session_state["batch_results"].empty:
                    results_df = st.session_state["batch_results"]

                    st.write("### Hasil Batch")
                    st.dataframe(results_df, use_container_width=True)

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download CSV",
                        data=csv_bytes,
                        file_name="batch_inference_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
