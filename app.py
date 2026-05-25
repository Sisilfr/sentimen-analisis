import os
import re
import json
import time
import logging
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

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

# =========================================================
# REFERENCE CONFIG (sesuai notebook training + inference)
# =========================================================
LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TASK2ID = {"news": 0, "sosmed": 1}
ID2TASK = {v: k for k, v in TASK2ID.items()}

DEFAULT_BASE_MODEL = "indobenchmark/indobert-base-p1"
DEFAULT_DROPOUT = 0.20
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_MC_PASSES = 7

# selective confidence threshold (sesuai notebook inference)
AUTO_GT_MIN_CONF = 0.80
AUTO_GT_MIN_AGREEMENT = 0.80
AUTO_GT_MAX_NORM_ENTROPY = 0.35
AUTO_GT_MIN_MARGIN = 0.20

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
        "base_model_name": get_secret("BASE_MODEL_NAME", DEFAULT_BASE_MODEL),
    }

    missing = [
        k for k, v in cfg.items()
        if k in {"tracking_uri", "dagshub_token", "run_id"} and not v
    ]
    if missing:
        raise ValueError(f"Secrets belum lengkap: {missing}")

    return cfg


# =========================================================
# DAGSHUB / MLFLOW REST LIGHT CLIENT
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

    # DagsHub kadang balas 500 untuk file opsional yang tidak ada
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
# PREPROCESSING (helper-based, bukan dictionary slang)
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
# MODEL SESUAI TRAINING NOTEBOOK
# =========================================================
class MultiTaskIndoBERT(nn.Module):
    def __init__(
        self,
        base_model: str,
        num_labels: int = 3,
        task_names=("news", "sosmed"),
        dropout: float = DEFAULT_DROPOUT,
    ):
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
            task_mask = (task_ids == task_id)
            if task_mask.any():
                task_logits = self.heads[task_name](pooled[task_mask]).float()
                logits[task_mask] = task_logits

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

    dropout_prob = float(runtime_config.get("dropout_prob", runtime_config.get("dropout", DEFAULT_DROPOUT)))
    max_length = int(runtime_config.get("max_length", DEFAULT_MAX_LENGTH))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        logger.info("Tokenizer loaded from local artifact folder.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        logger.info("Tokenizer local tidak ada. Fallback ke base model HF: %s", base_model_name)

    model = MultiTaskIndoBERT(
        base_model=base_model_name,
        dropout=dropout_prob,
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
# MC DROPOUT INFERENCE (sesuai notebook inference)
# =========================================================
def normalized_entropy(prob_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    entropy = -np.sum(prob_matrix * np.log(prob_matrix + eps), axis=1)
    max_entropy = np.log(prob_matrix.shape[1])
    return entropy / max_entropy


def margin_top2(prob_matrix: np.ndarray) -> np.ndarray:
    sorted_probs = -np.sort(-prob_matrix, axis=1)
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def confidence_tier_from_metrics(max_prob: float, agreement: float) -> str:
    if (max_prob >= 0.90) and (agreement >= 0.90):
        return "sangat_kuat"
    if (max_prob >= 0.80) and (agreement >= 0.80):
        return "kuat"
    if max_prob >= 0.70:
        return "menengah"
    return "lemah"


def is_auto_gt_candidate(max_prob: float, agreement: float, norm_entropy: float, margin: float) -> bool:
    return (
        (max_prob >= AUTO_GT_MIN_CONF) and
        (agreement >= AUTO_GT_MIN_AGREEMENT) and
        (norm_entropy <= AUTO_GT_MAX_NORM_ENTROPY) and
        (margin >= AUTO_GT_MIN_MARGIN)
    )


def apply_task_post_rule(
    task_name: str,
    pred_label_raw: str,
    max_prob: float,
    agreement: float,
    norm_entropy: float,
    margin: float,
    neg_prob: float,
    net_prob: float,
    pos_prob: float,
    clean_text: str,
) -> str:
    """
    Rule deployment konservatif.
    Tujuannya bukan memalsukan confidence, tapi menahan output ambigu.
    """

    final_label = pred_label_raw
    text = clean_text.lower()

    negative_cues = [
        "anjlok", "melemah", "turun", "krisis", "macet", "mencekik",
        "mahal", "error", "lemot", "keluhan", "ngeluh", "mengeluhkan",
        "gangguan", "phishing", "penipuan", "scam", "rugi", "beban",
        "likuiditas", "tertekan", "depresiasi", "biaya naik", "saldo hilang"
    ]

    neutral_cues = [
        "mengumumkan", "menginformasikan", "informasi", "penyesuaian",
        "rapat umum", "rups", "jadwal", "standar", "kebijakan",
        "meluncurkan", "merilis", "edukasi", "imbauan", "sosialisasi"
    ]

    has_negative_cue = any(c in text for c in negative_cues)
    has_neutral_cue = any(c in text for c in neutral_cues)

    if task_name == "sosmed":
        if max_prob < 0.55:
            return "netral"
        if agreement < 0.65:
            return "netral"
        if margin < 0.10:
            return "netral"
        if norm_entropy > 0.60:
            return "netral"

        if pred_label_raw == "positif":
            if pos_prob < 0.58:
                return "netral"
            if (pos_prob - net_prob) < 0.10:
                return "netral"
            if has_negative_cue:
                return "netral"

        if pred_label_raw == "negatif":
            if neg_prob < 0.52 and (neg_prob - net_prob) < 0.08 and not has_negative_cue:
                return "netral"

        if has_neutral_cue and net_prob >= 0.32:
            return "netral"

    if task_name == "news":
        if has_neutral_cue and net_prob >= 0.30 and margin < 0.10:
            return "netral"

        if pred_label_raw == "positif":
            if pos_prob < 0.60:
                return "netral"
            if margin < 0.12:
                return "netral"

        if pred_label_raw == "negatif" and has_negative_cue and neg_prob >= 0.45:
            return "negatif"

        if net_prob >= 0.33 and margin < 0.08:
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
        self.task2id = TASK2ID.copy()
        self.max_length = int(bundle.runtime_config.get("resolved_max_length", DEFAULT_MAX_LENGTH))

@torch.no_grad()
def predict_with_mc_dropout_df(
    self,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mc_passes: int = DEFAULT_MC_PASSES,
) -> pd.DataFrame:
    all_pass_probs = []

    for _ in range(mc_passes):
        self.bundle.model.train()  # aktifkan dropout
        pass_probs = []

        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start + batch_size].copy()
            texts = batch_df["text"].astype(str).tolist()
            task_ids = torch.tensor(batch_df["task_id"].tolist(), dtype=torch.long, device=self.bundle.device)

            enc = self.bundle.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.bundle.device) for k, v in enc.items()}

            logits = self.bundle.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                task_ids=task_ids,
            )
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pass_probs.append(probs)

        pass_probs = np.vstack(pass_probs)
        all_pass_probs.append(pass_probs)

    all_pass_probs = np.stack(all_pass_probs, axis=0)
    mean_probs = all_pass_probs.mean(axis=0)
    std_probs = all_pass_probs.std(axis=0)

    vote_preds = all_pass_probs.argmax(axis=-1)
    final_pred_ids = mean_probs.argmax(axis=1)

    agreement = (vote_preds == final_pred_ids[None, :]).mean(axis=0)
    max_prob = mean_probs.max(axis=1)
    ent = normalized_entropy(mean_probs)
    margin = margin_top2(mean_probs)

    out = df.copy()
    out["pred_id_raw"] = final_pred_ids
    out["pred_label_raw"] = out["pred_id_raw"].map(ID2LABEL)

    for label_name, label_id in LABEL2ID.items():
        out[f"prob_{label_name}"] = mean_probs[:, label_id]
        out[f"std_{label_name}"] = std_probs[:, label_id]

    out["max_prob"] = max_prob
    out["agreement"] = agreement
    out["norm_entropy"] = ent
    out["margin_top2"] = margin

    final_labels = []
    confidence_tiers = []
    auto_gt_candidates = []

    for _, row in out.iterrows():
        final_label = apply_task_post_rule(
            task_name=row["task_key"],
            pred_label_raw=row["pred_label_raw"],
            max_prob=float(row["max_prob"]),
            agreement=float(row["agreement"]),
            norm_entropy=float(row["norm_entropy"]),
            margin=float(row["margin_top2"]),
            neg_prob=float(row["prob_negatif"]),
            net_prob=float(row["prob_netral"]),
            pos_prob=float(row["prob_positif"]),
            clean_text=str(row["text"]),
        )
        final_labels.append(final_label)

        tier = confidence_tier_from_metrics(float(row["max_prob"]), float(row["agreement"]))
        confidence_tiers.append(tier)

        candidate = is_auto_gt_candidate(
            float(row["max_prob"]),
            float(row["agreement"]),
            float(row["norm_entropy"]),
            float(row["margin_top2"]),
        ) and (final_label == row["pred_label_raw"])
        auto_gt_candidates.append(candidate)

    out["pred_label"] = final_labels
    out["confidence_tier"] = confidence_tiers
    out["auto_gt_candidate"] = auto_gt_candidates
    out["auto_gt_label"] = np.where(out["auto_gt_candidate"], out["pred_label"], "")

    score_tensor = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    out["sentiment_score"] = (mean_probs * score_tensor[None, :]).sum(axis=1)

    self.bundle.model.eval()
    return out

    def predict_one(self, text: str, task_name: str, mc_passes: int = DEFAULT_MC_PASSES) -> Dict:
        clean_text = self.preprocessor.clean_text(text)

        df = pd.DataFrame([
            {
                "text": clean_text,
                "task_key": task_name,
                "task_id": TASK2ID[task_name],
            }
        ])

        pred_df = self.predict_with_mc_dropout_df(df, batch_size=1, mc_passes=mc_passes)
        row = pred_df.iloc[0].to_dict()

        row["token_length"] = int(
            len(
                self.bundle.tokenizer.encode(
                    clean_text,
                    truncation=True,
                    max_length=self.max_length
                )
            )
        )
        row["raw_text"] = text
        row["clean_text"] = clean_text
        return row

def predict_batch(
    self,
    df: pd.DataFrame,
    text_col: str,
    task_col: str,
    mc_passes: int = DEFAULT_MC_PASSES,
) -> pd.DataFrame:
    proc = df.copy()
    proc["text"] = proc[text_col].astype(str).map(self.preprocessor.clean_text)
    proc["task_key"] = proc[task_col].astype(str).str.strip().str.lower()
    proc["task_key"] = proc["task_key"].where(proc["task_key"].isin(["news", "sosmed"]), "news")
    proc["task_id"] = proc["task_key"].map(TASK2ID)

    pred_df = self.predict_with_mc_dropout_df(
        proc,
        batch_size=DEFAULT_BATCH_SIZE,
        mc_passes=mc_passes,
    )
    return pred_df


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
        mc_passes = st.slider("MC Dropout passes", 3, 9, DEFAULT_MC_PASSES, 2)

        if st.button("🔍 Analyze", type="primary"):
            if input_text.strip():
                with st.spinner("Processing..."):
                    result = engine.predict_one(input_text, task_name, mc_passes=mc_passes)

                st.subheader("🔸 Model Results")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Final Label", result["pred_label"])
                    st.metric("Raw Label", result["pred_label_raw"])
                    st.metric("Confidence (max_prob)", f"{float(result['max_prob']) * 100:.2f}%")

                with c2:
                    st.metric("Agreement", f"{float(result['agreement']) * 100:.2f}%")
                    st.metric("Norm Entropy", f"{float(result['norm_entropy']):.4f}")
                    st.metric("Margin Top-2", f"{float(result['margin_top2']):.4f}")

                with c3:
                    st.metric("Confidence Tier", result["confidence_tier"])
                    st.metric("Auto-GT Candidate", "YES" if bool(result["auto_gt_candidate"]) else "NO")
                    st.metric("Token Length", result["token_length"])

                st.write("### Probability Mean (MC Dropout)")
                prob_df = pd.DataFrame({
                    "label": ["negatif", "netral", "positif"],
                    "probability": [
                        float(result["prob_negatif"]),
                        float(result["prob_netral"]),
                        float(result["prob_positif"]),
                    ],
                    "std": [
                        float(result["std_negatif"]),
                        float(result["std_netral"]),
                        float(result["std_positif"]),
                    ],
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                st.write("### Detail")
                st.json({
                    "sentiment_score": float(result["sentiment_score"]),
                    "auto_gt_label": result["auto_gt_label"],
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
                            mc_passes=mc_passes_batch,
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
                        "text",
                        "task_key",
                        "pred_label",
                        "pred_label_raw",
                        "max_prob",
                        "agreement",
                        "norm_entropy",
                        "margin_top2",
                        "confidence_tier",
                        "auto_gt_candidate",
                        "auto_gt_label",
                        "prob_negatif",
                        "prob_netral",
                        "prob_positif",
                        "std_negatif",
                        "std_netral",
                        "std_positif",
                        "sentiment_score",
                    ] if c in results_df.columns]

                    st.dataframe(results_df[display_cols], use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv_data,
                        file_name="sentiment_multitask_mc_dropout_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
