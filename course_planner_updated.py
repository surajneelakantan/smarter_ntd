#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import math
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import zipper_prep
import cli_streamlit_compat
import ollama
from sentence_transformers import SentenceTransformer
from database import DatabaseManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("smarter-rag")


@dataclass
class Config:
    data_candidates: Tuple[str, ...] = ("modules.xlsx", "modules.csv")
    minutes_per_page: float = 8.0
    max_pdf_minutes: Optional[float] = None
    video_overhead_minutes: float = 30.0
    weight_embed: float = 0.50
    weight_tfidf: float = 0.35
    weight_terms: float = 0.35
    relevance_threshold: float = 0.35
    allow_overage_minutes: int = 15
    both_select: str = "shorter"
    llm_model: str = os.environ.get(
        "OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")
    llm_temp: float = 0.1
    llm_on: bool = True
    verbose: bool = True
    max_input_length: int = 5000
    llm_cache_size: int = 100
    batch_explanations: bool = True
    explanation_batch_size: int = 5


# Add modules thhat should not be considered
EXCLUDED_MODULES = {
    "SMARTER DAI Blackboards.pdf",
    "Smarter_TeacherIntro_H2024.pdf",
}

# Consolidated topic configuration with caching
TOPIC_CONFIG = {
    "svm": {
        "canonical": "support vector machine",
        "core_terms": ["support", "vector", "machine"],
        "allowed": ["support", "vector", "machine", "kernel", "margin", "classification", "hyperplane", "rbf", "linear", "polynomial", "hinge", "loss", "supervised"],
        "banned": ["retrieval", "rag", "augmented", "document"],
        "heuristics": ["support vector machine", "kernel trick", "rbf kernel", "linear kernel", "margin maximization", "hinge loss", "svc", "classification algorithm"]
    },
    "rag": {
        "canonical": "retrieval augmented generation",
        "core_terms": ["retrieval", "augment", "generation"],
        "allowed": ["retrieval", "augmented", "generation", "embed", "embedding", "vector", "semantic", "search", "index", "chunk", "rerank", "dense", "sparse", "context", "document", "query", "knowledge", "database", "language", "model", "llm", "transformer", "attention", "neural", "machine", "learning"],
        "banned": ["svm", "support vector", "kernel trick", "margin"],
        "heuristics": ["retrieval augmented generation", "vector search", "semantic search", "document retrieval", "embedding index", "knowledge base", "dense retrieval", "context retrieval", "reranking", "llm", "transformer", "attention mechanism"]
    },
    "llm": {
        "canonical": "large language model",
        "core_terms": ["language", "model", "transformer"],
        "allowed": ["language", "model", "large", "transformer", "attention", "token", "prompt", "fine", "tuning", "instruction", "alignment", "inference", "context", "window", "embedding", "encoder", "decoder", "neural", "generative"],
        "banned": ["svm", "support vector", "kernel"],
        "heuristics": ["large language model", "transformer", "self attention", "prompt engineering", "fine-tuning", "tokenization", "instruction following", "generative model"]
    },
    "nlp": {
        "canonical": "natural language processing",
        "core_terms": ["natural", "language", "processing", "text"],
        "allowed": ["natural", "language", "processing", "text", "token", "embedding", "classification", "entity", "recognition", "syntax", "semantic", "parsing", "sentiment", "analysis"],
        "banned": [],
        "heuristics": ["natural language processing", "text classification", "named entity recognition", "tokenization", "embeddings", "sentiment analysis"]
    },
    "classification": {
        "canonical": "machine learning classification",
        "core_terms": ["classify", "class", "predict"],
        "allowed": ["classification", "classify", "supervised", "learning", "logistic", "regression", "decision", "tree", "random", "forest", "neural", "network", "naive", "bayes", "ensemble", "precision", "recall", "accuracy"],
        "banned": ["retrieval", "rag", "document"],
        "heuristics": ["supervised learning", "logistic regression", "decision tree", "random forest", "neural network classification", "cross validation", "precision recall", "feature selection"]
    },
    "regression": {
        "canonical": "machine learning regression",
        "core_terms": ["regression", "predict", "linear"],
        "allowed": ["regression", "linear", "polynomial", "ridge", "lasso", "gradient", "descent", "mean", "squared", "error", "prediction", "supervised", "learning", "feature", "engineering"],
        "banned": ["retrieval", "rag", "document"],
        "heuristics": ["linear regression", "polynomial regression", "ridge regression", "lasso regression", "gradient descent", "mean squared error", "feature engineering"]
    },
    "clustering": {
        "canonical": "clustering",
        "core_terms": ["cluster", "group", "unsupervised"],
        "allowed": ["clustering", "cluster", "unsupervised", "learning", "means", "hierarchical", "density", "based", "centroid", "silhouette", "analysis", "grouping", "partition"],
        "banned": [],
        "heuristics": ["k means clustering", "hierarchical clustering", "density based clustering", "unsupervised learning", "cluster analysis", "centroid", "silhouette score"]
    },
    "neural": {
        "canonical": "neural networks",
        "core_terms": ["neural", "network", "deep"],
        "allowed": ["neural", "network", "artificial", "deep", "learning", "backpropagation", "activation", "function", "convolutional", "recurrent", "feedforward", "layer", "neuron"],
        "banned": [],
        "heuristics": ["artificial neural network", "deep learning", "backpropagation", "activation function", "convolutional neural", "recurrent neural", "feedforward network"]
    },
    "reinforcement": {
        "canonical": "reinforcement learning",
        "core_terms": ["reinforcement", "reward", "policy"],
        "allowed": ["reinforcement", "learning", "policy", "gradient", "markov", "decision", "process", "reward", "function", "exploration", "exploitation", "temporal", "difference", "agent"],
        "banned": [],
        "heuristics": ["reinforcement learning", "policy gradient", "q learning", "markov decision process", "reward function", "exploration exploitation", "temporal difference"]
    }
}

REQUIRED = ["course_name", "pdf_name", "pdf_summary", "number of pages",
            "video_related_to_pdf", "video_transcription_summary", "video_duration"]


# LRU Cache implementation
class LRUCache:
    """Thread-safe LRU cache with size limit"""

    def __init__(self, maxsize: int = 100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)


# Global cache with size limit
_LLM_CACHE = None
_ST_MODEL = None
_ST_MODEL_LOCK = threading.Lock()


def get_llm_cache(maxsize: int = 100) -> LRUCache:
    """Get or create LLM cache"""
    global _LLM_CACHE
    if _LLM_CACHE is None:
        _LLM_CACHE = LRUCache(maxsize)
    return _LLM_CACHE


def get_st_model():
    """Thread-safe model loading"""
    global _ST_MODEL
    if _ST_MODEL is None:
        with _ST_MODEL_LOCK:
            if _ST_MODEL is None:  # Double-check pattern
                log.info("Loading SentenceTransformer model...")
                _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def ensure_ollama_ready(cfg: Config) -> None:
    try:
        ollama.list()
        log.info("Ollama connection established")
    except Exception as e:
        raise SystemExit(
            "Failed to connect to Ollama. Start it from: https://ollama.com/download") from e


def clean_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"_x000D_", " ", str(s))).strip()


def parse_video_minutes(s: Any) -> float:
    if s is None or s == "" or (isinstance(s, str) and s.lower() in ['n/a', 'na', 'none', 'null', 'invalid']):
        return 0.0
    if isinstance(s, (int, float)):
        try:
            return max(0.0, float(s))
        except Exception:
            return 0.0

    txt = str(s).strip().lower()

    # MM:SS format (most common in Excel)
    if re.match(r'^\d{1,2}:\d{2}$', txt):
        parts = txt.split(':')
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except Exception:
            return 0.0

    # HH:MM:SS format
    parts = txt.split(":")
    if len(parts) == 3:
        try:
            h, m, sec = [float(p) for p in parts]
            return int(h) * 60 + int(m) + (sec / 60.0)
        except Exception:
            return 0.0

    try:
        return max(0.0, float(txt))
    except Exception:
        return 0.0


def is_module_excluded(pdf_name: str, video_name: str) -> bool:
    """Check if a module should be excluded based on the exclusion list"""
    if pdf_name and pdf_name.strip() in EXCLUDED_MODULES:
        return True
    if video_name and video_name.strip() in EXCLUDED_MODULES:
        return True
    return False


def load_data(cfg: Config) -> pd.DataFrame:
    path = None
    for name in cfg.data_candidates:
        p = Path(name)
        if p.exists():
            path = p
            break
    if not path:
        raise FileNotFoundError(
            f"No metadata file found. Tried: {cfg.data_candidates}")

    try:
        df = pd.read_excel(path) if path.suffix.lower() in (
            ".xlsx", ".xls") else pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load data file: {e}")

    df.columns = [c.strip().lower() for c in df.columns]

    # Column mapping
    alias = {
        "number_of_pages": "number of pages",
        "video_file": "video_related_to_pdf",
        "video_summary": "video_transcription_summary",
        "pdf_title": "pdf_name",
        "course": "course_name"
    }
    for src, dst in alias.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["course_name", "pdf_name", "pdf_summary", "video_related_to_pdf", "video_transcription_summary"]:
        if c in df.columns:
            df[c] = df[c].map(clean_text)

    df["number of pages"] = pd.to_numeric(
        df["number of pages"], errors="coerce").fillna(0.0)
    df["video_duration"] = df["video_duration"].map(parse_video_minutes)
    if "keywords" not in df.columns:
        df["keywords"] = ""

    df["pdf_minutes"] = df["number of pages"].apply(lambda p: min(
        p * cfg.minutes_per_page, cfg.max_pdf_minutes) if cfg.max_pdf_minutes and p > 0 else p * cfg.minutes_per_page)
    df["video_minutes"] = df["video_duration"].astype(float)
    df["has_pdf"] = df["pdf_name"].str.len().gt(0) & df["pdf_minutes"].gt(0)
    df["has_video"] = df["video_related_to_pdf"].str.len().gt(
        0) & df["video_minutes"].gt(0)

    # NEW: Apply exclusion filter
    df["excluded"] = df.apply(lambda row: is_module_excluded(
        row["pdf_name"], row["video_related_to_pdf"]), axis=1)

    if df["excluded"].sum() > 0:
        log.info(
            f"Excluding {df['excluded'].sum()} modules based on exclusion list")

    df = df[~df["excluded"]].copy()

    # Include all relevant fields
    df["search_text"] = (
        df["course_name"] + " " +
        df["pdf_name"] + " " +
        df["pdf_summary"] + " " +
        df["video_transcription_summary"].fillna("") + " " +
        df["key_words_pdf"].fillna("") + " " +
        df["key_words_video"].fillna("") + " " +
        df.get("keywords", "")
    ).str.lower().map(clean_text)

    result_df = df[(df["has_pdf"] | df["has_video"])].drop_duplicates(subset=[
        "course_name", "pdf_name", "video_related_to_pdf", "search_text"]).reset_index(drop=True)

    if len(result_df) == 0:
        raise ValueError(
            "No valid modules found in data file (all modules have missing PDFs and videos)")

    return result_df


def parse_json_strict(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        best = None
        for s in starts:
            depth = 0
            for i in range(s, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[s: i + 1]
                        if best is None or len(cand) > len(best):
                            best = cand
                        break
        if not best:
            raise ValueError("No JSON object found in LLM reply.")
        return json.loads(best)


def ollama_chat_json(model: str, prompt: str, temp: float = 0.1, retries: int = 2, use_cache: bool = True, cache_size: int = 100) -> Any:
    # Use LRU cache
    cache = get_llm_cache(cache_size)
    cache_key = hash(prompt + str(temp)) if use_cache else None

    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    for attempt in range(retries):
        try:
            out = ollama.chat(model=model, options={"temperature": max(0.0, min(
                temp, 0.4)), "num_predict": 1200}, messages=[{"role": "user", "content": prompt}])
            raw = (out["message"]["content"] or "").strip()
            if not raw:
                raise RuntimeError("Empty reply from Ollama.")

            data = parse_json_strict(raw)
            if not isinstance(data, (dict, list)):
                raise RuntimeError("Parsed JSON is not an object/array.")

            if cache_key:
                cache.set(cache_key, data)
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Ollama JSON failed after {retries} attempts: {e}")
            log.warning(f"Ollama attempt {attempt+1} failed, retrying: {e}")

    raise RuntimeError("Ollama failed all retry attempts")


def ollama_chat_text(model: str, prompt: str, temp: float = 0.2, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            out = ollama.chat(model=model, options={"temperature": max(0.0, min(
                temp, 0.4)), "num_predict": 2500}, messages=[{"role": "user", "content": prompt}])
            text = (out["message"]["content"] or "").strip()
            if not text:
                raise RuntimeError("Empty reply from Ollama.")
            return text
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Ollama text failed after {retries} attempts: {e}")
            log.warning(f"Ollama attempt {attempt+1} failed, retrying: {e}")

    raise RuntimeError("Ollama failed all retry attempts")


class HybridRanker:
    def __init__(self, texts: List[str], cfg: Config):
        self.cfg = cfg
        self.texts = texts
        self.tfidf = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=90000, ngram_range=(1, 3))
        self.X = self.tfidf.fit_transform(texts)
        self.model = get_st_model()
        self.embed = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False)

    def _minmax(self, a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a
        lo, hi = float(np.min(a)), float(np.max(a))
        if math.isclose(lo, hi, abs_tol=1e-9):
            return np.ones_like(a)
        return (a - lo) / (hi - lo + 1e-12)

    def tfidf_scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.tfidf.transform([query]), self.X).ravel()

    def embed_scores(self, query: str) -> np.ndarray:
        qv = self.model.encode([query], normalize_embeddings=True)
        return (self.embed @ qv.T).ravel()


def build_term_weights(texts: List[str], pos_idx: np.ndarray, off_idx: np.ndarray) -> np.ndarray:
    vec = TfidfVectorizer(lowercase=True, stop_words="english",
                          max_features=120000, ngram_range=(1, 3))
    X = vec.fit_transform(texts)
    feats = np.array(vec.get_feature_names_out())

    def mean_vec(rows: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[1], dtype=float) if rows.size == 0 else np.asarray(X[rows].mean(axis=0)).ravel()

    pos_mean, off_mean = mean_vec(pos_idx), mean_vec(off_idx)
    global_mean = np.asarray(X.mean(axis=0)).ravel()
    pos_score = pos_mean - global_mean
    strong_terms = feats[np.argsort(-pos_score)[:24]]

    strong_pat = re.compile(
        r"\b(" + "|".join(map(re.escape, strong_terms.tolist())) + r")\b", re.I) if strong_terms.size else None
    term_weights = np.zeros(len(texts), dtype=float)
    for i, t in enumerate(texts):
        if strong_pat and strong_pat.search(t):
            term_weights[i] += 1.0
    return term_weights


@dataclass
class UserProfile:
    topic: str = ""
    hours: float = 0.0
    profession: str = ""
    format: str = ""


def validate_hours(hours: Any) -> Optional[float]:
    """Validate hours input"""
    try:
        h = float(hours)
        if 0.1 <= h <= 100:
            return h
        return None
    except (ValueError, TypeError):
        return None


def validate_format(format_str: str) -> Optional[str]:
    """Validate format input"""
    f = str(format_str).lower().strip()
    if f in ["pdf", "video", "both"]:
        return f
    return None


def validate_topic(topic: str) -> Optional[str]:
    """Validate topic input"""
    t = str(topic).strip()
    if t and len(t) > 1:
        return t
    return None


def validate_role(role: str) -> Optional[str]:
    """Validate role input"""
    r = str(role).strip()
    if r and len(r) > 1 and r.lower() not in ['null', 'none', 'n/a', '']:
        return r
    return None


@dataclass
class Module:
    idx: int
    course: str
    title: str
    summary: str
    pdf_minutes: float
    video_minutes: float
    has_pdf: bool
    has_video: bool
    pdf_name: str
    video_name: str
    search_text: str


@dataclass
class Hit:
    module: Module
    score: float
    reasons: Dict[str, float]


class MandatoryInfoCollector:
    """collection of required information"""

    REQUIRED_FIELDS = {
        "topic": {
            "prompt": "What topic would you like to learn about?",
            "examples": ["SVM", "RAG", "NLP", "classification", "neural networks", "machine learning", "LLM", "deep learning"],
            "validation": validate_topic
        },
        "hours": {
            "prompt": "How many hours do you have available for learning?",
            "examples": ["6 hours", "2.5 hours", "10h", "3-4 hours"],
            "validation": validate_hours
        },
        "role": {
            "prompt": "What's your professional role or background?",
            "examples": ["data scientist", "software engineer", "student", "researcher", "doctor", "analyst", "project coordinator"],
            "validation": validate_role
        },
        "format": {
            "prompt": "Do you prefer learning from PDFs, videos, or both?",
            "examples": ["PDF", "video", "both", "either"],
            "validation": validate_format
        }
    }

    def get_missing_fields(self, user_profile: UserProfile) -> List[str]:
        missing = []
        if not user_profile.topic:
            missing.append("topic")
        if user_profile.hours <= 0:
            missing.append("hours")
        if not user_profile.profession or len(user_profile.profession.strip()) == 0:
            missing.append("role")
        if not user_profile.format or len(user_profile.format.strip()) == 0:
            missing.append("format")
        return missing

    def create_collection_prompt(self, message: str, missing_fields: List[str], user_profile: UserProfile) -> str:
        context = self._build_context(user_profile)

        prompt = f"""You are helping someone plan their learning. You need to extract information and ask for missing required details.

REQUIRED INFORMATION STILL NEEDED:
{self._format_required_fields(missing_fields)}

CURRENT CONTEXT: {context}
USER MESSAGE: "{message}"

INSTRUCTIONS:
1. Extract ANY information you can from their message (even partial matches)
2. If information is still missing after extraction, ask for ONE missing piece conversationally
3. Be helpful and provide examples when asking
4. Use natural language, not robotic responses
5. Validate extracted information makes sense
6. For roles: accept ANY professional title, job role, or occupation mentioned (e.g., "project coordinator", "project co-ordinator", "project manager", "data scientist", etc.)
7. For role extraction: look for "I am a...", "I'm a...", "project manager", "data scientist", "software engineer", job titles, professions, occupations
8. IMPORTANT: Extract roles even with spelling variations (e.g., "co-ordinator" vs "coordinator")

RESPOND WITH JSON:
{{
    "extracted": {{
        "topic": "extracted topic or null",
        "hours": extracted_number_or_null, 
        "role": "extracted role/profession or null",
        "format": "pdf/video/both or null"
    }},
    "response": "Your conversational response to the user",
    "complete": true_or_false
}}

EXAMPLES:
- If they say "I want to learn about machine learning for 8 hours" → extract topic="machine learning", hours=8
- If they say "I'm a doctor" → extract role="doctor"
- If they say "I am a project co-ordinator" → extract role="project coordinator"
- If they say "videos please" → extract format="video"
- Always be conversational in your response"""

        return prompt

    def _build_context(self, profile: UserProfile) -> str:
        context_parts = []
        if profile.topic:
            context_parts.append(f"Topic: {profile.topic}")
        if profile.hours > 0:
            context_parts.append(f"Hours: {profile.hours}")
        if profile.profession:
            context_parts.append(f"Role: {profile.profession}")
        if profile.format:
            context_parts.append(f"Format: {profile.format}")
        return ", ".join(context_parts) if context_parts else "No information collected yet"

    def _format_required_fields(self, missing_fields: List[str]) -> str:
        formatted = []
        for field in missing_fields:
            field_info = self.REQUIRED_FIELDS[field]
            formatted.append(
                f"- {field}: {field_info['prompt']} (examples: {', '.join(field_info['examples'])})")
        return "\n".join(formatted)


class FormatHandler:
    """Consolidated format handling with optimization"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def normalize_format_token(self, text: str) -> Optional[str]:
        low = text.lower()
        if "video" in low and "pdf" not in low:
            return "video"
        if "pdf" in low and "video" not in low:
            return "pdf"
        if any(w in low for w in ["both", "either", "all"]):
            return "both"
        return None

    def effective_video_minutes(self, m: Module) -> float:
        base = m.video_minutes if m.video_minutes > 0 else 0.0
        return base + (self.cfg.video_overhead_minutes if m.has_video else 0.0)

    def format_ok(self, m: Module, user_format: str) -> bool:
        f = (user_format or "").lower()
        if f == "pdf":
            return m.has_pdf and m.pdf_minutes > 0
        if f == "video":
            return m.has_video and self.effective_video_minutes(m) > 0
        return (m.has_pdf and m.pdf_minutes > 0) or (m.has_video and self.effective_video_minutes(m) > 0)

    def duration_for_format(self, m: Module, user_format: str) -> float:
        f = (user_format or "both").lower()
        if f == "pdf":
            return m.pdf_minutes if m.has_pdf else 0.0
        if f == "video":
            return self.effective_video_minutes(m) if m.has_video else 0.0

        # For "both" format
        pdf_d = m.pdf_minutes if m.has_pdf else 0.0
        vid_d = self.effective_video_minutes(m) if m.has_video else 0.0

        if pdf_d == 0.0 and vid_d == 0.0:
            return 0.0
        if pdf_d == 0.0:
            return vid_d
        if vid_d == 0.0:
            return pdf_d

        if self.cfg.both_select == "shorter":
            return min(pdf_d, vid_d)
        elif self.cfg.both_select == "longer":
            return max(pdf_d, vid_d)
        else:
            return pdf_d + vid_d

    def resolved_format(self, m: Module, user_format: str) -> str:
        f = (user_format or "both").lower()
        if f in ("pdf", "video"):
            return f
        pdf_d = m.pdf_minutes if m.has_pdf else 0.0
        vid_d = self.effective_video_minutes(m) if m.has_video else 0.0
        if self.cfg.both_select == "shorter":
            if pdf_d and not vid_d:
                return "pdf"
            if vid_d and not pdf_d:
                return "video"
            return "pdf" if pdf_d <= vid_d else "video"
        return "pdf" if pdf_d >= vid_d else "video"

    def display_title_resolved(self, m: Module, user_format: str) -> str:
        rf = self.resolved_format(m, user_format)
        if rf == "video" and m.has_video:
            return m.video_name or m.title
        if rf == "pdf" and m.has_pdf:
            return m.pdf_name or m.title
        return m.title


class SMARTERChatbot:
    def __init__(self, cfg: Optional[Config] = None, db=None, session_id=None):
        self.cfg = cfg or Config()
        ensure_ollama_ready(self.cfg)
        self.df = load_data(self.cfg)
        self.texts = self.df["search_text"].tolist()
        self.format_handler = FormatHandler(self.cfg)
        self.info_collector = MandatoryInfoCollector()

        self.rank = HybridRanker(self.texts, self.cfg)

        # Pre-compute course embeddings for efficiency
        model = get_st_model()
        self.course_embs: Dict[str, np.ndarray] = {}
        for c in self.df["course_name"].unique():
            indices = self.df[self.df["course_name"] == c].index.tolist()
            emb = self.rank.embed[np.array(indices)]
            meanv = np.mean(emb, axis=0)
            self.course_embs[c] = meanv / (np.linalg.norm(meanv) + 1e-9)

        self.user = UserProfile()
        self.last: List[Hit] = []
        self.modules: List[Module] = []
        self._pending_topic_choice: Optional[List[str]] = None
        self.db = db
        self.session_id = session_id

        for i, row in self.df.iterrows():
            self.modules.append(Module(
                idx=i, course=row["course_name"],
                title=row["pdf_name"] if row["pdf_name"] else row["video_related_to_pdf"],
                summary=row["pdf_summary"] or row["video_transcription_summary"],
                pdf_minutes=float(row["pdf_minutes"]), video_minutes=float(row["video_minutes"]),
                has_pdf=bool(row["has_pdf"]), has_video=bool(row["has_video"]),
                pdf_name=row["pdf_name"], video_name=row["video_related_to_pdf"],
                search_text=row["search_text"]))

    def _get_topic_config(self, topic: str) -> Dict[str, Any]:
        """Get topic configuration with fallback"""
        topic_lower = topic.lower().strip()

        # Direct match
        if topic_lower in TOPIC_CONFIG:
            return TOPIC_CONFIG[topic_lower]

        # Partial match
        for key, config in TOPIC_CONFIG.items():
            if key in topic_lower or topic_lower in key:
                return config

        # Fallback configuration
        words = [w for w in re.findall(r'\b\w{3,}\b', topic_lower)
                 if w not in {'the', 'and', 'for', 'with', 'learning', 'machine'}]
        return {
            "canonical": topic,
            "core_terms": words[:3],
            "allowed": words + ["learning", "model", "algorithm", "data"],
            "banned": [],
            "heuristics": [topic]
        }

    def _detect_multiple_topics(self, message: str) -> Optional[List[str]]:
        """Detect multiple topics in message"""
        low = message.lower()
        found = []

        for topic in TOPIC_CONFIG.keys():
            if topic in low:
                found.append(topic)

        # Check for conjunctions and return max 2 topics
        if len(found) >= 2 and re.search(r'\b(?:and|or|vs|versus)\b', low):
            return sorted(found)[:2]

        return None

    def _expand_query(self, topic: str, role: Optional[str]) -> List[Dict[str, Any]]:
        """Optimized query expansion with caching"""
        config = self._get_topic_config(topic)
        canonical = config["canonical"]

        # Base terms
        base_terms = [{"text": canonical, "weight": 1.0, "source": "topic"}]

        # Add heuristic terms
        heuristic_terms = [{"text": term, "weight": 0.6, "source": "heuristic"}
                           for term in config["heuristics"]]

        # LLM expansion with caching
        llm_terms = []
        if self.cfg.llm_on:
            try:
                prompt = f"""Generate 8 ML/AI search terms for "{canonical}". Focus ONLY on this topic. Return JSON: {{"terms": [{{"text": "term", "weight": 0.8}}]}}"""
                resp = ollama_chat_json(
                    self.cfg.llm_model, prompt, temp=0.05, use_cache=True, cache_size=self.cfg.llm_cache_size)
                if isinstance(resp, dict) and "terms" in resp:
                    for t in resp["terms"][:8]:
                        if isinstance(t, dict) and "text" in t:
                            txt = str(t["text"]).strip().lower()
                            if txt and any(allow in txt for allow in config["allowed"]) and not any(ban in txt for ban in config["banned"]):
                                w = float(t.get("weight", 0.5))
                                llm_terms.append({"text": txt, "weight": max(
                                    0.3, min(w, 1.0)), "source": "llm"})
            except Exception as e:
                log.warning(f"LLM expansion failed: {e}")

        # Combine and deduplicate
        all_terms = base_terms + heuristic_terms + llm_terms
        dedup = {}
        for t in all_terms:
            txt = t["text"].strip().lower()
            if txt:
                dedup[txt] = max(dedup.get(txt, 0.0), t["weight"])

        result = [{"text": k, "weight": w, "source": "mix"}
                  for k, w in dedup.items()]
        result.sort(key=lambda x: x["weight"], reverse=True)
        return result

    def _score(self, topic: str, expanded: List[Dict[str, Any]]) -> List[Hit]:
        """Optimized scoring with better matching"""
        config = self._get_topic_config(topic)
        exp_terms = [t["text"] for t in expanded]
        expansion_query = (topic + " " + " ".join(exp_terms)).strip()

        s_tfidf = self.rank._minmax(self.rank.tfidf_scores(expansion_query))
        s_emb = self.rank._minmax(self.rank.embed_scores(expansion_query))

        order = np.argsort(-s_emb)
        pos_idx = order[:min(50, len(self.texts))]
        low_cut = float(np.quantile(s_emb, 0.25))
        off_idx = np.where(s_emb <= max(low_cut, 0.10))[0]

        term_weights = build_term_weights(
            self.texts, np.array(pos_idx), np.array(off_idx))
        s_terms = self.rank._minmax(term_weights)

        model = get_st_model()
        qv = model.encode([expansion_query], normalize_embeddings=True)[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        course_guard = np.array(
            [float(np.dot(self.course_embs[m.course], qv)) for m in self.modules])
        course_guard = self.rank._minmax(course_guard)
        guard_thr = float(np.quantile(course_guard, 0.3))

        base = self.cfg.weight_tfidf * s_tfidf + self.cfg.weight_embed * \
            s_emb + self.cfg.weight_terms * s_terms
        base = self.rank._minmax(base)
        thr = max(self.cfg.relevance_threshold, float(np.quantile(base, 0.40)))

        hits = []
        for i, m in enumerate(self.modules):
            if not self.format_handler.format_ok(m, self.user.format) or self.format_handler.duration_for_format(m, self.user.format) <= 0.5:
                continue

            text = m.search_text.lower()
            title_blob = f"{m.title} {m.pdf_name} {m.video_name}".lower()

            # Enhanced matching with core terms fallback
            core_terms = config["core_terms"]
            topic_hits = sum(1 for term in exp_terms[:8] if re.search(r"(?:^|[^a-zA-Z])" + re.escape(
                term.replace('-', '[-\s]?')) + r"(?:[^a-zA-Z]|$)", text + " " + title_blob, re.I))
            fallback_hits = sum(
                1 for term in core_terms if term in text.lower())

            if topic_hits == 0 and fallback_hits == 0:
                continue

            # Content relevance boost
            content_relevance = 0
            if self.user.topic.lower() in text:
                content_relevance += 0.3
            if any(self.user.topic.lower() in field.lower() for field in [m.title, m.pdf_name, m.video_name] if field):
                content_relevance += 0.2

            guard = course_guard[i]
            penalty = -0.25 * (guard_thr - guard) if guard < guard_thr else 0.0
            score = float(base[i]) + content_relevance + penalty

            if score >= thr:
                hits.append(Hit(module=m, score=score, reasons={
                    "base": float(base[i]), "tfidf": float(s_tfidf[i]), "embed": float(s_emb[i]),
                    "terms": float(s_terms[i]), "course_guard": guard, "penalty": penalty, "content_rel": content_relevance}))

        hits.sort(key=lambda h: h.score, reverse=True)
        if self.cfg.verbose:
            log.info("Selected %d modules (threshold: %.3f)", len(hits), thr)
        return hits

    def _pack_time_optimal(self, hits: List[Hit], target_minutes: int) -> List[Hit]:
        """Optimized time packing algorithm"""
        seen = set()
        uniq = []
        for h in hits:
            m = h.module
            d = self.format_handler.duration_for_format(m, self.user.format)
            if d <= 0:
                continue
            rf = self.format_handler.resolved_format(m, self.user.format)
            title_key = (m.title or "").strip().lower()
            key = ("video", (m.video_name.strip().lower() if m.video_name else title_key)) if rf == "video" else (
                "pdf", (m.pdf_name.strip().lower() if m.pdf_name else title_key))
            if key not in seen:
                seen.add(key)
                uniq.append(h)

        if len(uniq) == 0:
            return []

        if len(uniq) <= 60:
            # Dynamic programming for optimal packing
            W = target_minutes
            n = len(uniq)
            dp = [0.0] * (W + 1)
            keep = [[False] * (W + 1) for _ in range(n)]
            durations = [max(1, min(int(round(self.format_handler.duration_for_format(
                h.module, self.user.format))), W)) for h in uniq]
            values = [max(0.0, h.score) for h in uniq]

            for i in range(n):
                w = durations[i]
                v = values[i]
                for cap in range(W, w - 1, -1):
                    if dp[cap - w] + v > dp[cap]:
                        dp[cap] = dp[cap - w] + v
                        keep[i][cap] = True

            cap = W
            chosen_idx = []
            for i in range(n - 1, -1, -1):
                if keep[i][cap]:
                    chosen_idx.append(i)
                    cap -= durations[i]
                    if cap < 0:
                        break
            chosen_idx.reverse()
            return [uniq[i] for i in chosen_idx]
        else:
            # Greedy approach for large datasets
            allow = self.cfg.allow_overage_minutes
            scored = [(max(0.0, h.score) / max(1.0, math.sqrt(self.format_handler.duration_for_format(h.module, self.user.format))),
                      -int(round(self.format_handler.duration_for_format(h.module, self.user.format))), h) for h in uniq]
            scored.sort(key=lambda t: (-t[0], t[1]))
            total, sel = 0, []
            for dens, negd, h in scored:
                d = -negd
                if total + d <= target_minutes:
                    sel.append(h)
                    total += d
                elif total < target_minutes and total + d <= target_minutes + allow:
                    sel.append(h)
                    total += d
                    break
            return sel

    def _generate_explanations_batch(self, topic: str, role: str, modules: List[Module]) -> List[str]:
        """NEW: Batch explanation generation to reduce LLM calls"""
        if not self.cfg.llm_on:
            return [f"This module covers relevant concepts for {topic}."] * len(modules)

        if not role or role.lower() in ('null', 'none', ''):
            role = "learner"

        results = []
        batch_size = self.cfg.explanation_batch_size if self.cfg.batch_explanations else 1

        for batch_start in range(0, len(modules), batch_size):
            batch_end = min(batch_start + batch_size, len(modules))
            batch_modules = modules[batch_start:batch_end]

            if batch_size == 1 or not self.cfg.batch_explanations:
                # Single module explanation (original behavior)
                m = batch_modules[0]
                rf = self.format_handler.resolved_format(m, self.user.format)
                if rf == "video" and m.has_video:
                    title, summ = m.video_name or m.title, clean_text(
                        self.df.iloc[m.idx]["video_transcription_summary"])
                elif rf == "pdf" and m.has_pdf:
                    title, summ = m.pdf_name or m.title, clean_text(
                        self.df.iloc[m.idx]["pdf_summary"])
                else:
                    title, summ = m.title, m.summary

                if len(title) > 200:
                    title = title[:197] + "..."

                prompt = f"""Explain this module to a {role} learning {topic}. 

    Module: {title}
    Summary: {summ[:400]}

    Write 2-3 sentences explaining how this teaches {topic} and include one actionable step. Be specific and practical."""

                try:
                    text = ollama_chat_text(
                        self.cfg.llm_model, prompt, temp=0.2)
                    expl = re.sub(r'^(Here is|Here\'s|This is).*?:\s*',
                                  '', text.strip(), flags=re.I)
                    expl = re.sub(r'\s+', ' ', expl).strip()
                    if len(expl) >= 40:
                        results.append(expl)
                    else:
                        raise ValueError("Response too short")
                except Exception:
                    results.append(
                        f"This module teaches {topic} concepts through {title}. You'll learn practical applications and techniques. Action: Review the key concepts and try a hands-on exercise.")
            else:
                # Batch processing - TEXT-BASED VERSION (More Reliable)
                module_infos = []
                for idx, m in enumerate(batch_modules, 1):
                    rf = self.format_handler.resolved_format(
                        m, self.user.format)
                    if rf == "video" and m.has_video:
                        title, summ = m.video_name or m.title, clean_text(
                            self.df.iloc[m.idx]["video_transcription_summary"])
                    elif rf == "pdf" and m.has_pdf:
                        title, summ = m.pdf_name or m.title, clean_text(
                            self.df.iloc[m.idx]["pdf_summary"])
                    else:
                        title, summ = m.title, m.summary

                    if len(title) > 120:
                        title = title[:117] + "..."
                    if len(summ) > 250:
                        summ = summ[:247] + "..."

                    module_infos.append(
                        f"MODULE {idx}:\nTitle: {title}\nSummary: {summ}")

                # TEXT-BASED prompt with clear delimiters
                batch_prompt = f"""You are explaining {len(batch_modules)} learning modules to a {role} learning about {topic}.

    For EACH module below, write 6-7 sentences explaining how it teaches {topic} and include one actionable step. Be specific and practical.

    {chr(10).join(module_infos)}

    IMPORTANT: Start each explanation with "EXPLANATION {idx}:" followed by your explanation. Use this exact format:

    EXPLANATION 1: [your explanation here]
    EXPLANATION 2: [your explanation here]
    EXPLANATION 3: [your explanation here]
    etc."""

                try:
                    response_text = ollama_chat_text(
                        self.cfg.llm_model, batch_prompt, temp=0.2)

                    # Parse text-based response using regex
                    explanations = []
                    pattern = r'EXPLANATION\s+\d+:\s*(.+?)(?=EXPLANATION\s+\d+:|$)'
                    matches = re.findall(
                        pattern, response_text, re.DOTALL | re.IGNORECASE)

                    if len(matches) >= len(batch_modules):
                        explanations = matches[:len(batch_modules)]
                    else:
                        # Fallback: split by newlines and filter
                        lines = response_text.split('\n')
                        current_expl = []
                        for line in lines:
                            line = line.strip()
                            if re.match(r'^EXPLANATION\s+\d+:', line, re.IGNORECASE):
                                if current_expl:
                                    explanations.append(' '.join(current_expl))
                                current_expl = [
                                    re.sub(r'^EXPLANATION\s+\d+:\s*', '', line, flags=re.IGNORECASE)]
                            elif line and current_expl:
                                current_expl.append(line)
                        if current_expl:
                            explanations.append(' '.join(current_expl))

                    # Process and clean explanations
                    for expl in explanations[:len(batch_modules)]:
                        # CHANGED: Remove more prefixes including "MODULE X:"
                        cleaned = re.sub(r'^(Here is|Here\'s|This is|Module \d+:|MODULE \d+:).*?:\s*',
                                         '', expl.strip(), flags=re.I)
                        # CHANGED: Also remove standalone "MODULE X:" at the start
                        cleaned = re.sub(r'^MODULE\s+\d+:\s*',
                                         '', cleaned, flags=re.I)
                        # CHANGED: Ensure we start with "This module" or "The [name] module"
                        if not re.match(r'^(This module|The .+? module)', cleaned, re.I):
                            cleaned = f"This module {cleaned}"
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        if len(cleaned) >= 40:
                            results.append(cleaned)
                        else:
                            results.append(
                                f"This module teaches {topic} concepts with practical applications. Action: Review key concepts and try exercises.")

                    # If we didn't get enough explanations, fill with defaults
                    while len(results) < batch_start + len(batch_modules):
                        m = batch_modules[len(results) - batch_start]
                        title = self.format_handler.display_title_resolved(
                            m, self.user.format)
                        if len(title) > 80:
                            title = title[:77] + "..."
                        results.append(
                            f"This module teaches {topic} concepts through {title}. You'll learn practical applications and techniques. Action: Review the key concepts and try a hands-on exercise.")

                except Exception as e:
                    log.warning(
                        f"Batch explanation failed: {e}, falling back to individual")
                    # Fallback to individual explanations with proper API calls
                    for m in batch_modules:
                        rf = self.format_handler.resolved_format(
                            m, self.user.format)
                        if rf == "video" and m.has_video:
                            title, summ = m.video_name or m.title, clean_text(
                                self.df.iloc[m.idx]["video_transcription_summary"])
                        elif rf == "pdf" and m.has_pdf:
                            title, summ = m.pdf_name or m.title, clean_text(
                                self.df.iloc[m.idx]["pdf_summary"])
                        else:
                            title, summ = m.title, m.summary

                        if len(title) > 200:
                            title = title[:197] + "..."

                        prompt = f"""Explain this module to a {role} learning {topic}. 

    Module: {title}
    Summary: {summ[:800]}

    Write 8-10 sentences explaining how this teaches {topic} and include one actionable step. Be specific and practical."""

                        try:
                            text = ollama_chat_text(
                                self.cfg.llm_model, prompt, temp=0.2)
                            expl = re.sub(r'^(Here is|Here\'s|This is).*?:\s*',
                                          '', text.strip(), flags=re.I)
                            expl = re.sub(r'\s+', ' ', expl).strip()
                            if len(expl) >= 40:
                                results.append(expl)
                            else:
                                raise ValueError("Response too short")
                        except Exception:
                            results.append(
                                f"This module teaches {topic} concepts through {title}. You'll learn practical applications and techniques. Action: Review the key concepts and try a hands-on exercise.")

        return results

    def _generate_explanations(self, topic: str, role: str, modules: List[Module]) -> List[str]:
        """Wrapper to use batch or individual explanation generation"""
        if self.cfg.batch_explanations and len(modules) > 1:
            log.info(
                f"Generating explanations in batches of {self.cfg.explanation_batch_size}...")
            return self._generate_explanations_batch(topic, role, modules)
        else:
            return self._generate_explanations_batch(topic, role, modules)

    def _create_learning_path(self, selected: List[Hit], explanations: List[str]) -> str:
        """Create formatted learning path output"""
        if len(selected) == 0:
            return "No modules selected. This shouldn't happen!"

        fmt_label = self.user.format or "both"
        response = [
            f"Here's your {self.user.topic} learning path ({fmt_label}):\n"]

        total_minutes = 0
        for i, (h, expl) in enumerate(zip(selected, explanations), 1):
            m = h.module
            dur = int(self.format_handler.duration_for_format(
                m, self.user.format))
            total_minutes += dur
            title = self.format_handler.display_title_resolved(
                m, self.user.format)
            if len(title) > 150:
                title = title[:147] + "..."

            # Format display
            format_tag = ""
            if self.user.format == "both":
                tags = []
                if m.has_pdf and m.pdf_minutes > 0:
                    tags.append("PDF")
                if m.has_video and self.format_handler.effective_video_minutes(m) > 0:
                    tags.append("Video")
                format_tag = " — " + ", ".join(tags) if tags else ""

            response.append(f"{i}. {title} — {dur} min{format_tag}")
            response.append(f"   {expl}\n")

        response.append(
            f"Total: {total_minutes} min (~{total_minutes/60:.1f} hrs)")
        requested = max(1, int(self.user.hours * 60))
        coverage = total_minutes / requested

        if coverage >= 0.8:
            response.append(" Good coverage!")
        elif coverage >= 0.6:
            response.append("Note: Prioritized quality over quantity.")
        else:
            response.append(
                " Limited matches. Try: broader topic, more time, or different format.")

        response.append("\nCommands: 'save' | 'remove N' | 'reset'")
        return "\n".join(response)

    def _is_command(self, msg: str) -> bool:
        """Check if message is a command"""
        low = msg.lower()
        return (low in ("help", "commands", "?", "reset", "clear", "restart", "save", "export") or
                re.search(r"\bremove\s+(\d+)\b", low) is not None)

    def _handle_command(self, msg: str) -> str:
        """Handle system commands"""
        low = msg.lower()

        if low in ("help", "commands", "?"):
            return "Commands: 'save' | 'remove N' | 'reset'\nJust chat naturally and I'll help you create a learning plan!"

        if low in ("reset", "clear", "restart"):
            self.user = UserProfile()
            self.last = []
            self._pending_topic_choice = None
            return "Profile reset. What would you like to learn?"

        if low in ("save", "export"):
            return self._save_plan()

        m_rm = re.search(r"\bremove\s+(\d+)\b", low)
        if m_rm:
            n = int(m_rm.group(1))
            if not self.last:
                return "Nothing to remove yet."
            if 1 <= n <= len(self.last):
                removed = self.format_handler.display_title_resolved(
                    self.last[n-1].module, self.user.format)
                del self.last[n - 1]
                return f"Removed: {removed[:80]}"
            return f"Choose 1-{len(self.last)}."

        return "Unknown command."

    def _update_user_profile(self, extracted: dict):
        """Update user profile with extracted information and validation"""
        if extracted.get("topic"):
            validated_topic = validate_topic(extracted["topic"])
            if validated_topic and not self.user.topic:
                self.user.topic = validated_topic
                log.debug(f"Updated topic to: {self.user.topic}")

        if extracted.get("hours"):
            validated_hours = validate_hours(extracted["hours"])
            if validated_hours and self.user.hours <= 0:
                self.user.hours = validated_hours
                log.debug(f"Updated hours to: {self.user.hours}")

        if extracted.get("role"):
            validated_role = validate_role(extracted["role"])
            if validated_role and not self.user.profession:
                self.user.profession = validated_role
                log.debug(f"Updated role to: {self.user.profession}")

        if extracted.get("format"):
            validated_format = validate_format(extracted["format"])
            if validated_format and not self.user.format:
                self.user.format = validated_format
                log.debug(f"Updated format to: {self.user.format}")

    def _ask_for_next_missing_field(self, field: str) -> str:
        """Fallback method for asking for missing information"""
        field_info = self.info_collector.REQUIRED_FIELDS[field]
        examples = ", ".join(field_info["examples"])
        return f"{field_info['prompt']} (examples: {examples})"

    def process_message(self, user_input: str) -> str:
        """Main message processing with LLM-first approach"""
        # Input validation
        if len(user_input) > self.cfg.max_input_length:
            return f"Input too long (max {self.cfg.max_input_length} characters). Please shorten your message."

        msg = user_input.strip()

        if not msg:
            return "Hi! Tell me what you want to learn and I'll help you create a personalized learning plan.\n\nExamples:\n• 'I want to learn about machine learning for 10 hours'\n• 'RAG for data scientists, 6 hours, videos only'"

        # Handle commands first
        if self._is_command(msg):
            return self._handle_command(msg)

        # Handle pending topic choice
        if self._pending_topic_choice:
            topics = self._pending_topic_choice
            low = msg.lower()
            if low in ('1', 'first', topics[0]):
                self.user.topic = topics[0]
                self._pending_topic_choice = None
                return self._continue_after_topic_selection()
            elif low in ('2', 'second', topics[1]) and len(topics) > 1:
                self.user.topic = topics[1]
                self._pending_topic_choice = None
                return self._continue_after_topic_selection()
            elif low in ('both', '3', 'all'):
                self.user.topic = "machine learning"
                self._pending_topic_choice = None
                return self._continue_after_topic_selection()
            else:
                return f"Please choose:\n  1. {topics[0].title()}\n  2. {topics[1].title()}\n  3. Both (broader search)\n\nType: 1, 2, or 3"

        # Check what information we still need
        missing_fields = self.info_collector.get_missing_fields(self.user)

        if missing_fields:
            # Use LLM to extract info and ask for missing pieces
            prompt = self.info_collector.create_collection_prompt(
                msg, missing_fields, self.user)

            try:
                llm_response = ollama_chat_json(
                    self.cfg.llm_model, prompt, temp=0.1, cache_size=self.cfg.llm_cache_size)

                # Update user profile with extracted information
                extracted = llm_response.get("extracted", {})
                self._update_user_profile(extracted)

                # Check for multiple topics after extraction
                if extracted.get("topic"):
                    multiple_topics = self._detect_multiple_topics(msg)
                    if multiple_topics:
                        self._pending_topic_choice = multiple_topics
                        topics = multiple_topics
                        return (f"I detected multiple topics: **{topics[0]}** and **{topics[1]}**\n\n"
                                f"Which would you like to focus on?\n"
                                f"  1. {topics[0].title()} only\n"
                                f"  2. {topics[1].title()} only\n"
                                f"  3. Both (broader search)\n\n"
                                f"Type: 1, 2, or 3")

                # Check if we're now complete
                still_missing = self.info_collector.get_missing_fields(
                    self.user)
                if not still_missing:
                    if self.db and self.session_id:
                        self.db.update_session_slots(
                            self.session_id,
                            self.user.topic,
                            self.user.profession,
                            self.user.hours,
                            self.user.format
                        )
                    return self._generate_learning_path()
                else:
                    response = llm_response.get(
                        "response", "I need more information to help you.")
                    # Add current profile for context
                    if any([self.user.topic, self.user.hours > 0, self.user.profession, self.user.format]):
                        profile_parts = []
                        if self.user.topic:
                            profile_parts.append(f"Topic: {self.user.topic}")
                        if self.user.hours > 0:
                            profile_parts.append(f"Hours: {self.user.hours}")
                        if self.user.profession:
                            profile_parts.append(
                                f"Role: {self.user.profession}")
                        if self.user.format:
                            profile_parts.append(f"Format: {self.user.format}")
                        response += f"\n\n✓ Current: {' | '.join(profile_parts)}"
                    return response

            except Exception as e:
                log.error(f"LLM extraction failed: {e}")
                # Fallback to simple prompting
                return self._ask_for_next_missing_field(missing_fields[0])
        else:
            # All required info collected, generate the learning path
            return self._generate_learning_path()

    def _continue_after_topic_selection(self) -> str:
        """Continue flow after topic selection"""
        missing_fields = self.info_collector.get_missing_fields(self.user)
        if not missing_fields:
            return self._generate_learning_path()
        else:
            return self._ask_for_next_missing_field(missing_fields[0])

    def _generate_learning_path(self) -> str:
        """Generate the actual learning path after all slots are filled"""
        log.info(
            f"Generating learning path for: {self.user.topic}, {self.user.hours}h, {self.user.profession}, {self.user.format}")

        target_minutes = max(1, int(round(self.user.hours * 60)))
        expanded = self._expand_query(self.user.topic, self.user.profession)

        if len(expanded) == 0:
            return f"Could not expand query for '{self.user.topic}'. Try a different topic?"

        hits = self._score(self.user.topic, expanded)
        if not hits:
            format_msg = f" in {self.user.format} format" if self.user.format != "both" else ""
            return f"No modules found for '{self.user.topic}'{format_msg}.\n\nTry:\n  • Broader topic (e.g., 'machine learning' instead of specific algorithm)\n  • Different format\n  • Check spelling\n  • Type 'reset' to start from the beginning"

        selected = self._pack_time_optimal(hits, target_minutes)
        if not selected:
            return f"Couldn't fit modules into {self.user.hours} hours.\n\nTry:\n  • Increasing time budget\n  • Broader topic search"

        num_batches = math.ceil(len(
            selected) / self.cfg.explanation_batch_size) if self.cfg.batch_explanations else len(selected)
        log.info(
            f"Generating explanations for {len(selected)} modules in ~{num_batches} API call(s)...")

        exps = self._generate_explanations(self.user.topic, self.user.profession, [
                                           h.module for h in selected])
        self.last = selected
        return self._create_learning_path(selected, exps)

    def _save_plan(self) -> str:
        """Save the current learning plan"""
        if not self.last:
            return "No plan to save! Generate a plan first."

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smarter_plan_{ts}.json"
        names = [self.format_handler.display_title_resolved(
            h.module, self.user.format) or h.module.title or "" for h in self.last if h.module.title]

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(names, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Failed to save: {e}"

        if cli_streamlit_compat.is_streamlit() and zipper_prep is not None:
            is_pdf, is_video = self.user.format in (
                'pdf', 'both'), self.user.format in ('video', 'both')
            df_course = pd.DataFrame({
                'course_name': [h.module.course for h in self.last],
                'pdf_name': [h.module.pdf_name if is_pdf else np.nan for h in self.last],
                'video_related_to_pdf': [h.module.video_name if is_video else np.nan for h in self.last]})
            zipper_prep.prepare_json_and_button(df_course)
            sys.exit(0)
        return f"Saved {len(names)} modules to {filename}"


def main():
    """Main entry point"""
    cfg = Config()
    try:
        ensure_ollama_ready(cfg)
        bot = SMARTERChatbot(cfg=cfg)
        db = DatabaseManager() ## This is to initialize the memory when the chatbot is started.
    except Exception as e:
        print(f"\nStartup error: {e}")
        sys.exit(1)


    ## adding the user identification
    try:
        identifier = input("Enter your user ID or name: ").strip()
        if not identifier:
            identifier = "anonymous_user"
    except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            sys.exit(0)

    user = db.get_or_create_user(identifier)
    user_id = str(user["user_id"])

    ## check for past sessions
    past_sessions = db.get_user_sessions(user_id)
    if past_sessions:
        print(f"\nWelcome back, {identifier}! You have {len(past_sessions)} past session(s).")
        for i, s in enumerate(past_sessions[:5], 1):
            topic = s.get("topic", "unknown topic")
            date = s["last_active"].strftime("%Y-%m-%d %H:%M")
            print(f"  {i}. {topic} (last active: {date})")
        print(f"N. Start a new session")
        print()
    else:
        print(f"\nWelcome, {identifier}! Let's create your first learning plan.\n")

    session_id = db.create_session(user_id)
    bot.db = db
    bot.session_id = session_id

    print("SMARTER Course Planner - Örebro University")
    print("\nWhat do you want to learn?")
    # print("\nCommands: save | remove N | reset | help | quit")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            db.touch_session(session_id)
            db.update_last_seen(user_id)
            print("\n\nGoodbye! Happy learning!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "goodbye", "q"):
            db.touch_session(session_id)  # Update last active time
            db.update_last_seen(user_id)  # Update user last seen
            print("\nGoodbye! Happy learning!")
            break

        try:
            db.save_message(session_id, "user", user_input)
            resp = bot.process_message(user_input)
            db.save_message(session_id, "assistant", resp)
            print(f"\nAssistant: {resp}")
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
            print(f"\nSorry, something went wrong: {e}")
            print("Try: 'reset' to start over or 'help' for commands")


if __name__ == "__main__":
    main()
