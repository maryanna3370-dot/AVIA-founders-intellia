#!/usr/bin/env python3
"""AI Founders intellia - feedback collection tool for entrepreneurs/medical/students.

GUI app to collect feedback from 4 stakeholders on a problem.
Supports input via text, PDF, WAV, or microphone.
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, simpledialog
import argparse
import json
import os
import re
import sqlite3
import hashlib
import shutil
import subprocess
import time
import queue
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import threading
from getpass import getpass

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def extract_text_from_pdf(path):
    if PdfReader is None:
        raise RuntimeError('PyPDF2 is not installed. Install via pip install PyPDF2.')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ''
        text.append(page_text)
    return '\n'.join(text).strip()


def extract_text_from_wav(path):
    if sr is None:
        raise RuntimeError('speech_recognition is not installed. Install via pip install SpeechRecognition.')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    # offline pocketsphinx engine
    try:
        text = r.recognize_sphinx(audio)
    except sr.RequestError as e:
        raise RuntimeError(f'Pocketsphinx recognition unavailable: {e}')
    except sr.UnknownValueError:
        return ''
    return text.strip()


def load_pdf(problem_text):
    if PdfReader is None:
        messagebox.showerror("Error", "PyPDF2 not installed")
        return
    path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if path:
        try:
            text = extract_text_from_pdf(path)
            problem_text.delete(1.0, tk.END)
            problem_text.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Error", str(e))


def load_wav(problem_text):
    if sr is None:
        messagebox.showerror("Error", "SpeechRecognition not installed")
        return
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        try:
            text = extract_text_from_wav(path)
            problem_text.delete(1.0, tk.END)
            problem_text.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Error", str(e))


WORKSPACE_ROOT = os.environ.get('AI_WORKSPACE_ROOT', os.getcwd())
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
DB_PATH = os.path.join(REPORTS_DIR, 'reports.db')
INDEX_QUEUE = queue.Queue()
INDEX_WORKER_CALLBACK = None
_index_worker_started = False

# Retrieval metric counters and stats
METRICS = {
    'query_count': 0,
    'response_count': 0,
    'last_query_latency_ms': 0.0,
    'avg_query_latency_ms': 0.0,
    'total_query_latency_ms': 0.0,
    'last_response_latency_ms': 0.0,
    'avg_response_latency_ms': 0.0,
    'total_response_latency_ms': 0.0,
    'top_k_requested': 0,
    'top_k_returned': 0,
    'search_candidates': 0,
    'db_embeddings_loaded': 0,
    'chunk_count': 0,
    'chunk_min_words': 0,
    'chunk_max_words': 0,
    'chunk_avg_words': 0.0,
    'embedding_calls': 0,
    'embedding_cache_hits': 0,
    'embedding_cache_misses': 0,
    'cache_hit_rate': 0.0,
}

# Lightweight embedding models: BGE-small outperforms MiniLM
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
EMBEDDING_INSTRUCTION = 'Represent this sentence for searching relevant passages: '
LLM_MODEL_NAME = 'distilgpt2'
OLLAMA_MODEL_NAME = os.environ.get('OLLAMA_MODEL_NAME', 'llama2')
OLLAMA_MAX_TOKENS = 256
OLLAMA_TEMPERATURE = 0.0
CONVERSATION_HISTORY_LIMIT = 1000

# Device configuration (CUDA acceleration + CPU fallback)
_embedding_model = None
_llm_pipeline = None
_embedding_cache = {}  # Cache for embeddings to avoid recomputation
_device = None  # Detected device (cuda or cpu)
_cuda_available = False
_max_batch_size = 32  # Adjust based on available GPU memory

# Current session user (id and name)
CURRENT_USER_ID = None
CURRENT_USERNAME = None
CURRENT_CONVERSATION_ID = None

# Continuous voice conversation state
CONTINUOUS_VOICE_STOP_EVENT = None
CONTINUOUS_VOICE_THREAD = None

# Agent logging
AGENT_LOG_PATH = None
AGENT_LOGGER = None
AGENT_LOG_MAX_BYTES = 5 * 1024 * 1024
AGENT_LOG_BACKUP_COUNT = 5


def detect_device():
    """Detect and return available device (CUDA or CPU)."""
    global _device, _cuda_available
    if _device is not None:
        return _device
    
    if torch is not None:
        try:
            if torch.cuda.is_available():
                _device = torch.device('cuda')
                _cuda_available = True
                device_name = torch.cuda.get_device_name(0)
                return _device
        except Exception:
            pass
    
    _device = torch.device('cpu') if torch is not None else None
    _cuda_available = False
    return _device


def get_device_info():
    """Get human-readable device information."""
    device = detect_device()
    if device is None:
        return "CPU (torch not available)"
    
    if device.type == 'cuda' and torch is not None:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA - {gpu_name} ({gpu_mem:.1f}GB)"
        except Exception:
            return "CUDA (unknown GPU)"
    return "CPU"


def get_optimal_batch_size():
    """Get optimal batch size based on device and available memory."""
    device = detect_device()
    if device is None or device.type == 'cpu':
        return 16  # Conservative for CPU
    
    if torch is not None and device.type == 'cuda':
        try:
            # Get available GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory
            reserved_mem = torch.cuda.memory_reserved(0)
            available_mem = (total_mem - reserved_mem) / (1024**3)  # GB
            
            # Estimate: ~1GB per batch for BGE-small
            if available_mem > 8:
                return 64  # Large batch for high-end GPUs
            elif available_mem > 4:
                return 48
            elif available_mem > 2:
                return 32
            else:
                return 16
        except Exception:
            return 32
    
    return 32


def ensure_reports_dir():
    if not os.path.isdir(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)


def set_workspace_root(path):
    """Set the data workspace root directory for reports and database storage."""
    global WORKSPACE_ROOT, REPORTS_DIR, DB_PATH
    if path:
        WORKSPACE_ROOT = os.path.abspath(path)
    else:
        WORKSPACE_ROOT = os.getcwd()
    REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
    DB_PATH = os.path.join(REPORTS_DIR, 'reports.db')
    ensure_reports_dir()
    return WORKSPACE_ROOT


def index_queue_size():
    return INDEX_QUEUE.qsize()


def enqueue_indexing_job(report_id, chunks, created_at):
    """Queue a report indexing job for background processing."""
    ensure_reports_dir()
    INDEX_QUEUE.put((report_id, chunks, created_at))
    if INDEX_WORKER_CALLBACK:
        INDEX_WORKER_CALLBACK(
            f'Queued report {report_id} for indexing ({len(chunks)} chunks)',
            queue_size=index_queue_size()
        )
    start_indexing_worker()


def start_indexing_worker(update_callback=None):
    global _index_worker_started, INDEX_WORKER_CALLBACK
    if update_callback is not None:
        INDEX_WORKER_CALLBACK = update_callback
    if _index_worker_started:
        return
    _index_worker_started = True
    threading.Thread(target=_indexing_worker, daemon=True).start()


def _indexing_worker():
    while True:
        report_id, chunks, created_at = INDEX_QUEUE.get()
        init_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            try:
                conn.execute('PRAGMA journal_mode = WAL')
                conn.execute('PRAGMA synchronous = NORMAL')
                conn.execute('PRAGMA temp_store = MEMORY')
            except Exception:
                pass
            cur = conn.cursor()
            cur.execute('DELETE FROM embeddings WHERE report_id = ?', (report_id,))
            total = len(chunks)
            if INDEX_WORKER_CALLBACK:
                INDEX_WORKER_CALLBACK(
                    f'Starting indexing report {report_id} ({total} chunks)',
                    queue_size=max(index_queue_size(), 0)
                )
            batch_size = 8
            for batch_start in range(0, total, batch_size):
                batch = chunks[batch_start:batch_start + batch_size]
                embeddings = batch_embed_texts(batch, batch_size=16)
                rows = []
                for offset, (chunk_text, embedding) in enumerate(zip(batch, embeddings)):
                    if embedding is None:
                        continue
                    chunk_index = batch_start + offset
                    checksum = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                    token_count = len(re.findall(r'\S+', chunk_text))
                    rows.append((report_id, chunk_index, chunk_text, 'report', EMBEDDING_MODEL_NAME, token_count, '', checksum, sqlite3.Binary(embedding.encode('utf-8')), created_at, get_current_user_id()))
                if rows:
                    cur.executemany(
                        'INSERT INTO embeddings (report_id, chunk_index, chunk_text, source_type, embedding_model, token_count, tags, checksum, embedding, created_at, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        rows
                    )
                    conn.commit()
                if INDEX_WORKER_CALLBACK:
                    indexed = min(batch_start + len(batch), total)
                    percent = int((indexed / total) * 100) if total else 100
                    INDEX_WORKER_CALLBACK(
                        f'Indexed {indexed}/{total} chunks for report {report_id} ({percent}%)',
                        queue_size=max(index_queue_size(), 0)
                    )
            if INDEX_WORKER_CALLBACK:
                INDEX_WORKER_CALLBACK(
                    f'Finished indexing report {report_id}',
                    queue_size=max(index_queue_size(), 0)
                )
        except Exception as e:
            if INDEX_WORKER_CALLBACK:
                INDEX_WORKER_CALLBACK(
                    f'Indexing failed for report {report_id}: {e}',
                    queue_size=max(index_queue_size(), 0)
                )
        finally:
            conn.close()
            INDEX_QUEUE.task_done()


def increment_metric(name, amount=1):
    METRICS[name] = METRICS.get(name, 0) + amount


def set_metric(name, value):
    METRICS[name] = value


def calculate_cache_hit_rate():
    hits = METRICS.get('embedding_cache_hits', 0)
    misses = METRICS.get('embedding_cache_misses', 0)
    total = hits + misses
    return float(hits) / total if total else 0.0


def load_embedding_model():
    """Load embedding model with CUDA acceleration and CPU fallback."""
    global _embedding_model, _device
    
    if _embedding_model is not None:
        return _embedding_model
    
    if SentenceTransformer is None:
        return None
    
    try:
        device = detect_device()
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Move to GPU if available
        if device and device.type == 'cuda':
            try:
                _embedding_model = _embedding_model.to(device)
                # Enable memory-efficient inference
                if hasattr(_embedding_model, 'eval'):
                    _embedding_model.eval()
            except RuntimeError as e:
                # Fallback to CPU if GPU runs out of memory
                try:
                    _embedding_model = _embedding_model.to('cpu')
                except Exception:
                    pass
        
        # Ensure model is in eval mode
        if hasattr(_embedding_model, 'eval'):
            _embedding_model.eval()
            
    except Exception as e:
        _embedding_model = None
    
    return _embedding_model


def get_embedding(text, dim=384):
    """Get embedding with instruction-based approach, caching, and device management."""
    if not text:
        return None
    
    increment_metric('embedding_calls')
    # Check cache first
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _embedding_cache:
        increment_metric('embedding_cache_hits')
        set_metric('cache_hit_rate', calculate_cache_hit_rate())
        return _embedding_cache[text_hash]
    increment_metric('embedding_cache_misses')
    set_metric('cache_hit_rate', calculate_cache_hit_rate())
    
    model = load_embedding_model()
    if model is not None:
        try:
            device = detect_device()
            # Use instruction for better semantic matching
            query_with_instruction = f"{EMBEDDING_INSTRUCTION}{text}"
            embedded = model.encode(
                query_with_instruction,
                normalize_embeddings=True,
                device=device
            )
            if hasattr(embedded, 'tolist'):
                embedded = embedded.tolist()
            result = json.dumps([float(x) for x in embedded])
            _embedding_cache[text_hash] = result
            return result
        except RuntimeError as e:
            # Handle CUDA out of memory - fall back to hash-based
            if 'cuda' in str(e).lower() or 'out of memory' in str(e).lower():
                clear_gpu_memory()
        except Exception:
            pass
    
    # Fallback: hash-based embedding
    text_bytes = text.encode('utf-8')
    vector = [0.0] * dim
    for index, byte in enumerate(text_bytes):
        idx = byte % dim
        vector[idx] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    result = json.dumps(vector)
    _embedding_cache[text_hash] = result
    return result


def batch_embed_texts(texts, batch_size=None):
    """Efficiently embed multiple texts in batches with CUDA support and memory management."""
    if not texts:
        return []
    
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    model = load_embedding_model()
    if model is None:
        return [get_embedding(text) for text in texts]
    
    try:
        embeddings = []
        device = detect_device()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Add instruction prefix for better retrieval
            batch_with_instruction = [f"{EMBEDDING_INSTRUCTION}{text}" for text in batch]
            
            try:
                # Try encoding with optimal batch size
                batch_embeddings = model.encode(
                    batch_with_instruction,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=device
                )
                for emb in batch_embeddings:
                    embeddings.append(json.dumps([float(x) for x in emb]))
                    
            except RuntimeError as e:
                # Handle CUDA out of memory errors
                if 'cuda' in str(e).lower() or 'out of memory' in str(e).lower():
                    # Reduce batch size and try again
                    reduced_batch = batch_size // 2
                    if reduced_batch > 0:
                        sub_embeddings = batch_embed_texts(batch, batch_size=reduced_batch)
                        embeddings.extend(sub_embeddings)
                    else:
                        # Fall back to single embeddings
                        for text in batch:
                            embeddings.append(get_embedding(text))
                else:
                    # Other runtime errors, fall back
                    for text in batch:
                        embeddings.append(get_embedding(text))
            
            # Clear CUDA cache if available
            if torch is not None and device and device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        
        return embeddings
    except Exception:
        # Complete fallback to single embeddings
        return [get_embedding(text) for text in texts]


def clear_gpu_memory():
    """Clear GPU memory cache if CUDA is available."""
    if torch is not None and detect_device() and detect_device().type == 'cuda':
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        except Exception:
            return False
    return False


def set_embedding_model(model_name):
    """Switch embedding model. Supported models:
    - 'BAAI/bge-small-en-v1.5' (default, recommended)
    - 'BAAI/bge-base-en-v1.5' (larger, better quality)
    - 'all-MiniLM-L6-v2' (lightweight, older)
    - 'sentence-transformers/all-mpnet-base-v2' (high quality)
    """
    global _embedding_model, EMBEDDING_MODEL_NAME
    if model_name not in [
        'BAAI/bge-small-en-v1.5',
        'BAAI/bge-base-en-v1.5', 
        'all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2'
    ]:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Clean up old model and GPU memory
    _embedding_model = None
    clear_gpu_memory()
    
    # Reset caches
    EMBEDDING_MODEL_NAME = model_name
    _embedding_cache.clear()
    
    # Pre-load new model
    load_embedding_model()
    device_info = get_device_info()
    return f"Embedding model switched to {model_name} (Using: {device_info})"


def parse_embedding(value):
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    embedding = json.loads(value)
    if np is not None:
        return np.array(embedding, dtype=np.float32)
    return embedding


def split_sentences(text):
    pieces = re.split(r'(?<=[.!?])\s+', text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(text, max_chars=500, overlap=100):
    if not text:
        return []
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = split_sentences(cleaned)
    chunks = []
    current = ''
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())

    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
            else:
                prev = chunks[idx - 1].split()[-overlap:] if overlap else []
                overlapped_chunks.append(' '.join(prev + chunk.split()))
        chunks = overlapped_chunks
    return chunks


def build_report_chunks(data, max_chars=500, overlap=100):
    text_pieces = [data.get('problem', '')]
    for fb in data.get('feedback', []):
        for field_name in ['resonate', 'aspects', 'questions', 'missing']:
            value = fb.get(field_name)
            if value:
                text_pieces.append(f"{field_name}: {value}")
    full_text = '\n'.join([piece for piece in text_pieces if piece]).strip()
    return chunk_text(full_text, max_chars=max_chars, overlap=overlap)


def normalize_text(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def tokenize_text(text):
    return [token for token in normalize_text(text).split() if token]


def keyword_score(query, text):
    query_tokens = tokenize_text(query)
    text_tokens = tokenize_text(text)
    if not query_tokens or not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    text_set = set(text_tokens)
    matches = sum(1 for token in query_tokens if token in text_set)
    recall = matches / len(query_tokens)
    precision = sum(1 for token in query_tokens if token in text_tokens) / max(len(text_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def combine_search_scores(vector_score, keyword_score_value, alpha=0.5):
    return vector_score * alpha + keyword_score_value * (1 - alpha)


def calculate_query_complexity(query):
    """Estimate query complexity to adapt semantic weighting.
    
    Complex queries benefit from higher semantic weighting.
    Simple keyword queries benefit from higher keyword weighting.
    """
    query_tokens = tokenize_text(query)
    
    # Factors indicating query complexity
    num_tokens = len(query_tokens)
    avg_token_length = sum(len(t) for t in query_tokens) / max(num_tokens, 1)
    
    # Simple queries: short, few tokens
    # Complex queries: longer, more tokens, rare words
    if num_tokens < 3:
        return 0.5  # Simple → favor keywords
    elif num_tokens < 8 and avg_token_length < 6:
        return 0.6  # Medium simple
    elif num_tokens < 15:
        return 0.75  # Medium complex
    else:
        return 0.85  # Complex → favor semantics


def calculate_semantic_confidence(vector_score, keyword_score):
    """Calculate confidence in semantic vs keyword match.
    
    High semantic score + low keyword score = strong semantic match
    High keyword score + low semantic score = strong keyword match
    """
    if vector_score > 0.8 and keyword_score < 0.3:
        return 'semantic_strong'  # Trust semantics
    elif keyword_score > 0.7 and vector_score < 0.4:
        return 'keyword_strong'   # Trust keywords
    elif vector_score > 0.6 and keyword_score > 0.5:
        return 'both_strong'      # Both signals agree
    else:
        return 'weak'             # Uncertain


def calculate_diversity_penalty(item, ranked_items, penalty_factor=0.1):
    """Penalize items similar to already-ranked items (avoid redundancy).
    
    If many top results are from the same report, lower-ranked items
    from that report get penalized to show diversity.
    """
    item_report_id = item['report_id']
    same_report_count = sum(1 for r in ranked_items if r['report_id'] == item_report_id)
    
    # Penalty increases with number of items from same report
    # First item: no penalty
    # Second item: 10% penalty
    # Third item: 20% penalty, etc.
    penalty = 1.0 - (penalty_factor * max(0, same_report_count))
    return max(penalty, 0.6)  # Minimum 60% of original score


def extract_context_topics(history, top_n=10):
    """Extract key topics and keywords from conversation history."""
    if not history:
        return {}
    
    # Combine all history content
    full_text = ' '.join([msg['content'] for msg in history if msg['content']])
    
    # Extract tokens and weight recent messages more
    tokens = tokenize_text(full_text)
    token_freq = {}
    
    # Weight recent messages higher
    for i, msg in enumerate(reversed(history)):
        weight = (i + 1) / len(history) if history else 1.0
        msg_tokens = tokenize_text(msg['content'])
        for token in msg_tokens:
            token_freq[token] = token_freq.get(token, 0) + weight
    
    # Return top tokens as context topics
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {token: score for token, score in sorted_tokens}


def extract_mentioned_reports(history):
    """Extract report IDs or filenames mentioned in conversation history."""
    if not history:
        return set()
    
    mentioned = set()
    for msg in history:
        content = msg.get('content', '')
        # Match patterns like "Report X", "report_id: 123", "[1]", etc.
        import re
        
        # Extract explicit report references [1], [2], etc.
        ids = re.findall(r'\[(\d+)\]', content)
        mentioned.update(ids)
        
        # Extract "report X" patterns
        report_refs = re.findall(r'[Rr]eport\s*(\d+)', content)
        mentioned.update(report_refs)
    
    return mentioned


def context_relevance_boost(text, context_topics, boost_factor=1.3):
    """Boost relevance score if text matches conversation context."""
    if not context_topics:
        return 1.0
    
    text_tokens = set(tokenize_text(text))
    matching_topics = sum(context_topics[token] for token in text_tokens if token in context_topics)
    
    # Apply boost if there are matching topics
    if matching_topics > 0:
        # Scale boost by number and frequency of matches
        boost = 1.0 + min(matching_topics / 10, boost_factor - 1.0)
        return boost
    return 1.0


def rerank_candidates(query, candidates, top_n=50, alpha=0.5, context_topics=None, context_boost=1.3, mentioned_reports=None):
    reranked = []
    for item in candidates:
        kw = keyword_score(query, item['text'])
        combined = combine_search_scores(item['vector_score'], kw, alpha=alpha)
        item['keyword_score'] = kw
        
        # Apply context-based boost if available
        if context_topics:
            boost = context_relevance_boost(item['text'], context_topics, boost_factor=context_boost)
            combined = combined * boost
        
        # Further boost if this report was mentioned in conversation
        if mentioned_reports and str(item['report_id']) in mentioned_reports:
            combined = combined * 1.5  # 50% boost for explicitly mentioned reports
        
        item['score'] = combined
        reranked.append(item)
    return sorted(reranked, key=lambda item: item['score'], reverse=True)[:top_n]


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    if np is not None:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return sum(x * y for x, y in zip(a, b))


def load_local_llm():
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline
    if pipeline is None:
        return None
    try:
        device = 0 if torch is not None and torch.cuda.is_available() else -1
        _llm_pipeline = pipeline('text-generation', model=LLM_MODEL_NAME, device=device, return_full_text=False)
    except Exception:
        _llm_pipeline = None
    return _llm_pipeline


def ollama_cli_available():
    return shutil.which('ollama') is not None


def ollama_run(prompt, model=None, callback=None, timeout=600):
    if model is None:
        model = OLLAMA_MODEL_NAME
    if not ollama_cli_available():
        return None
    cmd = ['ollama', 'run', model, '--format', 'text', '--hidethinking']
    try:
        if callback is None:
            result = subprocess.run(
                cmd + [prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return result.stdout.strip()

        process = subprocess.Popen(
            cmd + [prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        output = ''
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ''):
                if line == '':
                    break
                callback(line)
                output += line
            process.stdout.close()
        process.wait(timeout=timeout)
        if process.returncode != 0:
            return None
        return output.strip()
    except subprocess.CalledProcessError:
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def local_llm_generate(prompt, max_tokens=200):
    llm = load_local_llm()
    if llm is not None:
        try:
            max_length = min(len(prompt.split()) + max_tokens, 512)
            output = llm(prompt, max_length=max_length, do_sample=False, num_return_sequences=1)
            if output and isinstance(output, list):
                return output[0].get('generated_text', '').strip()
            return str(output)
        except Exception:
            pass

    if ollama_cli_available():
        ollama_result = ollama_run(prompt)
        if ollama_result:
            return ollama_result

    return local_llm_summarize(prompt, max_sentences=6)


def local_llm_generate_stream(prompt, max_tokens=200, callback=None):
    if callback is None:
        return local_llm_generate(prompt, max_tokens=max_tokens)

    if ollama_cli_available():
        result = ollama_run(prompt, callback=callback)
        if result is not None:
            return result

    output = local_llm_generate(prompt, max_tokens=max_tokens)
    if output:
        chunk_size = 64
        for i in range(0, len(output), chunk_size):
            callback(output[i:i + chunk_size])
    return output


def build_vector_index(embeddings):
    if np is None or len(embeddings) == 0:
        return None
    matrix = np.vstack([np.asarray(emb, dtype=np.float32) for emb in embeddings])
    if faiss is not None:
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index
    try:
        from sklearn.neighbors import NearestNeighbors
        return NearestNeighbors(n_neighbors=min(len(matrix), 50), metric='cosine').fit(matrix)
    except Exception:
        return None


def index_search(query_embedding, candidates, top_k=50):
    """Optimized vector search with better similarity computation."""
    if np is None or len(candidates) == 0:
        return candidates[:top_k]
    
    embeddings = [np.asarray(c['embedding'], dtype=np.float32) for c in candidates]
    
    if faiss is not None:
        try:
            index = build_vector_index(embeddings)
            query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            # Use inner product for normalized embeddings (faster and more accurate)
            distances, indices = index.search(query_vec, min(len(embeddings), top_k * 2))
            selected = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    candidates[idx]['vector_score'] = float(distances[0][i])
                    selected.append(candidates[idx])
            return selected
        except Exception:
            pass
    
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(
            n_neighbors=min(len(embeddings), top_k * 2),
            metric='cosine'
        ).fit(np.vstack(embeddings))
        distances, indices = nn.kneighbors(
            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1),
            return_distance=True
        )
        selected = []
        for i, idx in enumerate(indices[0]):
            candidates[idx]['vector_score'] = float(1 - distances[0][i])  # Convert distance to similarity
            selected.append(candidates[idx])
        return selected
    except Exception:
        return sorted(candidates, key=lambda item: item['vector_score'], reverse=True)[:top_k]


def search_reports(query, top_k=5, context_topics=None, mentioned_reports=None):
    """Enhanced search with conversation context awareness."""
    search_start = time.time()
    increment_metric('query_count')
    increment_metric('top_k_requested', top_k)

    query_embedding = parse_embedding(get_embedding(query))
    if query_embedding is None:
        return []
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT report_id, chunk_index, chunk_text, embedding FROM embeddings WHERE user_id = ?', (uid,))
    rows = c.fetchall()
    conn.close()

    set_metric('db_embeddings_loaded', len(rows))

    candidates = []
    for report_id, chunk_index, chunk_text, embedding in rows:
        emb = parse_embedding(embedding)
        if emb is None:
            continue
        score = cosine_similarity(query_embedding, emb)
        candidates.append({
            'report_id': report_id,
            'chunk_index': chunk_index,
            'text': chunk_text,
            'vector_score': score,
            'embedding': emb
        })

    if not candidates:
        set_metric('search_candidates', 0)
        search_elapsed = (time.time() - search_start) * 1000
        set_metric('last_query_latency_ms', search_elapsed)
        increment_metric('total_query_latency_ms', search_elapsed)
        set_metric('avg_query_latency_ms', METRICS['total_query_latency_ms'] / METRICS['query_count'] if METRICS['query_count'] else 0.0)
        return []

    set_metric('search_candidates', len(candidates))
    word_counts = [len(item['text'].split()) for item in candidates]
    if word_counts:
        set_metric('chunk_count', len(word_counts))
        set_metric('chunk_min_words', min(word_counts))
        set_metric('chunk_max_words', max(word_counts))
        set_metric('chunk_avg_words', sum(word_counts) / len(word_counts))

    # Use expanded candidate pool for better reranking
    vector_candidates = index_search(query_embedding, candidates, top_k=max(top_k * 10, 100))
    
    # Improved reranking with context awareness (favor semantic similarity)
    reranked = rerank_candidates(query, vector_candidates, top_n=top_k, alpha=0.7, 
                                 context_topics=context_topics, context_boost=1.3,
                                 mentioned_reports=mentioned_reports)

    top_results = []
    for item in reranked:
        report = load_db_report(item['report_id'])
        top_results.append({
            'report_id': item['report_id'],
            'filename': report['filename'],
            'created_at': report['created_at'],
            'score': item['score'],
            'vector_score': item['vector_score'],
            'keyword_score': item['keyword_score'],
            'chunk_index': item['chunk_index'],
            'field_name': 'report_chunk',
            'text': item['text'],
            'chunk_context': load_surrounding_chunks(item['report_id'], item['chunk_index'], window=2),
            'problem': report['problem']
        })

    set_metric('top_k_returned', len(top_results))
    search_elapsed = (time.time() - search_start) * 1000
    set_metric('last_query_latency_ms', search_elapsed)
    increment_metric('total_query_latency_ms', search_elapsed)
    set_metric('avg_query_latency_ms', METRICS['total_query_latency_ms'] / METRICS['query_count'] if METRICS['query_count'] else 0.0)
    return top_results


def load_chunk_with_neighbors(report_id, chunk_index, window=1):
    """Return chunk text for a given index plus neighbor chunks."""
    return load_surrounding_chunks(report_id, chunk_index, window=window)


def load_chunk_number_with_neighbors(report_id, chunk_number, window=1):
    """Retrieve a chunk and neighbors using 1-based chunk numbering."""
    zero_based_index = max(chunk_number - 1, 0)
    return load_surrounding_chunks(report_id, zero_based_index, window=window)


def load_surrounding_chunks(report_id, chunk_index, window=1):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    lower = max(chunk_index - window, 0)
    upper = chunk_index + window
    c.execute(
        'SELECT chunk_index, chunk_text FROM embeddings WHERE report_id = ? AND user_id = ? AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index',
        (report_id, uid, lower, upper)
    )
    rows = c.fetchall()
    conn.close()
    return [{'chunk_index': idx, 'chunk_text': text} for idx, text in rows]


def local_llm_summarize(prompt, max_sentences=4):
    if pipeline is not None:
        try:
            summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=-1)
            summary = summarizer(prompt, max_length=200, min_length=80, do_sample=False)
            return summary[0]['summary_text'].strip()
        except Exception:
            pass

    sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
    if len(sentences) <= max_sentences:
        return ' '.join(sentences).strip()
    return ' '.join(sentences[:max_sentences]).strip()


def summarize_search_results(query, top_k=5):
    results = search_reports(query, top_k)
    if not results:
        return [], 'No matching reports found.'

    context_parts = []
    for result in results:
        chunk_context_text = '\n'.join(
            f"Chunk {ctx['chunk_index']}: {ctx['chunk_text']}"
            for ctx in result.get('chunk_context', [])
        )
        context_parts.append(
            f"Report {result['report_id']} ({result['created_at']}): chunk {result['chunk_index']} match.\n"
            f"Text: {result['text']}\n"
            f"Context chunks:\n{chunk_context_text}\n"
            f"Problem: {result['problem']}"
        )
    prompt = f"Question: {query}\n\nTop matching reports:\n\n" + '\n\n'.join(context_parts)
    summary = local_llm_summarize(prompt)
    return results, summary


def build_memory_context(query, memories):
    if not memories:
        return ''
    lines = [f"Memory {m['id']} ({m['name']}): {m['snippet']}" for m in memories]
    return 'Relevant memories:\n' + '\n'.join(lines) + '\n\n'


def build_reasoning_prompt(query, cited_context, history_text='', memory_text='', context_text=''):
    return (
        "You are a local reasoning assistant.\n"
        "First generate a short reasoning plan, then provide the final answer.\n"
        "Use only the provided sources and memories, and cite sources by their IDs.\n"
        "If the answer is not supported by the sources, say you cannot answer.\n\n"
        f"{history_text}"
        f"{memory_text}"
        f"{context_text}"
        "Sources:\n"
        + '\n'.join(cited_context)
        + "\n\n"
        f"Question: {query}\n\n"
        "Reasoning plan:\n"
    )


def generate_rag_answer(query, top_k=5, stream_callback=None, save_to_memory=True, speak_answer=False):
    """Generate RAG answer with conversation context influencing retrieval."""
    response_start = time.time()
    if CURRENT_CONVERSATION_ID is None:
        create_conversation('session')
    save_conversation_message('user', query)
    
    # Get conversation context to influence ranking
    history = get_conversation_history(CURRENT_CONVERSATION_ID, limit=CONVERSATION_HISTORY_LIMIT)
    context_topics = extract_context_topics(history)
    mentioned_reports = extract_mentioned_reports(history)
    
    # Search with context-aware ranking
    results = search_reports(query, top_k, context_topics=context_topics, mentioned_reports=mentioned_reports)
    memories = find_relevant_memories(query, top_k=3)
    
    if not results:
        return [], 'No relevant reports found to answer this question.'

    cited_context = []
    source_citations = {}
    for idx, result in enumerate(results, 1):
        source_id = f"[{idx}]"
        source_citations[source_id] = {
            'report_id': result['report_id'],
            'filename': result['filename'],
            'created_at': result['created_at'],
            'score': result['score'],
            'chunk_index': result['chunk_index']
        }
        cited_context.append(
            f"{source_id} {result['filename']} ({result['created_at']}) chunk {result['chunk_index']}: {result['text'][:300]}"
        )

    history_text = ''
    if history:
        history_text = 'Conversation:\n' + '\n'.join(f"{item['role']}: {item['content']}" for item in history) + '\n\n'
    memory_text = build_memory_context(query, memories)
    
    # Show context topics and mentioned reports used for retrieval
    context_text = ''
    debug_info = []
    if context_topics:
        top_topics = sorted(context_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_str = ', '.join([f"{topic} ({score:.2f})" for topic, score in top_topics])
        debug_info.append(f"Topics: {topics_str}")
    if mentioned_reports:
        debug_info.append(f"Mentioned reports: {', '.join(sorted(mentioned_reports))}")
    
    if debug_info:
        context_text = f"(Using conversation context: {' | '.join(debug_info)})\n\n"
    
    prompt = build_reasoning_prompt(query, cited_context, history_text=history_text, memory_text=memory_text, context_text=context_text)

    if stream_callback is not None:
        answer = local_llm_generate_stream(prompt, max_tokens=300, callback=stream_callback)
    else:
        answer = local_llm_generate(prompt, max_tokens=300)

    if answer is None:
        answer = ''

    if save_to_memory and answer:
        memory_name = f"Memory {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        memory_content = f"Question: {query}\nAnswer: {answer}"
        save_memory(memory_name, memory_content, tags='auto,voice')

    if speak_answer and answer:
        threading.Thread(target=speak_text, args=(answer,), daemon=True).start()

    save_conversation_message('assistant', answer)

    citations_text = "\n\nSources cited:\n"
    for source_id, citation in source_citations.items():
        citations_text += f"{source_id} {citation['filename']} (ID: {citation['report_id']}, chunk {citation['chunk_index']}, Score: {citation['score']:.3f})\n"
        citations_text += f"    Navigate: python3 ai_founders_intellia.py --chunk-with-neighbors {citation['report_id']} {citation['chunk_index']}\n"

    final_answer = answer + citations_text
    if stream_callback is not None:
        stream_callback(citations_text)
    response_elapsed = (time.time() - response_start) * 1000
    increment_metric('response_count')
    set_metric('last_response_latency_ms', response_elapsed)
    increment_metric('total_response_latency_ms', response_elapsed)
    set_metric('avg_response_latency_ms', METRICS['total_response_latency_ms'] / METRICS['response_count'] if METRICS['response_count'] else 0.0)
    return results, final_answer


# -----------------------------
# Autonomous agent helpers
# -----------------------------
def agent_print(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[AGENT] {ts} {msg}"
    print(line)
    global AGENT_LOGGER
    if AGENT_LOGGER is not None:
        try:
            AGENT_LOGGER.info(msg)
        except Exception:
            pass


def agent_stream(text):
    print(text, end='', flush=True)
    global AGENT_LOGGER
    if AGENT_LOGGER is not None:
        try:
            AGENT_LOGGER.info(text)
        except Exception:
            pass


def agent_index_callback(msg, queue_size=None):
    ts = datetime.utcnow().isoformat()
    if queue_size is not None:
        line = f"[AGENT INDEX] {ts} {msg} (queue={queue_size})"
    else:
        line = f"[AGENT INDEX] {ts} {msg}"
    print(line)
    global AGENT_LOGGER
    if AGENT_LOGGER is not None:
        try:
            AGENT_LOGGER.info(f"INDEX: {msg} (queue={queue_size})")
        except Exception:
            pass


def reindex_all_reports():
    """Load all JSON files in reports/ and enqueue them for indexing."""
    init_db()
    ensure_reports_dir()
    files = sorted([f for f in os.listdir(REPORTS_DIR) if f.lower().endswith('.json')])
    if not files:
        agent_print('No JSON reports found to reindex.')
        return 0
    count = 0
    for fname in files:
        try:
            data = load_json_report(fname)
            save_report_to_db(fname, data)
            agent_print(f'Queued report {fname} for indexing')
            count += 1
        except Exception as e:
            agent_print(f'Failed to load {fname}: {e}')
    return count


def init_agent_logger(path=None, max_bytes=None, backup_count=None):
    """Initialize rotating agent logger. Defaults to REPORTS_DIR/agent.log.

    Parameters:
    - path: file path
    - max_bytes: rotation size in bytes
    - backup_count: number of rotated files to keep
    """
    global AGENT_LOG_PATH, AGENT_LOGGER, AGENT_LOG_MAX_BYTES, AGENT_LOG_BACKUP_COUNT
    if path is None:
        path = os.path.join(REPORTS_DIR, 'agent.log')
    AGENT_LOG_PATH = os.path.abspath(path)
    if max_bytes is not None:
        AGENT_LOG_MAX_BYTES = int(max_bytes)
    if backup_count is not None:
        AGENT_LOG_BACKUP_COUNT = int(backup_count)

    try:
        os.makedirs(os.path.dirname(AGENT_LOG_PATH), exist_ok=True)
        logger = logging.getLogger('ai_agent')
        logger.setLevel(logging.INFO)
        # Remove existing handlers
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        handler = RotatingFileHandler(AGENT_LOG_PATH, maxBytes=AGENT_LOG_MAX_BYTES, backupCount=AGENT_LOG_BACKUP_COUNT)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        AGENT_LOGGER = logger
        print(f'[AGENT] Agent logging initialized at {AGENT_LOG_PATH} (rotating {AGENT_LOG_MAX_BYTES} bytes, backups={AGENT_LOG_BACKUP_COUNT})')
    except Exception as e:
        AGENT_LOGGER = None
        print(f'[AGENT] Failed to initialize agent logger {path}: {e}')


def close_agent_logger():
    global AGENT_LOGGER
    try:
        if AGENT_LOGGER is not None:
            for h in list(AGENT_LOGGER.handlers):
                try:
                    AGENT_LOGGER.removeHandler(h)
                    h.close()
                except Exception:
                    pass
    except Exception:
        pass
    AGENT_LOGGER = None


def run_agent(query=None, loop=False, retries=2, retry_delay=2, log_path=None):
    """Enhanced autonomous CLI agent.

    - Reindexes reports (with retries)
    - Starts background indexer
    - Optionally runs a single RAG query or enters an interactive REPL
    - Uses simple retry/backoff for transient failures
    """
    agent_print('Agent starting')
    # Initialize agent logger with rotation options
    try:
        if log_path is not None:
            init_agent_logger(log_path)
        else:
            init_agent_logger()
    except Exception:
        pass
    init_db()  # Initialize the database
    ensure_reports_dir()

    # Reindex with retries
    attempt = 0
    queued = 0
    while attempt < max(1, retries):
        try:
            agent_print(f'Reindexing JSON reports (attempt {attempt+1}/{max(1, retries)})...')
            queued = reindex_all_reports()
            break
        except Exception as e:
            agent_print(f'Reindex attempt {attempt+1} failed: {e}')
            attempt += 1
            if attempt < retries:
                time.sleep(retry_delay)
            else:
                agent_print('Proceeding despite reindex failures')

    # Start background indexing worker with agent callback
    try:
        start_indexing_worker(update_callback=agent_index_callback)
    except Exception as e:
        agent_print(f'Failed to start indexing worker: {e}')

    # Allow some time for indexing to start
    time.sleep(1)

    def run_query_with_retries(q):
        attempt_q = 0
        while attempt_q < max(1, retries):
            try:
                agent_print(f'Running RAG for query (attempt {attempt_q+1}/{max(1, retries)})...')
                results, answer = generate_rag_answer(q, top_k=3, stream_callback=agent_stream)
                agent_print('\nRAG run complete.')
                return results, answer
            except Exception as e:
                agent_print(f'RAG attempt {attempt_q+1} failed: {e}')
                attempt_q += 1
                if attempt_q < retries:
                    time.sleep(retry_delay)
        agent_print('All RAG attempts failed.')
        return None, None

    try:
        if loop:
            agent_print('Entering interactive REPL mode. Type "exit" or Ctrl-C to quit.')
            while True:
                try:
                    q = input('Agent query> ').strip()
                except (EOFError, KeyboardInterrupt):
                    agent_print('\nREPL terminated by user')
                    break
                if not q:
                    continue
                if q.lower() in ('exit', 'quit'):
                    agent_print('Exiting REPL.')
                    break
                run_query_with_retries(q)
        else:
            if query:
                run_query_with_retries(query)
            else:
                agent_print('No query provided; agent finished after enqueueing reports.')
    except KeyboardInterrupt:
        agent_print('\nAgent interrupted by user')

    agent_print('Agent finished')
    # Close logger
    try:
        close_agent_logger()
    except Exception:
        pass


def init_db():
    ensure_reports_dir()
    conn = sqlite3.connect(DB_PATH)
    # Performance-minded pragmas for faster bulk inserts/indexing
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA cache_size = -2000')
    except Exception:
        pass
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            created_at TEXT,
            problem TEXT,
            project_name TEXT,
            raw_json TEXT,
            user_id INTEGER
        )'''
    )
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_filename_user ON reports(filename, user_id)')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            report_id INTEGER,
            stakeholder INTEGER,
            resonate TEXT,
            aspects TEXT,
            questions TEXT,
            missing TEXT,
            FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
        )'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            report_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT,
            source_type TEXT,
            embedding_model TEXT,
            token_count INTEGER,
            tags TEXT,
            checksum TEXT,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
        )'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            name TEXT,
            content TEXT,
            tags TEXT,
            created_at TEXT,
            last_accessed TEXT,
            user_id INTEGER
        )'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            name TEXT,
            created_at TEXT,
            last_accessed TEXT
        )'''
    )
    c.execute(
        '''CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )'''
    )
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id ON conversation_messages(conversation_id, id DESC)')
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_report_chunk ON embeddings(report_id, chunk_index)')
    conn.commit()
    migrate_embeddings_table(conn)
    migrate_reports_table(conn)
    # Add users table if missing
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            salt TEXT,
            created_at TEXT,
            is_admin INTEGER DEFAULT 0
        )'''
    )
    conn.commit()
    # Ensure user_id columns exist on main tables (add if missing)
    try:
        c.execute("PRAGMA table_info(reports)")
        report_cols = [r[1] for r in c.fetchall()]
        if 'project_name' not in report_cols:
            c.execute('ALTER TABLE reports ADD COLUMN project_name TEXT')
        if 'user_id' not in report_cols:
            c.execute('ALTER TABLE reports ADD COLUMN user_id INTEGER')
        c.execute("PRAGMA table_info(embeddings)")
        emb_cols = [r[1] for r in c.fetchall()]
        if 'user_id' not in emb_cols:
            c.execute('ALTER TABLE embeddings ADD COLUMN user_id INTEGER')
        c.execute("PRAGMA table_info(memories)")
        mem_cols = [r[1] for r in c.fetchall()]
        if 'user_id' not in mem_cols:
            c.execute('ALTER TABLE memories ADD COLUMN user_id INTEGER')
        conn.commit()
    except Exception:
        pass
    conn.close()


def migrate_embeddings_table(conn):
    c = conn.cursor()
    c.execute("PRAGMA table_info(embeddings)")
    columns = [row[1] for row in c.fetchall()]
    if 'chunk_text' in columns:
        return
    if not columns:
        return

    c.execute('ALTER TABLE embeddings RENAME TO embeddings_old')
    c.execute(
        '''CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            report_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT,
            source_type TEXT,
            embedding_model TEXT,
            token_count INTEGER,
            tags TEXT,
            checksum TEXT,
            embedding BLOB,
            created_at TEXT,
            user_id INTEGER,
            FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
        )'''
    )
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_report_chunk ON embeddings(report_id, chunk_index)')
    c.execute('SELECT report_id, chunk_index, text, embedding FROM embeddings_old')
    rows = c.fetchall()
    for report_id, chunk_index, text, embedding in rows:
        if isinstance(embedding, str):
            embedding = sqlite3.Binary(embedding.encode('utf-8'))
        checksum = hashlib.sha256((text or '').encode('utf-8')).hexdigest()
        token_count = len(re.findall(r'\S+', text or ''))
        c.execute(
            'INSERT INTO embeddings (report_id, chunk_index, chunk_text, source_type, embedding_model, token_count, tags, checksum, embedding, created_at, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (report_id, chunk_index, text, 'report', 'local_hash', token_count, '', checksum, embedding, datetime.now().isoformat(), None)
        )
    c.execute('DROP TABLE embeddings_old')
    conn.commit()


def migrate_reports_table(conn):
    c = conn.cursor()
    c.execute("PRAGMA table_info(reports)")
    columns = [row[1] for row in c.fetchall()]
    if 'user_id' not in columns:
        c.execute('ALTER TABLE reports ADD COLUMN user_id INTEGER')
        conn.commit()

    c.execute("PRAGMA index_list(reports)")
    indexes = [row for row in c.fetchall() if row[2]]
    filename_unique = False
    composite_unique = False
    for idx in indexes:
        idx_name = idx[1]
        c.execute(f"PRAGMA index_info({idx_name})")
        idx_columns = [col[2] for col in c.fetchall()]
        if idx_columns == ['filename']:
            filename_unique = True
        if idx_columns == ['filename', 'user_id'] or idx_columns == ['user_id', 'filename']:
            composite_unique = True
    if filename_unique and not composite_unique:
        c.execute('ALTER TABLE reports RENAME TO reports_old')
        c.execute(
            '''CREATE TABLE reports (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                created_at TEXT,
                problem TEXT,
                project_name TEXT,
                raw_json TEXT,
                user_id INTEGER
            )'''
        )
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_filename_user ON reports(filename, user_id)')
        c.execute('INSERT INTO reports (id, filename, created_at, problem, project_name, raw_json, user_id) SELECT id, filename, created_at, problem, project_name, raw_json, user_id FROM reports_old')
        c.execute('DROP TABLE reports_old')
        conn.commit()
    elif not composite_unique:
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_filename_user ON reports(filename, user_id)')
        conn.commit()


def hash_password(password, salt_hex=None):
    if salt_hex is None:
        salt = os.urandom(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt_hex, dk.hex()


def create_user(username, password, is_admin=0):
    init_db()
    salt, pwdhash = hash_password(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    try:
        c.execute('INSERT INTO users (username, password_hash, salt, created_at, is_admin) VALUES (?, ?, ?, ?, ?)',
                  (username, pwdhash, salt, now, int(is_admin)))
        conn.commit()
        uid = c.lastrowid
    except Exception:
        uid = None
    conn.close()
    return uid


def verify_user(username, password):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, password_hash, salt FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    uid, stored_hash, salt = row
    _, check_hash = hash_password(password, salt_hex=salt)
    if check_hash == stored_hash:
        return uid
    return None


def ensure_default_user():
    global CURRENT_USER_ID, CURRENT_USERNAME
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username FROM users ORDER BY id LIMIT 1')
    row = c.fetchone()
    if row:
        CURRENT_USER_ID, CURRENT_USERNAME = row[0], row[1]
    else:
        uid = create_user('local', '')
        CURRENT_USER_ID, CURRENT_USERNAME = uid, 'local'
    conn.close()
    return CURRENT_USER_ID


def create_conversation(name='session'):
    global CURRENT_CONVERSATION_ID
    user_id = get_current_user_id()
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('INSERT INTO conversations (user_id, name, created_at, last_accessed) VALUES (?, ?, ?, ?)',
              (user_id, name, now, now))
    conn.commit()
    CURRENT_CONVERSATION_ID = c.lastrowid
    conn.close()
    return CURRENT_CONVERSATION_ID


def get_conversation_history(conversation_id=None, limit=CONVERSATION_HISTORY_LIMIT):
    if conversation_id is None:
        conversation_id = CURRENT_CONVERSATION_ID
    if conversation_id is None:
        return []
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role, content, created_at FROM conversation_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?',
              (conversation_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{'role': row[0], 'content': row[1], 'created_at': row[2]} for row in reversed(rows)]


def save_conversation_message(role, content, conversation_id=None):
    if conversation_id is None:
        conversation_id = CURRENT_CONVERSATION_ID
    if conversation_id is None:
        conversation_id = create_conversation()
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('INSERT INTO conversation_messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
              (conversation_id, role, content, now))
    c.execute('UPDATE conversations SET last_accessed = ? WHERE id = ?', (now, conversation_id))
    conn.commit()
    conn.close()
    return conversation_id


def get_current_user_id():
    global CURRENT_USER_ID
    if CURRENT_USER_ID is None:
        ensure_default_user()
    return CURRENT_USER_ID


def prompt_initial_login(root):
    global CURRENT_USER_ID, CURRENT_USERNAME
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT username FROM users ORDER BY id')
    users = [row[0] for row in c.fetchall()]
    conn.close()

    if not users:
        while True:
            username = simpledialog.askstring('Create First User', 'Enter username:', parent=root)
            if username is None:
                break
            password = simpledialog.askstring('Create First User', 'Enter password:', show='*', parent=root)
            if password is None:
                continue
            if username.strip() and password:
                uid = create_user(username.strip(), password)
                if uid:
                    CURRENT_USER_ID = uid
                    CURRENT_USERNAME = username.strip()
                    break
    else:
        while True:
            username = simpledialog.askstring('Login', 'Username:', parent=root)
            if username is None:
                break
            password = simpledialog.askstring('Login', 'Password:', show='*', parent=root)
            if password is None:
                continue
            uid = verify_user(username.strip(), password)
            if uid:
                CURRENT_USER_ID = uid
                CURRENT_USERNAME = username.strip()
                break
            if messagebox.askyesno('Create user', 'User not found or wrong password. Create new user?', parent=root):
                uid = create_user(username.strip(), password)
                if uid:
                    CURRENT_USER_ID = uid
                    CURRENT_USERNAME = username.strip()
                    break
    if CURRENT_USER_ID is None:
        ensure_default_user()


def logout_user():
    global CURRENT_USER_ID, CURRENT_USERNAME
    CURRENT_USER_ID = None
    CURRENT_USERNAME = None
    ensure_default_user()
    return CURRENT_USER_ID


def list_users():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, created_at, is_admin FROM users ORDER BY id')
    rows = c.fetchall()
    conn.close()
    return rows


def delete_user(username):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE username = ?', (username,))
    changed = c.rowcount
    conn.commit()
    conn.close()
    return changed


def change_password(username):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    uid = row[0]
    conn.close()
    old = simpledialog.askstring('Old Password', 'Enter your current password:', show='*')
    if old is None:
        return False
    if verify_user(username, old) is None:
        messagebox.showerror('Error', 'Current password incorrect')
        return False
    new = simpledialog.askstring('New Password', 'Enter a new password:', show='*')
    if not new:
        return False
    salt, pwdhash = hash_password(new)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET password_hash = ?, salt = ? WHERE id = ?', (pwdhash, salt, uid))
    conn.commit()
    conn.close()
    messagebox.showinfo('Password Changed', 'Password updated successfully')
    return True


def migrate_existing_rows_to_current_user(assign_user_id=None):
    init_db()
    if assign_user_id is None:
        uid = get_current_user_id()
    else:
        uid = assign_user_id
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Assign reports, embeddings, memories with NULL user_id to the chosen user
    c.execute('UPDATE reports SET user_id = ? WHERE user_id IS NULL', (uid,))
    c.execute('UPDATE embeddings SET user_id = ? WHERE user_id IS NULL', (uid,))
    c.execute('UPDATE memories SET user_id = ? WHERE user_id IS NULL', (uid,))
    conn.commit()
    changed = conn.total_changes
    conn.close()
    return changed


def list_db_reports():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT id, created_at, project_name, substr(problem, 1, 80) FROM reports WHERE user_id = ? ORDER BY created_at DESC', (uid,))
    rows = c.fetchall()
    conn.close()
    return rows


def load_db_report(report_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT id, filename, created_at, problem, project_name, raw_json FROM reports WHERE id = ? AND user_id = ?', (report_id, uid))
    row = c.fetchone()
    if row is None:
        conn.close()
        raise FileNotFoundError(f'Report id {report_id} not found')
    report_id, filename, created_at, problem, project_name, raw_json = row
    c.execute('SELECT stakeholder, resonate, aspects, questions, missing FROM feedback WHERE report_id = ? ORDER BY stakeholder', (report_id,))
    feedback_rows = c.fetchall()
    conn.close()
    feedback = []
    for fb_row in feedback_rows:
        feedback.append({
            'stakeholder': fb_row[0],
            'resonate': fb_row[1],
            'aspects': fb_row[2],
            'questions': fb_row[3],
            'missing': fb_row[4]
        })
    report = {
        'id': report_id,
        'filename': filename,
        'created_at': created_at,
        'problem': problem,
        'project_name': project_name,
        'raw_json': json.loads(raw_json) if raw_json else None,
        'feedback': feedback
    }
    return report


def save_report_to_db(filename, data):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    current_user = get_current_user_id()
    c.execute('SELECT id FROM reports WHERE filename = ? AND user_id = ?', (filename, current_user))
    row = c.fetchone()
    if row:
        report_id = row[0]
        c.execute(
            'UPDATE reports SET created_at = ?, problem = ?, project_name = ?, raw_json = ? WHERE id = ?',
            (data.get('created_at'), data.get('problem'), data.get('project_name', ''), json.dumps(data, indent=2), report_id)
        )
    else:
        c.execute(
            'INSERT INTO reports (filename, created_at, problem, project_name, raw_json, user_id) VALUES (?, ?, ?, ?, ?, ?)',
            (filename, data.get('created_at'), data.get('problem'), data.get('project_name', ''), json.dumps(data, indent=2), current_user)
        )
        report_id = c.lastrowid
    c.execute('DELETE FROM feedback WHERE report_id = ?', (report_id,))
    for fb in data['feedback']:
        c.execute(
            'INSERT INTO feedback (report_id, stakeholder, resonate, aspects, questions, missing) VALUES (?, ?, ?, ?, ?, ?)',
            (report_id, fb['stakeholder'], fb['resonate'], fb['aspects'], fb['questions'], fb['missing'])
        )
    # Defer embeddings indexing to a background worker for responsiveness
    report_chunks = build_report_chunks(data, max_chars=500, overlap=100)
    enqueue_indexing_job(report_id, report_chunks, data['created_at'])
    conn.commit()
    conn.close()
    return report_id


def load_json_report(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


## Memory API (persistent multi-turn memory)
def save_memory(name, content, tags=''):
    ensure_reports_dir()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('INSERT INTO memories (name, content, tags, created_at, last_accessed, user_id) VALUES (?, ?, ?, ?, ?, ?)',
              (name, content, tags, now, now, get_current_user_id()))
    conn.commit()
    mem_id = c.lastrowid
    conn.close()
    return mem_id


def load_memory(mem_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT id, name, content, tags, created_at, last_accessed FROM memories WHERE id = ? AND user_id = ?', (mem_id, uid))
    row = c.fetchone()
    if row:
        c.execute('UPDATE memories SET last_accessed = ? WHERE id = ?', (datetime.now().isoformat(), mem_id))
        conn.commit()
    conn.close()
    if not row:
        return None
    return {
        'id': row[0],
        'name': row[1],
        'content': row[2],
        'tags': row[3],
        'created_at': row[4],
        'last_accessed': row[5]
    }


def list_memories():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT id, name, substr(content,1,120), tags, created_at FROM memories WHERE user_id = ? ORDER BY last_accessed DESC, created_at DESC', (uid,))
    rows = c.fetchall()
    conn.close()
    return rows


def delete_memory(mem_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('DELETE FROM memories WHERE id = ? AND user_id = ?', (mem_id, uid))
    conn.commit()
    conn.close()


def get_all_memories():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uid = get_current_user_id()
    c.execute('SELECT id, name, content, tags, created_at FROM memories WHERE user_id = ? ORDER BY last_accessed DESC, created_at DESC', (uid,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            'id': row[0],
            'name': row[1],
            'content': row[2],
            'tags': row[3],
            'created_at': row[4]
        }
        for row in rows
    ]


def find_relevant_memories(query, top_k=3):
    # Use semantic memory retrieval combined with keyword relevance.
    query_emb = parse_embedding(get_embedding(query))
    memories = get_all_memories()
    candidates = []
    for mem in memories:
        content = mem['content'] or ''
        mem_emb = parse_embedding(get_embedding(content))
        semantic_score = cosine_similarity(query_emb, mem_emb) if query_emb is not None and mem_emb is not None else 0.0
        text_score = keyword_score(query, content)
        combined_score = 0.7 * semantic_score + 0.3 * text_score
        candidates.append({
            'id': mem['id'],
            'name': mem['name'],
            'snippet': content[:180],
            'score': combined_score,
            'semantic_score': semantic_score,
            'text_score': text_score,
            'content': content
        })
    ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return ranked[:top_k]


def extract_text_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_pieces = []
    feedback = []

    def walk(item, prefix=''):
        if isinstance(item, dict):
            for key, value in item.items():
                if key in ('problem', 'title', 'summary', 'description', 'notes') and isinstance(value, str):
                    text_pieces.append(value)
                elif key == 'feedback' and isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            fb = {
                                'stakeholder': entry.get('stakeholder', 0),
                                'resonate': entry.get('resonate', ''),
                                'aspects': entry.get('aspects', ''),
                                'questions': entry.get('questions', ''),
                                'missing': entry.get('missing', '')
                            }
                            feedback.append(fb)
                        else:
                            text_pieces.append(str(entry))
                elif isinstance(value, (dict, list)):
                    walk(value, prefix + f"{key}: ")
                elif isinstance(value, str):
                    text_pieces.append(prefix + f"{key}: {value}")
                else:
                    text_pieces.append(prefix + f"{key}: {value}")
        elif isinstance(item, list):
            for value in item:
                walk(value, prefix)
        elif item is not None:
            text_pieces.append(prefix + str(item))

    walk(data)
    if not text_pieces:
        text_pieces.append(json.dumps(data, ensure_ascii=False, indent=2))
    return '\n'.join(text_pieces).strip(), feedback


def import_source_files():
    paths = filedialog.askopenfilenames(
        title='Import source files',
        filetypes=[('Data files', '*.json *.pdf'), ('JSON files', '*.json'), ('PDF files', '*.pdf')]
    )
    if not paths:
        return

    imported = []
    for path in paths:
        try:
            if path.lower().endswith('.json'):
                text, feedback = extract_text_from_json(path)
            elif path.lower().endswith('.pdf'):
                text = extract_text_from_pdf(path)
                feedback = []
            else:
                continue
        except Exception as e:
            messagebox.showerror('Import Error', f'Could not import {os.path.basename(path)}:\n{e}')
            continue

        data = {
            'problem': text or f'Imported content from {os.path.basename(path)}',
            'feedback': feedback,
            'created_at': datetime.now().isoformat()
        }
        report_id = save_report_to_db(path, data)
        imported.append(f'{os.path.basename(path)} (ID {report_id})')

    if imported:
        messagebox.showinfo('Import Complete', 'Imported sources:\n' + '\n'.join(imported))
        try:
            refresh_report_list()
        except NameError:
            pass


def record_mic(problem_text):
    if sr is None:
        messagebox.showerror("Error", "SpeechRecognition not installed")
        return
    def _record():
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                messagebox.showinfo("Recording", "Speak now...")
                audio = r.listen(source, timeout=10)
            text = r.recognize_sphinx(audio)
            problem_text.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    threading.Thread(target=_record).start()


def tts_available():
    if pyttsx3 is not None:
        return True
    for cmd in ('say', 'espeak', 'spd-say'):
        if shutil.which(cmd):
            return True
    return False


def init_tts_engine():
    global _TTS_ENGINE
    if _TTS_ENGINE is not None:
        return _TTS_ENGINE
    if pyttsx3 is None:
        return None
    try:
        _TTS_ENGINE = pyttsx3.init()
        _TTS_ENGINE.setProperty('rate', 170)
        return _TTS_ENGINE
    except Exception:
        _TTS_ENGINE = None
        return None


def speak_text(text):
    if not text:
        return False
    engine = init_tts_engine()
    if engine is not None:
        try:
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            pass
    for cmd in ('say', 'espeak', 'spd-say'):
        path = shutil.which(cmd)
        if path:
            try:
                if cmd == 'say':
                    subprocess.run([path, text], check=True)
                elif cmd == 'espeak':
                    subprocess.run([path, text.replace('\n', ' ')], check=True)
                elif cmd == 'spd-say':
                    subprocess.run([path, text.replace('\n', ' ')], check=True)
                return True
            except Exception:
                pass
    return False


def continuous_voice_conversation(callback=None, top_k=5, save_to_memory=True, speak_answer=True):
    global CONTINUOUS_VOICE_STOP_EVENT
    if sr is None:
        raise RuntimeError('SpeechRecognition not installed')
    if CONTINUOUS_VOICE_STOP_EVENT is None:
        CONTINUOUS_VOICE_STOP_EVENT = threading.Event()
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while not CONTINUOUS_VOICE_STOP_EVENT.is_set():
                try:
                    if callback:
                        callback('Listening for your question...')
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
                    transcript = recognizer.recognize_sphinx(audio).strip()
                    if not transcript:
                        continue
                    if callback:
                        callback(f'You said: {transcript}')
                    if CURRENT_CONVERSATION_ID is None:
                        create_conversation('session')
                    save_conversation_message('user', transcript)
                    results, answer = generate_rag_answer(transcript, top_k=top_k, save_to_memory=save_to_memory, speak_answer=speak_answer)
                    if callback:
                        callback(f'Assistant response:\n{answer}')
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    if callback:
                        callback('Could not understand audio, continuing...')
                    continue
                except Exception as e:
                    if callback:
                        callback(f'Voice conversation error: {e}')
                    break
    except Exception as e:
        if callback:
            callback(f'Voice session failed: {e}')


def create_gui():
    root = tk.Tk()
    root.title("AI Founders Toolkit")
    root.geometry("780x1040")
    
    # Text field heights (adjust these to change all form field heights at once)
    TEXT_HEIGHT_PROBLEM = 3
    TEXT_HEIGHT_QUESTION = 1
    
    # Set window background color
    root.configure(bg="#f8f9fa")

    # Header Frame
    header_frame = tk.Frame(root, bg="#2c3e50", height=100)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    
    title_label = tk.Label(header_frame, text="AI Founders Toolkit", font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
    title_label.pack(pady=15)

    subtitle_label = tk.Label(header_frame, text="Start to collect feedback on my problem.", font=("Arial", 11), bg="#2c3e50", fg="#ecf0f1")
    subtitle_label.pack(pady=(0, 10))

    # Ensure a default user exists and show login controls
    root.withdraw()
    prompt_initial_login(root)
    root.deiconify()
    user_frame = tk.Frame(header_frame, bg="#2c3e50")
    user_frame.pack(side=tk.RIGHT, padx=10)
    user_label = tk.Label(user_frame, text=f"User: {CURRENT_USERNAME}", bg="#2c3e50", fg="white")
    user_label.pack(side=tk.LEFT, padx=(0, 8))

    def login_prompt():
        nonlocal user_label
        uname = simpledialog.askstring('Login', 'Username:')
        if not uname:
            return
        pwd = simpledialog.askstring('Password', 'Password:', show='*')
        if pwd is None:
            return
        uid = verify_user(uname, pwd)
        if uid:
            global CURRENT_USER_ID, CURRENT_USERNAME
            CURRENT_USER_ID = uid
            CURRENT_USERNAME = uname
            user_label.config(text=f"User: {CURRENT_USERNAME}")
            refresh_report_list()
            refresh_memory_list()
            messagebox.showinfo('Login', f'Logged in as {CURRENT_USERNAME}')
            return
        if messagebox.askyesno('Create user', 'User not found or wrong password. Create new user?'):
            uid = create_user(uname, pwd)
            if uid:
                CURRENT_USER_ID = uid
                CURRENT_USERNAME = uname
                user_label.config(text=f"User: {CURRENT_USERNAME}")
                refresh_report_list()
                refresh_memory_list()
                messagebox.showinfo('User Created', f'User {CURRENT_USERNAME} created and logged in')

    login_btn = tk.Button(user_frame, text='Login/Create', command=login_prompt, bg='#111827', fg='white')
    login_btn.pack(side=tk.LEFT)
    logout_btn = tk.Button(user_frame, text='Logout', command=lambda: (logout_user(), user_label.config(text=f"User: {CURRENT_USERNAME}"), refresh_report_list(), refresh_memory_list()), bg='#111827', fg='white')
    logout_btn.pack(side=tk.LEFT, padx=(6,0))

    def change_pwd():
        if not CURRENT_USERNAME:
            messagebox.showwarning('No User', 'No user logged in')
            return
        change_password(CURRENT_USERNAME)

    pwd_btn = tk.Button(user_frame, text='Change Password', command=change_pwd, bg='#111827', fg='white')
    pwd_btn.pack(side=tk.LEFT, padx=(6,0))

    def migrate_prompt():
        if messagebox.askyesno('Migrate Data', 'Assign existing unowned data to the current user?'):
            changed = migrate_existing_rows_to_current_user()
            messagebox.showinfo('Migration', f'Updated {changed} rows to user {CURRENT_USERNAME}')
            try:
                refresh_report_list()
                refresh_memory_list()
            except NameError:
                pass

    migrate_btn = tk.Button(user_frame, text='Migrate Data', command=migrate_prompt, bg='#065f46', fg='white')
    migrate_btn.pack(side=tk.LEFT, padx=(6,0))

    # Main Content Frame
    content_frame = tk.Frame(root, bg="#f8f9fa")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    problem_frame = tk.Frame(content_frame, bg="white", bd=1, relief=tk.RAISED)
    problem_frame.pack(pady=10, fill=tk.X)
    problem_label = tk.Label(problem_frame, text="Describe Your Problem:", font=("Arial", 11, "bold"), bg="white", fg="#2c3e50")
    problem_label.pack(anchor=tk.W, padx=10, pady=(8, 5))
    problem_text = tk.Text(problem_frame, height=TEXT_HEIGHT_PROBLEM, width=60, font=("Arial", 10), wrap=tk.WORD)
    problem_text.pack(padx=10, pady=(0, 10))

    project_frame = tk.Frame(content_frame, bg="#f8f9fa")
    project_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
    project_label = tk.Label(project_frame, text="Project Name:", font=("Arial", 10, "bold"), bg="#f8f9fa", fg="#2c3e50")
    project_label.pack(side=tk.LEFT)
    project_entry = tk.Entry(project_frame, width=30, font=("Arial", 10))
    project_entry.pack(side=tk.LEFT, padx=(10, 0))

    buttons_frame = tk.Frame(content_frame, bg="#f8f9fa")
    buttons_frame.pack(pady=10)
    load_pdf_btn = tk.Button(buttons_frame, text="📄 Load from PDF", command=lambda: load_pdf(problem_text), bg="#3498db", fg="white", padx=10)
    load_pdf_btn.pack(side=tk.LEFT, padx=5)
    load_wav_btn = tk.Button(buttons_frame, text="🔊 Load from WAV", command=lambda: load_wav(problem_text), bg="#3498db", fg="white", padx=10)
    load_wav_btn.pack(side=tk.LEFT, padx=5)
    record_mic_btn = tk.Button(buttons_frame, text="🎤 Record", command=lambda: record_mic(problem_text), bg="#3498db", fg="white", padx=10)
    record_mic_btn.pack(side=tk.LEFT, padx=5)

    voice_status_label = tk.Label(buttons_frame, text="Voice status: idle", font=("Arial", 9), bg="#f8f9fa", fg="#111827")
    voice_status_label.pack(side=tk.LEFT, padx=(10, 0))

    def update_voice_status(text):
        root.after(0, lambda: voice_status_label.config(text=text))

    def append_voice_output(message):
        def _append():
            try:
                rag_output_text.config(state=tk.NORMAL)
                rag_output_text.insert(tk.END, message + '\n')
                rag_output_text.see(tk.END)
                rag_output_text.config(state=tk.DISABLED)
            except Exception:
                pass
        root.after(0, _append)

    def start_continuous_voice():
        global CONTINUOUS_VOICE_STOP_EVENT, CONTINUOUS_VOICE_THREAD
        if sr is None:
            messagebox.showerror('Error', 'SpeechRecognition is not installed.')
            return
        if CONTINUOUS_VOICE_THREAD is not None and CONTINUOUS_VOICE_THREAD.is_alive():
            return
        CONTINUOUS_VOICE_STOP_EVENT = threading.Event()
        voice_btn.config(text='⏹️ Stop Voice')
        update_voice_status('Listening...')
        def _voice_worker():
            continuous_voice_conversation(
                callback=append_voice_output,
                top_k=int(top_k_var.get()),
                save_to_memory=auto_memory_var.get(),
                speak_answer=speech_output_var.get()
            )
            root.after(0, lambda: update_voice_status('Stopped'))
            root.after(0, lambda: voice_btn.config(text='🎙️ Continuous Voice'))
        CONTINUOUS_VOICE_THREAD = threading.Thread(target=_voice_worker, daemon=True)
        CONTINUOUS_VOICE_THREAD.start()

    def stop_continuous_voice():
        global CONTINUOUS_VOICE_STOP_EVENT
        if CONTINUOUS_VOICE_STOP_EVENT is not None:
            CONTINUOUS_VOICE_STOP_EVENT.set()
        update_voice_status('Stopping...')

    def toggle_voice_conversation():
        if CONTINUOUS_VOICE_THREAD is not None and CONTINUOUS_VOICE_THREAD.is_alive():
            stop_continuous_voice()
        else:
            start_continuous_voice()

    voice_btn = tk.Button(buttons_frame, text="🎙️ Continuous Voice", command=toggle_voice_conversation, bg="#8b5cf6", fg="white", padx=10)
    voice_btn.pack(side=tk.LEFT, padx=5)

    speech_output_var = tk.BooleanVar(value=tts_available())
    auto_memory_var = tk.BooleanVar(value=True)
    speech_toggle = tk.Checkbutton(buttons_frame, text='Speech out', variable=speech_output_var, bg='#f8f9fa', fg='#111827')
    speech_toggle.pack(side=tk.LEFT, padx=5)
    memory_toggle = tk.Checkbutton(buttons_frame, text='Auto-memory', variable=auto_memory_var, bg='#f8f9fa', fg='#111827')
    memory_toggle.pack(side=tk.LEFT, padx=5)

    def import_files_async():
        paths = filedialog.askopenfilenames(
            title='Import source files',
            filetypes=[('Data files', '*.json *.pdf'), ('JSON files', '*.json'), ('PDF files', '*.pdf')]
        )
        if not paths:
            return

        progress_win = tk.Toplevel(root)
        progress_win.title('Importing...')
        progress_win.geometry('320x100')
        progress_label = tk.Label(progress_win, text='Importing sources...', wraplength=300)
        progress_label.pack(padx=10, pady=10)

        def _update_progress(message):
            root.after(0, lambda: progress_label.config(text=message))

        def _worker(paths):
            imported = []
            for index, path in enumerate(paths, start=1):
                _update_progress(f'Importing {index}/{len(paths)}: {os.path.basename(path)}')
                try:
                    if path.lower().endswith('.json'):
                        text, feedback = extract_text_from_json(path)
                    elif path.lower().endswith('.pdf'):
                        text = extract_text_from_pdf(path)
                        feedback = []
                    else:
                        continue
                except Exception as e:
                    root.after(0, lambda p=path, e=e: messagebox.showerror('Import Error', f'Could not import {os.path.basename(p)}:\n{e}'))
                    continue

                data = {
                    'problem': text or f'Imported content from {os.path.basename(path)}',
                    'feedback': feedback,
                    'created_at': datetime.now().isoformat()
                }
                report_id = save_report_to_db(path, data)
                imported.append(f'{os.path.basename(path)} (ID {report_id})')

            def _finish():
                try:
                    progress_win.destroy()
                except Exception:
                    pass
                if imported:
                    messagebox.showinfo('Import Complete', 'Imported sources:\n' + '\n'.join(imported))
                    try:
                        refresh_report_list()
                    except NameError:
                        pass

            root.after(0, _finish)

        threading.Thread(target=_worker, args=(paths,), daemon=True).start()

    import_btn = tk.Button(buttons_frame, text="📥 Import sources", command=import_files_async, bg="#10b981", fg="white", padx=10)
    import_btn.pack(side=tk.LEFT, padx=5)

    index_progress_frame = tk.Frame(content_frame, bg="#f8fafc")
    index_progress_frame.pack(fill=tk.BOTH, pady=(10, 0))
    top_index_frame = tk.Frame(index_progress_frame, bg="#f8fafc")
    top_index_frame.pack(fill=tk.X, padx=10)
    index_progress_label = tk.Label(top_index_frame, text="Indexing progress:", font=("Arial", 11, "bold"), bg="#f8fafc", fg="#2c3e50")
    index_progress_label.pack(side=tk.LEFT)
    index_queue_label = tk.Label(top_index_frame, text="Queue: 0", font=("Arial", 10), bg="#f8fafc", fg="#374151")
    index_queue_label.pack(side=tk.RIGHT)
    index_progress_text = scrolledtext.ScrolledText(index_progress_frame, height=4, width=70, font=("Courier", 9), bg="#ffffff", fg="#111827")
    index_progress_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
    index_progress_text.config(state=tk.DISABLED)

    def update_index_progress(message, queue_size=None, clear=False):
        def _append():
            if queue_size is not None:
                index_queue_label.config(text=f"Queue: {queue_size}")
            index_progress_text.config(state=tk.NORMAL)
            if clear:
                index_progress_text.delete(1.0, tk.END)
            index_progress_text.insert(tk.END, message + '\n')
            index_progress_text.see(tk.END)
            index_progress_text.config(state=tk.DISABLED)
        root.after(0, _append)

    # Stakeholders Section
    stakeholders_label = tk.Label(content_frame, text="Stakeholder Feedback", font=("Arial", 12, "bold"), bg="#f8f9fa", fg="#2c3e50")
    stakeholders_label.pack(anchor=tk.W, pady=(15, 10))
    
    # Create scrollable frame for stakeholders
    canvas = tk.Canvas(content_frame, bg="#f8f9fa", highlightthickness=0)
    scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Stakeholders
    stakeholders = []
    for i in range(1, 5):
        frame = tk.Frame(scrollable_frame, bg="white", bd=1, relief=tk.RAISED)
        frame.pack(pady=8, padx=0, fill=tk.X)
        label = tk.Label(frame, text=f"Stakeholder {i}", font=("Arial", 12, "bold"), bg="white", fg="#2c3e50")
        label.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        q1_label = tk.Label(frame, text="Does this problem resonate with them?", font=("Arial", 9, "italic"), bg="white", fg="#555")
        q1_label.pack(anchor=tk.W, padx=15, pady=(0, 2))
        q1_text = tk.Text(frame, height=TEXT_HEIGHT_QUESTION, width=60, font=("Arial", 9), wrap=tk.WORD)
        q1_text.pack(padx=15, pady=(0, 8), fill=tk.X)
        
        q2_label = tk.Label(frame, text="What aspects do they think matter most?", font=("Arial", 9, "italic"), bg="white", fg="#555")
        q2_label.pack(anchor=tk.W, padx=15, pady=(0, 2))
        q2_text = tk.Text(frame, height=TEXT_HEIGHT_QUESTION, width=60, font=("Arial", 9), wrap=tk.WORD)
        q2_text.pack(padx=15, pady=(0, 8), fill=tk.X)
        
        q3_label = tk.Label(frame, text="What questions or concerns come to mind?", font=("Arial", 9, "italic"), bg="white", fg="#555")
        q3_label.pack(anchor=tk.W, padx=15, pady=(0, 2))
        q3_text = tk.Text(frame, height=TEXT_HEIGHT_QUESTION, width=60, font=("Arial", 9), wrap=tk.WORD)
        q3_text.pack(padx=15, pady=(0, 8), fill=tk.X)
        
        q4_label = tk.Label(frame, text="What are you missing about this problem?", font=("Arial", 9, "italic"), bg="white", fg="#555")
        q4_label.pack(anchor=tk.W, padx=15, pady=(0, 2))
        q4_text = tk.Text(frame, height=TEXT_HEIGHT_QUESTION, width=60, font=("Arial", 9), wrap=tk.WORD)
        q4_text.pack(padx=15, pady=(0, 10), fill=tk.X)
        
        stakeholders.append({
            'resonate': q1_text,
            'aspects': q2_text,
            'questions': q3_text,
            'missing': q4_text
        })
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def save(problem_text, stakeholders):
        problem = problem_text.get(1.0, tk.END).strip()
        
        # Validation
        if not problem:
            messagebox.showwarning("Missing Information", "Please enter a problem statement.")
            return
        
        feedback = []
        for i, st in enumerate(stakeholders, 1):
            fb = {
                'stakeholder': i,
                'resonate': st['resonate'].get(1.0, tk.END).strip(),
                'aspects': st['aspects'].get(1.0, tk.END).strip(),
                'questions': st['questions'].get(1.0, tk.END).strip(),
                'missing': st['missing'].get(1.0, tk.END).strip()
            }
            feedback.append(fb)
        
        # Save to JSON export and SQLite storage
        data = {
            'problem': problem,
            'project_name': project_entry.get().strip(),
            'feedback': feedback,
            'created_at': datetime.now().isoformat()
        }
        ensure_reports_dir()
        filename = os.path.join(REPORTS_DIR, f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        report_id = save_report_to_db(filename, data)
        
        # Display in output - formatted nicely
        output = "="*70 + "\n"
        output += "AI FOUNDERS INTELLIA - FEEDBACK REPORT\n"
        output += "="*70 + "\n\n"
        output += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if project_entry.get().strip():
            output += f"Project: {project_entry.get().strip()}\n"
        output += f"File: {filename}\n"
        output += "-"*70 + "\n\n"
        output += f"PROBLEM STATEMENT:\n{problem}\n\n"
        output += "-"*70 + "\n\n"
        output += "STAKEHOLDER FEEDBACK:\n"
        for fb in feedback:
            has_feedback = any([fb['resonate'], fb['aspects'], fb['questions'], fb['missing']])
            if has_feedback:
                output += f"\n▸ Stakeholder {fb['stakeholder']}:\n"
                if fb['resonate']:
                    output += f"  • Resonates: {fb['resonate']}\n"
                if fb['aspects']:
                    output += f"  • Key Aspects: {fb['aspects']}\n"
                if fb['questions']:
                    output += f"  • Questions/Concerns: {fb['questions']}\n"
                if fb['missing']:
                    output += f"  • Missing Elements: {fb['missing']}\n"
        
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, output)
        output_text.config(state=tk.DISABLED)
        messagebox.showinfo(
            "✓ Saved Successfully",
            f"Your feedback has been saved to:\n\n{filename}\n\nSQLite report id: {report_id}"
        )
        refresh_report_list()

    # Save Button
    save_btn = tk.Button(content_frame, text="💾 Save Problem and Feedback", command=lambda: save(problem_text, stakeholders), 
                         bg="#27ae60", fg="white", font=("Arial", 11, "bold"), padx=20, pady=10)
    save_btn.pack(pady=15)

    # Output Section
    output_label = tk.Label(content_frame, text="Feedback Report:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    output_label.pack(anchor=tk.W, pady=(15, 5))
    output_frame = tk.Frame(content_frame, bg="white", bd=1, relief=tk.SUNKEN)
    output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    output_text = scrolledtext.ScrolledText(output_frame, height=12, width=70, font=("Courier", 9), bg="white", fg="#2c3e50")
    output_text.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)
    output_text.config(state=tk.DISABLED)

    def perform_search():
        query = search_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("Question", "Please enter a question first.")
            return
        top_k = int(top_k_var.get())
        results, summary = summarize_search_results(query, top_k=top_k)

        search_output_text.config(state=tk.NORMAL)
        search_output_text.delete(1.0, tk.END)
        if not results:
            search_output_text.insert(tk.END, "No matching reports found.")
        else:
            for idx, result in enumerate(results, 1):
                loaded_chunks = ', '.join(str(ctx['chunk_index']) for ctx in result.get('chunk_context', []))
                search_output_text.insert(tk.END, f"{idx}. Report ID {result['report_id']} | {result['created_at']} | Combined {result['score']:.3f} | Vector {result['vector_score']:.3f} | Keyword {result['keyword_score']:.3f}\n")
                search_output_text.insert(tk.END, f"   File: {result['filename']}\n")
                search_output_text.insert(tk.END, f"   Best match: chunk {result['chunk_index']} (loaded {loaded_chunks})\n")
                search_output_text.insert(tk.END, f"   Text: {result['text'][:200]}\n\n")
        search_output_text.config(state=tk.DISABLED)

        insight_output_text.config(state=tk.NORMAL)
        insight_output_text.delete(1.0, tk.END)
        insight_output_text.insert(tk.END, summary)
        insight_output_text.config(state=tk.DISABLED)
        update_metrics_display()

    def refresh_conversation_history():
        if CURRENT_CONVERSATION_ID is None:
            create_conversation('session')
        history = get_conversation_history(CURRENT_CONVERSATION_ID, limit=CONVERSATION_HISTORY_LIMIT)
        conv_text.config(state=tk.NORMAL)
        conv_text.delete(1.0, tk.END)
        if not history:
            conv_text.insert(tk.END, "No conversation history yet. Ask a question to start a session.")
        else:
            for item in history:
                role = 'You' if item['role'] == 'user' else 'Assistant'
                conv_text.insert(tk.END, f"{item['created_at']} - {role}: {item['content']}\n\n")
        conv_text.config(state=tk.DISABLED)

    def create_new_conversation_session():
        create_conversation('session')
        refresh_conversation_history()
        messagebox.showinfo('New Session', 'Started a new conversation session.')

    def perform_rag_answer():
        query = search_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("Question", "Please enter a question first.")
            return
        top_k = int(top_k_var.get())

        rag_output_text.config(state=tk.NORMAL)
        rag_output_text.delete(1.0, tk.END)
        rag_output_text.insert(tk.END, "Generating answer...\n")
        rag_output_text.config(state=tk.DISABLED)

        def append_text(text):
            def _append():
                rag_output_text.config(state=tk.NORMAL)
                rag_output_text.insert(tk.END, text)
                rag_output_text.see(tk.END)
                rag_output_text.config(state=tk.DISABLED)
            root.after(0, _append)

        def _worker():
            results, answer = generate_rag_answer(
                query,
                top_k=top_k,
                stream_callback=append_text,
                save_to_memory=auto_memory_var.get(),
                speak_answer=speech_output_var.get()
            )
            root.after(0, refresh_conversation_history)
            root.after(0, update_metrics_display)

        threading.Thread(target=_worker, daemon=True).start()

    search_frame = tk.Frame(content_frame, bg="#f8f9fa", bd=1, relief=tk.SOLID)
    search_frame.pack(fill=tk.X, pady=(15, 10))

    metrics_frame = tk.Frame(content_frame, bg="#f8f9fa", bd=1, relief=tk.SOLID)
    metrics_frame.pack(fill=tk.X, pady=(0, 10))
    metrics_label = tk.Label(metrics_frame, text="Retrieval Metrics:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    metrics_label.pack(anchor=tk.W, padx=10, pady=(8, 4))
    metrics_text = scrolledtext.ScrolledText(metrics_frame, height=8, width=70, font=("Courier", 9), bg="#ffffff", fg="#111827")
    metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    metrics_text.config(state=tk.DISABLED)

    def update_metrics_display():
        stats = [
            f"Queries: {METRICS.get('query_count', 0)}",
            f"Top-K requested: {METRICS.get('top_k_requested', 0)}",
            f"Top-K returned: {METRICS.get('top_k_returned', 0)}",
            f"Search candidates: {METRICS.get('search_candidates', 0)}",
            f"DB embeddings loaded: {METRICS.get('db_embeddings_loaded', 0)}",
            f"Chunk count: {METRICS.get('chunk_count', 0)}",
            f"Chunk words (min/max/avg): {METRICS.get('chunk_min_words', 0)}/{METRICS.get('chunk_max_words', 0)}/{METRICS.get('chunk_avg_words', 0):.1f}",
            f"Embedding calls: {METRICS.get('embedding_calls', 0)}",
            f"Embedding cache hits: {METRICS.get('embedding_cache_hits', 0)}",
            f"Embedding cache misses: {METRICS.get('embedding_cache_misses', 0)}",
            f"Cache hit rate: {METRICS.get('cache_hit_rate', 0.0):.2%}",
            f"Last query latency: {METRICS.get('last_query_latency_ms', 0.0):.1f} ms",
            f"Avg query latency: {METRICS.get('avg_query_latency_ms', 0.0):.1f} ms",
            f"Last response latency: {METRICS.get('last_response_latency_ms', 0.0):.1f} ms",
            f"Avg response latency: {METRICS.get('avg_response_latency_ms', 0.0):.1f} ms",
        ]
        metrics_text.config(state=tk.NORMAL)
        metrics_text.delete(1.0, tk.END)
        metrics_text.insert(tk.END, "\n".join(stats))
        metrics_text.config(state=tk.DISABLED)

    conversation_frame = tk.Frame(content_frame, bg="#f8f9fa", bd=1, relief=tk.SOLID)
    conversation_frame.pack(fill=tk.BOTH, pady=(0, 10))
    conversation_label = tk.Label(conversation_frame, text="Conversation History:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    conversation_label.pack(anchor=tk.W, pady=(8, 4), padx=8)
    conv_text = scrolledtext.ScrolledText(conversation_frame, height=8, width=70, font=("Courier", 9), bg="#ffffff", fg="#111827")
    conv_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
    conv_text.config(state=tk.DISABLED)
    conv_btn_frame = tk.Frame(conversation_frame, bg="#f8f9fa")
    conv_btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
    refresh_conv_btn = tk.Button(conv_btn_frame, text="🔄 Refresh History", command=refresh_conversation_history, bg="#4b5563", fg="white", font=("Arial", 10), padx=10, pady=6)
    refresh_conv_btn.pack(side=tk.LEFT, padx=(0, 6))
    new_session_btn = tk.Button(conv_btn_frame, text="🆕 New Session", command=create_new_conversation_session, bg="#10b981", fg="white", font=("Arial", 10), padx=10, pady=6)
    new_session_btn.pack(side=tk.LEFT)

    search_label = tk.Label(search_frame, text="Ask a question:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    search_label.pack(anchor=tk.W, padx=10, pady=(10, 4))
    search_text = tk.Text(search_frame, height=3, width=60, font=("Arial", 10), wrap=tk.WORD)
    search_text.pack(padx=10, pady=(0, 10), fill=tk.X)

    control_frame = tk.Frame(search_frame, bg="#f8f9fa")
    control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
    top_k_label = tk.Label(control_frame, text="Top sources:", font=("Arial", 10), bg="#f8f9fa", fg="#2c3e50")
    top_k_label.pack(side=tk.LEFT, padx=(0, 5))
    top_k_var = tk.IntVar(value=5)
    top_k_spinbox = tk.Spinbox(control_frame, from_=1, to=10, textvariable=top_k_var, width=3, font=("Arial", 10))
    top_k_spinbox.pack(side=tk.LEFT, padx=(0, 15))

    search_btn = tk.Button(control_frame, text="🔎 Search & Summarize", command=perform_search, bg="#2563eb", fg="white", font=("Arial", 10), padx=12, pady=8)
    search_btn.pack(side=tk.LEFT, padx=5)
    rag_btn = tk.Button(control_frame, text="🧠 RAG Answer", command=perform_rag_answer, bg="#f97316", fg="white", font=("Arial", 10), padx=12, pady=8)
    rag_btn.pack(side=tk.LEFT, padx=5)

    search_output_text = scrolledtext.ScrolledText(search_frame, height=8, width=70, font=("Courier", 9), bg="#f8fafc", fg="#111827")
    search_output_text.pack(padx=10, pady=(0, 12), fill=tk.BOTH, expand=True)
    search_output_text.config(state=tk.DISABLED)

    summary_label = tk.Label(search_frame, text="AI Insight Response:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    summary_label.pack(anchor=tk.W, padx=10, pady=(0, 4))
    insight_output_text = scrolledtext.ScrolledText(search_frame, height=8, width=70, font=("Courier", 9), bg="#f8fafc", fg="#111827")
    insight_output_text.pack(padx=10, pady=(0, 12), fill=tk.BOTH, expand=True)
    insight_output_text.config(state=tk.DISABLED)

    rag_label = tk.Label(search_frame, text="RAG Answer (with source citations):", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    rag_label.pack(anchor=tk.W, padx=10, pady=(0, 4))
    rag_output_text = scrolledtext.ScrolledText(search_frame, height=10, width=70, font=("Courier", 9), bg="#f8fafc", fg="#111827")
    rag_output_text.pack(padx=10, pady=(0, 12), fill=tk.BOTH, expand=True)
    rag_output_text.config(state=tk.DISABLED)

    def refresh_report_list():
        report_listbox.delete(0, tk.END)
        for report_id, created_at, project_name, snippet in list_db_reports():
            project_segment = f"[{project_name}] " if project_name else ''
            label = f"{report_id} | {created_at} | {project_segment}{snippet}"
            report_listbox.insert(tk.END, label)

    def load_selected_report():
        selection = report_listbox.curselection()
        if not selection:
            messagebox.showwarning("Select Report", "Please select a saved report to load.")
            return
        text = report_listbox.get(selection[0])
        report_id = int(text.split('|', 1)[0].strip())
        try:
            data = load_db_report(report_id)
            title = f"Loaded saved report: {data['filename']} (ID {data['id']})"
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output = f"{title}\n\n"
            output += f"Problem:\n{data.get('problem', '')}\n\n"
            output += "Stakeholder Feedback:\n"
            for fb in data.get('feedback', []):
                output += f"\nStakeholder {fb.get('stakeholder')}:\n"
                output += f"  • Does this problem resonate with them? {fb.get('resonate', '')}\n"
                output += f"  • What aspects do they think matter most? {fb.get('aspects', '')}\n"
                output += f"  • What questions or concerns come to mind? {fb.get('questions', '')}\n"
                output += f"  • What are you missing about this problem? {fb.get('missing', '')}\n"
            output_text.insert(tk.END, output)
            output_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    saved_frame = tk.Frame(content_frame, bg="#f8f9fa")
    saved_frame.pack(fill=tk.BOTH, padx=0, pady=(0, 10))
    saved_label = tk.Label(saved_frame, text="Saved Reports:", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    saved_label.pack(anchor=tk.W, pady=(10, 4))
    report_listbox = tk.Listbox(saved_frame, height=6, font=("Arial", 10), activestyle='none')
    report_listbox.pack(fill=tk.BOTH, padx=0, pady=(0, 6), expand=True)
    load_report_btn = tk.Button(saved_frame, text="Load Selected Report", command=load_selected_report,
                                bg="#3498db", fg="white", font=("Arial", 10), padx=10, pady=6)
    load_report_btn.pack(anchor=tk.E)

    refresh_report_list()

    # Memory Panel
    memory_frame = tk.Frame(content_frame, bg="#f8f9fa", bd=1, relief=tk.SOLID)
    memory_frame.pack(fill=tk.BOTH, padx=0, pady=(6, 10))
    memory_label = tk.Label(memory_frame, text="Memories (multi-turn):", font=("Arial", 11, "bold"), bg="#f8f9fa", fg="#2c3e50")
    memory_label.pack(anchor=tk.W, pady=(8, 4), padx=8)

    mem_listbox = tk.Listbox(memory_frame, height=5, font=("Arial", 10), activestyle='none')
    mem_listbox.pack(fill=tk.BOTH, padx=8, pady=(0, 6), expand=True)

    def refresh_memory_list():
        mem_listbox.delete(0, tk.END)
        for mem in list_memories():
            mid, name, snippet, tags, created = mem
            label = f"{mid} | {name} | {tags} | {created}"
            mem_listbox.insert(tk.END, label)

    def save_problem_as_memory():
        name = simpledialog.askstring("Memory name", "Enter a short name for this memory:")
        if not name:
            return
        content = problem_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "Problem text is empty; nothing to save.")
            return
        save_memory(name, content)
        refresh_memory_list()
        messagebox.showinfo("Saved", f"Memory '{name}' saved.")

    def load_selected_memory_to_output():
        sel = mem_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Please select a memory first.")
            return
        text = mem_listbox.get(sel[0])
        mem_id = int(text.split('|', 1)[0].strip())
        mem = load_memory(mem_id)
        if not mem:
            messagebox.showerror("Error", "Memory not found.")
            return
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Memory: {mem['name']}\n\n{mem['content']}")
        output_text.config(state=tk.DISABLED)

    def insert_memory_into_problem():
        sel = mem_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Please select a memory first.")
            return
        text = mem_listbox.get(sel[0])
        mem_id = int(text.split('|', 1)[0].strip())
        mem = load_memory(mem_id)
        if not mem:
            messagebox.showerror("Error", "Memory not found.")
            return
        problem_text.insert(tk.END, "\n" + mem['content'])

    def delete_selected_memory():
        sel = mem_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Please select a memory first.")
            return
        text = mem_listbox.get(sel[0])
        mem_id = int(text.split('|', 1)[0].strip())
        if messagebox.askyesno("Delete", "Delete selected memory?"):
            delete_memory(mem_id)
            refresh_memory_list()

    mem_btn_frame = tk.Frame(memory_frame, bg="#f8f9fa")
    mem_btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
    save_mem_btn = tk.Button(mem_btn_frame, text="💾 Save Problem as Memory", command=save_problem_as_memory, bg="#6b7280", fg="white")
    save_mem_btn.pack(side=tk.LEFT, padx=(0, 6))
    load_mem_btn = tk.Button(mem_btn_frame, text="📂 Load Selected", command=load_selected_memory_to_output, bg="#3498db", fg="white")
    load_mem_btn.pack(side=tk.LEFT, padx=(0, 6))
    insert_mem_btn = tk.Button(mem_btn_frame, text="📥 Insert into Problem", command=insert_memory_into_problem, bg="#10b981", fg="white")
    insert_mem_btn.pack(side=tk.LEFT, padx=(0, 6))
    del_mem_btn = tk.Button(mem_btn_frame, text="🗑️ Delete", command=delete_selected_memory, bg="#ef4444", fg="white")
    del_mem_btn.pack(side=tk.LEFT)

    refresh_memory_list()
    refresh_conversation_history()
    start_indexing_worker(update_callback=update_index_progress)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Founders Toolkit')
    parser.add_argument('--run-gui', action='store_true', help='Launch the Tkinter GUI')
    parser.add_argument('--workspace-root', metavar='PATH', help='Set the workspace root for reports and database storage')
    parser.add_argument('--agent', action='store_true', help='Run autonomous agent CLI (reindex, RAG, memory ops)')
    parser.add_argument('--agent-query', metavar='QUERY', help='Question for agent to answer during run')
    parser.add_argument('--agent-loop', action='store_true', help='Run agent in interactive REPL loop for multiple queries')
    parser.add_argument('--agent-retries', type=int, default=2, help='Number of retries for transient failures in agent operations')
    parser.add_argument('--agent-retry-delay', type=int, default=2, help='Seconds to wait between agent retries')
    parser.add_argument('--agent-log', metavar='PATH', help='Path to agent log file (default: reports/agent.log)')
    parser.add_argument('--agent-log-max-bytes', type=int, default=5*1024*1024, help='Max bytes before rotating agent log')
    parser.add_argument('--agent-log-backups', type=int, default=5, help='Number of rotated log files to keep')
    parser.add_argument('--list-reports', action='store_true', help='List saved reports for the current user')
    parser.add_argument('--list-memories', action='store_true', help='List saved memories for the current user')
    parser.add_argument('--show-memory', metavar='MEMORY_ID', help='Show memory content for a memory entry')
    parser.add_argument('--show-report', metavar='REPORT_ID', help='Show details for a saved report')
    parser.add_argument('--ask', metavar='QUERY', help='Ask a question and get a RAG answer')
    parser.add_argument('--list-users', action='store_true', help='List application users')
    parser.add_argument('--create-user', metavar='USERNAME', help='Create a new user')
    parser.add_argument('--delete-user', metavar='USERNAME', help='Delete an existing user')
    parser.add_argument('--change-password', metavar='USERNAME', help='Change password for an existing user')
    parser.add_argument('--migrate-null-data', action='store_true', help='Assign any unowned data rows to the current user')
    parser.add_argument('--chunk-with-neighbors', nargs=2, metavar=('REPORT_ID','CHUNK_NUMBER'), help='Retrieve a specific chunk plus its neighboring chunks using 1-based chunk numbering')
    args = parser.parse_args()

    if args.workspace_root:
        set_workspace_root(args.workspace_root)

    if args.list_reports:
        for report_id, created_at, project_name, snippet in list_db_reports():
            project_segment = f"[{project_name}] " if project_name else ''
            print(f"{report_id}: {created_at} {project_segment}{snippet}")
    elif args.list_memories:
        for mem in list_memories():
            mid, name, snippet, tags, created = mem
            print(f"{mid}: {name} ({tags}) {created}\n  {snippet}")
    elif args.show_memory:
        mem = load_memory(int(args.show_memory))
        if not mem:
            print('Memory not found.')
        else:
            print(f"Memory {mem['id']}: {mem['name']}\nTags: {mem['tags']}\nCreated: {mem['created_at']}\n\n{mem['content']}")
    elif args.show_report:
        report = load_db_report(int(args.show_report))
        print(f"Report {report['id']}: {report['filename']}\nCreated: {report['created_at']}\nProject: {report['project_name']}\n\nProblem:\n{report['problem']}\n\nFeedback:\n")
        for fb in report.get('feedback', []):
            print(f"Stakeholder {fb['stakeholder']}:\n  Resonate: {fb['resonate']}\n  Aspects: {fb['aspects']}\n  Questions: {fb['questions']}\n  Missing: {fb['missing']}\n")
    elif args.ask:
        results, answer = generate_rag_answer(args.ask, top_k=5)
        print(answer)
    elif args.agent:
        run_agent(args.agent_query, loop=args.agent_loop, retries=args.agent_retries, retry_delay=args.agent_retry_delay)
    elif args.list_users:
        for uid, username, created, is_admin in list_users():
            print(f'{uid}: {username} (created: {created}, admin: {bool(is_admin)})')
    elif args.create_user:
        password = getpass(f'Password for {args.create_user}: ')
        uid = create_user(args.create_user, password)
        if uid:
            print(f'Created user {args.create_user} (id={uid})')
        else:
            print('Failed to create user; the username may already exist.')
    elif args.delete_user:
        confirm = input(f'Delete user {args.delete_user}? [y/N]: ').strip().lower()
        if confirm == 'y':
            deleted = delete_user(args.delete_user)
            print(f'Deleted {deleted} user(s)')
    elif args.change_password:
        password = getpass(f'Current password for {args.change_password}: ')
        if verify_user(args.change_password, password) is None:
            print('Invalid username or password')
        else:
            new_password = getpass('New password: ')
            salt, pwdhash = hash_password(new_password)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('UPDATE users SET password_hash = ?, salt = ? WHERE username = ?', (pwdhash, salt, args.change_password))
            conn.commit()
            conn.close()
            print('Password updated')
    elif args.migrate_null_data:
        uid = get_current_user_id()
        changed = migrate_existing_rows_to_current_user(uid)
        print(f'Migrated {changed} rows to user id {uid}')
    elif args.chunk_with_neighbors:
        report_id = int(args.chunk_with_neighbors[0])
        chunk_number = int(args.chunk_with_neighbors[1])
        chunks = load_chunk_number_with_neighbors(report_id, chunk_number, window=1)
        if not chunks:
            print(f'No chunks found for report {report_id} around chunk {chunk_number}')
        else:
            for chunk in chunks:
                print(f"Chunk {chunk['chunk_index']}:")
                print(chunk['chunk_text'])
                print('-' * 80)
    else:
        create_gui()
