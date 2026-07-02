from typing import Optional
import os

_FALLBACK = os.environ.get("FALLBACK_MODE", "1")

try:
    if _FALLBACK == "0":
        raise ImportError("Fallback disabled; attempting real model import")
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

    class LocalGenerator:
        def __init__(self, model_name: Optional[str] = None):
            model_name = model_name or os.environ.get("MODEL_NAME", "distilgpt2")
            model_path = os.environ.get("MODEL_PATH")

            source = model_path or model_name
            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModelForCausalLM.from_pretrained(source)

            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        def generate(self, prompt: str, max_length: int = 128) -> str:
            out = self._pipe(prompt, max_length=max_length, do_sample=True, num_return_sequences=1)
            return out[0]["generated_text"]

except Exception:
    # Fallback lightweight generator (no heavy deps)
    class LocalGenerator:
        def __init__(self, model_name: Optional[str] = None):
            pass

        def generate(self, prompt: str, max_length: int = 128) -> str:
            # Very small deterministic fallback: echo with a note
            prefix = "[fallback reply] "
            out = prefix + prompt
            if len(out) > max_length:
                return out[:max_length]
            return out


_GEN: Optional[LocalGenerator] = None


def get_generator() -> LocalGenerator:
    global _GEN
    if _GEN is None:
        _GEN = LocalGenerator()
    return _GEN


def generate_text(prompt: str, max_length: int = 128) -> str:
    return get_generator().generate(prompt, max_length=max_length)
