#!/usr/bin/env python3
"""Same interactive chat as task_breakdown_chat.py ('load <id>', 'correct <note>',
'analyse dependency between <id> and <id>'), but answered by the local LoRA
fine-tune (training/adapters/) instead of Ollama.

The fine-tune only exists in MLX format right now (not converted to GGUF/Ollama),
so this talks to it directly via the mlx_lm Python API rather than the Ollama
HTTP API task_breakdown_chat.py normally uses. It reuses that script's exact
prompt-building/command logic unchanged, swapping out only the model call.

Setup (one-time):
  export AZURE_DEVOPS_PAT="..."   # PAT with Work Items (Read) scope

Usage:
  python3 task_breakdown_chat_finetuned.py
  python3 task_breakdown_chat_finetuned.py 97061   # load a work item immediately
"""

import sys

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

import rq_agent.task_breakdown_chat as tbc

MODEL_REPO = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
ADAPTER_PATH = "training/adapters"
# Matches ado_common.call_ollama's production temperature/repeat_penalty/
# repeat_last_n, so behavior here is comparable to the Ollama-served model --
# without this, generation can fall into the same repetition loop we saw and
# fixed for Ollama earlier (see ado_common.call_ollama).
SAMPLER = make_sampler(temp=0.3)
LOGITS_PROCESSORS = make_logits_processors(repetition_penalty=1.3, repetition_context_size=64)

_model = None
_tokenizer = None


def call_finetuned(messages: list, model: str = None, show_progress: bool = True) -> str:
    prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    chunks = []
    for response in stream_generate(
        _model, _tokenizer, prompt=prompt, max_tokens=800,
        sampler=SAMPLER, logits_processors=LOGITS_PROCESSORS,
    ):
        if show_progress:
            print(response.text, end="", flush=True)
        chunks.append(response.text)
    if show_progress:
        print()
    return "".join(chunks).strip()


def main():
    global _model, _tokenizer
    print(f"Loading fine-tuned model ({MODEL_REPO} + {ADAPTER_PATH})...")
    _model, _tokenizer = load(MODEL_REPO, adapter_path=ADAPTER_PATH)

    # task_breakdown_chat.load_item/analyse_dependency call the module-level
    # `call_ollama` name in its own module -- swap it for our mlx-backed one.
    tbc.call_ollama = call_finetuned

    tbc.main()


if __name__ == "__main__":
    main()
