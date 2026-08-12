# Portions adapted from LatentMAS (https://github.com/Gen-Verse/LatentMAS).
# Licensed under Apache-2.0 and modified by the StateBridge authors.
# See THIRD_PARTY_NOTICES.md.

"""
Model wrapper for HuggingFace Transformers backend.

Provides unified interface for model loading, chat rendering,
tokenization, and text generation.
"""

import os
import glob
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


# Local model search paths (checked before HuggingFace download)
LOCAL_MODEL_DIRS = [
    "./models",
    "../models",
    os.path.expanduser("~/.cache/huggingface/hub"),
]


def _resolve_model_path(model_name: str) -> str:
    """Resolve model path, preferring local models over HuggingFace downloads.

    Args:
        model_name: Model name or path, e.g. "Qwen/Qwen3-4B" or "./models/Qwen3-4B"

    Returns:
        Resolved model path (local absolute path or HuggingFace model name).
    """
    # If it's an absolute/relative path that exists, use directly
    if os.path.isdir(model_name):
        print(f"[ModelWrapper] Using local model: {os.path.abspath(model_name)}")
        return model_name

    # Extract base name (Qwen/Qwen3-4B -> Qwen3-4B)
    base_name = model_name.split("/")[-1] if "/" in model_name else model_name

    # Search local directories
    for search_dir in LOCAL_MODEL_DIRS:
        local_path = os.path.join(search_dir, base_name)
        if os.path.isdir(local_path):
            has_config = os.path.exists(os.path.join(local_path, "config.json"))
            has_model = (
                os.path.exists(os.path.join(local_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(local_path, "model.safetensors")) or
                len(glob.glob(os.path.join(local_path, "model-*.safetensors"))) > 0
            )
            if has_config and has_model:
                print(f"[ModelWrapper] Found local model: {os.path.abspath(local_path)}")
                return local_path

    # No local model found; use HuggingFace
    print(f"[ModelWrapper] Using HuggingFace model: {model_name}")
    return model_name


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"


class ModelWrapper:
    """Wrapper around HuggingFace Transformers model for generation tasks."""

    def __init__(self, model_name: str, device: torch.device, args=None):
        resolved_path = _resolve_model_path(model_name)

        self.model_name = resolved_path
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, use_fast=True)
        _ensure_pad_token(self.tokenizer)

        # Load model
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

    def render_chat(
        self,
        messages: List[Dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        """Render chat messages into model input format.

        Args:
            messages: Chat message list.
            add_generation_prompt: Whether to add generation prompt.
            enable_thinking: Whether to enable thinking mode (Qwen3-specific).
                - True: Model generates <think>...</think> reasoning process.
                - False: Model outputs answer directly, skipping thinking.
        """
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            if "enable_thinking" in tpl:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking
                )
            else:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
