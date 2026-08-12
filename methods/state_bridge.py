"""
StateBridge: Latent-State Communication for Multi-Agent Systems

Features:
- Multi-GPU parallel inference
- Procrustes alignment for cross-agent hidden state transfer
- Real-time logging and resume support
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse
import math
import re
import json
import os
import sys
from queue import Empty
from tqdm import tqdm
from datetime import datetime
import time

from . import default_agents
from models import ModelWrapper
from prompts import (
    build_agent_message_embedding_mas,
    EMBEDDING_HINT_MARKER,
)

from utils import extract_gsm8k_answer, extract_gsm8k_answer_1, auto_device, normalize_answer, extract_markdown_python_block, run_with_timeout, set_seed
from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
    load_winogrande,
)


class TeeLogger:
    """Logger that writes to both file and stdout with immediate sync to disk."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.file = open(log_file, 'a', encoding='utf-8')
        
    def log(self, msg: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
    
    def close(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def parse_log_for_completed(log_file: str) -> Dict[int, Dict]:
    """Parse log file to extract completed samples for resume support."""
    completed = {}
    if not os.path.exists(log_file):
        return completed
    
    pattern = re.compile(
        r'\[\s*(\d+)/\d+\]\s+(?:GPU\d+\s+)?(OK|WRONG)\s+Pred:\s*(\S+)\s*\|\s*Gold:\s*(\S+)'
    )
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                idx = int(match.group(1)) - 1
                correct = match.group(2) == 'OK'
                pred = match.group(3)
                gold = match.group(4)
                if pred == "N/A":
                    pred = None
                completed[idx] = {
                    "idx": idx,
                    "prediction": pred,
                    "gold": gold,
                    "correct": correct,
                }
    
    return completed


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    text = re.sub(r'^</think>\s*', '', text)
    return text.strip()


class HiddenStateCapture:
    """Capture last-layer hidden states via forward hook.
    
    Reduces memory from GB to MB compared to output_hidden_states=True.
    Only saves the last position of the last layer.
    """
    
    def __init__(self, device='cpu'):
        self.hidden_states = []
        self.device = device
        self._step = 0
    
    def hook(self, module, input, output):
        """Forward hook: called on each forward pass."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        if hidden.dim() == 3:
            last_hidden = hidden[:, -1:, :].detach().clone()
        elif hidden.dim() == 2:
            last_hidden = hidden.unsqueeze(1).detach().clone()
        else:
            return
        
        self.hidden_states.append(last_hidden)
        self._step += 1
    
    def get_all(self):
        """Get all captured hidden states, shape: [batch, num_steps, hidden_dim]."""
        if self.hidden_states:
            return torch.cat(self.hidden_states, dim=1)
        return None
    
    def clear(self):
        """Clear captured data."""
        self.hidden_states = []
        self._step = 0


class StateBridge:
    """StateBridge: Procrustes-aligned latent-state communication between agents.
    
    Core pipeline: Planner -> Critic -> Refiner -> Judger
    Each agent's hidden states are aligned to the embedding space via
    whitened Orthogonal Procrustes before being injected as prefix
    embeddings into the next agent.
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_prefix_tokens: int = 64,
        enable_thinking: bool = True,
        prefix_strategy: str = "scale",
        noise_std: float = 0,
        pool_size: int = 4,
        alignment_mode: str = "adaptive",
        adaptive_reg: float = 1e-3,
        snap_ratio: float = 0.3,
        debug_mode: bool = False,
        use_hook: bool = True,
        collect_viz: bool = False,
        args=None,
    ) -> None:
        self.model = model
        self.args = args
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_prefix_tokens = max_prefix_tokens
        self.enable_thinking = enable_thinking
        
        self.prefix_strategy = prefix_strategy
        self.prefix_scale = 1.0
        self.noise_std = noise_std
        self.pool_size = pool_size
        
        self.alignment_mode = alignment_mode
        self.adaptive_reg = adaptive_reg
        self.snap_ratio = snap_ratio
        
        self.debug_mode = debug_mode
        self.use_hook = use_hook
        self.agents = default_agents()
        self.prompt_style = getattr(args, "prompt", "sequential") if args else "sequential"
        self.task = getattr(args, "task", "medqa") if args else "medqa"
        self.device = model.device
        
        self.collect_viz = collect_viz
        self.viz_data = {"planner": [], "critic": [], "refiner": []}
        
        self.embedding_layer = model.model.get_input_embeddings()
        self.hidden_size = self.embedding_layer.embedding_dim
        self.dtype = self.embedding_layer.weight.dtype
        
        self.vocab_embeds = self.embedding_layer.weight.detach().float()
        self.target_norm = self.vocab_embeds.norm(dim=-1).mean().item()

    def _align_hidden_sequence(self, hidden_seq: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Whiten + Orthogonal Procrustes alignment.
        
        Maps hidden states to the embedding space via:
        1. Center both distributions
        2. Whiten with regularized covariance
        3. Orthogonal Procrustes rotation
        4. Reconstruct in embedding space
        5. Norm calibration + vocabulary snapping
        """
        batch, seq_len, hidden = hidden_seq.shape
        
        target_device = self.device
        if hidden_seq.device != target_device:
            hidden_seq = hidden_seq.to(target_device)
        
        hidden_float = hidden_seq.float()

        if token_ids is None or token_ids.numel() == 0:
            return hidden_float.to(self.dtype)

        token_ids = token_ids.to(target_device)
        token_embeds = self.embedding_layer(token_ids).float()

        T = min(hidden_float.shape[1], token_embeds.shape[1])
        hidden_float = hidden_float[:, :T, :]
        token_embeds = token_embeds[:, :T, :]

        H = hidden_float.reshape(-1, hidden)
        E = token_embeds.reshape(-1, hidden)
        N = H.shape[0]
        if N < 2:
            return hidden_float.to(self.dtype)

        reg = float(getattr(self, "adaptive_reg", 1e-3))

        def _sym_power(A: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
            """Symmetric matrix power via eigendecomposition."""
            A = (A + A.T) * 0.5
            evals, evecs = torch.linalg.eigh(A)
            evals = evals.clamp_min(eps).pow(power)
            return (evecs * evals.unsqueeze(0)) @ evecs.T

        # Step 1: Center
        mu_H = H.mean(dim=0, keepdim=True)
        mu_E = E.mean(dim=0, keepdim=True)
        Hc = H - mu_H
        Ec = E - mu_E

        # Step 2: Regularized covariances
        I = torch.eye(hidden, device=H.device, dtype=H.dtype)
        Cov_H = (Hc.T @ Hc) / N + reg * I
        Cov_E = (Ec.T @ Ec) / N + reg * I

        # Step 3: Whiten
        H_w = Hc @ _sym_power(Cov_H, -0.5)
        E_w = Ec @ _sym_power(Cov_E, -0.5)

        # Step 4: Orthogonal Procrustes on whitened spaces
        M = H_w.T @ E_w
        U, _, Vt = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vt

        # Enforce proper rotation (det = +1) to avoid reflection
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Step 5: Reconstruct in embedding space
        sqrt_Cov_E = _sym_power(Cov_E, 0.5)
        aligned_flat = (Hc @ _sym_power(Cov_H, -0.5)) @ R @ sqrt_Cov_E + mu_E
        aligned = aligned_flat.reshape(batch, T, hidden)

        # Norm calibration
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        aligned = aligned * (self.target_norm / aligned_norm)

        # Vocabulary snapping
        if self.snap_ratio > 0:
            flat = aligned.reshape(-1, hidden)
            sims = torch.matmul(
                F.normalize(flat, dim=-1),
                F.normalize(self.vocab_embeds, dim=-1).T,
            )
            top_ids = sims.argmax(dim=-1)
            snapped = self.vocab_embeds[top_ids].reshape(batch, T, hidden)
            aligned = (1 - self.snap_ratio) * aligned + self.snap_ratio * snapped

        return aligned.to(self.dtype)

    def _process_prefix(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        """Apply prefix processing strategy (scaling, noise, pooling)."""
        batch, seq_len, hidden = prefix_embeds.shape
        strategy = self.prefix_strategy
        
        if strategy == "none":
            return prefix_embeds
        
        elif strategy == "scale":
            return prefix_embeds * self.prefix_scale
        
        elif strategy == "noise":
            noise = torch.randn_like(prefix_embeds) * self.noise_std * self.target_norm
            return prefix_embeds + noise
            
        elif strategy == "scale_noise":
            scaled = prefix_embeds * self.prefix_scale
            noise = torch.randn_like(scaled) * self.noise_std * self.target_norm * self.prefix_scale
            return scaled + noise
        
        elif strategy == "weighted_avg":
            positions = torch.arange(seq_len, device=self.device, dtype=torch.float32)
            weights = F.softmax(positions / math.sqrt(seq_len), dim=0)
            weights = weights.view(1, seq_len, 1).to(prefix_embeds.dtype)
            pooled = (prefix_embeds * weights).sum(dim=1, keepdim=True)
            return pooled
        
        elif strategy == "mean_pool":
            if seq_len <= self.pool_size:
                return prefix_embeds
            chunk_size = seq_len // self.pool_size
            remainder = seq_len % self.pool_size
            pooled_chunks = []
            start = 0
            for i in range(self.pool_size):
                end = start + chunk_size + (1 if i < remainder else 0)
                chunk = prefix_embeds[:, start:end, :]
                pooled_chunks.append(chunk.mean(dim=1, keepdim=True))
                start = end
            return torch.cat(pooled_chunks, dim=1)
            
        return prefix_embeds

    def _generate_with_prefix(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        prefix_embeds: Optional[torch.Tensor] = None,
        insert_position: Optional[int] = None,
        need_hidden_states: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Generate text and extract hidden states with full context.
        
        Args:
            prompt_embeds: Current prompt embeddings (B, L, H)
            prompt_mask: Attention mask for prompt
            prefix_embeds: Optional prefix embeddings from previous agent (B, Lp, H)
            insert_position: If specified, insert prefix at this token position.
                             If None, prepend prefix at the beginning.
            need_hidden_states: If False, skip hidden states output to save memory.
        """
        
        if prefix_embeds is not None and prefix_embeds.numel() > 0:
            batch_size = prompt_embeds.shape[0]
            prefix_len = prefix_embeds.shape[1]
            
            if insert_position is not None and insert_position > 0:
                left_embeds = prompt_embeds[:, :insert_position, :]
                right_embeds = prompt_embeds[:, insert_position:, :]
                full_embeds = torch.cat([left_embeds, prefix_embeds, right_embeds], dim=1)
                
                left_mask = prompt_mask[:, :insert_position]
                right_mask = prompt_mask[:, insert_position:]
                prefix_mask = torch.ones(
                    (batch_size, prefix_len),
                    dtype=prompt_mask.dtype,
                    device=self.device
                )
                full_mask = torch.cat([left_mask, prefix_mask, right_mask], dim=1)
            else:
                full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
                prefix_mask = torch.ones(
                    (batch_size, prefix_len),
                    dtype=prompt_mask.dtype,
                    device=self.device
                )
                full_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        else:
            full_embeds = prompt_embeds
            full_mask = prompt_mask

        input_len = full_embeds.shape[1]
        
        # Hidden state capture via forward hook
        capture = None
        hook_handle = None
        use_hook_mode = self.use_hook and need_hidden_states
        
        if use_hook_mode:
            capture = HiddenStateCapture(device=self.device)
            model_inner = self.model.model
            if hasattr(model_inner, 'model') and hasattr(model_inner.model, 'layers'):
                last_layer = model_inner.model.layers[-1]
            elif hasattr(model_inner, 'layers'):
                last_layer = model_inner.layers[-1]
            elif hasattr(model_inner, 'transformer') and hasattr(model_inner.transformer, 'h'):
                last_layer = model_inner.transformer.h[-1]
            else:
                raise ValueError(f"Cannot find layers in model architecture: {type(model_inner)}")
            hook_handle = last_layer.register_forward_hook(capture.hook)
        
        try:
            with torch.no_grad():
                gen_outputs = self.model.model.generate(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    max_new_tokens=input_len + self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=need_hidden_states and not self.use_hook,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        
        seqs = gen_outputs.sequences
        total_len = seqs.shape[1]
        actual_gen = total_len - full_embeds.shape[1] if total_len > full_embeds.shape[1] else total_len
        
        if total_len < input_len:
            gen_len = total_len
            gen_token_ids = seqs
        else:
            gen_len = total_len - input_len
            gen_token_ids = seqs[:, input_len:] if gen_len > 0 else seqs[:, :0]
        
        texts = []
        for idx, seq in enumerate(seqs):
            if total_len < input_len:
                new_tokens = seq
            else:
                new_tokens = seq[input_len:] if gen_len > 0 else seq
            text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text = strip_thinking(text)
            texts.append(text)
        
        # Get hidden states
        if need_hidden_states and gen_len > 0:
            if use_hook_mode:
                gen_hidden_seq = capture.get_all()
                if gen_hidden_seq is not None:
                    h_len = gen_hidden_seq.shape[1]
                    if h_len > gen_len:
                        gen_hidden_seq = gen_hidden_seq[:, -gen_len:, :]
                        h_len = gen_len
                    t_len = gen_token_ids.shape[1]
                    if h_len > t_len:
                        gen_hidden_seq = gen_hidden_seq[:, -t_len:, :]
                    elif h_len < t_len:
                        gen_token_ids = gen_token_ids[:, :h_len]
                else:
                    gen_hidden_seq = torch.zeros(
                        (full_embeds.shape[0], 1, self.hidden_size),
                        dtype=self.dtype, device=self.device
                    )
            else:
                if hasattr(gen_outputs, 'hidden_states') and gen_outputs.hidden_states:
                    all_hidden = []
                    for i, step_hidden in enumerate(gen_outputs.hidden_states):
                        last_layer = step_hidden[-1]
                        all_hidden.append(last_layer[:, -1:, :])
                    if all_hidden:
                        gen_hidden_seq = torch.cat(all_hidden, dim=1)
                        h_len = gen_hidden_seq.shape[1]
                        t_len = gen_token_ids.shape[1]
                        if h_len > t_len:
                            gen_hidden_seq = gen_hidden_seq[:, -t_len:, :]
                        elif h_len < t_len:
                            gen_token_ids = gen_token_ids[:, :h_len]
                    else:
                        gen_hidden_seq = torch.zeros(
                            (full_embeds.shape[0], 1, self.hidden_size),
                            dtype=self.dtype, device=self.device
                        )
                else:
                    gen_hidden_seq = torch.zeros(
                        (full_embeds.shape[0], 1, self.hidden_size),
                        dtype=self.dtype, device=self.device
                    )
        else:
            gen_hidden_seq = torch.zeros(
                (full_embeds.shape[0], 1, self.hidden_size),
                dtype=self.dtype, device=self.device
            )
            gen_token_ids = torch.zeros(
                (full_embeds.shape[0], 0), dtype=torch.long, device=self.device
            )
        
        last_token = seqs[0, -1].item() if seqs.shape[1] > 0 else None
        eos_token_id = self.model.tokenizer.eos_token_id
        gen_info = {
            "actual_gen": actual_gen,
            "max_tokens": self.max_new_tokens,
            "hit_eos": last_token == eos_token_id if last_token is not None else False,
        }
        
        return texts, gen_hidden_seq, gen_token_ids, gen_info

    @torch.no_grad()
    def run_item(self, item: Dict) -> Dict:
        """Run the full 4-agent pipeline on a single item."""
        current_prefix = None
        final_text = ""
        agent_traces = []
        
        for agent in self.agents:
            has_prefix = current_prefix is not None and current_prefix.numel() > 0
            
            msg = build_agent_message_embedding_mas(
                role=agent.role, 
                question=item["question"], 
                context="", 
                method="embedding_mas", 
                args=self.args,
                has_prefix=has_prefix,
            )
            
            prompts, _, _, _ = self.model.prepare_chat_batch(
                [msg], add_generation_prompt=True
            )
            
            if self.enable_thinking:
                wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
            else:
                wrapped_prompts = prompts
            
            # Compute embedding insertion position based on marker
            insert_position = None
            if has_prefix:
                prompt_text = wrapped_prompts[0]
                marker_idx = prompt_text.find(EMBEDDING_HINT_MARKER)
                if marker_idx != -1:
                    left_text = prompt_text[:marker_idx]
                    insert_position = len(self.model.tokenizer(left_text, add_special_tokens=False)['input_ids'])
                    wrapped_prompts = [prompt_text.replace(EMBEDDING_HINT_MARKER + "\n\n", "").replace(EMBEDDING_HINT_MARKER, "")]
            
            wrapped_encoded = self.model.tokenizer(
                wrapped_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = wrapped_encoded["input_ids"].to(self.device)
            attention_mask = wrapped_encoded["attention_mask"].to(self.device)
            
            received_prefix_tok = current_prefix.shape[1] if has_prefix else 0
            
            prompt_embeds = self.embedding_layer(input_ids)
            
            need_hidden = (agent.role != "judger")
            gen_start = time.time()
            generated_texts, gen_hidden_seq, gen_token_ids, gen_info = self._generate_with_prefix(
                prompt_embeds, attention_mask, current_prefix, insert_position, need_hidden
            )
            gen_time = time.time() - gen_start
            
            if self.debug_mode and agent.role == "judger":
                clean_output = strip_thinking(generated_texts[0])
                print(f"\n{'='*30} Judger Output {'='*30}")
                print(f"{clean_output}")
                print(f"{'='*70}\n")

            align_time = 0.0
            if agent.role != "judger":
                align_start = time.time()
                # Find </think> token position; pass only post-</think> embeddings
                think_end_token_ids = self.model.tokenizer.encode("</think>", add_special_tokens=False)
                
                think_end_pos = -1
                if len(think_end_token_ids) > 0:
                    search_len = len(think_end_token_ids)
                    token_list = gen_token_ids[0].tolist()
                    for i in range(len(token_list) - search_len + 1):
                        if token_list[i:i + search_len] == think_end_token_ids:
                            think_end_pos = i + search_len
                            break
                
                if think_end_pos > 0 and think_end_pos < gen_hidden_seq.shape[1]:
                    filtered_hidden = gen_hidden_seq[:, think_end_pos:, :]
                    filtered_token_ids = gen_token_ids[:, think_end_pos:]
                    
                    if filtered_hidden.shape[1] > self.max_prefix_tokens:
                        filtered_hidden = filtered_hidden[:, -self.max_prefix_tokens:, :]
                        filtered_token_ids = filtered_token_ids[:, -self.max_prefix_tokens:]
                else:
                    if gen_hidden_seq.shape[1] > self.max_prefix_tokens:
                        filtered_hidden = gen_hidden_seq[:, -self.max_prefix_tokens:, :]
                        filtered_token_ids = gen_token_ids[:, -self.max_prefix_tokens:]
                    else:
                        filtered_hidden = gen_hidden_seq
                        filtered_token_ids = gen_token_ids
                
                if filtered_hidden.shape[1] > 0:
                    aligned_embeds = self._align_hidden_sequence(filtered_hidden, filtered_token_ids)
                    if self.collect_viz:
                        _et = self.embedding_layer(filtered_token_ids).float().reshape(-1, self.hidden_size).detach().cpu()
                        _ht = filtered_hidden.float().reshape(-1, self.hidden_size).detach().cpu()
                        _et1 = aligned_embeds.float().reshape(-1, self.hidden_size).detach().cpu()
                        self.viz_data[agent.role].append({"e_t": _et, "h_t": _ht, "e_t1": _et1})
                    current_prefix = self._process_prefix(aligned_embeds)
                align_time = time.time() - align_start
            
            if agent.role == "judger":
                final_text = generated_texts[0]
            
            prompt_tok = int(attention_mask[0].sum())
            agent_traces.append({
                "role": agent.role,
                "output": generated_texts[0],
                "gen_info": gen_info,
                "prompt_tokens": prompt_tok,
                "prefix_tokens": received_prefix_tok,
                "generated_tokens": gen_info.get("actual_gen", 0),
                "inference_time": round(gen_time, 4),
                "alignment_time": round(align_time, 4),
            })
            
            del gen_hidden_seq, gen_token_ids, generated_texts
            if 'aligned_embeds' in dir():
                del aligned_embeds
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Per-agent generation summary
        role_abbrev = {"planner": "P", "critic": "C", "refiner": "R", "judger": "J"}
        parts = []
        for trace in agent_traces:
            info = trace.get("gen_info", {})
            abbr = role_abbrev.get(trace["role"], trace["role"][0].upper())
            actual = info.get("actual_gen", "?")
            eos_mark = "EOS" if info.get("hit_eos", False) else ""
            parts.append(f"{abbr}:{actual}{eos_mark}")
        print(f"[GEN] {' '.join(parts)}")

        pred = self._extract_answer(final_text, item)
        pred_1 = self._extract_answer_1(final_text, item)
        gold = self._get_gold_answer(item)
        
        # Task-specific correctness evaluation
        if self.task in ['mbppplus', 'humanevalplus']:
            if pred is None:
                correct = False
            else:
                python_code_to_exe = pred + "\n" + gold
                correct, _ = run_with_timeout(python_code_to_exe, timeout=10)
            correct_1 = correct
        elif self.task in ["aime2024", "aime2025"]:
            try:
                pred_int = int(pred) if pred else None
                gold_int = int(gold) if gold else None
                correct = (pred_int == gold_int) if (pred_int is not None and gold_int is not None) else False
            except ValueError:
                correct = False
            try:
                pred_1_int = int(pred_1) if pred_1 else None
                correct_1 = (pred_1_int == gold_int) if (pred_1_int is not None and gold_int is not None) else False
            except ValueError:
                correct_1 = False
        else:
            correct = (pred == gold) if (pred and gold) else False
            correct_1 = (pred_1 == gold) if (pred_1 and gold) else False
        
        total_prompt = sum(t.get("prompt_tokens", 0) for t in agent_traces)
        total_gen = sum(t.get("generated_tokens", 0) for t in agent_traces)
        total_prefix = sum(t.get("prefix_tokens", 0) for t in agent_traces)
        total_inference_time = sum(t.get("inference_time", 0) for t in agent_traces)
        total_alignment_time = sum(t.get("alignment_time", 0) for t in agent_traces)

        return {
            "question": item["question"],
            "gold": gold,
            "solution": item.get("solution", ""),
            "prediction": pred,
            "prediction_1": pred_1,
            "correct": correct,
            "correct_1": correct_1,
            "final_response": final_text,
            "idx": item.get("idx", -1),
            "trace": agent_traces,
            "efficiency": {
                "prompt_tokens": total_prompt,
                "generated_tokens": total_gen,
                "prefix_tokens": total_prefix,
                "total_tokens": total_prompt + total_gen + total_prefix,
                "inference_time": round(total_inference_time, 4),
                "alignment_time": round(total_alignment_time, 4),
                "per_agent": [{"role": t["role"], "prompt_tokens": t.get("prompt_tokens", 0), "generated_tokens": t.get("generated_tokens", 0), "prefix_tokens": t.get("prefix_tokens", 0), "inference_time": t.get("inference_time", 0), "alignment_time": t.get("alignment_time", 0)} for t in agent_traces],
            },
        }

    def _extract_answer(self, text: str, item: Dict) -> Optional[str]:
        """Extract answer using task-appropriate logic."""
        if self.task in ['mbppplus', 'humanevalplus']:
            return extract_markdown_python_block(text)
        elif self.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            return normalize_answer(extract_gsm8k_answer(text))
        elif self.task in ["winogrande"]:
            return normalize_answer(extract_gsm8k_answer(text))
        else:
            return normalize_answer(extract_gsm8k_answer(text))

    def _extract_answer_1(self, text: str, item: Dict) -> Optional[str]:
        """Extract answer using the simplified method (only looks for numbers)."""
        if self.task in ['mbppplus', 'humanevalplus']:
            return extract_markdown_python_block(text)
        else:
            return normalize_answer(extract_gsm8k_answer_1(text))

    def _get_gold_answer(self, item: Dict) -> str:
        """Get gold answer from unified data format."""
        return item.get("gold", "")


# ============================================================================
# Multi-GPU Parallel Extension
# ============================================================================


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: Dict,
):
    """Worker process for a single GPU."""
    if "seed" in config and config["seed"] is not None:
        set_seed(config["seed"] + gpu_id)
    
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    args = argparse.Namespace(
        model=config["model_name"],
        model_name=config["model_name"],
        task=config["task"],
        prompt=config["prompt"],
        batch_size=1,
    )
    
    model = ModelWrapper(config["model_name"], device=device)
    
    evaluator = StateBridge(
        model,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        max_prefix_tokens=config["max_prefix_tokens"],
        prefix_strategy=config.get("prefix_strategy", "scale"),
        enable_thinking=config["enable_thinking"],
        adaptive_reg=config.get("adaptive_reg", 1e-3),
        snap_ratio=config.get("snap_ratio", 0.3),
        debug_mode=config.get("debug_mode", False),
        use_hook=config.get("use_hook", True),
        collect_viz=config.get("collect_viz", False),
        args=args,
    )
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            
            idx, item = task
            start_t = time.time()
            result = evaluator.run_item(item)
            end_t = time.time()
            result["duration"] = end_t - start_t
            result["global_idx"] = idx
            result_queue.put((idx, result, gpu_id))
            
            torch.cuda.empty_cache()
            
        except Empty:
            continue
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            result_queue.put((idx, {"error": str(e), "traceback": error_msg, "global_idx": idx, "idx": idx}, gpu_id))
            torch.cuda.empty_cache()
    
    if config.get("collect_viz", False):
        viz_data = evaluator.viz_data
        if any(viz_data[k] for k in viz_data):
            viz_base = config.get("viz_output", "results/alignment_viz.pt")
            gpu_path = viz_base.replace(".pt", f"_gpu{gpu_id}.pt")
            torch.save({"viz_data": viz_data}, gpu_path)


class ParallelEvaluator:
    """Multi-GPU parallel evaluation coordinator."""
    
    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        config: Dict,
        log_file: str,
    ):
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.config = config
        self.log_file = log_file
        self.logger = TeeLogger(log_file)
        
    def run_evaluation(self, data: List[Dict], output_file: str = None) -> Dict:
        """Run parallel evaluation across multiple GPUs."""
        mp.set_start_method('spawn', force=True)
        
        completed = parse_log_for_completed(self.log_file)
        completed_indices = set(completed.keys())
        pending_indices = [i for i in range(len(data)) if i not in completed_indices]
        
        self.logger.log("=" * 60)
        self.logger.log("Multi-GPU Parallel Evaluation")
        self.logger.log(f"Total samples: {len(data)}")
        self.logger.log(f"Already completed (from log): {len(completed)}")
        self.logger.log(f"Pending: {len(pending_indices)}")
        self.logger.log(f"GPUs: {self.gpu_ids}")
        self.logger.log(f"Prefix Strategy: scale (scale=1.0)")
        self.logger.log("=" * 60)
        
        if not pending_indices:
            self.logger.log("All samples already completed!")
            return self._finalize_results(completed, data, output_file)
        
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        workers = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, task_queue, result_queue, self.config)
            )
            p.start()
            workers.append(p)
        
        for i in pending_indices:
            item = data[i].copy()
            item["idx"] = i
            task_queue.put((i, item))
        
        for _ in workers:
            task_queue.put(None)
        
        results_dict = dict(completed)
        received = 0
        total_pending = len(pending_indices)
        
        gpu_stats = {gpu_id: {"processed": 0, "correct": 0} for gpu_id in self.gpu_ids}
        start_time = time.time()
        
        while received < total_pending:
            try:
                idx, result, gpu_id = result_queue.get(timeout=300)
                received += 1
                
                if "error" in result:
                    self.logger.log(f"[ERROR] Sample {idx} on GPU {gpu_id}: {result['error']}")
                    result["correct"] = False
                    result["prediction"] = None
                    result["gold"] = data[idx].get("gold", data[idx].get("answer", ""))
                
                results_dict[result["idx"]] = result
                gpu_stats[gpu_id]["processed"] += 1
                if result.get("correct", False):
                    gpu_stats[gpu_id]["correct"] += 1
                
                total_done = len(results_dict)
                total_correct = sum(1 for r in results_dict.values() if r.get("correct", False))
                acc = total_correct / total_done * 100 if total_done > 0 else 0
                
                elapsed = time.time() - start_time
                samples_per_sec = received / elapsed if elapsed > 0 else 0
                remaining = total_pending - received
                eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else 0
                
                elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
                if eta_seconds >= 3600:
                    eta_str = f"{int(eta_seconds // 3600)}h{int((eta_seconds % 3600) // 60)}m"
                elif eta_seconds >= 60:
                    eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60)}s"
                else:
                    eta_str = f"{int(eta_seconds)}s"
                
                status = "OK" if result.get("correct", False) else "WRONG"
                is_code_task = self.config.get('task', '') in ['mbppplus', 'humanevalplus']
                pred_str = "[Code]" if is_code_task else (result.get('prediction') if result.get('prediction') else "N/A")
                gold_str = "[Code]" if is_code_task else result.get('gold', 'N/A')
                duration = result.get('duration', 0.0)
                
                self.logger.log(
                    f"[{total_done:3d}/{len(data)}] GPU{gpu_id} {status} "
                    f"Pred: {pred_str} | Gold: {gold_str} | "
                    f"Acc: {total_correct}/{total_done} = {acc:.1f}% | "
                    f"Dur: {duration:.2f}s | "
                    f"Time: {elapsed_str} | ETA: {eta_str} ({samples_per_sec:.2f} it/s)"
                )
                    
            except Empty:
                self.logger.log("Timeout waiting for results, checking workers...")
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    self.logger.log("All workers finished")
                    break
        
        total_elapsed = time.time() - start_time
        total_elapsed_str = f"{int(total_elapsed // 60)}m{int(total_elapsed % 60)}s"
        avg_speed = received / total_elapsed if total_elapsed > 0 else 0
        
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()
        
        # Merge per-GPU viz data if collect_viz is enabled
        if self.config.get("collect_viz", False):
            merged_viz = {"planner": [], "critic": [], "refiner": []}
            viz_base = self.config.get("viz_output", "results/alignment_viz.pt")
            for gid in self.gpu_ids:
                gpu_path = viz_base.replace(".pt", f"_gpu{gid}.pt")
                if os.path.exists(gpu_path):
                    gpu_data = torch.load(gpu_path, map_location="cpu", weights_only=False)
                    for key in merged_viz:
                        merged_viz[key].extend(gpu_data.get("viz_data", {}).get(key, []))
                    os.remove(gpu_path)
            final_viz = {}
            for key in merged_viz:
                if merged_viz[key]:
                    final_viz[key] = {
                        "e_t": torch.cat([d["e_t"] for d in merged_viz[key]], dim=0),
                        "h_t": torch.cat([d["h_t"] for d in merged_viz[key]], dim=0),
                        "e_t1": torch.cat([d["e_t1"] for d in merged_viz[key]], dim=0),
                    }
            if final_viz:
                torch.save({
                    "model_name": self.config["model_name"],
                    "task": self.config["task"],
                    "transitions": final_viz,
                }, viz_base)
                self.logger.log(f"Alignment viz data saved to {viz_base}")
        
        self.logger.log("")
        self.logger.log("GPU Statistics:")
        for gpu_id in self.gpu_ids:
            stats = gpu_stats[gpu_id]
            gpu_acc = stats["correct"] / stats["processed"] * 100 if stats["processed"] > 0 else 0
            self.logger.log(f"  GPU {gpu_id}: {stats['processed']} samples, {stats['correct']} correct ({gpu_acc:.1f}%)")
        
        self.logger.log("")
        self.logger.log(f"Total time: {total_elapsed_str} ({avg_speed:.2f} samples/sec)")
        
        return self._finalize_results(results_dict, data, output_file)
    
    def _finalize_results(self, results_dict: Dict, data: List[Dict], output_file: str) -> Dict:
        """Finalize and save results."""
        results_sorted = [results_dict[i] for i in sorted(results_dict.keys())]
        
        correct_count = sum(1 for r in results_sorted if r.get("correct", False))
        correct_count_1 = sum(1 for r in results_sorted if r.get("correct_1", False))
        accuracy = correct_count / len(results_sorted) if results_sorted else 0
        accuracy_1 = correct_count_1 / len(results_sorted) if results_sorted else 0
        
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log("Evaluation Complete")
        self.logger.log(f"Total: {len(results_sorted)}")
        self.logger.log(f"Correct (method): {correct_count}")
        self.logger.log(f"Accuracy (method): {accuracy * 100:.2f}%")
        self.logger.log(f"Correct (method_1): {correct_count_1}")
        self.logger.log(f"Accuracy (method_1): {accuracy_1 * 100:.2f}%")
        self.logger.log("=" * 60)
        
        summary = {
            "total": len(results_sorted),
            "correct": correct_count,
            "accuracy": accuracy,
            "correct_1": correct_count_1,
            "accuracy_1": accuracy_1,
            "config": self.config,
            "results": results_sorted,
            "timestamp": datetime.now().isoformat(),
        }
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.log(f"Results saved to {output_file}")
        
        self.logger.close()
        return summary


# All supported tasks
ALL_TASKS = ["gsm8k", "aime2024", "aime2025", "gpqa", 
             "arc_challenge", "mbppplus", "humanevalplus", "medqa"]

# Task-specific hyperparameters
TASK_CONFIG = {
    "gsm8k": {"max_new_tokens": 2048},
    "aime2024": {"max_new_tokens": 20000},
    "aime2025": {"max_new_tokens": 20000},
    "gpqa": {"max_new_tokens": 8192},
    "arc_easy": {"max_new_tokens": 2048},
    "arc_challenge": {"max_new_tokens": 2048},
    "mbppplus": {"max_new_tokens": 4096},
    "humanevalplus": {"max_new_tokens": 4096},
    "medqa": {"max_new_tokens": 8192},
    "winogrande": {"max_new_tokens": 2048},
}


def load_dataset_by_name(task_name: str) -> List[Dict]:
    """Load dataset by task name."""
    if task_name == "gsm8k":
        dataset_iter = load_gsm8k(split="test")
    elif task_name == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif task_name == "aime2025":
        dataset_iter = load_aime2025(split="train")
    elif task_name == "gpqa":
        dataset_iter = load_gpqa_diamond(split="test")
    elif task_name == "arc_easy":
        dataset_iter = load_arc_easy(split="test")
    elif task_name == "arc_challenge":
        dataset_iter = load_arc_challenge(split="test")
    elif task_name == "mbppplus":
        dataset_iter = load_mbppplus(split="test")
    elif task_name == "humanevalplus":
        dataset_iter = load_humanevalplus(split="test")
    elif task_name == "medqa":
        dataset_iter = load_medqa(split="test")
    elif task_name == "winogrande":
        dataset_iter = load_winogrande(split="validation")
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    
    return list(dataset_iter)


def run_single_task(cli_args, task_name: str, gpu_ids: List[int], timestamp: str) -> Dict:
    """Run evaluation on a single task and save results."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    print(f"Loading {task_name} dataset...")
    data = load_dataset_by_name(task_name)
    if cli_args.limit:
        data = data[:cli_args.limit]
    print(f"Loaded {len(data)} samples")
    
    model_size_match = re.search(r'(\d+[Bb])', cli_args.model)
    model_size = model_size_match.group(1).upper() if model_size_match else "unknown"
    
    if hasattr(cli_args, 'result_prefix') and cli_args.result_prefix:
        run_name = f"{cli_args.result_prefix}_{task_name}_{timestamp}"
    else:
        run_name = f"statebridge_{task_name}_{model_size}_{timestamp}"
    log_file = os.path.join(cli_args.log_dir, f"{run_name}.log")
    output_file = f"results/{run_name}.json"
    
    print(f"Log file: {log_file}")
    print(f"Output file: {output_file}")
    
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Task '{task_name}' not in TASK_CONFIG.")
    if getattr(cli_args, 'max_new_tokens', None) is not None:
        task_max_tokens = cli_args.max_new_tokens
    else:
        task_max_tokens = TASK_CONFIG[task_name]["max_new_tokens"]
    print(f"Max new tokens for {task_name}: {task_max_tokens}")
    
    config = {
        "model_name": cli_args.model,
        "task": task_name,
        "prompt": cli_args.prompt,
        "max_new_tokens": task_max_tokens,
        "temperature": cli_args.temperature,
        "max_prefix_tokens": cli_args.max_prefix_tokens,
        "prefix_strategy": "scale",
        "enable_thinking": True,
        "adaptive_reg": cli_args.adaptive_reg,
        "snap_ratio": cli_args.snap_ratio,
        "debug_mode": cli_args.debug,
        "use_hook": cli_args.use_hook,
        "seed": cli_args.seed,
        "collect_viz": getattr(cli_args, 'collect_viz', False),
        "viz_output": getattr(cli_args, 'viz_output', None),
    }
    
    evaluator = ParallelEvaluator(
        model_name=cli_args.model,
        gpu_ids=gpu_ids,
        config=config,
        log_file=log_file,
    )
    
    summary = evaluator.run_evaluation(data, output_file)
    return summary


def main():
    import argparse as ap
    
    parser = ap.ArgumentParser(description="StateBridge - Latent-State Communication MAS Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--task", type=str, default="medqa",
                        choices=ALL_TASKS,
                        help="Dataset/task to evaluate (ignored if --run_all is set)")
    parser.add_argument("--run_all", action="store_true", 
                        help="Run all datasets sequentially")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset")
    parser.add_argument("--output", type=str, default=None, help="Output file path (single task mode)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--max_prefix_tokens", type=int, default=64, help="Max prefix tokens")
    parser.add_argument("--adaptive_reg", type=float, default=1e-3, help="Regularization for covariance whitening")
    parser.add_argument("--snap_ratio", type=float, default=0.3, help="Snap-to-nearest-embedding ratio")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], 
                        default="sequential", help="Multi-agent architecture")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs")
    parser.add_argument("--log_dir", type=str, default="logs_new", help="Log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from specific log file")
    parser.add_argument("--debug", action="store_true", help="Print detailed output for each agent")
    parser.add_argument("--use_hook", action="store_true", default=True,
                        help="Use hook to capture hidden states (saves memory)")
    parser.add_argument("--no_hook", dest="use_hook", action="store_false",
                        help="Use output_hidden_states instead of hook")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens")
    parser.add_argument("--result_prefix", type=str, default=None, 
                        help="Custom prefix for result files")
    parser.add_argument("--collect_viz", action="store_true",
                        help="Collect alignment vectors for visualization")
    parser.add_argument("--viz_output", type=str, default=None,
                        help="Output path for viz data (.pt file)")
    
    cli_args = parser.parse_args()
    
    if cli_args.collect_viz and cli_args.viz_output is None:
        model_tag = cli_args.model.split("/")[-1].lower().replace("-", "_")
        cli_args.viz_output = f"results/alignment_viz_{model_tag}.pt"
    
    if cli_args.seed is not None:
        set_seed(cli_args.seed)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs(cli_args.log_dir, exist_ok=True)
    
    if cli_args.gpus:
        gpu_ids = [int(x.strip()) for x in cli_args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    if not gpu_ids:
        raise ValueError("No GPUs available")
    print(f"Using GPUs: {gpu_ids}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cli_args.run_all:
        print(f"\n{'#'*60}")
        print(f"RUN-ALL MODE: Running all {len(ALL_TASKS)} datasets")
        print(f"{'#'*60}")
        
        all_summaries = {}
        for task_name in ALL_TASKS:
            try:
                summary = run_single_task(cli_args, task_name, gpu_ids, timestamp)
                all_summaries[task_name] = {
                    "accuracy": summary.get("accuracy", 0),
                    "correct": summary.get("correct", 0),
                    "total": summary.get("total", 0),
                }
            except Exception as e:
                print(f"Error running task {task_name}: {e}")
                all_summaries[task_name] = {"error": str(e)}
        
        print(f"\n{'='*60}")
        print("ALL TASKS COMPLETE - Summary")
        print(f"{'='*60}")
        for task_name, result in all_summaries.items():
            if "error" in result:
                print(f"  {task_name}: ERROR - {result['error']}")
            else:
                acc = result['accuracy'] * 100
                print(f"  {task_name}: {result['correct']}/{result['total']} = {acc:.2f}%")
        
        if cli_args.result_prefix:
            combined_output = f"results/{cli_args.result_prefix}_all_tasks_{timestamp}.json"
        else:
            combined_output = f"results/all_tasks_{timestamp}.json"
        with open(combined_output, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nCombined summary saved to {combined_output}")
        
        return all_summaries
    
    # Single task mode
    if cli_args.resume:
        log_file = cli_args.resume
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        if cli_args.output is None:
            cli_args.output = f"results/{base_name}.json"
        
        data = load_dataset_by_name(cli_args.task)
        if cli_args.limit:
            data = data[:cli_args.limit]
        print(f"Loaded {len(data)} samples")
        
        if cli_args.task not in TASK_CONFIG:
            raise ValueError(f"Task '{cli_args.task}' not in TASK_CONFIG.")
        if cli_args.max_new_tokens is not None:
            task_max_tokens = cli_args.max_new_tokens
        else:
            task_max_tokens = TASK_CONFIG[cli_args.task]["max_new_tokens"]
        
        config = {
            "model_name": cli_args.model,
            "task": cli_args.task,
            "prompt": cli_args.prompt,
            "max_new_tokens": task_max_tokens,
            "temperature": cli_args.temperature,
            "max_prefix_tokens": cli_args.max_prefix_tokens,
            "prefix_strategy": "scale",
            "enable_thinking": True,
            "adaptive_reg": cli_args.adaptive_reg,
            "snap_ratio": cli_args.snap_ratio,
            "debug_mode": cli_args.debug,
            "use_hook": cli_args.use_hook,
            "seed": cli_args.seed,
            "collect_viz": getattr(cli_args, 'collect_viz', False),
            "viz_output": getattr(cli_args, 'viz_output', None),
        }
        
        evaluator = ParallelEvaluator(
            model_name=cli_args.model,
            gpu_ids=gpu_ids,
            config=config,
            log_file=log_file,
        )
        
        summary = evaluator.run_evaluation(data, cli_args.output)
        return summary
    
    summary = run_single_task(cli_args, cli_args.task, gpu_ids, timestamp)
    return summary


if __name__ == "__main__":
    main()
