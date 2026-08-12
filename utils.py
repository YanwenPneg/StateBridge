# Portions adapted from LatentMAS (https://github.com/Gen-Verse/LatentMAS).
# Licensed under Apache-2.0 and modified by the StateBridge authors.
# See THIRD_PARTY_NOTICES.md.

import os
import random
import re
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_gsm8k_answer_1(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} or from text tail.

    Priority: \\boxed{} content > standalone letter in tail > number in tail.
    """
    # Step 1: Try to extract from \\boxed{...} (highest priority)
    boxes = re.findall(r"\\boxed\{([^}\]]*?)(?:\}|\]\]|\])", text)
    if boxes:
        content = boxes[-1].strip()
        # Check if boxed content is a single letter
        letter_match = re.search(r"^[A-Da-d]$", content)
        if letter_match:
            return letter_match.group(0)
        # Try to extract LAST letter from content like "b)" or "(a)"
        letters_in_content = re.findall(r"([A-Da-d])", content)
        if letters_in_content:
            return letters_in_content[-1]  # Return the LAST letter
        # Otherwise extract number from boxed
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content

    # Step 2: No \\boxed{}, check if </think> exists; otherwise treat as truncated
    if "</think>" not in text:
        return None

    # Only search in the last ~10 characters
    tail_text = text[-10:] if len(text) > 10 else text

    # Step 3: Find the LAST standalone letter (A/B/C/D/a/b/c/d) in tail
    standalone_letters = re.findall(r"(?<![a-zA-Z])([A-Da-d])(?![a-zA-Z])", tail_text)
    if standalone_letters:
        return standalone_letters[-1]

    # Step 4: Fall back to last number in tail
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", tail_text)
    if numbers:
        return numbers[-1]

    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Simplified answer extraction: only looks for numbers."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    """Extract gold answer from GSM8K-style #### delimiter."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    ans = ans.strip().lower()
    return ans


def extract_markdown_python_block(text: str) -> Optional[str]:
    """Extract the last ```python ... ``` code block from text."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def run_with_timeout(code: str, timeout: int) -> Tuple[bool, Optional[str]]:
    """Execute `code` in a fresh Python subprocess with a timeout.

    Returns: (ok, error_message). error_message is None when ok=True.
    """
    with tempfile.TemporaryDirectory() as td:
        prog = os.path.join(td, "prog.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            r = subprocess.run(
                [sys.executable, prog],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Execution exceeded {timeout} seconds"
        except Exception as e:
            return False, f"SubprocessError: {e}"

        if r.returncode == 0:
            return True, None

        err = (r.stderr or r.stdout or "").strip()
        if not err:
            err = f"NonZeroExit: returncode={r.returncode}"
        # Truncate overly long error messages
        return False, err[:8000]
