"""Model, tokenizer and calibration-slice loading (design.md -> Calibration protocol).

HuggingFace objects stay at this boundary; the rest of the package sees tensors.
Heavy imports happen inside functions so the decomposition core is importable
without ``transformers``/``datasets`` installed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# The archived slice definition (change phase0-calibration-robustness D3): exactly
# 128 non-empty documents. It is fixed by the recorded results, not a tunable count.
DOCS128_COUNT = 128


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config dtype name to a :class:`torch.dtype`.

    Args:
        name: One of ``bfloat16``, ``float16``, ``float32``.

    Returns:
        The matching torch dtype.

    Raises:
        ValueError: If the name is unknown.
    """
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}"
        ) from exc


def layer_module_path(layer_index: int, module_name: str) -> str:
    """Return the dotted path of an MLP projection inside the base model.

    Args:
        layer_index: Decoder layer index.
        module_name: Module name inside the MLP block, e.g. ``gate_proj``.

    Returns:
        Path accepted by ``model.get_submodule``.
    """
    return f"model.layers.{layer_index}.mlp.{module_name}"


def load_tokenizer(model_id: str) -> Any:
    """Load the model tokenizer.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        The tokenizer object (HuggingFace boundary type).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str, dtype: str) -> Any:
    """Load a causal LM in evaluation mode on the requested dtype.

    Args:
        model_id: HuggingFace model identifier.
        dtype: Config dtype name; see :func:`resolve_dtype`.

    Returns:
        The model in ``eval`` mode (HuggingFace boundary type).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=resolve_dtype(dtype))
    model.eval()
    return model


def load_documents(
    dataset_id: str,
    dataset_config: str,
    split: str,
    n_documents: int,
) -> list[str]:
    """Return the first ``n_documents`` non-empty documents of a split.

    Args:
        dataset_id: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split: Split name.
        n_documents: Number of non-empty documents to keep.

    Returns:
        The selected document texts.

    Raises:
        ValueError: If fewer than ``n_documents`` non-empty documents exist.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, dataset_config, split=split)
    documents: list[str] = []
    for text in dataset["text"]:
        if text.strip():
            documents.append(text)
        if len(documents) >= n_documents:
            break
    if len(documents) < n_documents:
        raise ValueError(
            f"dataset {dataset_id}/{dataset_config} has only {len(documents)} documents"
        )
    return documents


def encode_documents(
    tokenizer: Any,
    documents: list[str],
    seq_len: int,
) -> tuple[Tensor, Tensor]:
    """Tokenize documents to fixed-length id/mask tensors.

    Args:
        tokenizer: HuggingFace tokenizer.
        documents: Document texts.
        seq_len: Truncation and padding length.

    Returns:
        ``(input_ids, attention_mask)``, both int64 ``[n_documents, seq_len]``.
    """
    encoded = tokenizer(
        documents,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(torch.long)
    attention_mask = encoded["attention_mask"].to(torch.long)
    return input_ids, attention_mask


def load_text_stream(dataset_id: str, dataset_config: str, split: str) -> str:
    """Concatenate a split's non-empty ``text`` values in order.

    Args:
        dataset_id: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split: Split name.

    Returns:
        The concatenated text; entries are joined with a newline so adjacent
        documents stay separated (design.md D1).

    Raises:
        ValueError: If the split has no non-empty text.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, dataset_config, split=split)
    texts = [text for text in dataset["text"] if text.strip()]
    if not texts:
        raise ValueError(f"dataset {dataset_id}/{dataset_config} split {split} has no text")
    return "\n".join(texts)


def build_dense_windows(
    tokenizer: Any,
    text: str,
    *,
    seq_len: int,
    n_windows: int,
) -> tuple[Tensor, Tensor]:
    """Tokenize a text stream once and cut it into full-length windows.

    The stream is tokenized once without padding or truncation, then the first
    ``n_windows * seq_len`` tokens are reshaped into ``n_windows`` non-overlapping
    windows (design.md D1). Every window is exactly ``seq_len`` tokens, so no
    padding is produced and the attention mask is all ones.

    Args:
        tokenizer: HuggingFace tokenizer or duck-typed equivalent.
        text: Concatenated document text.
        seq_len: Window length in tokens; must be >= 1.
        n_windows: Number of complete windows to keep; must be >= 1.

    Returns:
        ``(input_ids, attention_mask)``, both int64 ``[n_windows, seq_len]``.

    Raises:
        ValueError: If ``seq_len`` or ``n_windows`` is below 1, or the stream is
            shorter than ``n_windows * seq_len`` tokens.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    encoded = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(torch.long)
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    required = seq_len * n_windows
    if int(input_ids.numel()) < required:
        raise ValueError(
            f"text stream has {int(input_ids.numel())} tokens, need {required} "
            f"for {n_windows} windows of {seq_len}"
        )
    windows = input_ids[:required].reshape(n_windows, seq_len)
    attention_mask = torch.ones_like(windows)
    return windows, attention_mask


def split_slice(
    input_ids: Tensor,
    attention_mask: Tensor,
    eval_fraction: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a slice by order into contiguous fit and held-out portions.

    The first ``round((1 - eval_fraction) * n_items)`` items are the fit portion
    and the remaining items are held out (design.md D4). No shuffling and no RNG
    are used, so repeating the split yields identical portions.

    Args:
        input_ids: Token ids ``[n_items, seq_len]`` int64.
        attention_mask: Attention mask ``[n_items, seq_len]`` int64.
        eval_fraction: Held-out fraction in ``[0, 1)``.

    Returns:
        ``(fit_ids, fit_mask, heldout_ids, heldout_mask)``, each a contiguous
        slice of the corresponding input rows.

    Raises:
        ValueError: If the shapes disagree, ``eval_fraction`` is outside
            ``[0, 1)``, or either portion would be empty.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and mask {tuple(attention_mask.shape)} differ"
        )
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    n_items = int(input_ids.shape[0])
    n_fit = round((1.0 - eval_fraction) * n_items)
    if n_fit < 1 or n_items - n_fit < 1:
        raise ValueError(
            f"eval_fraction={eval_fraction} on {n_items} items leaves an empty portion "
            f"(fit={n_fit}, held-out={n_items - n_fit})"
        )
    return input_ids[:n_fit], attention_mask[:n_fit], input_ids[n_fit:], attention_mask[n_fit:]
