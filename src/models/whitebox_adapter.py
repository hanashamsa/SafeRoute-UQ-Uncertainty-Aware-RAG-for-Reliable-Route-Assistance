import torch
import math
import numpy as np
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from lm_polygraph.utils.model import WhiteboxModel
    HAVE_POLYGRAPH = True
except Exception:
    HAVE_POLYGRAPH = False




def softmax(x: torch.Tensor):
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)

def compute_token_entropy(logits: torch.Tensor) -> float:
    p = softmax(logits)
    return float(-(p * (p + 1e-20).log()).sum().item())


def load_hf_model(model_name: str, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model.eval()
    return model, tokenizer


def hf_generate_and_extract(
    model,
    tokenizer,
    prompt: str,
    device="cuda",
    max_new_tokens=128
) -> Dict[str, Any]:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False
    )

    scores = out.scores
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    entropies, top1_probs, topk_mean_probs = [], [], []

    for s in scores:
        logits = s[0]
        ent = compute_token_entropy(logits)
        entropies.append(ent)

        p = softmax(logits)
        top1_probs.append(float(p.max().item()))
        topk_mean_probs.append(float(p.topk(5).values.mean().item()))

    geom_mean_top1 = float(
        math.exp(sum(math.log(max(1e-12, p)) for p in top1_probs) / len(top1_probs))
    ) if top1_probs else None

    mean_entropy = float(sum(entropies) / len(entropies)) if entropies else None

    return {
        "text": gen_text,
        "entropies": entropies,
        "top1_probs": top1_probs,
        "topk_mean_probs": topk_mean_probs,
        "geom_mean_top1": geom_mean_top1,
        "mean_entropy": mean_entropy,
    }




def poly_generate_and_extract(
    model_name: str,
    prompt: str,
    device="cuda",
    max_new_tokens=128
) -> Dict[str, Any]:

    if not HAVE_POLYGRAPH:
        raise RuntimeError("LM-Polygraph not installed")

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    wb = WhiteboxModel(base, tokenizer, model_path=model_name)

    texts = wb.generate_texts([prompt], max_new_tokens=max_new_tokens)
    poly = wb.get_uncertainty()

    return {
        "text": texts[0],
        "poly_perplexity": float(poly.get("perplexity", None)),
        "poly_samples": int(poly.get("n_samples", 0)),
    }




def generate_and_extract(
    model_name: str,
    prompt: str,
    device="cuda",
    max_new_tokens=128
) -> Dict[str, Any]:
    """
    Unified white-box generation API.
    Uses LM-Polygraph if available, otherwise HF white-box.
    """

    if HAVE_POLYGRAPH:
        try:
            return poly_generate_and_extract(
                model_name=model_name,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens
            )
        except Exception:
            pass  # fall back safely

    model, tokenizer = load_hf_model(model_name, device=device)
    return hf_generate_and_extract(
        model,
        tokenizer,
        prompt,
        device=device,
        max_new_tokens=max_new_tokens
    )
