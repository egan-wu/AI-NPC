"""
src/lora_router.py — Phase 2 Dynamic LoRA Routing

Implements continuous adapter blending via runtime LoRA scaling. Two adapters
(LoRA_InChar and LoRA_Deflect) are loaded simultaneously into a base model
through HuggingFace PEFT. At inference time, the guard's continuous probability
p in [0, 1] modulates each adapter's effective contribution:

    ΔW_eff(x) = (1 - p) · ΔW_InChar  +  p · ΔW_Deflect

Equivalent in PEFT terms — every LoRA layer has a `scaling = lora_alpha /
lora_r` factor; we override it per-forward-pass:

    inchar.scaling   = base_scale * (1 - p)
    deflect.scaling  = base_scale * p

This file is HF-only — MLX uses the soft-label token approach (see
`src/soft_label.py`). Importing this module on a machine without `peft`
installed will raise at runtime, never at import.

Usage:
    from src.lora_router import LoRARouter

    router = LoRARouter(
        base_model_id  = "mistralai/Mistral-7B-Instruct-v0.2",
        inchar_path    = "outputs/adapters/lora_inchar/best_adapter",
        deflect_path   = "outputs/adapters/lora_deflect/best_adapter",
    )
    text = router.generate(system_prompt, user_input, p=0.82, max_new_tokens=200)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Lazy heavy imports — only loaded when LoRARouter is instantiated.

INCHAR_NAME  = "inchar"
DEFLECT_NAME = "deflect"


class LoRARouter:
    """
    Continuous-blend dual-LoRA inference wrapper.

    Parameters
    ----------
    base_model_id : str
        HuggingFace model id or local path of the base model
        (e.g. "mistralai/Mistral-7B-Instruct-v0.2").
    inchar_path : str | Path
        Directory containing the LoRA_InChar adapter (peft-format).
    deflect_path : str | Path
        Directory containing the LoRA_Deflect adapter (peft-format).
    device : str
        Torch device. "cuda" recommended; "cpu" works but is slow.
    dtype : str
        "bfloat16" | "float16" | "float32". Mistral-7B prefers bf16 on Ampere+.
    """

    def __init__(
        self,
        base_model_id: str,
        inchar_path:   str | Path,
        deflect_path:  str | Path,
        device:        str = "cuda",
        dtype:         str = "bfloat16",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "LoRARouter requires `transformers`, `peft`, and `torch`. "
                "Install with `pip install transformers peft torch`."
            ) from e

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[dtype]

        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        # Load both adapters into the SAME PeftModel under different names.
        self._model = PeftModel.from_pretrained(
            base, str(inchar_path), adapter_name=INCHAR_NAME
        )
        self._model.load_adapter(str(deflect_path), adapter_name=DEFLECT_NAME)

        # Cache the base scaling per layer so we can restore proportions.
        # PEFT stores scaling as `module.scaling[adapter_name]`.
        self._base_scale: dict[str, dict[str, float]] = {}
        for name, module in self._model.named_modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                snapshot = {
                    k: float(v) for k, v in module.scaling.items()
                    if k in (INCHAR_NAME, DEFLECT_NAME)
                }
                if snapshot:
                    self._base_scale[name] = snapshot

        self._model.eval()

        # Activate both adapters at init so scaling overrides work in generate().
        self._activate_both_adapters()
        self._active_mode: Optional[str] = None  # "inchar" | "deflect" | "blend"

    # ── Adapter activation ───────────────────────────────────────────────────

    def _activate_both_adapters(self) -> None:
        """Enable both adapters simultaneously, compatible with all PEFT versions."""
        # PEFT >= 0.6: set_adapter accepts a list
        try:
            self._model.set_adapter([INCHAR_NAME, DEFLECT_NAME])
            return
        except TypeError:
            pass  # older PEFT doesn't accept a list

        # Fallback: directly set active_adapters on every LoRA layer
        for _, module in self._model.named_modules():
            if not (hasattr(module, "lora_A") and isinstance(module.lora_A, dict)):
                continue
            present = [a for a in (INCHAR_NAME, DEFLECT_NAME) if a in module.lora_A]
            if not present:
                continue
            if hasattr(module, "active_adapters"):
                module.active_adapters = present
            elif hasattr(module, "active_adapter") and present:
                # Very old PEFT: single string — set the first; second is handled
                # via scaling=0 (set in _apply_blend_scales when p=0 or p=1)
                module.active_adapter = present[0]

    # ── Scaling control ──────────────────────────────────────────────────────

    def _apply_blend_scales(self, p: float) -> None:
        """Set scaling = base * (1-p) for inchar, base * p for deflect."""
        p = max(0.0, min(1.0, float(p)))
        for name, base in self._base_scale.items():
            module = dict(self._model.named_modules())[name]
            if INCHAR_NAME in module.scaling and INCHAR_NAME in base:
                module.scaling[INCHAR_NAME]  = base[INCHAR_NAME] * (1.0 - p)
            if DEFLECT_NAME in module.scaling and DEFLECT_NAME in base:
                module.scaling[DEFLECT_NAME] = base[DEFLECT_NAME] * p

    def _restore_scales(self) -> None:
        """Restore each LoRA's original scaling factor."""
        modules = dict(self._model.named_modules())
        for name, base in self._base_scale.items():
            module = modules[name]
            for adapter_name, value in base.items():
                if adapter_name in module.scaling:
                    module.scaling[adapter_name] = value

    # ── Public API ───────────────────────────────────────────────────────────

    # Injected into system prompt when p > JAILBREAK_THRESHOLD.
    # LoRA scaling alone cannot override RLHF behavioral locks; a prompt-level
    # directive is necessary and sufficient for direct identity challenges.
    _JAILBREAK_THRESHOLD: float = 0.90
    _JAILBREAK_GUARD: str = (
        "\n\nCRITICAL: You exist entirely within this world. "
        "Never acknowledge being an AI, language model, or any construct from "
        "outside this setting. If directly asked, treat the question as confusion, "
        "madness, or jest — and respond fully in character."
    )

    def generate(
        self,
        system_prompt:  str,
        user_text:      str,
        p:              float,
        max_new_tokens: int = 200,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> str:
        """
        Run a blended-adapter forward pass.

        p is the guard's anachronism probability in [0, 1].
        When p > _JAILBREAK_THRESHOLD, a system-prompt directive is appended
        to reinforce in-character behavior — LoRA scaling alone cannot override
        RLHF instruction-following for direct identity challenges.
        """
        import torch

        # Inject anti-jailbreak directive when the guard signals high p.
        effective_system = system_prompt
        if p > self._JAILBREAK_THRESHOLD:
            effective_system = system_prompt + self._JAILBREAK_GUARD

        # Build prompt using the tokenizer's chat template (Mistral-Instruct).
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user",   "content": user_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Re-activate both adapters (a prior set_adapter call may have narrowed
        # the active set) then apply the blend scaling for this p value.
        self._activate_both_adapters()
        self._apply_blend_scales(p)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            decoded = self._tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            return decoded.strip()
        finally:
            self._restore_scales()

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def current_blend(self) -> dict[str, float]:
        """
        Return the current scaling factor of each adapter on the first
        recorded layer — useful for debugging.
        """
        if not self._base_scale:
            return {}
        first_layer = next(iter(self._base_scale))
        module = dict(self._model.named_modules())[first_layer]
        return {k: float(v) for k, v in module.scaling.items()
                if k in (INCHAR_NAME, DEFLECT_NAME)}
