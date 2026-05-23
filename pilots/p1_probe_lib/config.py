"""Backbone configs for the committed-SAE-trace pipeline.

Introduced to let Phase-2 extend beyond Qwen3. Existing Qwen3 scripts keep
their hard-coded constants for byte-level reproducibility of paper logs.
New scripts (Gemma pilot/full) should import the configs here.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackboneCfg:
    tag: str                  # e.g. "qwen3-1.7b-L14" or "gemma2-2b-L12"
    model_id: str             # HF repo id
    sae_release: str          # SAE-Lens release
    sae_id: str               # SAE-Lens sae_id inside the release
    layer: int
    d_model: int
    d_sae: int
    top_k: int = 32
    hook_kind: Literal["mlp_in", "residual_post"] = "mlp_in"


QWEN3_L14 = BackboneCfg(
    tag="qwen3-1.7b-L14",
    model_id="Qwen/Qwen3-1.7B",
    sae_release="mwhanna-qwen3-1.7b-transcoders-lowl0",
    sae_id="layer_14",
    layer=14, d_model=2048, d_sae=163840,
    hook_kind="mlp_in",
)

GEMMA2_L12 = BackboneCfg(
    tag="gemma2-2b-L12",
    model_id="google/gemma-2-2b",
    sae_release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    layer=12, d_model=2304, d_sae=16384,
    hook_kind="residual_post",
)

GEMMA2_9B_L20 = BackboneCfg(
    tag="gemma2-9b-L20",
    model_id="google/gemma-2-9b",
    sae_release="gemma-scope-9b-pt-res-canonical",
    sae_id="layer_20/width_131k/canonical",
    layer=20, d_model=3584, d_sae=131072,
    hook_kind="residual_post",
)


ALL = {QWEN3_L14.tag: QWEN3_L14, GEMMA2_L12.tag: GEMMA2_L12, GEMMA2_9B_L20.tag: GEMMA2_9B_L20}
