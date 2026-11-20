# oMLX 0.8.0 bundle patches

Patched overlays built on top of upstream oMLX **0.8.0** (main). These files add
GLM-MTP Qwen3.5-122B load support and streaming-imatrix handling to the 0.8.0 bundle.

## Patches (all target the GLM-MTP Qwen3.5-122B load + streaming-imatrix)

1. **`omlx/patches/mlx_lm_mtp/qwen35_model.py` — GlmMtpHead**
   Add `GlmMtpHead` (enorm/hnorm RMSNorms + `nn.Linear(2h,h) eh_proj` +
   `shared_head_norm` + MTP decoder layers); route `mtp_head_style == "glm"` to it
   instead of the stock Qwen3.5 `MTPModule`. Restores the GLM-style MTP head that
    0.8.0 removed (the stock build only knows the Qwen3.5-style head).

2. **`omlx/patches/mlx_vlm_qwen35_base_compat.py` — last-layer-full-attn**
   In `decoder_layer_init`, force the LAST layer (idx 48) to full attention
   (`is_linear = False`): the 122B source has full-attn through layer 48 but the
   periodic schedule (interval 4) marks it linear. CAVEAT: base_compat only runs
   when `layer_types` is set (`needs_base_compat` returns False otherwise) — for
   models with `layer_types: None` the fix is dead code and must live in the stock
   `language.py` is_linear default instead.

3. **`omlx/oq.py` — streaming-imatrix allowlist (step3p7 + jamba)**
   Add `step3p7` + `jamba` to `_STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES` and the
   matching `_streamed_text_args` branches, so Step-3.7/AI21-Jamba2 use real
   streaming imatrix instead of the RAM-safe proxy that drops `lm_head`.
