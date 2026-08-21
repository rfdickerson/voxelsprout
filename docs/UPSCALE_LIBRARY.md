# Temporal upscaling

The renderer exposes native rendering, its built-in temporal reconstruction path,
and optional Intel XeSS. Temporal reconstruction is always compiled and is the
fallback when an XeSS SDK/runtime is unavailable.

```bash
odai --upscaler off
odai --upscaler temporal --upscaler-quality quality
odai --upscaler xess --upscaler-quality balanced
```

`ODAI_ENABLE_XESS=ON` and `ODAI_XESS_SDK_DIR` select the optional build-time
integration. XeSS availability is reported at runtime; it is never a hard launch
dependency.

All temporal paths consume camera jitter, depth, and the retained skinned-actor
velocity buffer. `native` quality runs the temporal resolve without reducing the
internal render resolution.
