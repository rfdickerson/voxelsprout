#pragma once

// This engine's own temporal upscaler: the always-available backend, and the
// reference implementation of IUpscaler.
//
// It is the fallback every other backend falls back TO, so it has no SDK, no
// runtime probe and no way to be unavailable -- if it could fail, the "asking
// for DLSS on a machine without it still works" contract in upscale_policy.h
// would have nothing to land on.
//
// Two pipelines, one layout. The same descriptor set feeds a same-resolution
// TAA resolve and a reconstructing upscale, and which one runs is decided by
// the extents rather than by a separate mode flag -- so a host that sets its
// render scale by hand gets the upscaler without asking for it, and a host at
// native resolution gets plain TAA without paying for reconstruction.

#include "render/upscale/upscaler_backend.h"

namespace odai::render::upscale {

std::unique_ptr<IUpscaler> createTemporalUpscaler(const HostServices& host);

}  // namespace odai::render::upscale
