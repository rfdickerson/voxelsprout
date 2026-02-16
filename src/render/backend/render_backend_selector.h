#pragma once

#if defined(VOXEL_RENDER_BACKEND_VULKAN)
#include "render/backend/vulkan/renderer_backend.h"
#elif defined(VOXEL_RENDER_BACKEND_DX12)
#error "DX12 backend is not implemented yet."
#elif defined(VOXEL_RENDER_BACKEND_METAL)
#error "Metal backend is not implemented yet."
#else
#error "VOXEL_RENDER_BACKEND is not configured to a supported backend"
#endif
