// odai_ui_vulkan takes a caller-owned VmaAllocator (see UiRenderer::InitInfo) and
// does not itself provide the VMA header-only implementation, so exactly one
// translation unit in the consuming app must instantiate it. This mirrors
// src/render/backend/vulkan/vma_usage.cc, which serves the same role for odai.
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
