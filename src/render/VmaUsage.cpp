#if defined(VOXEL_HAS_VMA) && defined(VOXEL_VMA_HEADER_ONLY)
#define VMA_IMPLEMENTATION
#if defined(__has_include)
#if __has_include(<vk_mem_alloc.h>)
#include <vk_mem_alloc.h>
#elif __has_include(<vma/vk_mem_alloc.h>)
#include <vma/vk_mem_alloc.h>
#else
#error "VOXEL_HAS_VMA is set but vk_mem_alloc.h was not found"
#endif
#else
#include <vk_mem_alloc.h>
#endif
#endif
