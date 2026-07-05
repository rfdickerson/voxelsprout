// VMA is header-only and needs exactly one implementation translation unit
// somewhere in the final binary. Providing it here means contestants linking
// ui_stress_capture don't need to know this quirk exists (see
// examples/vulkan_ui_integration/vma_impl.cc for the same pattern).
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
