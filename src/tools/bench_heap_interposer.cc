// Linux/glibc-only allocation census for the separate benchmark allocation pass.
// Uses glibc's underlying allocator to avoid dlsym recursion in malloc hooks.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

extern "C" void* __libc_malloc(std::size_t);
extern "C" void* __libc_calloc(std::size_t, std::size_t);
extern "C" void* __libc_realloc(void*, std::size_t);
extern "C" void* __libc_memalign(std::size_t, std::size_t);

namespace {
std::atomic<bool> active{false};
std::atomic<std::uint64_t> calls{0};
int output = -1;
void count() {
    if (active.load(std::memory_order_relaxed)) calls.fetch_add(1, std::memory_order_relaxed);
}
}

extern "C" void* malloc(std::size_t size) { count(); return __libc_malloc(size); }
extern "C" void* calloc(std::size_t countItems, std::size_t size) {
    count(); return __libc_calloc(countItems, size);
}
extern "C" void* realloc(void* ptr, std::size_t size) {
    if (size != 0) count();
    return __libc_realloc(ptr, size);
}
extern "C" void* aligned_alloc(std::size_t alignment, std::size_t size) {
    count(); return __libc_memalign(alignment, size);
}
extern "C" int posix_memalign(void** ptr, std::size_t alignment, std::size_t size) {
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) return 22;
    void* result = __libc_memalign(alignment, size);
    if (!result) return 12;
    *ptr = result;
    count();
    return 0;
}

extern "C" void odai_bench_frame_begin(std::uint64_t) {
    if (output < 0) {
        if (const char* path = std::getenv("ODAI_HEAP_ALLOC_CSV")) {
            output = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (output >= 0) {
                constexpr char header[] = "frame,allocations\n";
                (void)write(output, header, sizeof(header) - 1);
            }
        }
    }
    calls.store(0, std::memory_order_relaxed);
    active.store(true, std::memory_order_release);
}

extern "C" void odai_bench_frame_end(std::uint64_t frame) {
    active.store(false, std::memory_order_release);
    if (output < 0) return;
    char line[80];
    const int size = std::snprintf(line, sizeof(line), "%llu,%llu\n",
        static_cast<unsigned long long>(frame),
        static_cast<unsigned long long>(calls.load(std::memory_order_relaxed)));
    if (size > 0 && size < static_cast<int>(sizeof(line))) (void)write(output, line, size);
}
