#pragma once

// Suballocator for the imported-scene geometry arenas.
//
// Streaming replaces the old "one exact-fit buffer per scene" model: cells are
// added and removed continuously, so their vertex and index ranges have to be
// carved out of, and handed back to, one long-lived buffer. This is the
// bookkeeping half of that -- Vulkan-free, so it can be unit tested (no test
// executable in this project links Vulkan), and because a suballocator that
// silently fails to coalesce looks exactly like one that works until the arena
// runs out of room hours into a session.
//
// First-fit over a sorted free list, with neighbour coalescing on free. First-fit
// rather than best-fit because cell geometry is fairly uniform in size, so the
// scan is short and best-fit's extra bookkeeping buys little.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace odai::render {

class GpuArenaAllocator {
public:
    static constexpr std::uint64_t kInvalidOffset = ~0ull;

    // Sets the arena's total size and drops every existing allocation. Growing
    // an arena that still has live allocations is NOT supported here: the
    // backend grows by creating a bigger buffer and re-uploading, because the
    // old device memory has to be copied anyway.
    void reset(std::uint64_t capacity) {
        m_capacity = capacity;
        m_freeBlocks.clear();
        if (capacity > 0) {
            m_freeBlocks.push_back({0, capacity});
        }
        m_used = 0;
    }

    // Extends the arena, keeping every live allocation at its current offset.
    // This is what the backend calls when a streamed cell does not fit: it
    // recreates the GPU buffer larger and copies the old contents across at the
    // same offsets, so no chunk has to be re-uploaded or renumbered.
    // Shrinking is not supported and is ignored.
    void grow(std::uint64_t newCapacity) {
        if (newCapacity <= m_capacity) {
            return;
        }
        const std::uint64_t added = newCapacity - m_capacity;
        // The new space starts exactly where the arena used to end, so if the
        // arena already ended in free space the two are contiguous and must
        // merge -- otherwise repeated growth would leave a seam at every old
        // boundary that no single allocation could ever span.
        if (!m_freeBlocks.empty() &&
            m_freeBlocks.back().offset + m_freeBlocks.back().size == m_capacity) {
            m_freeBlocks.back().size += added;
        } else {
            m_freeBlocks.push_back({m_capacity, added});
        }
        m_capacity = newCapacity;
    }

    // Returns the offset of a range of `size` bytes aligned to `alignment`, or
    // kInvalidOffset when the arena cannot satisfy it. `alignment` must be a
    // power of two; 0 or 1 both mean unaligned.
    [[nodiscard]] std::uint64_t allocate(std::uint64_t size, std::uint64_t alignment) {
        if (size == 0) {
            return kInvalidOffset;
        }
        const std::uint64_t align = (alignment < 2) ? 1 : alignment;
        for (std::size_t i = 0; i < m_freeBlocks.size(); ++i) {
            const Block block = m_freeBlocks[i];
            const std::uint64_t alignedOffset = (block.offset + align - 1) & ~(align - 1);
            // Alignment padding comes out of this block, so the usable span
            // shrinks by however much we skipped.
            const std::uint64_t padding = alignedOffset - block.offset;
            if (block.size < padding || block.size - padding < size) {
                continue;
            }

            const std::uint64_t tailOffset = alignedOffset + size;
            const std::uint64_t tailSize = block.size - padding - size;
            // Rebuild the block as up to two remainders: the alignment padding
            // in front stays free (it is reusable by a less-aligned request),
            // and whatever is left behind the allocation stays free too.
            m_freeBlocks.erase(m_freeBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            if (tailSize > 0) {
                m_freeBlocks.insert(
                    m_freeBlocks.begin() + static_cast<std::ptrdiff_t>(i), {tailOffset, tailSize});
            }
            if (padding > 0) {
                m_freeBlocks.insert(
                    m_freeBlocks.begin() + static_cast<std::ptrdiff_t>(i), {block.offset, padding});
            }
            m_used += size;
            return alignedOffset;
        }
        return kInvalidOffset;
    }

    // Returns a range to the arena, merging it with any adjacent free blocks.
    // The (offset, size) pair must match what allocate() returned; freeing a
    // range twice, or one that was never allocated, corrupts the arena.
    void free(std::uint64_t offset, std::uint64_t size) {
        if (size == 0) {
            return;
        }
        m_used -= std::min(m_used, size);

        // Insert in offset order so coalescing only ever has to look at the
        // immediate neighbours.
        const auto insertAt = std::lower_bound(
            m_freeBlocks.begin(), m_freeBlocks.end(), offset,
            [](const Block& block, std::uint64_t value) { return block.offset < value; });
        const auto inserted = m_freeBlocks.insert(insertAt, {offset, size});
        const std::size_t index = static_cast<std::size_t>(inserted - m_freeBlocks.begin());

        // Merge forward first: merging backward would invalidate `index`.
        if (index + 1 < m_freeBlocks.size()) {
            Block& current = m_freeBlocks[index];
            const Block& next = m_freeBlocks[index + 1];
            if (current.offset + current.size == next.offset) {
                current.size += next.size;
                m_freeBlocks.erase(m_freeBlocks.begin() + static_cast<std::ptrdiff_t>(index) + 1);
            }
        }
        if (index > 0) {
            Block& previous = m_freeBlocks[index - 1];
            const Block& current = m_freeBlocks[index];
            if (previous.offset + previous.size == current.offset) {
                previous.size += current.size;
                m_freeBlocks.erase(m_freeBlocks.begin() + static_cast<std::ptrdiff_t>(index));
            }
        }
    }

    [[nodiscard]] std::uint64_t capacity() const { return m_capacity; }
    [[nodiscard]] std::uint64_t used() const { return m_used; }
    // Number of separate free ranges. One (or zero) means no fragmentation; a
    // number that climbs steadily over a session is the signal that coalescing
    // is not working.
    [[nodiscard]] std::size_t freeBlockCount() const { return m_freeBlocks.size(); }
    [[nodiscard]] std::uint64_t largestFreeBlock() const {
        std::uint64_t largest = 0;
        for (const Block& block : m_freeBlocks) {
            largest = std::max(largest, block.size);
        }
        return largest;
    }

private:
    struct Block {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    std::vector<Block> m_freeBlocks;
    std::uint64_t m_capacity = 0;
    std::uint64_t m_used = 0;
};

}  // namespace odai::render
