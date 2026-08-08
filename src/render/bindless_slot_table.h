#pragma once

// Refcounted slot bookkeeping for the bindless imported-texture table.
//
// Deliberately Vulkan-free. The backend owns the actual VkImages; this only
// answers "which slot does this key live in, and is this the first or last
// reference to it". Keeping it separate is what makes it testable at all --
// no test executable in this project links Vulkan, and this is precisely the
// kind of bookkeeping (free lists, reference counts, key maps) that fails
// silently rather than loudly when it is wrong.
//
// Slot numbering: indices here are *table indices*, not bindless indices. The
// backend adds kBindlessTextureStaticCount to get the descriptor array index.
// Indices are stable for the lifetime of the table -- a released slot is
// recycled, never renumbered, because renumbering would silently repoint live
// draws at whatever texture landed in the vacated position.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::render {

inline constexpr std::uint32_t kInvalidSlotIndex = 0xFFFFFFFFu;

class BindlessSlotTable {
public:
    struct Acquisition {
        std::uint32_t slotIndex = kInvalidSlotIndex;
        // True only on the 0 -> 1 transition, i.e. the caller must actually
        // create and upload the image. False means it was already resident and
        // the reference count was simply bumped.
        bool needsUpload = false;
    };

    // Maximum number of slots the descriptor array can address. Acquiring past
    // this returns kInvalidSlotIndex rather than growing.
    void setCapacity(std::uint32_t capacity) { m_capacity = capacity; }
    [[nodiscard]] std::uint32_t capacity() const { return m_capacity; }

    // Reserves a slot for `key`. An empty key is never deduplicated: it always
    // takes a fresh slot and can only be released by index.
    Acquisition acquire(const std::string& key) {
        if (!key.empty()) {
            const auto existing = m_slotByKey.find(key);
            if (existing != m_slotByKey.end()) {
                ++m_slots[existing->second].refCount;
                return {existing->second, false};
            }
        }

        std::uint32_t slotIndex = kInvalidSlotIndex;
        if (!m_freeSlots.empty()) {
            slotIndex = m_freeSlots.back();
            m_freeSlots.pop_back();
        } else if (static_cast<std::uint32_t>(m_slots.size()) < m_capacity) {
            slotIndex = static_cast<std::uint32_t>(m_slots.size());
            m_slots.emplace_back();
        } else {
            return {kInvalidSlotIndex, false};
        }

        m_slots[slotIndex].refCount = 1;
        m_slots[slotIndex].key = key;
        if (!key.empty()) {
            m_slotByKey[key] = slotIndex;
        }
        return {slotIndex, true};
    }

    // Undoes an acquire whose upload then failed, so a half-built slot is never
    // left addressable. Only valid immediately after an acquire that returned
    // needsUpload -- that is the only case where no one else can hold a
    // reference yet.
    void abandon(std::uint32_t slotIndex) {
        if (slotIndex >= m_slots.size() || m_slots[slotIndex].refCount == 0) {
            return;
        }
        releaseSlot(slotIndex);
    }

    // Drops one reference. Returns true when that was the last one, meaning the
    // caller now owns destroying the image behind it.
    bool release(std::uint32_t slotIndex) {
        if (slotIndex >= m_slots.size()) {
            return false;
        }
        Slot& slot = m_slots[slotIndex];
        if (slot.refCount == 0) {
            return false;
        }
        if (--slot.refCount > 0) {
            return false;
        }
        releaseSlot(slotIndex);
        return true;
    }

    [[nodiscard]] std::uint32_t refCount(std::uint32_t slotIndex) const {
        return slotIndex < m_slots.size() ? m_slots[slotIndex].refCount : 0u;
    }
    // Number of slots ever allocated, including recycled ones. This is the
    // high-water mark, and the bound the descriptor writer iterates to.
    [[nodiscard]] std::size_t slotCount() const { return m_slots.size(); }
    // Slots currently holding a live texture.
    [[nodiscard]] std::size_t residentCount() const { return m_slotByKey.size(); }
    [[nodiscard]] std::size_t freeCount() const { return m_freeSlots.size(); }

    void clear() {
        m_slots.clear();
        m_slotByKey.clear();
        m_freeSlots.clear();
    }

private:
    struct Slot {
        std::uint32_t refCount = 0;
        std::string key;
    };

    void releaseSlot(std::uint32_t slotIndex) {
        Slot& slot = m_slots[slotIndex];
        if (!slot.key.empty()) {
            m_slotByKey.erase(slot.key);
        }
        slot = Slot{};
        m_freeSlots.push_back(slotIndex);
    }

    std::vector<Slot> m_slots;
    std::unordered_map<std::string, std::uint32_t> m_slotByKey;
    std::vector<std::uint32_t> m_freeSlots;
    std::uint32_t m_capacity = 0;
};

}  // namespace odai::render
