#include "ui/font.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <zlib.h>
namespace {
using Bytes = std::vector<std::uint8_t>;
void le(Bytes &b, std::uint32_t n, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
        b.push_back(n >> (8 * i));
}
struct Bits {
    Bytes bytes;
    unsigned count = 0;
    void put(std::uint32_t value, unsigned n) {
        for (unsigned i = n; i > 0; --i) {
            if (count % 8 == 0)
                bytes.push_back(0);
            bytes.back() |= ((value >> (i - 1)) & 1) << (7 - count % 8);
            ++count;
        }
    }
};
Bytes fixture() {
    Bits shape;
    shape.put(1, 4);
    shape.put(0, 4); // one fill, no line styles
    shape.put(0, 1);
    shape.put(3, 5);
    shape.put(1, 5);
    shape.put(0, 1);
    shape.put(0, 1);
    shape.put(1, 1);
    auto edge = [&](int x, int y) {
        shape.put(1, 1);
        shape.put(1, 1);
        shape.put(9, 4);
        shape.put(1, 1);
        shape.put(x, 11);
        shape.put(y, 11);
    };
    edge(300, -700);
    edge(300, 700);
    edge(-600, 0);
    shape.put(0, 6);
    Bytes font;
    le(font, 1, 2);
    le(font, 0x84, 1);
    le(font, 0, 1);
    le(font, 7, 1);
    for (char c : std::string("Fixture"))
        font.push_back(c);
    le(font, 1, 2);
    le(font, 4, 2);
    le(font, 4 + shape.bytes.size(), 2);
    font.insert(font.end(), shape.bytes.begin(), shape.bytes.end());
    le(font, 'A', 2);
    le(font, 800, 2);
    le(font, 200, 2);
    le(font, 0, 2);
    le(font, 640, 2);
    // Bounds and kerning count, unused for atlas placement but present in the record.
    font.push_back(0);
    le(font, 0, 2);
    Bytes file{'F', 'W', 'S', 10, 0, 0, 0, 0};
    file.push_back(0);
    le(file, 24 * 256, 2);
    le(file, 1, 2);
    le(file, (48 << 6) | 63, 2);
    le(file, font.size(), 4);
    file.insert(file.end(), font.begin(), font.end());
    le(file, 0, 2);
    auto size = file.size();
    for (int i = 0; i < 4; ++i)
        file[4 + i] = size >> (8 * i);
    return file;
}
} // namespace
int main() {
    auto bytes = fixture();
    odai::ui::Font font;
    std::string error;
    assert(font.loadSwfFont(bytes, "Fixture", 40, error));
    assert(font.valid());
    assert(font.glyph('A').advance > 25 && font.glyph('A').advance < 26);
    assert(std::any_of(font.atlasPixels().begin(), font.atlasPixels().end(),
                       [](auto p) { return p > 0; }));
    const auto advance = font.glyph('A').advance;
    auto truncated = bytes;
    truncated.resize(15);
    assert(!font.loadSwfFont(truncated, "Fixture", 40, error));
    assert(font.glyph('A').advance == advance); // Failed reads cannot destroy the current face.
    assert(!font.loadSwfFont(bytes, "Missing", 40, error));
    uLongf size = compressBound(bytes.size() - 8);
    Bytes compressed(8 + size);
    std::copy_n(bytes.begin(), 8, compressed.begin());
    compressed[0] = 'C';
    assert(compress(compressed.data() + 8, &size, bytes.data() + 8, bytes.size() - 8) == Z_OK);
    compressed.resize(8 + size);
    assert(font.loadSwfFont(compressed, "Fixture", 40, error));
    assert(font.glyph('A').advance == advance);
    assert(!font.loadSwfFont(bytes, "Fixture", 10000, error));
    std::cout << "SWF font tests passed\n";
}
