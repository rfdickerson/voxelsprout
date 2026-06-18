#include "ui/font.h"

#include "core/log.h"
#include "ui/ui_text_util.h"

#include <algorithm>
#include <fstream>
#include <ios>

#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

namespace {

// ---------------------------------------------------------------------------
// Big-endian binary parsing helpers (OpenType tables are big-endian).
// ---------------------------------------------------------------------------

inline std::uint16_t u16be(const std::uint8_t* p) {
    return static_cast<std::uint16_t>((p[0] << 8) | p[1]);
}
inline std::uint32_t u32be(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(u16be(p)) << 16) | u16be(p + 2);
}
inline std::int16_t i16be(const std::uint8_t* p) {
    return static_cast<std::int16_t>(u16be(p));
}

// Returns the absolute offset of a named 4-char table within the font binary
// (from the start of the data buffer), or 0 if not found.
std::uint32_t findOtTable(const std::uint8_t* data, std::uint32_t fontStart, const char tag[4]) {
    const std::uint16_t numTables = u16be(data + fontStart + 4);
    for (std::uint16_t i = 0; i < numTables; ++i) {
        const std::uint8_t* rec = data + fontStart + 12 + 16u * i;
        if (rec[0] == tag[0] && rec[1] == tag[1] && rec[2] == tag[2] && rec[3] == tag[3])
            return u32be(rec + 8);
    }
    return 0;
}

// Coverage table: returns the coverage index (>=0) if glyphId is covered, else -1.
int coverageIdx(const std::uint8_t* cov, std::uint32_t glyphId) {
    const std::uint16_t fmt = u16be(cov);
    if (fmt == 1) {
        const std::uint16_t n = u16be(cov + 2);
        int lo = 0, hi = static_cast<int>(n) - 1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            const std::uint16_t g = u16be(cov + 4 + 2u * mid);
            if (g == glyphId) return mid;
            if (g < glyphId) lo = mid + 1; else hi = mid - 1;
        }
    } else if (fmt == 2) {
        const std::uint16_t n = u16be(cov + 2);
        for (std::uint16_t i = 0; i < n; ++i) {
            const std::uint8_t* r = cov + 4 + 6u * i;
            const std::uint16_t s = u16be(r), e = u16be(r + 2);
            if (glyphId >= s && glyphId <= e)
                return static_cast<int>(u16be(r + 4)) + static_cast<int>(glyphId - s);
        }
    }
    return -1;
}

// ClassDef table: returns the class of glyphId (0 if not listed).
std::uint16_t classDef(const std::uint8_t* cdef, std::uint32_t glyphId) {
    const std::uint16_t fmt = u16be(cdef);
    if (fmt == 1) {
        const std::uint16_t start = u16be(cdef + 2), n = u16be(cdef + 4);
        if (glyphId >= start && glyphId < static_cast<std::uint32_t>(start + n))
            return u16be(cdef + 6 + 2u * (glyphId - start));
    } else if (fmt == 2) {
        const std::uint16_t n = u16be(cdef + 2);
        for (std::uint16_t i = 0; i < n; ++i) {
            const std::uint8_t* r = cdef + 4 + 6u * i;
            const std::uint16_t s = u16be(r), e = u16be(r + 2);
            if (glyphId >= s && glyphId <= e) return u16be(r + 4);
        }
    }
    return 0;
}

// Count bytes occupied by a ValueRecord given its ValueFormat bitmask.
int valueRecordBytes(std::uint16_t fmt) {
    int bits = 0;
    for (std::uint16_t f = fmt; f; f >>= 1) bits += (f & 1);
    return bits * 2;
}

// Extract XAdvance (bit 2 = 0x0004) from a ValueRecord.
std::int16_t xAdvance(const std::uint8_t* rec, std::uint16_t fmt) {
    if (!(fmt & 0x0004)) return 0;
    int off = 0;
    if (fmt & 0x0001) off += 2; // XPlacement
    if (fmt & 0x0002) off += 2; // YPlacement
    return i16be(rec + off);
}

struct LigatureRule {
    std::vector<std::uint32_t> seq; // Codepoint sequence (first glyph included).
    std::uint32_t result = 0;       // Result codepoint (Unicode ligature).
};

struct KernClass {
    std::vector<std::uint16_t> classDef1; // [glyphId] = class1
    std::vector<std::uint16_t> classDef2; // [glyphId] = class2
    std::uint32_t class2Count = 0;
    std::vector<float> matrix; // [c1 * class2Count + c2] = xAdv in pixels
};

}  // anonymous namespace

namespace odai::ui {

// ---------------------------------------------------------------------------
// ShapingData — GPOS kerning and GSUB ligature tables parsed from the TTF.
// ---------------------------------------------------------------------------

struct Font::ShapingData {
    std::vector<std::uint8_t> ttfData;
    stbtt_fontinfo fontInfo{};
    float scale = 1.0f;
    std::vector<LigatureRule> ligatures;
    std::unordered_map<std::uint64_t, float> kernPairs; // packed(g1<<16|g2) -> xAdv px
    std::vector<KernClass> kernClasses;

    float kernAdvance(std::uint32_t g1, std::uint32_t g2) const {
        float k = 0.0f;
        const auto it = kernPairs.find((static_cast<std::uint64_t>(g1) << 16) | g2);
        if (it != kernPairs.end()) k += it->second;
        for (const KernClass& kc : kernClasses) {
            const std::uint16_t c1 = (g1 < kc.classDef1.size()) ? kc.classDef1[g1] : 0;
            const std::uint16_t c2 = (g2 < kc.classDef2.size()) ? kc.classDef2[g2] : 0;
            const std::size_t class1Count = kc.class2Count > 0 ? kc.matrix.size() / kc.class2Count : 0;
            if (c1 < class1Count && c2 < kc.class2Count)
                k += kc.matrix[static_cast<std::size_t>(c1) * kc.class2Count + c2];
        }
        return k;
    }
};

// ---------------------------------------------------------------------------
// Special member functions — defined here so ~unique_ptr<ShapingData> works.
// ---------------------------------------------------------------------------

Font::Font() = default;
Font::~Font() = default;
Font::Font(Font&&) noexcept = default;
Font& Font::operator=(Font&&) noexcept = default;

// ---------------------------------------------------------------------------
// GPOS parser: collects PairAdjustment (type 2) kern lookups.
// ---------------------------------------------------------------------------

static void parseGpos(Font::ShapingData& sd, const std::uint8_t* data, std::uint32_t fontStart) {
    const std::uint32_t base = findOtTable(data, fontStart, "GPOS");
    if (!base) return;
    const std::uint8_t* gpos = data + base;
    const std::uint8_t* featureList = gpos + u16be(gpos + 6);
    const std::uint16_t featCount = u16be(featureList);

    // Collect LookupList indices for the 'kern' feature.
    std::vector<std::uint16_t> lookupIdx;
    for (std::uint16_t fi = 0; fi < featCount; ++fi) {
        const std::uint8_t* rec = featureList + 2 + 6u * fi;
        if (rec[0] == 'k' && rec[1] == 'e' && rec[2] == 'r' && rec[3] == 'n') {
            const std::uint8_t* feat = featureList + u16be(rec + 4);
            const std::uint16_t n = u16be(feat + 2);
            for (std::uint16_t j = 0; j < n; ++j)
                lookupIdx.push_back(u16be(feat + 4 + 2u * j));
        }
    }
    if (lookupIdx.empty()) return;

    const std::uint8_t* lookupList = gpos + u16be(gpos + 8);
    const std::uint16_t lookupCount = u16be(lookupList);

    for (const std::uint16_t li : lookupIdx) {
        if (li >= lookupCount) continue;
        const std::uint8_t* lookup = lookupList + u16be(lookupList + 2 + 2u * li);
        std::uint16_t ltype = u16be(lookup);
        const std::uint16_t subCount = u16be(lookup + 4);

        for (std::uint16_t si = 0; si < subCount; ++si) {
            const std::uint8_t* sub = lookup + u16be(lookup + 6 + 2u * si);
            std::uint16_t subltype = ltype;
            // Extension lookup (type 9): follow the extension pointer.
            if (subltype == 9 && u16be(sub) == 1) {
                subltype = u16be(sub + 2);
                sub = sub + u32be(sub + 4);
            }
            if (subltype != 2) continue; // Only handle PairAdjustment.

            const std::uint16_t fmt  = u16be(sub);
            const std::uint16_t vf1  = u16be(sub + 4);
            const std::uint16_t vf2  = u16be(sub + 6);
            const int vr1sz = valueRecordBytes(vf1);
            const int vr2sz = valueRecordBytes(vf2);

            if (fmt == 1) {
                // Format 1: individual glyph-pair sets.
                const std::uint8_t* cov = sub + u16be(sub + 2);
                const std::uint16_t psCount = u16be(sub + 8);

                // Build the covered glyph list from the coverage table.
                std::vector<std::uint16_t> covGlyphs;
                {
                    const std::uint16_t cf = u16be(cov);
                    if (cf == 1) {
                        const std::uint16_t n = u16be(cov + 2);
                        for (std::uint16_t i = 0; i < n; ++i)
                            covGlyphs.push_back(u16be(cov + 4 + 2u * i));
                    } else if (cf == 2) {
                        const std::uint16_t n = u16be(cov + 2);
                        for (std::uint16_t i = 0; i < n; ++i) {
                            const std::uint8_t* r = cov + 4 + 6u * i;
                            for (std::uint16_t g = u16be(r); g <= u16be(r + 2); ++g)
                                covGlyphs.push_back(g);
                        }
                    }
                }

                for (std::uint16_t pi = 0; pi < psCount && pi < static_cast<std::uint16_t>(covGlyphs.size()); ++pi) {
                    const std::uint16_t g1 = covGlyphs[pi];
                    const std::uint8_t* ps = sub + u16be(sub + 10 + 2u * pi);
                    const std::uint16_t pairCount = u16be(ps);
                    const int recSz = 2 + vr1sz + vr2sz;
                    for (std::uint16_t ki = 0; ki < pairCount; ++ki) {
                        const std::uint8_t* r = ps + 2 + recSz * ki;
                        const std::int16_t xa = xAdvance(r + 2, vf1);
                        if (xa) {
                            const std::uint64_t key = (static_cast<std::uint64_t>(g1) << 16) | u16be(r);
                            sd.kernPairs[key] = static_cast<float>(xa) * sd.scale;
                        }
                    }
                }
            } else if (fmt == 2) {
                // Format 2: class-based kern matrix.
                const std::uint16_t c1Count = u16be(sub + 12);
                const std::uint16_t c2Count = u16be(sub + 14);
                const std::uint8_t* cdef1 = sub + u16be(sub + 8);
                const std::uint8_t* cdef2 = sub + u16be(sub + 10);

                KernClass kc;
                kc.class2Count = c2Count;
                kc.matrix.resize(static_cast<std::size_t>(c1Count) * c2Count, 0.0f);

                // Read the kern matrix.
                const std::uint8_t* rows = sub + 16;
                for (std::uint16_t c1 = 0; c1 < c1Count; ++c1) {
                    for (std::uint16_t c2 = 0; c2 < c2Count; ++c2) {
                        const std::size_t off = (static_cast<std::size_t>(c1) * c2Count + c2) *
                                                static_cast<std::size_t>(vr1sz + vr2sz);
                        const std::int16_t xa = xAdvance(rows + off, vf1);
                        if (xa)
                            kc.matrix[static_cast<std::size_t>(c1) * c2Count + c2] =
                                static_cast<float>(xa) * sd.scale;
                    }
                }

                // Determine max glyph ID to size the class arrays.
                std::uint32_t maxG = 0;
                const auto scanCdef = [&](const std::uint8_t* cd) {
                    if (u16be(cd) == 1) {
                        const std::uint16_t s = u16be(cd + 2), n = u16be(cd + 4);
                        maxG = std::max(maxG, static_cast<std::uint32_t>(s + n - 1));
                    } else if (u16be(cd) == 2) {
                        const std::uint16_t n = u16be(cd + 2);
                        if (n) maxG = std::max(maxG, static_cast<std::uint32_t>(u16be(cd + 4 + 6u * (n - 1) + 2)));
                    }
                };
                scanCdef(cdef1);
                scanCdef(cdef2);
                maxG = std::min(maxG, std::uint32_t{65535});
                kc.classDef1.assign(maxG + 1, 0);
                kc.classDef2.assign(maxG + 1, 0);

                const auto fillCdef = [&](const std::uint8_t* cd, std::vector<std::uint16_t>& out) {
                    if (u16be(cd) == 1) {
                        const std::uint16_t s = u16be(cd + 2), n = u16be(cd + 4);
                        for (std::uint32_t g = s; g < static_cast<std::uint32_t>(s + n) && g <= maxG; ++g)
                            out[g] = u16be(cd + 6 + 2u * (g - s));
                    } else if (u16be(cd) == 2) {
                        const std::uint16_t n = u16be(cd + 2);
                        for (std::uint16_t i = 0; i < n; ++i) {
                            const std::uint8_t* r = cd + 4 + 6u * i;
                            const std::uint16_t s = u16be(r), e = u16be(r + 2), cls = u16be(r + 4);
                            for (std::uint32_t g = s; g <= e && g <= maxG; ++g) out[g] = cls;
                        }
                    }
                };
                fillCdef(cdef1, kc.classDef1);
                fillCdef(cdef2, kc.classDef2);
                sd.kernClasses.push_back(std::move(kc));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight pre-scan: collect all LigatureSubst output glyph IDs from
// 'liga' and 'dlig' features without building full rules.
// ---------------------------------------------------------------------------

static std::vector<std::uint16_t> prescanGsubLigaGlyphs(const std::uint8_t* data,
                                                         std::uint32_t fontStart) {
    std::vector<std::uint16_t> outGlyphs;
    const std::uint32_t base = findOtTable(data, fontStart, "GSUB");
    if (!base) return outGlyphs;
    const std::uint8_t* gsub = data + base;
    const std::uint8_t* featureList = gsub + u16be(gsub + 6);
    const std::uint16_t featCount = u16be(featureList);

    std::vector<std::uint16_t> lookupIdx;
    for (std::uint16_t fi = 0; fi < featCount; ++fi) {
        const std::uint8_t* rec = featureList + 2 + 6u * fi;
        const bool isLiga = (rec[0]=='l' && rec[1]=='i' && rec[2]=='g' && rec[3]=='a');
        const bool isDlig = (rec[0]=='d' && rec[1]=='l' && rec[2]=='i' && rec[3]=='g');
        if (isLiga || isDlig) {
            const std::uint8_t* feat = featureList + u16be(rec + 4);
            const std::uint16_t n = u16be(feat + 2);
            for (std::uint16_t j = 0; j < n; ++j)
                lookupIdx.push_back(u16be(feat + 4 + 2u * j));
        }
    }

    const std::uint8_t* lookupList = gsub + u16be(gsub + 8);
    const std::uint16_t lookupCount = u16be(lookupList);

    for (const std::uint16_t li : lookupIdx) {
        if (li >= lookupCount) continue;
        const std::uint8_t* lookup = lookupList + u16be(lookupList + 2 + 2u * li);
        std::uint16_t ltype = u16be(lookup);
        const std::uint16_t subCount = u16be(lookup + 4);
        for (std::uint16_t si = 0; si < subCount; ++si) {
            const std::uint8_t* sub = lookup + u16be(lookup + 6 + 2u * si);
            std::uint16_t subltype = ltype;
            if (subltype == 7 && u16be(sub) == 1) {
                subltype = u16be(sub + 2);
                sub = sub + u32be(sub + 4);
            }
            if (subltype != 4 || u16be(sub) != 1) continue;
            const std::uint16_t ligSetCount = u16be(sub + 4);
            for (std::uint16_t lsi = 0; lsi < ligSetCount; ++lsi) {
                const std::uint8_t* ligSet = sub + u16be(sub + 6 + 2u * lsi);
                const std::uint16_t ligCount = u16be(ligSet);
                for (std::uint16_t ki = 0; ki < ligCount; ++ki) {
                    const std::uint8_t* lig = ligSet + u16be(ligSet + 2 + 2u * ki);
                    outGlyphs.push_back(u16be(lig)); // LigGlyph
                }
            }
        }
    }

    // Deduplicate.
    std::sort(outGlyphs.begin(), outGlyphs.end());
    outGlyphs.erase(std::unique(outGlyphs.begin(), outGlyphs.end()), outGlyphs.end());
    return outGlyphs;
}

// ---------------------------------------------------------------------------
// GSUB parser: collects LigatureSubstitution (type 4) 'liga'/'dlig' rules.
// glyphToCP maps glyph ID -> codepoint for the loaded range.
// ligGlyphToCP maps ligature output glyph ID -> assigned codepoint (Unicode
// FB00-FB06 when the font has them in its cmap, or PUA U+E000+ otherwise).
// ---------------------------------------------------------------------------

static void parseGsub(Font::ShapingData& sd,
                      const std::unordered_map<std::uint32_t, std::uint32_t>& glyphToCP,
                      const std::unordered_map<std::uint32_t, std::uint32_t>& ligGlyphToCP,
                      const std::uint8_t* data, std::uint32_t fontStart) {
    const std::uint32_t base = findOtTable(data, fontStart, "GSUB");
    if (!base) return;
    const std::uint8_t* gsub = data + base;
    const std::uint8_t* featureList = gsub + u16be(gsub + 6);
    const std::uint16_t featCount = u16be(featureList);

    // Collect LookupList indices for 'liga' (standard) and 'dlig' (discretionary) features.
    std::vector<std::uint16_t> lookupIdx;
    for (std::uint16_t fi = 0; fi < featCount; ++fi) {
        const std::uint8_t* rec = featureList + 2 + 6u * fi;
        const bool isLiga = (rec[0]=='l' && rec[1]=='i' && rec[2]=='g' && rec[3]=='a');
        const bool isDlig = (rec[0]=='d' && rec[1]=='l' && rec[2]=='i' && rec[3]=='g');
        if (isLiga || isDlig) {
            const std::uint8_t* feat = featureList + u16be(rec + 4);
            const std::uint16_t n = u16be(feat + 2);
            for (std::uint16_t j = 0; j < n; ++j)
                lookupIdx.push_back(u16be(feat + 4 + 2u * j));
        }
    }
    if (lookupIdx.empty()) return;

    const std::uint8_t* lookupList = gsub + u16be(gsub + 8);
    const std::uint16_t lookupCount = u16be(lookupList);

    std::vector<LigatureRule> rules;

    for (const std::uint16_t li : lookupIdx) {
        if (li >= lookupCount) continue;
        const std::uint8_t* lookup = lookupList + u16be(lookupList + 2 + 2u * li);
        std::uint16_t ltype = u16be(lookup);
        const std::uint16_t subCount = u16be(lookup + 4);

        for (std::uint16_t si = 0; si < subCount; ++si) {
            const std::uint8_t* sub = lookup + u16be(lookup + 6 + 2u * si);
            std::uint16_t subltype = ltype;
            // GSUB Extension (type 7): follow the extension pointer.
            if (subltype == 7 && u16be(sub) == 1) {
                subltype = u16be(sub + 2);
                sub = sub + u32be(sub + 4);
            }
            if (subltype != 4 || u16be(sub) != 1) continue; // Only LigatureSubst format 1.

            const std::uint8_t* cov = sub + u16be(sub + 2);
            const std::uint16_t ligSetCount = u16be(sub + 4);

            // Build the first-glyph list from the coverage table.
            std::vector<std::uint16_t> firstGlyphs;
            {
                const std::uint16_t cf = u16be(cov);
                if (cf == 1) {
                    const std::uint16_t n = u16be(cov + 2);
                    for (std::uint16_t i = 0; i < n; ++i)
                        firstGlyphs.push_back(u16be(cov + 4 + 2u * i));
                } else if (cf == 2) {
                    const std::uint16_t n = u16be(cov + 2);
                    for (std::uint16_t i = 0; i < n; ++i) {
                        const std::uint8_t* r = cov + 4 + 6u * i;
                        for (std::uint16_t g = u16be(r); g <= u16be(r + 2); ++g)
                            firstGlyphs.push_back(g);
                    }
                }
            }

            for (std::uint16_t lsi = 0; lsi < ligSetCount && lsi < static_cast<std::uint16_t>(firstGlyphs.size()); ++lsi) {
                const std::uint16_t firstGlyph = firstGlyphs[lsi];
                auto it1 = glyphToCP.find(firstGlyph);
                if (it1 == glyphToCP.end()) continue;

                const std::uint8_t* ligSet = sub + u16be(sub + 6 + 2u * lsi);
                const std::uint16_t ligCount = u16be(ligSet);

                for (std::uint16_t ki = 0; ki < ligCount; ++ki) {
                    const std::uint8_t* lig = ligSet + u16be(ligSet + 2 + 2u * ki);
                    const std::uint16_t ligGlyph = u16be(lig);
                    const std::uint16_t compCount = u16be(lig + 2); // Total component count including first.
                    if (compCount == 0) continue; // Sanity guard.

                    auto itL = ligGlyphToCP.find(ligGlyph);
                    if (itL == ligGlyphToCP.end()) continue; // Not a Unicode ligature we support.

                    LigatureRule rule;
                    rule.result = itL->second;
                    rule.seq.push_back(it1->second); // First codepoint.
                    bool ok = true;
                    for (std::uint16_t ci = 0; ci < compCount - 1; ++ci) {
                        const std::uint16_t compGlyph = u16be(lig + 4 + 2u * ci);
                        auto itC = glyphToCP.find(compGlyph);
                        if (itC == glyphToCP.end()) { ok = false; break; }
                        rule.seq.push_back(itC->second);
                    }
                    if (ok && rule.seq.size() >= 2) {
                        rules.push_back(std::move(rule));
                    }
                }
            }
        }
    }

    // Sort longest-first so greedy matching prefers longer ligatures (e.g. ffi > fi).
    std::sort(rules.begin(), rules.end(),
              [](const LigatureRule& a, const LigatureRule& b) {
                  return a.seq.size() > b.seq.size();
              });
    sd.ligatures = std::move(rules);
}

// ---------------------------------------------------------------------------
// Font public API
// ---------------------------------------------------------------------------

const Glyph& Font::glyph(std::uint32_t codepoint) const {
    if (codepoint >= kAsciiFirst && codepoint < kAsciiFirst + kAsciiCount) {
        const std::size_t slot = codepoint - kAsciiFirst;
        if (m_asciiPresent[slot]) {
            return m_ascii[slot];
        }
    }
    const auto it = m_glyphs.find(codepoint);
    return it != m_glyphs.end() ? it->second : m_missing;
}

void Font::rebuildAsciiCache() {
    m_asciiPresent.fill(false);
    for (std::size_t slot = 0; slot < kAsciiCount; ++slot) {
        const auto it = m_glyphs.find(kAsciiFirst + static_cast<std::uint32_t>(slot));
        if (it != m_glyphs.end()) {
            m_ascii[slot] = it->second;
            m_asciiPresent[slot] = true;
        }
    }
}

float Font::measureText(std::string_view utf8) const {
    float width = 0.0f;
    for (const ShapedGlyph& sg : shape(utf8))
        width += glyph(sg.codepoint).advance + sg.kern;
    return width;
}

std::vector<ShapedGlyph> Font::shape(std::string_view utf8) const {
    // Decode UTF-8 to codepoints.
    std::vector<std::uint32_t> codepoints;
    std::size_t i = 0;
    while (i < utf8.size())
        codepoints.push_back(decodeUtf8(utf8, i));

    if (!m_shaping ||
        (m_shaping->ligatures.empty() && m_shaping->kernPairs.empty() && m_shaping->kernClasses.empty())) {
        // No shaping data — return plain codepoints.
        std::vector<ShapedGlyph> result(codepoints.size());
        for (std::size_t j = 0; j < codepoints.size(); ++j)
            result[j].codepoint = codepoints[j];
        return result;
    }

    const ShapingData& sd = *m_shaping;

    // GSUB: greedy ligature substitution (longest rule first).
    std::vector<std::uint32_t> shaped;
    shaped.reserve(codepoints.size());
    std::size_t pos = 0;
    while (pos < codepoints.size()) {
        bool matched = false;
        for (const LigatureRule& rule : sd.ligatures) {
            if (pos + rule.seq.size() > codepoints.size()) continue;
            bool eq = true;
            for (std::size_t k = 0; k < rule.seq.size(); ++k) {
                if (codepoints[pos + k] != rule.seq[k]) { eq = false; break; }
            }
            if (eq) {
                shaped.push_back(rule.result);
                pos += rule.seq.size();
                matched = true;
                break;
            }
        }
        if (!matched) shaped.push_back(codepoints[pos++]);
    }

    // GPOS: kern pairs between adjacent shaped glyphs.
    std::vector<ShapedGlyph> result(shaped.size());
    for (std::size_t j = 0; j < shaped.size(); ++j) {
        result[j].codepoint = shaped[j];
        if (j + 1 < shaped.size()) {
            const int g1 = stbtt_FindGlyphIndex(&sd.fontInfo, static_cast<int>(shaped[j]));
            const int g2 = stbtt_FindGlyphIndex(&sd.fontInfo, static_cast<int>(shaped[j + 1]));
            if (g1 > 0 && g2 > 0)
                result[j].kern = sd.kernAdvance(static_cast<std::uint32_t>(g1),
                                                 static_cast<std::uint32_t>(g2));
        }
    }
    return result;
}

void Font::setGlyph(std::uint32_t codepoint, const Glyph& glyph) {
    m_glyphs[codepoint] = glyph;
    if (codepoint >= kAsciiFirst && codepoint < kAsciiFirst + kAsciiCount) {
        const std::size_t slot = codepoint - kAsciiFirst;
        m_ascii[slot] = glyph;
        m_asciiPresent[slot] = true;
    }
}

void Font::initSyntheticMonospace(float advance, float ascent, float descent,
                                  std::uint32_t firstCodepoint, std::uint32_t lastCodepoint) {
    m_shaping.reset();
    m_glyphs.clear();
    m_atlas.clear();
    m_atlasWidth = 0;
    m_atlasHeight = 0;
    m_ascent = ascent;
    m_descent = descent;
    m_lineHeight = ascent + descent;
    for (std::uint32_t cp = firstCodepoint; cp <= lastCodepoint; ++cp) {
        Glyph g{};
        g.advance = advance;
        if (cp != ' ') {
            g.size = {advance * 0.6f, ascent * 0.7f};
            g.bearing = {advance * 0.1f, ascent * 0.7f};
        }
        m_glyphs[cp] = g;
    }
    m_missing = Glyph{};
    m_missing.advance = advance;
    rebuildAsciiCache();
}

bool Font::loadFromMemory(const std::uint8_t* ttfData, std::size_t ttfSize, float pixelHeight,
                          std::uint32_t atlasSize, std::uint32_t firstCodepoint,
                          std::uint32_t lastCodepoint) {
    if (ttfData == nullptr || ttfSize == 0 || lastCodepoint < firstCodepoint || atlasSize == 0) {
        return false;
    }

    stbtt_fontinfo info{};
    const int fontOffset = stbtt_GetFontOffsetForIndex(ttfData, 0);
    if (fontOffset < 0 || stbtt_InitFont(&info, ttfData, fontOffset) == 0) {
        return false;
    }

    const std::uint32_t glyphCount = (lastCodepoint - firstCodepoint) + 1u;
    m_atlas.assign(static_cast<std::size_t>(atlasSize) * static_cast<std::size_t>(atlasSize), 0u);
    m_atlasWidth = atlasSize;
    m_atlasHeight = atlasSize;

    stbtt_pack_context packContext{};
    if (stbtt_PackBegin(&packContext, m_atlas.data(), static_cast<int>(atlasSize),
                        static_cast<int>(atlasSize), 0, 1, nullptr) == 0) {
        return false;
    }

    int ascent = 0;
    int descent = 0;
    int lineGap = 0;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &lineGap);
    const float scale = stbtt_ScaleForPixelHeight(&info, pixelHeight);
    m_ascent = static_cast<float>(ascent) * scale;
    m_descent = static_cast<float>(-descent) * scale;
    m_lineHeight = static_cast<float>(ascent - descent + lineGap) * scale;

    // Pre-scan GSUB to find all ligature output glyph IDs (liga + dlig). We need
    // this before packing so that non-Unicode ligature glyphs (like EBGaramond's
    // "Th") can be manually rendered into the atlas and assigned PUA codepoints.
    const auto ligOutGlyphIds =
        prescanGsubLigaGlyphs(ttfData, static_cast<std::uint32_t>(fontOffset));

    // Build the ligGlyphToCP map: output glyph ID -> codepoint.
    // Prefer Unicode U+FB00-FB06 when the font has the glyph at those codepoints.
    // Fall back to PUA U+E000+ for font-specific glyphs (e.g. historical "Th").
    std::unordered_map<std::uint32_t, std::uint32_t> ligGlyphToCP;
    std::uint32_t puaNext = 0xE000u;
    {
        // Build inverse cmap for FB00-FB06 first.
        for (std::uint32_t cp = 0xFB00u; cp <= 0xFB06u; ++cp) {
            const int gid = stbtt_FindGlyphIndex(&info, static_cast<int>(cp));
            if (gid > 0) ligGlyphToCP[static_cast<std::uint32_t>(gid)] = cp;
        }
        // Assign PUA to any remaining ligature output glyphs not covered above.
        for (const std::uint16_t gid : ligOutGlyphIds) {
            if (!ligGlyphToCP.count(static_cast<std::uint32_t>(gid)))
                ligGlyphToCP[static_cast<std::uint32_t>(gid)] = puaNext++;
        }
    }

    // Pack range 0 (main) and range 1 (Unicode ligatures FB00-FB06).
    // Keep the pack context OPEN so we can add PUA glyphs immediately after.
    // Horizontal oversampling (3x) is the standard stb_truetype fix for jagged /
    // blocky text at UI sizes.
    std::vector<stbtt_packedchar> packed0(glyphCount);
    std::vector<stbtt_packedchar> packed1(7); // FB00-FB06

    stbtt_pack_range ranges[2];
    ranges[0] = {};
    ranges[0].font_size = pixelHeight;
    ranges[0].first_unicode_codepoint_in_range = static_cast<int>(firstCodepoint);
    ranges[0].num_chars = static_cast<int>(glyphCount);
    ranges[0].chardata_for_range = packed0.data();

    ranges[1] = {};
    ranges[1].font_size = pixelHeight;
    ranges[1].first_unicode_codepoint_in_range = 0xFB00;
    ranges[1].num_chars = 7;
    ranges[1].chardata_for_range = packed1.data();

    stbtt_PackSetOversampling(&packContext, 3, 1);
    const int packOk = stbtt_PackFontRanges(&packContext, ttfData, 0, ranges, 2);
    if (!packOk) {
        VOX_LOGW("Font") << "Atlas packing incomplete at " << pixelHeight
                         << "px (atlasSize=" << atlasSize
                         << "). Consider increasing atlasSize.";
    }

    // Pack PUA ligature glyphs (non-Unicode: no cmap entry) directly by glyph ID.
    // We render without horizontal oversampling to keep the manual rasterization
    // simple; these are rare historical ligatures (e.g. "Th") where 1x quality
    // is acceptable.
    struct PuaGlyphEntry { std::uint32_t glyphId; std::uint32_t codepoint; stbrp_rect rect; int x0, y0; };
    std::vector<PuaGlyphEntry> puaEntries;
    stbrp_context* rpCtx = static_cast<stbrp_context*>(packContext.pack_info);
    for (const auto& [glyphIdU32, cp] : ligGlyphToCP) {
        if (cp < 0xE000u) continue; // Unicode, already packed above.
        const int gid = static_cast<int>(glyphIdU32);
        int x0, y0, x1, y1;
        stbtt_GetGlyphBitmapBox(&info, gid, scale, scale, &x0, &y0, &x1, &y1);
        const int bw = x1 - x0;
        const int bh = y1 - y0;
        if (bw <= 0 || bh <= 0) continue;
        stbrp_rect r{};
        r.w = bw + 2; // +2 padding
        r.h = bh + 2;
        if (rpCtx && stbrp_pack_rects(rpCtx, &r, 1) && r.was_packed) {
            // Render 1x (no oversampling) directly into the atlas.
            stbtt_MakeGlyphBitmap(&info,
                m_atlas.data() + (r.y + 1) * atlasSize + (r.x + 1),
                bw, bh, static_cast<int>(atlasSize),
                scale, scale, gid);
            puaEntries.push_back(PuaGlyphEntry{glyphIdU32, cp, r, x0, y0});
        } else {
            VOX_LOGW("Font") << "No atlas space for PUA ligature glyph " << gid;
        }
    }

    stbtt_PackEnd(&packContext);

    m_glyphs.clear();

    // Range 0: main codepoint range.
    for (std::uint32_t i = 0; i < glyphCount; ++i) {
        // stbtt_GetPackedQuad derives the on-screen quad and atlas UVs together,
        // correctly accounting for the oversampling padding baked into the atlas
        // bitmap. Extracting geometry from packedchar fields by hand mismaps the
        // padded bitmap onto the quad and smears the glyph. Pen starts at the
        // origin so the returned quad is the glyph's offset box and UV rect.
        float penX = 0.0f;
        float penY = 0.0f;
        stbtt_aligned_quad q{};
        stbtt_GetPackedQuad(packed0.data(), static_cast<int>(atlasSize), static_cast<int>(atlasSize),
                            static_cast<int>(i), &penX, &penY, &q, /*align_to_integer=*/1);
        Glyph g{};
        g.size = {q.x1 - q.x0, q.y1 - q.y0};
        g.bearing = {q.x0, -q.y0};  // q.y0 is negative (above the baseline).
        g.advance = packed0[i].xadvance;
        g.uv = UiRect{q.s0, q.t0, q.s1, q.t1};
        m_glyphs[firstCodepoint + i] = g;
    }

    // Range 1: FB00-FB06 typographic ligatures.
    for (std::uint32_t i = 0; i < 7u; ++i) {
        if (packed1[i].xadvance == 0.0f) continue; // Font doesn't have this glyph.
        float penX = 0.0f, penY = 0.0f;
        stbtt_aligned_quad q{};
        stbtt_GetPackedQuad(packed1.data(), static_cast<int>(atlasSize), static_cast<int>(atlasSize),
                            static_cast<int>(i), &penX, &penY, &q, 1);
        Glyph g{};
        g.size = {q.x1 - q.x0, q.y1 - q.y0};
        g.bearing = {q.x0, -q.y0};
        g.advance = packed1[i].xadvance;
        g.uv = UiRect{q.s0, q.t0, q.s1, q.t1};
        m_glyphs[0xFB00u + i] = g;
    }

    // PUA ligature glyphs (manual 1x rasterization).
    {
        const float atlasF = static_cast<float>(atlasSize);
        for (const PuaGlyphEntry& pe : puaEntries) {
            const int bw = pe.rect.w - 2;
            const int bh = pe.rect.h - 2;
            int advance_i = 0, lsb = 0;
            stbtt_GetGlyphHMetrics(&info, static_cast<int>(pe.glyphId), &advance_i, &lsb);
            Glyph g{};
            g.size = {static_cast<float>(bw), static_cast<float>(bh)};
            g.bearing = {static_cast<float>(pe.x0), static_cast<float>(-pe.y0)};
            g.advance = static_cast<float>(advance_i) * scale;
            const float u0 = (pe.rect.x + 1) / atlasF;
            const float v0 = (pe.rect.y + 1) / atlasF;
            g.uv = UiRect{u0, v0, u0 + bw / atlasF, v0 + bh / atlasF};
            m_glyphs[pe.codepoint] = g;
        }
    }

    const auto spaceIt = m_glyphs.find(static_cast<std::uint32_t>(' '));
    m_missing = Glyph{};
    m_missing.advance = (spaceIt != m_glyphs.end()) ? spaceIt->second.advance : (pixelHeight * 0.4f);
    rebuildAsciiCache();

    // ---------------------------------------------------------------------------
    // OpenType shaping tables (GPOS kern + GSUB liga).
    // ---------------------------------------------------------------------------

    m_shaping = std::make_unique<ShapingData>();
    // Keep a copy of the TTF bytes so stbtt_FindGlyphIndex works at shape() time.
    m_shaping->ttfData.assign(ttfData, ttfData + ttfSize);
    stbtt_InitFont(&m_shaping->fontInfo, m_shaping->ttfData.data(), fontOffset);
    m_shaping->scale = scale;

    // Reverse map: glyph ID -> codepoint (for GSUB component lookup).
    std::unordered_map<std::uint32_t, std::uint32_t> glyphToCP;
    for (std::uint32_t cp = firstCodepoint; cp <= lastCodepoint; ++cp) {
        const int gid = stbtt_FindGlyphIndex(&info, static_cast<int>(cp));
        if (gid > 0) glyphToCP[static_cast<std::uint32_t>(gid)] = cp;
    }

    // ligGlyphToCP was built earlier (before atlas packing).
    // Narrow it to only the codepoints that were actually packed into the atlas.
    for (auto it = ligGlyphToCP.begin(); it != ligGlyphToCP.end(); ) {
        if (!m_glyphs.count(it->second))
            it = ligGlyphToCP.erase(it);
        else
            ++it;
    }

    VOX_LOGI("Font") << "Shaping: ligGlyphToCP=" << ligGlyphToCP.size()
                     << " glyphToCP=" << glyphToCP.size()
                     << " prescanLigGlyphs=" << ligOutGlyphIds.size()
                     << " (fi glyph=" << stbtt_FindGlyphIndex(&info, 0xFB01)
                     << " mapped=" << (ligGlyphToCP.count(static_cast<std::uint32_t>(stbtt_FindGlyphIndex(&info, 0xFB01))) > 0 ? "yes" : "no")
                     << ")";

    parseGpos(*m_shaping, m_shaping->ttfData.data(), static_cast<std::uint32_t>(fontOffset));
    parseGsub(*m_shaping, glyphToCP, ligGlyphToCP, m_shaping->ttfData.data(),
              static_cast<std::uint32_t>(fontOffset));

    VOX_LOGI("Font") << "Shaping: " << m_shaping->ligatures.size() << " liga rules, "
                     << m_shaping->kernPairs.size() << " kern pairs, "
                     << m_shaping->kernClasses.size() << " kern classes";

    return true;
}

bool Font::loadFromFile(const std::string& path, float pixelHeight, std::uint32_t atlasSize,
                        std::uint32_t firstCodepoint, std::uint32_t lastCodepoint) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        return false;
    }
    return loadFromMemory(bytes.data(), bytes.size(), pixelHeight, atlasSize, firstCodepoint, lastCodepoint);
}

}  // namespace odai::ui
