#pragma once

#include "core/frame_profiler.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <vector>

// Wall-clock collection for the headless sweep harnesses (odai_civ_sim,
// odai_stellaris_sim).
//
// Those sweeps already replay N deterministic seeded matches with no renderer
// and no window -- structurally they were already a CPU benchmark, they just
// never timed anything. This turns the same run into a throughput measurement
// without touching what the balance metrics report, so one command answers both
// "did the game get less fun" and "did the turn loop get slower".
//
// Report timings only from an optimized build: a Debug run of this tree is
// roughly 8x off (see "Optimized builds" in CLAUDE.md), which is why report()
// refuses to print a number without saying so when NDEBUG is absent.
namespace odai::tools {

class SimBench {
public:
    void addWorldgenMs(float ms) { m_worldgenMs.push_back(ms); }
    void addMatchMs(float ms) { m_matchMs.push_back(ms); }

    [[nodiscard]] std::size_t matchCount() const { return m_matchMs.size(); }

    void report(std::ostream& out, int turnsPerMatch) const {
        if (m_matchMs.empty() || turnsPerMatch <= 0) {
            return;
        }

        const double matchTotal = sum(m_matchMs);
        const double worldgenTotal = sum(m_worldgenMs);
        const double matches = static_cast<double>(m_matchMs.size());
        const double totalTurns = matches * static_cast<double>(turnsPerMatch);
        const double wallSeconds = (matchTotal + worldgenTotal) / 1000.0;

        const std::ios_base::fmtflags saved = out.flags();
        const std::streamsize savedPrecision = out.precision();

        out << "\n---- timing ----\n";
#ifndef NDEBUG
        // Release and RelWithDebInfo both define NDEBUG, so in a CMake build its
        // absence means Debug. Measured on this harness: ~5.8x slower.
        out << "  WARNING: built without NDEBUG (assertions live) -- for a CMake build\n"
               "           that means Debug, and these numbers measure the compiler more\n"
               "           than the simulation. Rebuild with the vcpkg-release preset.\n";
#endif
        out << std::fixed << std::setprecision(2);
        if (!m_worldgenMs.empty()) {
            out << "  worldgen   : mean " << (worldgenTotal / static_cast<double>(m_worldgenMs.size()))
                << " ms\n";
        }
        out << "  match      : mean " << (matchTotal / matches)
            << " ms   median " << percentile(m_matchMs, 0.50f)
            << "   p95 " << percentile(m_matchMs, 0.95f)
            << "   max " << percentile(m_matchMs, 1.0f) << "\n";
        out << "  turn       : mean " << ((matchTotal * 1000.0) / totalTurns) << " us\n";
        out << "  throughput : " << std::setprecision(0) << (totalTurns / (matchTotal / 1000.0))
            << " turns/sec   (" << m_matchMs.size() << " matches x " << turnsPerMatch
            // Three decimals so a fast sweep reads "0.002 s" instead of "0.00 s".
            << " turns in " << std::setprecision(3) << wallSeconds << " s)\n";

        out.flags(saved);
        out.precision(savedPrecision);
    }

private:
    static double sum(const std::vector<float>& values) {
        double total = 0.0;
        for (float v : values) total += static_cast<double>(v);
        return total;
    }

    // Nearest-rank, matching core::percentile's ceil-based index so numbers
    // reported here line up with the ones the renderer overlay reports.
    static double percentile(const std::vector<float>& values, float percentile01) {
        if (values.empty()) return 0.0;
        std::vector<float> scratch = values;
        const float clamped = std::clamp(percentile01, 0.0f, 1.0f);
        const auto targetIndex = static_cast<std::size_t>(
            std::ceil(clamped * static_cast<float>(scratch.size() - 1u)));
        auto first = scratch.begin();
        auto nth = first + static_cast<std::ptrdiff_t>(targetIndex);
        std::nth_element(first, nth, scratch.end());
        return static_cast<double>(*nth);
    }

    std::vector<float> m_worldgenMs;
    std::vector<float> m_matchMs;
};

}  // namespace odai::tools
