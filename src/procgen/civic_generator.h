#pragma once

#include <cstdint>

#include "procgen/mesh_emit.h"

// Civic building generator: recognizable, seeded silhouettes for the city
// builder's municipal services. Separate from the era/zone building generator
// because the era/level/wealth axes don't apply — a fire station is a fire
// station in any decade; identity comes from silhouette and accent color.
//
// Same local-space contract as generateBuilding: the lot spans
// [0, lotWidth] x [0, lotDepth] in XZ, ground at y = 0, street on the -Z side.
namespace odai::procgen {

enum class CivicKind : std::uint8_t {
    Police,        // flat-roof block + entry canopy + antenna mast
    Fire,          // garage doors on the street face + hose tower
    Clinic,        // white L-mass + rooftop cross sign
    School,        // gabled hall + lower wing + flagpole
    Park,          // gazebo or fountain centerpiece (small; sits among trees)
    Library,       // colonnade front + stepped entry
    Amphitheater,  // tiered seating stepping down to a stage (stays low)
    PowerPlant,    // turbine hall + smokestacks
};

struct CivicDesc {
    CivicKind kind = CivicKind::Police;
    float lotWidth = 1.8f;   // world units (2x2 tile footprint minus pad)
    float lotDepth = 1.8f;
    std::uint32_t seed = 0;  // same desc => bit-identical mesh
};

TriMesh generateCivicBuilding(const CivicDesc& desc);

}  // namespace odai::procgen
