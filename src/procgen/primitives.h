#pragma once

#include <array>
#include <vector>

#include "procgen/csg.h"

// Closed convex solid primitives for the CSG toolkit. All faces are convex
// polygons wound CCW from outside (right-handed, Y-up).
namespace odai::procgen {

CsgMesh makeBox(const Vector3& minCorner, const Vector3& maxCorner, const Color3& color);

// Gable prism over the XZ rectangle [minX,maxX] x [minZ,maxZ]: walls rise from
// y0 to eaveY, roof planes meet at a ridge at ridgeY running along X. Use
// rotateY for a Z-aligned ridge.
CsgMesh makeGablePrism(float minX, float minZ, float maxX, float maxZ,
                       float y0, float eaveY, float ridgeY, const Color3& color);

// Low-poly upright cylinder; the n-gon caps stay single convex polygons.
CsgMesh makeCylinder(const Vector3& baseCenter, float radius, float height,
                     int segments, const Color3& color);

// Extrude a convex CCW footprint (x,z pairs, CCW seen from +Y) from y0 to y1.
// taperTopScale scales the top ring about the footprint centroid (1 = straight
// walls); must be > 0.
CsgMesh makeConvexPrism(const std::vector<std::array<float, 2>>& footprintCcw,
                        float y0, float y1, const Color3& color,
                        float taperTopScale = 1.0f);

}  // namespace odai::procgen
