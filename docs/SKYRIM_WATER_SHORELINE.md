# Skyrim water and shoreline corrections

September 8, 2026. The existing imported water path remains in use.

- Water depth reconstruction now removes projection jitter, matching the depth
  prepass instead of moving the reconstructed riverbed every TAA phase.
- Refraction keeps the known, undistorted opaque riverbed as its attenuated
  fallback. A ray that leaves the viewport or fails at a silhouette no longer
  swaps that bottom for an unrelated opaque water-color patch.
- Shoreline coverage fades over a small depth interval scaled to FP16 precision.
  It composites against the existing pre-water HDR image after water fog. Unknown
  depth remains deep water; foreground ground remains dry. No extra render target
  or blend pass was added.
- Water with imported WATR appearance no longer receives the legacy green shore
  wash, exposure-stabilizing color grade, or hard luminance ceiling. Authored
  normals, colors, fog, and reflection parameters remain active. Other water keeps
  its existing fallback grading.

Both raster and ray-query shader variants compile. Focused imported rendering
policy tests cover missing depth, dry foreground, exact contact, continuous
shallow coverage, deep water, and distance-dependent depth precision. TAA policy
tests also pass. Daylight Riverwood validation reported no synchronization errors
or renderer-owned image leaks. No resource bindings or GPU dependencies changed;
the explicit ordering was checked against the
[Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).

Local comparisons: `captures/riverwood-water-before.*` and
`captures/riverwood-water-after.*`, 1536×864, fixed mill camera yaw 165/pitch -18.
The interactive window remains 768×432 logical points with a native 1536×864
framebuffer on the 2× display.

Final native-resolution run without validation: 40.56 fps average, p50 23.96 ms,
p95 24.93 ms, p99 25.66 ms; an initial streaming outlier reached 682 ms. The
60-fps target is not met in this water-heavy view at native DPI. These numbers
must not be compared directly to the earlier 768×432-pixel performance profile.
Dusk synchronization validation also completed without errors or image leaks.

This corrects confirmed rendering defects, not demonstrated retail parity.
Matched Skyrim SE reference footage, underwater presentation, screen-edge
reflection behavior, and temporal specular shimmer remain comparison work.
Screen-space refraction still cannot recover occluded or off-screen geometry.
