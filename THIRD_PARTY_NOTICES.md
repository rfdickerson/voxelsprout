# Third-party notices

ODAI source is MIT licensed; see LICENSE. Dependencies retain their own licenses.
The vcpkg manifest and baseline identify the dependency set. Installed packages
include vcpkg port copyright/license texts under `share/doc/odai/dependencies`.
Inter fonts include their SIL Open Font License in `assets/fonts/LICENSE-Inter.txt`.

Direct dependencies include GLFW, Dear ImGui, Jolt Physics, miniaudio,
nlohmann/json, stb, Vulkan headers/loader, Vulkan Memory Allocator and zlib.
Slang is a build tool used to compile the engine’s own shaders. Optional XeSS
requires a separately supplied SDK; it is not included in the Linux package.

The host supplies glibc, the Vulkan loader and GPU drivers. Review the staged
package’s resolved shared libraries and associated notices before publishing.
No Bethesda or mod-author assets are licensed by ODAI’s MIT license or included
in the binary package. See docs/ASSET_PROVENANCE.md for unresolved source-history
items; a dependency inventory is not a completed provenance review.
