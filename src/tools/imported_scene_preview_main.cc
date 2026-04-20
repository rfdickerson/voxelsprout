#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kLandSize = 65;
constexpr float kCellSizeUnits = 8192.0f;

enum class PreviewMode {
    TopDown,
    Isometric
};

struct PreviewBounds {
    float minX = 0.0f;
    float maxX = 0.0f;
    float minZ = 0.0f;
    float maxZ = 0.0f;
    float minHeight = 0.0f;
    float maxHeight = 0.0f;
};

struct ImageRgba {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

void printUsage() {
    std::cerr
        << "Usage: odai_imported_scene_preview <scene input path> <preview bmp output path> [width] [topdown|isometric]\n";
}

std::uint8_t clampToByte(float value) {
    const float clamped = std::clamp(value, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(clamped + 0.5f);
}

std::uint32_t landCellKey(int gridX, int gridY) {
    return (static_cast<std::uint32_t>(gridX) << 16u) ^ static_cast<std::uint32_t>(gridY & 0xffff);
}

PreviewBounds computeBounds(const odai::importer::ImportedScene& scene) {
    PreviewBounds bounds{};
    bounds.minX = std::numeric_limits<float>::max();
    bounds.maxX = -std::numeric_limits<float>::max();
    bounds.minZ = std::numeric_limits<float>::max();
    bounds.maxZ = -std::numeric_limits<float>::max();
    bounds.minHeight = std::numeric_limits<float>::max();
    bounds.maxHeight = -std::numeric_limits<float>::max();

    for (const auto& cell : scene.landscapeCells) {
        const float cellMinX = static_cast<float>(cell.gridX) * kCellSizeUnits;
        const float cellMinZ = static_cast<float>(cell.gridY) * kCellSizeUnits;
        bounds.minX = std::min(bounds.minX, cellMinX);
        bounds.maxX = std::max(bounds.maxX, cellMinX + kCellSizeUnits);
        bounds.minZ = std::min(bounds.minZ, cellMinZ);
        bounds.maxZ = std::max(bounds.maxZ, cellMinZ + kCellSizeUnits);
        for (const float height : cell.heights) {
            bounds.minHeight = std::min(bounds.minHeight, height);
            bounds.maxHeight = std::max(bounds.maxHeight, height);
        }
    }

    if (scene.landscapeCells.empty()) {
        bounds.minX = 0.0f;
        bounds.maxX = 1.0f;
        bounds.minZ = 0.0f;
        bounds.maxZ = 1.0f;
        bounds.minHeight = 0.0f;
        bounds.maxHeight = 1.0f;
    } else if (bounds.maxHeight <= bounds.minHeight) {
        bounds.maxHeight = bounds.minHeight + 1.0f;
    }

    return bounds;
}

float sampleLandHeight(
    const odai::importer::ImportedSceneLandscapeCell& cell,
    float localX,
    float localZ
) {
    if (cell.heights.size() != static_cast<std::size_t>(kLandSize * kLandSize)) {
        return 0.0f;
    }
    const float gridX = std::clamp(localX * static_cast<float>(kLandSize - 1), 0.0f, static_cast<float>(kLandSize - 1));
    const float gridZ = std::clamp(localZ * static_cast<float>(kLandSize - 1), 0.0f, static_cast<float>(kLandSize - 1));

    const int x0 = static_cast<int>(gridX);
    const int z0 = static_cast<int>(gridZ);
    const int x1 = std::min(x0 + 1, kLandSize - 1);
    const int z1 = std::min(z0 + 1, kLandSize - 1);
    const float tx = gridX - static_cast<float>(x0);
    const float tz = gridZ - static_cast<float>(z0);

    const auto sample = [&](int x, int z) {
        return cell.heights[static_cast<std::size_t>(z * kLandSize + x)];
    };

    const float h00 = sample(x0, z0);
    const float h10 = sample(x1, z0);
    const float h01 = sample(x0, z1);
    const float h11 = sample(x1, z1);
    const float hx0 = h00 + ((h10 - h00) * tx);
    const float hx1 = h01 + ((h11 - h01) * tx);
    return hx0 + ((hx1 - hx0) * tz);
}

std::array<std::uint8_t, 3> terrainColor(float normalizedHeight) {
    const float t = std::clamp(normalizedHeight, 0.0f, 1.0f);
    const float lowR = 56.0f;
    const float lowG = 84.0f;
    const float lowB = 72.0f;
    const float highR = 194.0f;
    const float highG = 176.0f;
    const float highB = 132.0f;
    return {
        clampToByte(lowR + ((highR - lowR) * t)),
        clampToByte(lowG + ((highG - lowG) * t)),
        clampToByte(lowB + ((highB - lowB) * t))
    };
}

void setPixel(ImageRgba& image, int x, int y, const std::array<std::uint8_t, 4>& color) {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>((y * image.width + x) * 4);
    image.pixels[index + 0] = color[0];
    image.pixels[index + 1] = color[1];
    image.pixels[index + 2] = color[2];
    image.pixels[index + 3] = color[3];
}

void drawDot(ImageRgba& image, int centerX, int centerY, int radius, const std::array<std::uint8_t, 4>& color) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if ((dx * dx) + (dy * dy) > (radius * radius)) {
                continue;
            }
            setPixel(image, centerX + dx, centerY + dy, color);
        }
    }
}

void blendPixel(ImageRgba& image, int x, int y, const std::array<std::uint8_t, 4>& color);

void drawLine(
    ImageRgba& image,
    int x0,
    int y0,
    int x1,
    int y1,
    const std::array<std::uint8_t, 4>& color
) {
    int dx = std::abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        blendPixel(image, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int e2 = err * 2;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

std::array<float, 3> transformPoint(
    const float matrix[16],
    const std::array<float, 3>& point
) {
    return {
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11]
    };
}

bool computeMeshBounds(
    const odai::importer::ImportedSceneMesh& mesh,
    std::array<float, 3>& outMin,
    std::array<float, 3>& outMax
) {
    if (mesh.vertices.empty()) {
        return false;
    }
    outMin = {
        mesh.vertices.front().position[0],
        mesh.vertices.front().position[1],
        mesh.vertices.front().position[2]
    };
    outMax = outMin;
    for (const auto& vertex : mesh.vertices) {
        outMin[0] = std::min(outMin[0], vertex.position[0]);
        outMin[1] = std::min(outMin[1], vertex.position[1]);
        outMin[2] = std::min(outMin[2], vertex.position[2]);
        outMax[0] = std::max(outMax[0], vertex.position[0]);
        outMax[1] = std::max(outMax[1], vertex.position[1]);
        outMax[2] = std::max(outMax[2], vertex.position[2]);
    }
    return true;
}

void blendPixel(ImageRgba& image, int x, int y, const std::array<std::uint8_t, 4>& color) {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>((y * image.width + x) * 4);
    const float srcAlpha = static_cast<float>(color[3]) / 255.0f;
    const float dstAlpha = static_cast<float>(image.pixels[index + 3]) / 255.0f;
    const float outAlpha = srcAlpha + (dstAlpha * (1.0f - srcAlpha));
    auto blendChannel = [&](int channel) {
        const float src = static_cast<float>(color[channel]);
        const float dst = static_cast<float>(image.pixels[index + channel]);
        const float out = (src * srcAlpha) + (dst * (1.0f - srcAlpha));
        image.pixels[index + channel] = clampToByte(out);
    };
    blendChannel(0);
    blendChannel(1);
    blendChannel(2);
    image.pixels[index + 3] = clampToByte(outAlpha * 255.0f);
}

bool writeBmp(const std::filesystem::path& outputPath, const ImageRgba& image) {
    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            std::cerr << "Failed to create preview output directory " << parentPath << ": " << mkdirError.message() << "\n";
            return false;
        }
    }

    const std::uint32_t rowStride = static_cast<std::uint32_t>(image.width * 4);
    const std::uint32_t pixelDataSize = rowStride * static_cast<std::uint32_t>(image.height);
    const std::uint32_t fileSize = 14u + 40u + pixelDataSize;

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "Failed to open preview BMP for writing: " << outputPath << "\n";
        return false;
    }

    const std::uint16_t fileType = 0x4D42u;
    const std::uint32_t reserved = 0u;
    const std::uint32_t pixelOffset = 14u + 40u;
    output.write(reinterpret_cast<const char*>(&fileType), sizeof(fileType));
    output.write(reinterpret_cast<const char*>(&fileSize), sizeof(fileSize));
    output.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    output.write(reinterpret_cast<const char*>(&pixelOffset), sizeof(pixelOffset));

    const std::uint32_t dibSize = 40u;
    const std::int32_t bmpWidth = image.width;
    const std::int32_t bmpHeight = -image.height;
    const std::uint16_t planes = 1u;
    const std::uint16_t bitsPerPixel = 32u;
    const std::uint32_t compression = 0u;
    const std::uint32_t ppm = 2835u;
    const std::uint32_t colorsUsed = 0u;
    const std::uint32_t colorsImportant = 0u;
    output.write(reinterpret_cast<const char*>(&dibSize), sizeof(dibSize));
    output.write(reinterpret_cast<const char*>(&bmpWidth), sizeof(bmpWidth));
    output.write(reinterpret_cast<const char*>(&bmpHeight), sizeof(bmpHeight));
    output.write(reinterpret_cast<const char*>(&planes), sizeof(planes));
    output.write(reinterpret_cast<const char*>(&bitsPerPixel), sizeof(bitsPerPixel));
    output.write(reinterpret_cast<const char*>(&compression), sizeof(compression));
    output.write(reinterpret_cast<const char*>(&pixelDataSize), sizeof(pixelDataSize));
    output.write(reinterpret_cast<const char*>(&ppm), sizeof(ppm));
    output.write(reinterpret_cast<const char*>(&ppm), sizeof(ppm));
    output.write(reinterpret_cast<const char*>(&colorsUsed), sizeof(colorsUsed));
    output.write(reinterpret_cast<const char*>(&colorsImportant), sizeof(colorsImportant));

    std::vector<std::uint8_t> bgra(pixelDataSize, 0u);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const std::size_t srcIndex = static_cast<std::size_t>((y * image.width + x) * 4);
            const std::size_t dstIndex = srcIndex;
            bgra[dstIndex + 0] = image.pixels[srcIndex + 2];
            bgra[dstIndex + 1] = image.pixels[srcIndex + 1];
            bgra[dstIndex + 2] = image.pixels[srcIndex + 0];
            bgra[dstIndex + 3] = image.pixels[srcIndex + 3];
        }
    }

    output.write(reinterpret_cast<const char*>(bgra.data()), static_cast<std::streamsize>(bgra.size()));
    return output.good();
}

ImageRgba buildPreviewImage(const odai::importer::ImportedScene& scene, int width) {
    const PreviewBounds bounds = computeBounds(scene);
    const float spanX = std::max(bounds.maxX - bounds.minX, 1.0f);
    const float spanZ = std::max(bounds.maxZ - bounds.minZ, 1.0f);
    const float aspect = spanZ / spanX;
    const int height = std::max(1, static_cast<int>(static_cast<float>(width) * aspect));

    ImageRgba image{};
    image.width = width;
    image.height = height;
    image.pixels.assign(static_cast<std::size_t>(width * height * 4), 255u);

    std::unordered_map<std::uint32_t, const odai::importer::ImportedSceneLandscapeCell*> cellByKey;
    for (const auto& cell : scene.landscapeCells) {
        cellByKey.emplace(landCellKey(cell.gridX, cell.gridY), &cell);
    }

    for (int py = 0; py < image.height; ++py) {
        for (int px = 0; px < image.width; ++px) {
            const float worldX = bounds.minX + (spanX * (static_cast<float>(px) + 0.5f) / static_cast<float>(image.width));
            const float worldZ = bounds.minZ + (spanZ * (static_cast<float>(py) + 0.5f) / static_cast<float>(image.height));
            const int gridX = static_cast<int>(std::floor(worldX / kCellSizeUnits));
            const int gridY = static_cast<int>(std::floor(worldZ / kCellSizeUnits));
            const auto cellIt = cellByKey.find(landCellKey(gridX, gridY));
            if (cellIt == cellByKey.end()) {
                setPixel(image, px, py, {18u, 22u, 28u, 255u});
                continue;
            }

            const auto& cell = *cellIt->second;
            const float cellOriginX = static_cast<float>(cell.gridX) * kCellSizeUnits;
            const float cellOriginZ = static_cast<float>(cell.gridY) * kCellSizeUnits;
            const float localX = (worldX - cellOriginX) / kCellSizeUnits;
            const float localZ = (worldZ - cellOriginZ) / kCellSizeUnits;
            const float heightValue = sampleLandHeight(cell, localX, localZ);
            const float normalized = (heightValue - bounds.minHeight) / (bounds.maxHeight - bounds.minHeight);
            const auto rgb = terrainColor(normalized);
            setPixel(image, px, py, {rgb[0], rgb[1], rgb[2], 255u});
        }
    }

    for (const auto& ref : scene.unresolvedRefs) {
        const int px = static_cast<int>(((ref.position[0] - bounds.minX) / spanX) * static_cast<float>(image.width));
        const int py = static_cast<int>(((ref.position[2] - bounds.minZ) / spanZ) * static_cast<float>(image.height));
        const bool hasModel = !ref.modelPath.empty();
        drawDot(
            image,
            px,
            py,
            hasModel ? 2 : 1,
            hasModel ? std::array<std::uint8_t, 4>{220u, 72u, 72u, 255u}
                     : std::array<std::uint8_t, 4>{250u, 214u, 92u, 255u}
        );
    }

    return image;
}

ImageRgba buildIsometricPreviewImage(const odai::importer::ImportedScene& scene, int width) {
    const PreviewBounds bounds = computeBounds(scene);
    const float centerX = 0.5f * (bounds.minX + bounds.maxX);
    const float centerZ = 0.5f * (bounds.minZ + bounds.maxZ);
    const float heightSpan = std::max(bounds.maxHeight - bounds.minHeight, 1.0f);
    const float worldSpan = std::max(bounds.maxX - bounds.minX, bounds.maxZ - bounds.minZ);
    const int height = std::max(1, width * 3 / 4);

    ImageRgba image{};
    image.width = width;
    image.height = height;
    image.pixels.assign(static_cast<std::size_t>(width * height * 4), 255u);
    for (int y = 0; y < image.height; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(std::max(image.height - 1, 1));
        const std::array<std::uint8_t, 4> sky{
            clampToByte(32.0f + (48.0f * (1.0f - t))),
            clampToByte(44.0f + (62.0f * (1.0f - t))),
            clampToByte(58.0f + (90.0f * (1.0f - t))),
            255u
        };
        for (int x = 0; x < image.width; ++x) {
            setPixel(image, x, y, sky);
        }
    }

    struct ProjectedPoint {
        float depth = 0.0f;
        float screenX = 0.0f;
        float screenY = 0.0f;
        std::array<std::uint8_t, 4> color{};
        int radius = 1;
    };

    std::vector<ProjectedPoint> points;
    points.reserve(scene.landscapeCells.size() * kLandSize * kLandSize);
    const float scale = (static_cast<float>(width) * 0.82f) / std::max(worldSpan, 1.0f);
    const float verticalScale = scale * 0.40f;
    const float originX = static_cast<float>(width) * 0.5f;
    const float originY = static_cast<float>(height) * 0.82f;

    auto project = [&](float worldX, float worldY, float worldZ) {
        const float relX = worldX - centerX;
        const float relZ = worldZ - centerZ;
        const float isoX = (relX - relZ) * 0.8660254f;
        const float isoY = ((relX + relZ) * 0.5f) - ((worldY - bounds.minHeight) * (heightSpan > 0.0f ? (worldSpan / heightSpan) * 0.12f : 0.0f));
        return std::array<float, 2>{
            originX + (isoX * scale),
            originY + (isoY * verticalScale)
        };
    };

    for (const auto& cell : scene.landscapeCells) {
        if (cell.heights.size() != static_cast<std::size_t>(kLandSize * kLandSize)) {
            continue;
        }
        const float cellOriginX = static_cast<float>(cell.gridX) * kCellSizeUnits;
        const float cellOriginZ = static_cast<float>(cell.gridY) * kCellSizeUnits;
        const float step = kCellSizeUnits / static_cast<float>(kLandSize - 1);
        for (int z = 0; z < kLandSize; ++z) {
            for (int x = 0; x < kLandSize; ++x) {
                const std::size_t index = static_cast<std::size_t>(z * kLandSize + x);
                const float worldX = cellOriginX + (static_cast<float>(x) * step);
                const float worldY = cell.heights[index];
                const float worldZ = cellOriginZ + (static_cast<float>(z) * step);
                const float normalized = (worldY - bounds.minHeight) / heightSpan;
                const auto rgb = terrainColor(normalized);
                const auto screen = project(worldX, worldY, worldZ);
                points.push_back(ProjectedPoint{
                    worldX + worldZ + (worldY * 0.35f),
                    screen[0],
                    screen[1],
                    {rgb[0], rgb[1], rgb[2], 220u},
                    1
                });
            }
        }
    }

    for (const auto& ref : scene.unresolvedRefs) {
        const auto screen = project(ref.position[0], ref.position[1] + 128.0f, ref.position[2]);
        const bool hasModel = !ref.modelPath.empty();
        points.push_back(ProjectedPoint{
            ref.position[0] + ref.position[2] + (ref.position[1] * 0.35f) + 2048.0f,
            screen[0],
            screen[1],
            hasModel ? std::array<std::uint8_t, 4>{232u, 98u, 78u, 255u}
                     : std::array<std::uint8_t, 4>{240u, 206u, 90u, 255u},
            hasModel ? 2 : 1
        });
    }

    std::sort(points.begin(), points.end(), [](const ProjectedPoint& lhs, const ProjectedPoint& rhs) {
        return lhs.depth < rhs.depth;
    });

    for (const ProjectedPoint& point : points) {
        const int px = static_cast<int>(std::lround(point.screenX));
        const int py = static_cast<int>(std::lround(point.screenY));
        for (int dy = -point.radius; dy <= point.radius; ++dy) {
            for (int dx = -point.radius; dx <= point.radius; ++dx) {
                if ((dx * dx) + (dy * dy) > (point.radius * point.radius)) {
                    continue;
                }
                blendPixel(image, px + dx, py + dy, point.color);
            }
        }
    }

    static constexpr std::array<std::array<int, 2>, 12> kBoxEdges{{
        {{0, 1}}, {{1, 3}}, {{3, 2}}, {{2, 0}},
        {{4, 5}}, {{5, 7}}, {{7, 6}}, {{6, 4}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}
    }};

    for (const auto& instance : scene.instances) {
        if (instance.meshIndex >= scene.meshes.size()) {
            continue;
        }
        const auto& mesh = scene.meshes[instance.meshIndex];
        if (mesh.name == "terrain") {
            continue;
        }
        std::array<float, 3> localMin{};
        std::array<float, 3> localMax{};
        if (!computeMeshBounds(mesh, localMin, localMax)) {
            continue;
        }
        const std::array<std::array<float, 3>, 8> corners{{
            {{localMin[0], localMin[1], localMin[2]}},
            {{localMax[0], localMin[1], localMin[2]}},
            {{localMin[0], localMin[1], localMax[2]}},
            {{localMax[0], localMin[1], localMax[2]}},
            {{localMin[0], localMax[1], localMin[2]}},
            {{localMax[0], localMax[1], localMin[2]}},
            {{localMin[0], localMax[1], localMax[2]}},
            {{localMax[0], localMax[1], localMax[2]}}
        }};
        std::array<std::array<float, 2>, 8> screenCorners{};
        bool valid = true;
        for (std::size_t i = 0; i < corners.size(); ++i) {
            const auto world = transformPoint(instance.transform, corners[i]);
            if (!std::isfinite(world[0]) || !std::isfinite(world[1]) || !std::isfinite(world[2])) {
                valid = false;
                break;
            }
            screenCorners[i] = project(world[0], world[1], world[2]);
        }
        if (!valid) {
            continue;
        }
        const std::array<std::uint8_t, 4> color{172u, 226u, 240u, 224u};
        for (const auto& edge : kBoxEdges) {
            drawLine(
                image,
                static_cast<int>(std::lround(screenCorners[edge[0]][0])),
                static_cast<int>(std::lround(screenCorners[edge[0]][1])),
                static_cast<int>(std::lround(screenCorners[edge[1]][0])),
                static_cast<int>(std::lround(screenCorners[edge[1]][1])),
                color);
        }
    }

    return image;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        printUsage();
        return 1;
    }

    const std::filesystem::path inputPath = argv[1];
    const std::filesystem::path outputPath = argv[2];
    const int width = (argc >= 4) ? std::max(256, std::atoi(argv[3])) : 1600;
    const std::string modeString = (argc >= 5) ? argv[4] : "topdown";
    PreviewMode mode = PreviewMode::TopDown;
    if (modeString == "isometric") {
        mode = PreviewMode::Isometric;
    } else if (modeString != "topdown") {
        printUsage();
        return 1;
    }

    odai::importer::ImportedScene scene{};
    if (!odai::importer::loadImportedScene(inputPath, scene)) {
        std::cerr << "Failed to load imported scene from " << inputPath << "\n";
        return 2;
    }

    std::cerr << "[scene preview] Loaded scene with "
              << scene.landscapeCells.size() << " landscape cells and "
              << scene.instances.size() << " imported instances, "
              << scene.unresolvedRefs.size() << " unresolved refs\n";

    const ImageRgba image =
        mode == PreviewMode::Isometric
            ? buildIsometricPreviewImage(scene, width)
            : buildPreviewImage(scene, width);
    std::cerr << "[scene preview] Built " << image.width << "x" << image.height << " "
              << (mode == PreviewMode::Isometric ? "isometric" : "top-down") << " preview\n";

    if (!writeBmp(outputPath, image)) {
        return 3;
    }

    std::cout << "Wrote preview BMP: " << outputPath << "\n";
    return 0;
}
