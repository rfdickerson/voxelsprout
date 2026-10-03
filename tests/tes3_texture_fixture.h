#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace tes3_texture_fixture {
using Bytes = std::vector<std::uint8_t>;
template<class T> inline void pod(Bytes& b, const T& value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    b.insert(b.end(), p, p + sizeof(T));
}
inline void string(Bytes& b, const std::string& s) {
    pod(b, std::uint32_t(s.size())); b.insert(b.end(), s.begin(), s.end());
}
inline void objectNet(Bytes& b, const std::string& name = "") {
    string(b, name); pod(b, std::int32_t(-1)); pod(b, std::int32_t(-1));
}
inline void write(const std::filesystem::path& path, const Bytes& bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}
// Independent asymmetric RGB quadrant image: red/green above blue/yellow.
inline Bytes tga(unsigned w = 4, unsigned h = 4, bool top = true, bool right = false, bool rle = false) {
    Bytes b(18); b[2] = rle ? 10 : 2; b[12] = w & 255; b[13] = w >> 8;
    b[14] = h & 255; b[15] = h >> 8; b[16] = 32; b[17] = 8 | (top ? 32 : 0) | (right ? 16 : 0);
    for (unsigned sy = 0; sy < h; ++sy) for (unsigned sx = 0; sx < w; ++sx) {
        const auto x = right ? w - 1 - sx : sx, y = top ? sy : h - 1 - sy;
        const std::array<std::array<std::uint8_t, 4>, 4> bgra = {{{0,0,255,255},{0,255,0,255},{255,0,0,255},{0,255,255,255}}};
        if (rle) b.push_back(0); // one raw pixel packet: deliberately asymmetric
        const auto& c = bgra[(y >= h / 2) * 2 + (x >= w / 2)]; b.insert(b.end(), c.begin(), c.end());
    }
    return b;
}
inline Bytes bmp(unsigned w = 4, unsigned h = 4, bool top = false) {
    const auto stride = (w * 3 + 3) & ~3u;
    Bytes b(54 + stride * h);
    const auto word = [&](unsigned at, std::uint32_t value) { std::memcpy(b.data() + at, &value, 4); };
    b[0] = 'B'; b[1] = 'M'; word(2, b.size()); word(10, 54); word(14, 40); word(18, w);
    const auto height = std::int32_t(top ? -int(h) : int(h)); std::memcpy(b.data() + 22, &height, 4);
    b[26] = 1; b[28] = 24; word(34, stride * h);
    const auto pixels = tga(w, h, top);
    for (unsigned y = 0; y < h; ++y) for (unsigned x = 0; x < w; ++x)
        std::memcpy(b.data() + 54 + y * stride + x * 3, pixels.data() + 18 + (y * w + x) * 4, 3);
    return b;
}
inline Bytes bc1Dds(unsigned size) {
    unsigned levels = 1;
    for (unsigned s = size; s > 1; s /= 2) ++levels;
    Bytes b(128);
    const auto word = [&](unsigned at, std::uint32_t v) { std::memcpy(b.data()+at,&v,4); };
    word(0,0x20534444); word(4,124); word(8,0x81007); word(12,size); word(16,size);
    word(20,((size+3)/4)*((size+3)/4)*8); word(28,levels); word(76,32); word(80,4);
    word(84,0x31545844); word(108,0x401008);
    for(unsigned s=size,level=0; level<levels; ++level,s=std::max(1u,s/2)) {
        for(unsigned y=0;y<(s+3)/4;++y) for(unsigned x=0;x<(s+3)/4;++x) {
            const std::array<std::uint16_t,4> colors{0xf800,0x07e0,0x001f,0xffe0};
            const auto color=colors[(y*4>=s/2)*2+(x*4>=s/2)];
            pod(b,color); pod(b,std::uint16_t(0)); pod(b,std::uint32_t(0));
        }
    }
    return b;
}
// Independently encoded opaque red BC2/3/7 blocks, with optional authored
// rectangular/sub-block mips. BC7 uses mode 6 with equal endpoints.
inline Bytes redDds(unsigned format, bool authoredMips) {
    Bytes b=bc1Dds(8); b.resize(format==7 ? 148 : 128);
    const auto word=[&](unsigned at,std::uint32_t v){std::memcpy(b.data()+at,&v,4);};
    word(12,4);word(28,authoredMips ? 4 : 1);
    word(84,format==2 ? 0x33545844 : format==3 ? 0x35545844 : 0x30315844);
    if(format==7) {word(128,98);word(132,3);word(136,0);word(140,1);word(144,0);}
    for(unsigned level=0;level<(authoredMips ? 4u : 1u);++level) {
        const auto w=std::max(1u,8u>>level),h=std::max(1u,4u>>level);
        for(unsigned i=0;i<((w+3)/4)*((h+3)/4);++i) {
            Bytes block(16);
            if(format==7) {
                unsigned bit=0;
                const auto bits=[&](unsigned v,unsigned n){for(unsigned j=0;j<n;++j,++bit) block[bit/8]|=((v>>j)&1)<<(bit%8);};
                bits(64,7);
                for(unsigned value:{127,0,0,127}) {bits(value,7);bits(value,7);}
                bits(1,1);bits(1,1);bits(0,3);for(int j=1;j<16;++j) bits(0,4);
            } else {
                block[9]=248;
                if(format==2) std::fill(block.begin(),block.begin()+8,255);
                else block[0]=255;
            }
            b.insert(b.end(),block.begin(),block.end());
        }
    }
    return b;
}
inline Bytes nif(const std::string& texture = "quadrants.tga", unsigned clamp = 3, unsigned uvSet = 0,
                 bool cutout = false, float tint = 1.f, bool particles = false, bool blend = false) {
    Bytes shape; objectNet(shape, "texture-plane"); pod(shape, std::uint16_t(0));
    for (float v : {0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,1.f,0.f,0.f,0.f}) pod(shape,v);
    pod(shape, std::uint32_t(cutout || blend ? 3 : 2)); pod(shape, std::int32_t(2)); pod(shape, std::int32_t(4));
    if (cutout || blend) pod(shape, std::int32_t(5));
    pod(shape, std::uint32_t(0)); pod(shape, std::int32_t(1)); pod(shape, std::int32_t(-1));
    Bytes geometry; pod(geometry, std::uint16_t(4)); pod(geometry, std::uint32_t(1));
    for (float v : {-1.f,0.f,-1.f, 1.f,0.f,-1.f, 1.f,0.f,1.f, -1.f,0.f,1.f}) pod(geometry,v);
    pod(geometry,std::uint32_t(1));
    for (int i=0;i<4;++i) for(float v:{0.f,-1.f,0.f}) pod(geometry,v);
    for (int i=0;i<4;++i) pod(geometry,0.f);
    pod(geometry,std::uint32_t(1));
    for(int i=0;i<4;++i) for(float v:{1.f,1.f,1.f,1.f}) pod(geometry,v);
    pod(geometry,std::uint16_t(2)); pod(geometry,std::uint32_t(1));
    for(float v:{0.f,1.f,1.f,1.f,1.f,0.f,0.f,0.f}) pod(geometry,v);
    for(float v:{0.f,2.f,2.f,2.f,2.f,0.f,0.f,0.f}) pod(geometry,v);
    pod(geometry,std::uint16_t(2)); pod(geometry,std::uint32_t(6));
    for(std::uint16_t v:{0,2,1,0,3,2}) pod(geometry,v);
    pod(geometry,std::uint16_t(0));
    Bytes property; objectNet(property); pod(property,std::uint16_t(0)); pod(property,std::uint32_t(2));
    pod(property,std::uint32_t(1)); pod(property,std::uint32_t(1)); pod(property,std::int32_t(3));
    pod(property,std::uint32_t(clamp)); pod(property,std::uint32_t(2)); pod(property,std::uint32_t(uvSet));
    pod(property,std::uint32_t(0)); pod(property,std::uint16_t(0));
    Bytes source; objectNet(source); pod(source,std::uint8_t(1)); string(source,texture);
    for(int i=0;i<3;++i) pod(source,std::uint32_t(0));
    pod(source,std::uint8_t(1));
    Bytes material; objectNet(material); pod(material,std::uint16_t(0));
    for(float v:{1.f,1.f,1.f,tint,tint,tint,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f}) pod(material,v);
    std::vector<std::pair<std::string,Bytes>> blocks{{"NiTriShape",shape},{"NiTriShapeData",geometry},
        {"NiTexturingProperty",property},{"NiSourceTexture",source},{"NiMaterialProperty",material}};
    if(cutout || blend) { Bytes alpha; objectNet(alpha); pod(alpha,std::uint16_t(cutout ? 0x200 | (4<<10) : 0xed)); pod(alpha,std::uint8_t(128)); blocks.push_back({"NiAlphaProperty",alpha}); }
    if (particles) {
        blocks.push_back({"NiParticleSystemController", Bytes(154)});
        blocks.push_back({"NiGravity", Bytes(44)});
        blocks.push_back({"NiParticleGrowFade", Bytes(16)});
        blocks.push_back({"NiParticleColorModifier", Bytes(12)});
    }
    const std::string header="NetImmerse File Format, Version 4.0.0.2\n";
    Bytes out(header.begin(),header.end()); pod(out,std::uint32_t(0x04000002)); pod(out,std::uint32_t(blocks.size()));
    for(const auto& [type,data]:blocks) { string(out,type); out.insert(out.end(),data.begin(),data.end()); }
    pod(out,std::uint32_t(1)); pod(out,std::int32_t(0)); return out;
}
} // namespace tes3_texture_fixture
