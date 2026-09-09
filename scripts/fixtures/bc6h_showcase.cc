// Synthetic BC6H material exhibit. No game assets and no alternate render path.
// Deliberately limited encoder: BC6H mode 11, two-endpoint 4x4 blocks.
// Layout reference: https://raw.githubusercontent.com/iOrange/bcdec/master/bcdec.h
#include "import/imported_scene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
using namespace odai::importer;
using V=std::array<float,3>;
float halfValue(unsigned h) { return (h>>10) ? std::ldexp(1.f+float(h&1023)/1024,int(h>>10)-15) : std::ldexp(float(h),-24); }
unsigned unquant(unsigned q) {return q==0?0:q==1023?65535:((q<<16)+32768)>>10;}
unsigned quant(float v) {
    static const auto table=[] {std::array<float,1024> t{};for(unsigned q=0;q<1024;++q)t[q]=halfValue((unquant(q)*31)>>6);return t;}();
    auto p=std::lower_bound(table.begin(),table.end(),v);unsigned q=p-table.begin();
    if(q==1024)return 1023;if(q && v-table[q-1]<table[q]-v)--q;return q;
}
V radiance(int f,float u,float v) {
    V d;
    switch(f) {case 0:d={1,-v,-u};break;case 1:d={-1,-v,u};break;
    case 2:d={u,1,v};break;case 3:d={u,-1,-v};break;
    case 4:d={u,-v,1};break;default:d={-u,-v,-1};}
    float l=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);for(auto&x:d)x/=l;
    // Z-up studio: broad cool fill, warm and cyan HDR softboxes.
    float warm=std::exp(-std::pow((d[0]+.40f)/.24f,8.f)-std::pow((d[2]-.48f)/.48f,8.f));
    float cool=std::exp(-std::pow((d[0]-.68f)/.12f,8.f)-std::pow((d[2]-.25f)/.65f,8.f));
    float top=std::pow(std::max(0.f,d[2]),5.f);
    return {.025f+warm*6.f+cool*.4f+top*.3f,
            .04f+warm*3.4f+cool*3.4f+top*.42f,
            .06f+warm*1.6f+cool*6.f+top*.7f};
}
ImportedSceneTexture cube(bool hdr) {
    ImportedSceneTexture t;t.sourcePath=hdr?"synthetic/studio_bc6h":"synthetic/studio_clamped";
    t.width=t.height=512;t.mipLevelCount=10;t.arrayLayers=6;t.linearData=true;
    t.format=hdr?TextureFormat::BC6HUfloat:TextureFormat::RGBA8;
    for(int f=0;f<6;++f)for(int n=512;n;n/=2) {
        int step=hdr?4:1;
        for(int y=0;y<n;y+=step)for(int x=0;x<n;x+=step) {
            V c{};
            for(int j=0;j<4;++j)for(int i=0;i<4;++i){
                V p=radiance(f,2*(x+(i+.5f)*step/4)/n-1,2*(y+(j+.5f)*step/4)/n-1);
                for(int k=0;k<3;++k)c[k]+=p[k]/16;
            }
            if(hdr){
                std::array<unsigned char,16>b{};unsigned bit=0;
                auto put=[&](unsigned value,unsigned count){for(unsigned i=0;i<count;++i,++bit)b[bit/8]|=((value>>i)&1)<<(bit%8);};
                std::array<V,16> pixels{}; V lo{1e9f,1e9f,1e9f},hi{};
                for(int j=0;j<4;++j)for(int i=0;i<4;++i){auto& p=pixels[j*4+i];
                    p=radiance(f,2*(x+i+.5f)/std::max(n,4)-1,2*(y+j+.5f)/std::max(n,4)-1);
                    if(n<4)p=c;
                    for(int k=0;k<3;++k){lo[k]=std::min(lo[k],p[k]);hi[k]=std::max(hi[k],p[k]);}}
                std::array<unsigned,3> qlo{},qhi{};for(int k=0;k<3;++k){qlo[k]=quant(lo[k]);qhi[k]=quant(hi[k]);}
                const unsigned weights[16]={0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64};
                auto indices=[&]{std::array<unsigned,16> indices{};for(int p=0;p<16;++p){float error=1e30f;
                    for(unsigned i=0;i<16;++i){float e=0;for(int k=0;k<3;++k){unsigned uq=((64-weights[i])*unquant(qlo[k])+weights[i]*unquant(qhi[k])+32)>>6;
                        float d=halfValue((uq*31)>>6)-pixels[p][k];e+=d*d;}if(e<error){error=e;indices[p]=i;}}}return indices;};
                auto ids=indices();if(ids[0]>=8){std::swap(qlo,qhi);ids=indices();}
                put(3,5);for(auto q:qlo)put(q,10);for(auto q:qhi)put(q,10);
                for(int p=0;p<16;++p)put(ids[p],p?4:3);
                t.rgba8.insert(t.rgba8.end(),b.begin(),b.end());
            }else{for(float v:c)t.rgba8.push_back(unsigned(std::clamp(v,0.f,1.f)*255+.5f));t.rgba8.push_back(255);}
        }
    }return t;
}
ImportedScene scene;
void vertex(V p,V n,unsigned mat,float u=0,float v=0){
    ImportedScenePackedVertex a;for(int k=0;k<3;++k){a.position[k]=p[k];a.normal[k]=n[k];a.color[k]=1;}
    a.uv[0]=u;a.uv[1]=v;a.textureIndex=0;
    scene.packedVertices.push_back(a);scene.packedLightingMaterialIndices.push_back(mat);
}
void sphere(float x,float y,float r,unsigned mat){
    unsigned base=scene.packedVertices.size(),first=scene.packedIndices.size();
    const int rings=64,slices=128;
    for(int j=0;j<=rings;++j)for(int i=0;i<=slices;++i){float a=3.14159265f*j/rings,b=6.2831853f*i/slices;
        V n={std::sin(a)*std::cos(b),std::cos(a),std::sin(a)*std::sin(b)};
        vertex({x+r*n[0],y+r*n[1],r*n[2]},n,mat,float(i)/slices,float(j)/rings);}
    for(int j=0;j<rings;++j)for(int i=0;i<slices;++i){unsigned a=base+j*(slices+1)+i,b=a+slices+1;
        for(auto idx:{a,a+1,b,b,a+1,b+1})scene.packedIndices.push_back(idx);}
    scene.packedDraws.push_back({first,unsigned(scene.packedIndices.size())-first});
}
void quad(V a,V b,V c,V d,V n,unsigned mat){unsigned base=scene.packedVertices.size(),first=scene.packedIndices.size();
    for(auto p:{a,b,c,d})vertex(p,n,mat);for(auto i:{0,1,2,0,2,3})scene.packedIndices.push_back(base+i);
    scene.packedDraws.push_back({first,6});}
int main(int argc,char**argv){if(argc!=2)return 1;scene.sourceTag="synthetic_bc6h_showcase";
    ImportedSceneTexture dark;dark.sourcePath="synthetic/charcoal";dark.width=dark.height=1;dark.rgba8={30,32,38,255};scene.textures.push_back(dark);
    scene.textures.push_back(cube(true));scene.textures.push_back(cube(false));
    for(int i=0;i<4;++i){ImportedNifLightingMaterial m;m.valid=1;m.shaderType=1;m.flags1=129;m.glossiness=i==1?128:1000000;
        m.environmentScale=i==3?0:.65f;m.textures[4]=i==2?2:1;m.specularStrength=.2f;scene.lightingMaterials.push_back(m);}
    sphere(-210,110,90,0);sphere(0,110,90,1);sphere(210,110,90,2);
    quad({-1200,20,1000},{1200,20,1000},{1200,20,-1000},{-1200,20,-1000},{0,1,0},3);
    quad({-1200,20,-200},{1200,20,-200},{1200,1200,-200},{-1200,1200,-200},{0,0,1},3);
    for(int k=0;k<3;++k){scene.boundsMin[k]=-1200;scene.boundsMax[k]=1200;}
    bool ok=saveImportedScene(scene,argv[1]);std::cout<<"BC6H studio radiance > 1, rough BC6H, clamped linear RGBA8 control; "<<scene.packedVertices.size()<<" vertices\n";return ok?0:1;
}
