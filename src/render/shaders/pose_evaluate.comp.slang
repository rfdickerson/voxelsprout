// Scalar word ABI is defined by anim/gpu_pose.cc. Matrix words are row-major;
// mathematical matrices are explicitly constructed before storage.
[[vk::binding(0,0)]] StructuredBuffer<uint> rig;
[[vk::binding(1,0)]] StructuredBuffer<uint> frame;
[[vk::binding(2,0)]] RWStructuredBuffer<float4> scratch;
[[vk::binding(3,0)]] RWStructuredBuffer<float4x4> palette;
[[vk::binding(4,0)]] RWStructuredBuffer<float4x4> previousPalette;
[[vk::binding(5,0)]] RWStructuredBuffer<float4x4> historyPalette;
struct Push { uint bones; uint nodes; uint output; uint phase; uint depth; uint pad0; uint pad1; uint pad2; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
struct Pose { float4 t; float4 q; float4 s; };
float4 qmul(float4 a,float4 b){return normalize(float4(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz)));}
float4 qinv(float4 q){return float4(-q.xyz,q.w);}
float4 qslerp(float4 a,float4 b,float t){
    float d=dot(a,b);if(d<0){b=-b;d=-d;}
    if(d>.9995)return normalize(lerp(a,b,t));
    float angle=acos(clamp(d,-1.0,1.0)),s=sin(angle);
    return a*(sin((1-t)*angle)/s)+b*(sin(t*angle)/s);
}
float4 value(uint offset){return float4(asfloat(rig[offset+1]),asfloat(rig[offset+2]),asfloat(rig[offset+3]),asfloat(rig[offset+4]));}
float4 channel(uint offset,uint count,float time,float duration,bool loop,bool rotation,float4 fallback){
    if(count==0)return fallback;
    if(count==1)return value(offset);
    uint a=0,b=count-1;float ta=asfloat(rig[offset]),tb=asfloat(rig[offset+b*8]);
    if(time<ta || time>tb){
        if(!loop)return time<ta?value(offset):value(offset+b*8);
        a=count-1;b=0;float first=ta;ta=tb;tb=duration+first;if(time<first)time+=duration;
    }else{
        // Upper bound on key times, matching the CPU's first nonzero interval.
        uint lo=0,hi=count-1;
        while(lo+1<hi){uint mid=(lo+hi)/2;if(asfloat(rig[offset+mid*8])<time)lo=mid;else hi=mid;}
        a=lo;b=hi;ta=asfloat(rig[offset+a*8]);tb=asfloat(rig[offset+b*8]);
    }
    float f=tb>ta?(time-ta)/(tb-ta):0;
    return rotation?qslerp(value(offset+a*8),value(offset+b*8),f):lerp(value(offset+a*8),value(offset+b*8),f);
}
Pose rest(uint bone){uint b=bone*32;Pose p;p.t=float4(asfloat(rig[b]),asfloat(rig[b+1]),asfloat(rig[b+2]),0);p.q=float4(asfloat(rig[b+4]),asfloat(rig[b+5]),asfloat(rig[b+6]),asfloat(rig[b+7]));p.s=float4(asfloat(rig[b+8]),asfloat(rig[b+9]),asfloat(rig[b+10]),0);return p;}
Pose sample(uint clip,uint bone,float time){
    float duration=asfloat(rig[clip]);bool loop=rig[clip+1]!=0,additive=rig[clip+2]!=0;
    time=loop?time-floor(time/duration)*duration:clamp(time,0.0,duration);
    Pose bind=rest(bone),p=bind;if(additive){p.t=0;p.q=float4(0,0,0,1);p.s=float4(1,1,1,0);}
    uint b=clip+4+bone*6;
    p.t=channel(rig[b],rig[b+1],time,duration,loop,false,p.t);
    if(rig[clip+3]==bone && rig[b+1]!=0)p.t=value(rig[b]);
    p.q=channel(rig[b+2],rig[b+3],time,duration,loop,true,p.q);
    p.s=channel(rig[b+4],rig[b+5],time,duration,loop,false,p.s);
    if(additive){p.t+=bind.t;p.q=qmul(bind.q,p.q);p.s*=bind.s;}
    return p;
}
Pose loadPose(uint node,uint bone){uint i=(node*pc.bones+bone)*3;Pose p;p.t=scratch[i];p.q=scratch[i+1];p.s=scratch[i+2];return p;}
void storePose(uint node,uint bone,Pose p){uint i=(node*pc.bones+bone)*3;scratch[i]=p.t;scratch[i+1]=p.q;scratch[i+2]=p.s;}
float4x4 localMatrix(Pose p){
    float4 q=p.q;float x=q.x,y=q.y,z=q.z,w=q.w;
    return float4x4((1-2*y*y-2*z*z)*p.s.x,(2*x*y-2*z*w)*p.s.y,(2*x*z+2*y*w)*p.s.z,p.t.x,
        (2*x*y+2*z*w)*p.s.x,(1-2*x*x-2*z*z)*p.s.y,(2*y*z-2*x*w)*p.s.z,p.t.y,
        (2*x*z-2*y*w)*p.s.x,(2*y*z+2*x*w)*p.s.y,(1-2*x*x-2*y*y)*p.s.z,p.t.z,0,0,0,1);
}
float4x4 bindMatrix(uint bone){uint b=bone*32+12;return float4x4(asfloat(rig[b]),asfloat(rig[b+1]),asfloat(rig[b+2]),asfloat(rig[b+3]),asfloat(rig[b+4]),asfloat(rig[b+5]),asfloat(rig[b+6]),asfloat(rig[b+7]),asfloat(rig[b+8]),asfloat(rig[b+9]),asfloat(rig[b+10]),asfloat(rig[b+11]),asfloat(rig[b+12]),asfloat(rig[b+13]),asfloat(rig[b+14]),asfloat(rig[b+15]));}
float4x4 actorMatrix(){return float4x4(asfloat(frame[0]),asfloat(frame[1]),asfloat(frame[2]),asfloat(frame[3]),asfloat(frame[4]),asfloat(frame[5]),asfloat(frame[6]),asfloat(frame[7]),asfloat(frame[8]),asfloat(frame[9]),asfloat(frame[10]),asfloat(frame[11]),asfloat(frame[12]),asfloat(frame[13]),asfloat(frame[14]),asfloat(frame[15]));}
float3 frame3(uint i){return float3(asfloat(frame[i]),asfloat(frame[i+1]),asfloat(frame[i+2]));}
float3 qlog(float4 q){q=normalize(q);if(q.w<0)q=-q;float len=length(q.xyz);return len>1e-7?q.xyz*(2*atan2(len,q.w)/len):q.xyz*2;}
float4 qexp(float3 v){float angle=length(v);if(angle<1e-7)return normalize(float4(v*.5,1));return float4(v*(sin(angle*.5)/angle),cos(angle*.5));}
float3 qrotate(float4 q,float3 v){return v+2*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
float4 qbetween(float3 a,float3 b){
    a=normalize(a);b=normalize(b);float d=clamp(dot(a,b),-1.0,1.0);
    if(d<-.99999){float3 axis=cross(a,float3(1,0,0));if(length(axis)<1e-5)axis=cross(a,float3(0,1,0));return float4(normalize(axis),0);}
    return normalize(float4(cross(a,b),1+d));
}
float4x4 modelAt(uint node,uint bone){
    float4x4 world=localMatrix(loadPose(node,bone));uint parent=rig[bone*32+28];
    while(parent!=0xffffffff){world=mul(localMatrix(loadPose(node,parent)),world);parent=rig[parent*32+28];}
    return world;
}
float3 positionAt(uint node,uint bone){return mul(modelAt(node,bone),float4(0,0,0,1)).xyz;}
float4 rotationAt(uint node,uint bone){
    float4 q=loadPose(node,bone).q;uint parent=rig[bone*32+28];
    while(parent!=0xffffffff){q=qmul(loadPose(node,parent).q,q);parent=rig[parent*32+28];}return q;
}
void turnBone(uint node,uint bone,float4 delta){
    uint parent=rig[bone*32+28];float4 world=qmul(delta,rotationAt(node,bone));Pose p=loadPose(node,bone);
    p.q=parent==0xffffffff?world:qmul(qinv(rotationAt(node,parent)),world);storePose(node,bone,p);
}
void applyModifier(uint node,uint at){
    uint kind=frame[at],upper=frame[at+1],lower=frame[at+2],end=frame[at+3];float3 target=frame3(at+4);float weight=saturate(asfloat(frame[at+7]));
    if(kind==0){Pose p=loadPose(node,upper);p.t.xyz+=target*weight;storePose(node,upper,p);return;}
    if(kind==2){
        float3 direction=target-positionAt(node,upper),forward=frame3(at+16);if(length(direction)<1e-5 || length(forward)<1e-5)return;
        float3 delta=qlog(qbetween(qrotate(rotationAt(node,upper),forward),direction));float angle=length(delta);
        if(angle>0)delta*=min(angle,clamp(asfloat(frame[at+15]),0.0,3.14159265))/angle*weight;
        turnBone(node,upper,qexp(delta));return;
    }
    float3 root=positionAt(node,upper),mid=positionAt(node,lower),tip=positionAt(node,end);
    float a=length(mid-root),b=length(tip-mid);float3 direction=target-root;float distance=length(direction);
    if(a<1e-5||b<1e-5||distance<1e-5)return;
    direction/=distance;float d=clamp(distance,abs(a-b)+1e-4,a+b-1e-4);
    float3 bend=mid-root-direction*dot(mid-root,direction);
    if(length(bend)<1e-4){float3 pole=frame3(at+8)-root;bend=pole-direction*dot(pole,direction);}
    if(length(bend)<1e-4)bend=cross(direction,abs(direction.y)<.9?float3(0,1,0):float3(1,0,0));
    bend=normalize(bend);float along=(a*a+d*d-b*b)/(2*d);float3 elbow=root+direction*along+bend*sqrt(max(0.0,a*a-along*along));
    float4 oldUpper=loadPose(node,upper).q,oldLower=loadPose(node,lower).q,oldEnd=loadPose(node,end).q;
    turnBone(node,upper,qbetween(mid-root,elbow-root));
    mid=positionAt(node,lower);tip=positionAt(node,end);turnBone(node,lower,qbetween(tip-mid,root+direction*d-mid));
    if(frame[at+11]!=0 && length(frame3(at+12))>1e-5)turnBone(node,end,qbetween(qrotate(rotationAt(node,end),float3(0,1,0)),frame3(at+12)));
    Pose p=loadPose(node,upper);p.q=qslerp(oldUpper,p.q,weight);storePose(node,upper,p);
    p=loadPose(node,lower);p.q=qslerp(oldLower,p.q,weight);storePose(node,lower,p);
    p=loadPose(node,end);p.q=qslerp(oldEnd,p.q,weight);storePose(node,end,p);
}
[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID){
    uint bone=tid.x;if(bone>=pc.bones)return;
    if(pc.phase==3){
        previousPalette[bone]=pc.pad0!=0?historyPalette[bone]:palette[bone];
        historyPalette[bone]=palette[bone];
    }else if(pc.phase==0){
        // Each invocation owns this bone across every node: intermediate pose
        // dependencies are invocation-local, while cached nodes evaluate once.
        {
            uint node=pc.pad2;
            uint b=16+node*12,kind=frame[b];Pose p;
            if(kind==0)p=sample(frame[b+1],bone,asfloat(frame[b+2]));
            else{
                uint input=frame[b+4],count=frame[b+5];p=loadPose(frame[input],bone);
                float total=asfloat(frame[input+1]);float mask=frame[b+6]==0?1:asfloat(frame[frame[b+6]+bone]);
                for(uint i=1;i<count;++i){Pose other=loadPose(frame[input+i*2],bone);float weight=asfloat(frame[input+i*2+1]);total+=weight;float a=total>0?weight/total:0;a=saturate(a*mask);
                    if(kind==2){Pose ref=sample(frame[b+7],bone,0);p.t+=(other.t-ref.t)*a;p.q=qmul(p.q,qslerp(float4(0,0,0,1),qmul(qinv(ref.q),other.q),a));float3 ratio=float3(abs(ref.s.x)>1e-6?other.s.x/ref.s.x:1,abs(ref.s.y)>1e-6?other.s.y/ref.s.y:1,abs(ref.s.z)>1e-6?other.s.z/ref.s.z:1);p.s.xyz*=lerp(float3(1,1,1),ratio,a);}
                    else{p.t=lerp(p.t,other.t,a);p.q=qslerp(p.q,other.q,a);p.s=lerp(p.s,other.s,a);}
                }
            }
            if(frame[b+8]!=0){
                uint at=frame[b+8]+bone*24;float h0=asfloat(frame[b+9]),h1=asfloat(frame[b+10]);
                p.t.xyz+=frame3(at)*h0+frame3(at+12)*h1;
                p.s.xyz+=frame3(at+8)*h0+frame3(at+20)*h1;
                p.q=qmul(qexp(frame3(at+4)*h0+frame3(at+16)*h1),p.q);
            }
            storePose(node,bone,p);
        }
    }else if(pc.phase==2){
        if(bone!=0)return;
        uint at=frame[16+pc.pad2*12+11];if(at==0)return;
        uint count=frame[at++];
        for(uint i=0;i<count;++i)applyModifier(pc.pad2,at+i*24);
    }else if(rig[bone*32+29]==pc.depth){
        float4x4 world=localMatrix(loadPose(pc.output,bone));uint parent=rig[bone*32+28];uint worldBase=pc.nodes*pc.bones*3;
        if(parent!=0xffffffff){uint p=worldBase+parent*4;world=mul(float4x4(scratch[p],scratch[p+1],scratch[p+2],scratch[p+3]),world);}
        uint b=worldBase+bone*4;scratch[b]=world[0];scratch[b+1]=world[1];scratch[b+2]=world[2];scratch[b+3]=world[3];
        palette[bone]=mul(mul(actorMatrix(),world),bindMatrix(bone));
    }
}
