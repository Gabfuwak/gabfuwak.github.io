import"./modulepreload-polyfill-B5Qt9EMX.js";/* empty css              */import{m as ne,a as Ut,b as Dt,c as It,d as Mt,e as St,f as Qe,p as re,r as Je,h as et,i as Rt,g as kt,j as Gt}from"./camera-Dvgm1X7S.js";import{i as Ot}from"./webgpu-D3-56OEV.js";function Tt(t){const[e,n,r]=t.translation,[i,a,o]=t.rotation;return ne(St(e,n,r),ne(Ut(a),ne(Dt(i),ne(It(o),Mt(t.scale)))))}function Vt(t,e){const n=new Float32Array(t.positions.length),r=new Float32Array(t.normals.length);for(let i=0;i<t.positions.length;i+=3){const a=t.positions[i],o=t.positions[i+1],c=t.positions[i+2];n[i]=e[0]*a+e[4]*o+e[8]*c+e[12],n[i+1]=e[1]*a+e[5]*o+e[9]*c+e[13],n[i+2]=e[2]*a+e[6]*o+e[10]*c+e[14]}for(let i=0;i<t.normals.length;i+=3){const a=t.normals[i],o=t.normals[i+1],c=t.normals[i+2];let d=e[0]*a+e[4]*o+e[8]*c,p=e[1]*a+e[5]*o+e[9]*c,f=e[2]*a+e[6]*o+e[10]*c;const u=Math.sqrt(d*d+p*p+f*f);u>1e-4&&(d/=u,p/=u,f/=u),r[i]=d,r[i+1]=p,r[i+2]=f}return{positions:n,normals:r,colors:t.colors,uvs:t.uvs,indices:t.indices}}function Lt(t){const e=[],n=[],r=[],i=[],a=[];let o=0;for(const c of t){for(let d=0;d<c.positions.length;d++)e.push(c.positions[d]);for(let d=0;d<c.normals.length;d++)n.push(c.normals[d]);for(let d=0;d<c.uvs.length;d++)r.push(c.uvs[d]);for(let d=0;d<c.colors.length;d++)i.push(c.colors[d]);for(let d=0;d<c.indices.length;d++)a.push(c.indices[d]+o);o+=c.positions.length/3}return{positions:new Float32Array(e),normals:new Float32Array(n),uvs:new Float32Array(r),colors:new Float32Array(i),indices:new Uint32Array(a)}}function z(t,e,n,r,i){const a=(_,g)=>{const B=new Float32Array(3),M=_[1]*g[2]-_[2]*g[1],U=_[2]*g[0]-_[0]*g[2],E=_[0]*g[1]-_[1]*g[0];return B[0]=M,B[1]=U,B[2]=E,B},o=(_,g)=>{const B=new Float32Array(3);return B[0]=_[0]-g[0],B[1]=_[1]-g[1],B[2]=_[2]-g[2],B},c=_=>{const g=Math.sqrt(_[0]*_[0]+_[1]*_[1]+_[2]*_[2]);return new Float32Array([_[0]/g,_[1]/g,_[2]/g])},d=[t,e,n,r].flat(),p=Array.from(c(a(o(t,e),o(t,r)))),f=new Array(4).fill(p).flat(),u=[0,1,2,0,2,3],x=[0,0,1,0,1,1,0,1],y=[i,i,i,i].flat();return{positions:new Float32Array(d),normals:new Float32Array(f),uvs:new Float32Array(x),colors:new Float32Array(y),indices:new Uint32Array(u)}}function qt(t,e=[1,1,1]){const n=[],r=[],i=[],a=[],o=[],c=[],d=[];for(const u of t.split(`
`)){const x=u.trim().split(/\s+/);if(x[0]==="v")n.push(+x[1],+x[2],+x[3]);else if(x[0]==="vn")r.push(+x[1],+x[2],+x[3]);else if(x[0]==="vt")i.push(+x[1],+x[2]);else if(x[0]==="f"){const y=x.slice(1);for(let b=1;b<y.length-1;b++)for(const _ of[y[0],y[b],y[b+1]]){const[g,B,M]=_.split("/"),U=(parseInt(g)-1)*3;if(a.push(n[U],n[U+1],n[U+2]),M){const E=(parseInt(M)-1)*3;o.push(r[E],r[E+1],r[E+2])}else o.push(0,0,0);if(B){const E=(parseInt(B)-1)*2;c.push(i[E],i[E+1])}else c.push(0,0);d.push(e[0],e[1],e[2])}}}if(r.length===0)for(let u=0;u<a.length;u+=9){const x=a[u+3]-a[u],y=a[u+4]-a[u+1],b=a[u+5]-a[u+2],_=a[u+6]-a[u],g=a[u+7]-a[u+1],B=a[u+8]-a[u+2],M=y*B-b*g,U=b*_-x*B,E=x*g-y*_,O=Math.sqrt(M*M+U*U+E*E)||1;for(let S=0;S<3;S++)o[u+S*3]=M/O,o[u+S*3+1]=U/O,o[u+S*3+2]=E/O}const p=a.length/3,f=new Uint32Array(p);for(let u=0;u<p;u++)f[u]=u;return{positions:new Float32Array(a),normals:new Float32Array(o),uvs:new Float32Array(c),colors:new Float32Array(d),indices:f}}function ot(t){const e=new Map,n=[],r=[],i=[];let a=0,o=0;for(const f of t.objects){const u=f.material;e.set(u.id,u);const x=Vt(f.mesh,f.transform);n.push(x);const y=f.mesh.positions.length/3;for(let b=0;b<y;b++)r.push(u.id);i.push({objectId:a,indexOffset:o,indexCount:f.mesh.indices.length}),o+=f.mesh.indices.length,a++}const c=e.size===0?-1:Math.max(...e.keys()),d=[];for(let f=0;f<=c;f++)d.push(e.get(f)??{id:f,diffuseAlbedo:new Float32Array(3),roughness:0,fresnel:new Float32Array(3),metalness:0,emission:0,basePerlinFreq:0,perlinOctaveAmp:0,perlinOctaveNb:0});const p=Lt(n);return{positions:p.positions,indices:p.indices,normals:p.normals,uvs:p.uvs,objectIds:new Uint32Array(r),materials:d,ranges:i}}function Nt(t){const e=[],n=[];for(const r of t){const i=e.length/3,a=r.minCorner,o=r.maxCorner;e.push(a[0],a[1],a[2],o[0],a[1],a[2],a[0],o[1],a[2],o[0],o[1],a[2],a[0],a[1],o[2],o[0],a[1],o[2],a[0],o[1],o[2],o[0],o[1],o[2]);const c=[0,1,1,3,3,2,2,0,4,5,5,7,7,6,6,4,0,4,1,5,2,6,3,7];n.push(...c.map(d=>i+d))}return{positions:new Float32Array(e),indices:new Uint32Array(n)}}function zt(t,e,n,r){t<r.minCorner[0]&&(r.minCorner[0]=t),e<r.minCorner[1]&&(r.minCorner[1]=e),n<r.minCorner[2]&&(r.minCorner[2]=n),t>r.maxCorner[0]&&(r.maxCorner[0]=t),e>r.maxCorner[1]&&(r.maxCorner[1]=e),n>r.maxCorner[2]&&(r.maxCorner[2]=n)}function Ht(t,e){const n=new Float32Array([1/0,1/0,1/0]),r=new Float32Array([-1/0,-1/0,-1/0]);for(let i=0;i<e.length;i++){const a=e[i];for(let o=0;o<3;o++){const c=t.indices[a*3+o],d=t.positions[c*3],p=t.positions[c*3+1],f=t.positions[c*3+2];zt(d,p,f,{minCorner:n,maxCorner:r})}}return{minCorner:n,maxCorner:r}}function tt(t,e){let n=t.maxCorner[0]-t.minCorner[0],r=t.maxCorner[1]-t.minCorner[1],i=t.maxCorner[2]-t.minCorner[2];return(n*(r+i)+r*i)*e}function nt(t,e,n){const r=e.length;let i=1e30,a=-1e30;const o=new Float32Array(r);for(let h=0;h<r;h++){const m=e[h],D=t.indices[m*3],C=t.indices[m*3+1],G=t.indices[m*3+2],I=(t.positions[D*3+n]+t.positions[C*3+n]+t.positions[G*3+n])/3;o[h]=I,I<i&&(i=I),I>a&&(a=I)}const c=12;if(a===i){const h=Math.floor(r/2);return{left:e.slice(0,h),right:e.slice(h),cost:1e30}}const d=new Uint32Array(c),p=new Float32Array(c*3).fill(1/0),f=new Float32Array(c*3).fill(-1/0),u=c/(a-i),x=new Uint8Array(r);for(let h=0;h<r;h++){const m=Math.min(Math.max(Math.floor((o[h]-i)*u),0),c-1);x[h]=m,d[m]++;const D=e[h],C=m*3;for(let G=0;G<3;G++){const I=t.indices[D*3+G],T=t.positions[I*3],R=t.positions[I*3+1],A=t.positions[I*3+2];T<p[C]&&(p[C]=T),R<p[C+1]&&(p[C+1]=R),A<p[C+2]&&(p[C+2]=A),T>f[C]&&(f[C]=T),R>f[C+1]&&(f[C+1]=R),A>f[C+2]&&(f[C+2]=A)}}const y=new Float32Array(c*3).fill(1/0),b=new Float32Array(c*3).fill(-1/0),_=new Uint32Array(c),g=(c-1)*3;y[g]=p[g],y[g+1]=p[g+1],y[g+2]=p[g+2],b[g]=f[g],b[g+1]=f[g+1],b[g+2]=f[g+2],_[c-1]=d[c-1];for(let h=c-2;h>=0;h--){const m=h*3,D=m+3;y[m]=y[D],y[m+1]=y[D+1],y[m+2]=y[D+2],b[m]=b[D],b[m+1]=b[D+1],b[m+2]=b[D+2],d[h]>0&&(p[m]<y[m]&&(y[m]=p[m]),p[m+1]<y[m+1]&&(y[m+1]=p[m+1]),p[m+2]<y[m+2]&&(y[m+2]=p[m+2]),f[m]>b[m]&&(b[m]=f[m]),f[m+1]>b[m+1]&&(b[m+1]=f[m+1]),f[m+2]>b[m+2]&&(b[m+2]=f[m+2])),_[h]=_[h+1]+d[h]}let B=1,M=1e30;{let h=p[0],m=p[1],D=p[2],C=f[0],G=f[1],I=f[2],T=d[0];for(let R=1;R<c;R++){const A=R*3,H=C-h,Y=G-m,W=I-D,me=b[A]-y[A],Z=b[A+1]-y[A+1],V=b[A+2]-y[A+2],Q=(H*(Y+W)+Y*W)*T+(me*(Z+V)+Z*V)*_[R];Q<M&&(M=Q,B=R),d[R]>0&&(p[A]<h&&(h=p[A]),p[A+1]<m&&(m=p[A+1]),p[A+2]<D&&(D=p[A+2]),f[A]>C&&(C=f[A]),f[A+1]>G&&(G=f[A+1]),f[A+2]>I&&(I=f[A+2])),T+=d[R]}}let U=0;for(let h=0;h<r;h++)x[h]<B&&U++;const E=new Uint32Array(U),O=new Uint32Array(r-U);let S=0,X=0;for(let h=0;h<r;h++)x[h]<B?E[S++]=e[h]:O[X++]=e[h];return{left:E,right:O,cost:M}}function jt(t,e){let n=nt(t,e,0);for(const r of[1,2]){const i=nt(t,e,r);i.cost<n.cost&&(n=i)}return{partition:[n.left,n.right],cost:n.cost}}function Pe(t,e,n){const r=Ht(t,e);if(e.length<=1||n>=32)return{kind:"leaf",boundingBox:r,triangles:new Uint32Array(e)};const i=jt(t,e),[a,o]=i.partition;return a.length===0||o.length===0?{kind:"leaf",boundingBox:r,triangles:new Uint32Array(e)}:i.cost+tt(r,1)>tt(r,e.length)?{kind:"leaf",boundingBox:r,triangles:new Uint32Array(e)}:{kind:"interior",boundingBox:r,left:Pe(t,a,n+1),right:Pe(t,o,n+1)}}function Ee(t,e,n,r){if(t.kind==="leaf"){const a=r.length;for(const o of t.triangles)r.push(o);return n[e]={minCorner:t.boundingBox.minCorner,maxCorner:t.boundingBox.maxCorner,isLeaf:!0,leftChild:-1,rightChild:-1,triangleIndex:a,nbTris:t.triangles.length},e+1}n[e]={minCorner:t.boundingBox.minCorner,maxCorner:t.boundingBox.maxCorner,isLeaf:!1,leftChild:e+1,rightChild:-1,triangleIndex:-1,nbTris:-1};const i=Ee(t.left,e+1,n,r);return n[e].rightChild=i,Ee(t.right,i,n,r)}function Xt(t){const e=[],n=[];return Ee(t,0,e,n),{nodes:e,primitives:new Uint32Array(n)}}function Yt(t){const e=ot(t),n={positions:new Float32Array(e.positions),indices:new Uint32Array(e.indices)},r=n.indices.length/3,i=new Uint32Array(Array.from({length:r},(a,o)=>o));return Pe(n,i,0)}const Wt=`struct PointLight{
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  _pad1: f32,
  direction: vec3<f32>,
  angle: f32,
}

struct Material{
  baseColor: vec3<f32>,
  roughness: f32,
  fresnel: vec3<f32>,
  metalness: f32,
  emission: f32,
  basePerlinFreq: f32,
  perlinOctaveAmp: f32,
  perlinOctaveNb: f32,
}

struct BVHNode{
  minCorner: vec3<f32>,
  isLeafOrTriNb: u32,
  maxCorner: vec3<f32>,
  triangleOrChildIdx: u32,
}

struct Uniforms {
  // Rasterizer (indices 0-15)
  mvp: mat4x4<f32>,
  // Shared (indices 16+)
  camera_pos: vec3<f32>,
  nbLights: f32,
  lights: array<PointLight, 4>,
  nbMaterials: f32,
  spp: f32,
  frame_count: f32,
  canvas_width: f32,
  materials: array<Material, 16>,
  // Raytracer (indices 200-211, fits in the existing UNIFORM_LENGTH buffer)
  camera_forward: vec3f,
  fov_factor: f32,
  camera_right: vec3f,
  aspect_ratio: f32,
  camera_up: vec3f,
  time: f32,
};

@group(0) @binding(0) var<uniform>          uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>    vertices  : array<f32>;
@group(0) @binding(2) var<storage, read>    indices   : array<u32>;
@group(0) @binding(3) var<storage, read>    objectIds : array<u32>;
@group(0) @binding(4) var<storage, read>    normals   : array<f32>;
@group(0) @binding(5) var<storage, read>    uvs       : array<f32>;
@group(0) @binding(6) var<storage, read>    bvh       : array<BVHNode>;
@group(0) @binding(7) var<storage, read_write> accum: array<vec4f>;

// ============================================================================
// Shared BRDF
// ============================================================================

fn GGX_distribution(material: Material, halfVector: vec3<f32>, normal:vec3<f32>) -> f32 {
  let roughness = max(0.045, material.roughness);
  let alpha = roughness * roughness;
  let alpha_sq = alpha * alpha;

  let pi = 3.14159265359;

  let NdotH = max(0.0, dot(normal, halfVector));
  let NdotH_sq = NdotH * NdotH;

  let denom_inner = NdotH_sq * (alpha_sq - 1.0) + 1.0;
  let denom = pi * denom_inner * denom_inner;

  return alpha_sq / max(1e-7, denom);
}

fn schlick_fresnel(material: Material, viewDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32>{
  let f_0: vec3<f32> = material.fresnel;
  let cosTheta = max(0.0, dot(viewDir, halfVector));
  return f_0 + (1.0 - f_0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}

fn G_1_schlick_approx(vec: vec3<f32>, normal: vec3<f32>, k: f32) -> f32{
  let NdotV = max(0.0, dot(normal, vec));
  let denom = NdotV * (1.0 - k) + k;
  return NdotV / max(0.0001, denom);
}

fn smith_geometric(material: Material, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>) -> f32 {
  let roughness = max(0.045, material.roughness);
  let alpha = roughness * roughness;
  let k = alpha * sqrt(2.0 / 3.14159265359);
  return G_1_schlick_approx(viewDir, normal, k) * G_1_schlick_approx(lightDir, normal, k);
}

fn microfacet_BRDF(material: Material, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32> {
  let d = GGX_distribution(material, halfVector, normal);
  let f = schlick_fresnel(material, viewDir, halfVector);
  let g = smith_geometric(material, normal, viewDir, lightDir);

  let numerator = d * f * g;
  let NdotL = max(0.0, dot(normal, lightDir));
  let NdotV = max(0.0, dot(normal, viewDir));
  let denominator = max(0.0001, 4.0 * NdotL * NdotV);

  return numerator / denominator;
}

fn evaluateRadiance(
  light: PointLight,
  worldPos: vec3<f32>,
  normal_in: vec3<f32>,
  viewDir: vec3<f32>,
  material: Material
) -> vec3<f32> {
  let normal = normalize(normal_in);
  let lightVec = light.position - worldPos;
  let lightDir = normalize(lightVec);
  let halfVector = normalize(viewDir + lightDir);

  let lightToPoint = -lightDir;
  let coneAngle = light.angle;
  let inCone = dot(normalize(light.direction), lightToPoint) >= cos(coneAngle / 2.0);

  if (!inCone) {
    return vec3f(material.emission);
  }

  let lightDistance = length(lightVec) / 100.0;
  let constant = 1.0;
  let linear = 0.09;
  let quadratic = 0.032;
  let attenuationFactor = 1.0 / (constant + linear * lightDistance + quadratic * lightDistance * lightDistance);

  let lambertFactor = max(0.0, dot(lightDir, normalize(normal)));

  let diffuse_term = material.baseColor * lambertFactor * light.color * light.intensity * attenuationFactor;
  let specular_term = microfacet_BRDF(material, normal, viewDir, lightDir, halfVector) * lambertFactor * light.color * light.intensity * attenuationFactor;

  let kD = (1.0 - material.metalness) * (1.0 - material.fresnel);

  return kD * diffuse_term + specular_term + material.emission;
}


// ============================================================================
// Rasterizer
// ============================================================================

struct RastVertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) @interpolate(flat) objectId: u32,
  @location(1) normal: vec4f,
  @location(2) worldPos: vec4f,
  @location(3) uv: vec2f,
};

@vertex
fn rastVertexMain(@builtin(vertex_index) vid: u32) -> RastVertexOutput {
  let idx = indices[vid];
  let pos    = vec3f(vertices[idx*3u], vertices[idx*3u+1u], vertices[idx*3u+2u]);
  let normal = vec3f(normals[idx*3u],  normals[idx*3u+1u],  normals[idx*3u+2u]);
  let uv     = vec2f(uvs[idx*2u], uvs[idx*2u+1u]);
  let objId  = objectIds[idx];

  var output: RastVertexOutput;
  output.pos      = uniforms.mvp * vec4f(pos, 1.0);
  output.worldPos = vec4f(pos, 1.0);
  output.objectId = objId;
  output.normal   = vec4f(normal, 0.0);
  output.uv       = uv;
  return output;
}

@fragment
fn rastFragmentMain(input: RastVertexOutput) -> @location(0) vec4f {
  var output_color = vec3f(0.0);

  if (u32(uniforms.nbLights) == 0u) {
    return vec4f(output_color, 1.0);
  }

  let material = resolve_material(input.objectId, input.uv);
  let viewDir  = normalize(uniforms.camera_pos - input.worldPos.xyz);

  for(var i = 0u; i < u32(uniforms.nbLights); i++) {
    output_color += evaluateRadiance(
      uniforms.lights[i],
      input.worldPos.xyz,
      input.normal.xyz,
      viewDir,
      material
    );
  }

  return vec4f(output_color, 1.0);
}


// ============================================================================
// Raytracer
// ============================================================================

struct RayVertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
};

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
};

struct Hit {
  triIndex: u32,
  barycentricCoords: vec3<f32>,
  t: f32
};

struct SurfacePoint {
  world_pos: vec3f,
  normal:    vec3f,
  uv:    vec2f,
  objectId:  u32,
  material:  Material,
};

fn resolve_material(material_id: u32, uv: vec2f) -> Material {
  var mat = uniforms.materials[material_id];
  switch(material_id) { // switch stays so that later i can add a custom material scheme, maybe with graphs we'll see
    default: {
      if(mat.perlinOctaveAmp > 1e-30){
        let t = uniforms.time / 100.0;
        mat.baseColor += vec3f(octavePerlin3D(vec3f(uv, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb)));
        mat.roughness += octavePerlin3D(vec3f(uv + 1.0, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb));
        mat.metalness += octavePerlin3D(vec3f(uv + 2.0, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb));
      }
      return mat;
    }
  }
}

fn resolve_hit(hit: Hit) -> SurfacePoint {
  let i0 = indices[hit.triIndex * 3u];
  let i1 = indices[hit.triIndex * 3u + 1u];
  let i2 = indices[hit.triIndex * 3u + 2u];

  let bary = hit.barycentricCoords;

  let p0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let p1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let p2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  let n0 = vec3f(normals[i0*3u], normals[i0*3u+1u], normals[i0*3u+2u]);
  let n1 = vec3f(normals[i1*3u], normals[i1*3u+1u], normals[i1*3u+2u]);
  let n2 = vec3f(normals[i2*3u], normals[i2*3u+1u], normals[i2*3u+2u]);


  let uv0 = vec2f(uvs[i0*2u], uvs[i0*2u+1u]);
  let uv1 = vec2f(uvs[i1*2u], uvs[i1*2u+1u]);
  let uv2 = vec2f(uvs[i2*2u], uvs[i2*2u+1u]);

  var sp: SurfacePoint;
  sp.world_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
  sp.normal    = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
  sp.uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
  sp.objectId  = objectIds[i0];
  sp.material  = resolve_material(sp.objectId, sp.uv);
  return sp;
}

fn ray_at(screenCoord: vec2<f32>) -> Ray {
  let horizontal = screenCoord.x * uniforms.fov_factor * uniforms.aspect_ratio;
  let vertical   = screenCoord.y * uniforms.fov_factor;

  var output: Ray;
  output.origin    = uniforms.camera_pos;
  output.direction = normalize(
    uniforms.camera_forward +
    horizontal * uniforms.camera_right +
    vertical   * uniforms.camera_up
  );
  return output;
}

fn rayBoxHit(ray: Ray, box: BVHNode) -> f32 {
  var tmin = 0.0f;
  var tmax = 1e30;

  for(var i = 0; i < 3; i++){
    let t1 = (box.minCorner[i] - ray.origin[i]) / ray.direction[i];
    let t2 = (box.maxCorner[i] - ray.origin[i]) / ray.direction[i];

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }


  if(tmax >= tmin && tmax >= 0.0f){
    return tmin;
  }else {
    return 1e30;
  }
}

fn rayTriangleHit(ray: Ray, tri_idx: u32) -> Hit {
  let i0 = indices[tri_idx * 3u];
  let i1 = indices[tri_idx * 3u + 1u];
  let i2 = indices[tri_idx * 3u + 2u];

  let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  let e1 = v1 - v0;
  let e2 = v2 - v0;

  let P   = cross(ray.direction, e2);
  let det = dot(e1, P);

  var no_hit: Hit;
  no_hit.t = -1.0;

  if (det < 0.00001) { return no_hit; }

  let inv_det = 1.0 / det;
  let T = ray.origin - v0;
  let u = dot(T, P) * inv_det;

  if (u < 0.0 || u > 1.0) { return no_hit; }

  let Q = cross(T, e1);
  let v = dot(ray.direction, Q) * inv_det;

  if (v < 0.0 || u + v > 1.0) { return no_hit; }

  let t = dot(e2, Q) * inv_det;

  if (t <= 0.00001) { return no_hit; }

  var hit: Hit;
  hit.triIndex = tri_idx;
  hit.barycentricCoords = vec3(1.0 - u - v, u, v);
  hit.t = t;
  return hit;
}

fn rayTrace(ray: Ray, hit: ptr<function, Hit>, is_shadow: bool, max_t: f32) -> bool {
  var closest_t = max_t;
  var found_hit = false;

  var stack: array<u32, 64>;
  var stack_ptr: i32 = 0;

  stack[0] = 0u;
  stack_ptr = 1;

  while (stack_ptr > 0) {
    stack_ptr -= 1;
    let curr_node_idx = stack[stack_ptr];

    if (bvh[curr_node_idx].isLeafOrTriNb != 0u) {
      for(var candidate_idx = 0u; candidate_idx < bvh[curr_node_idx].isLeafOrTriNb; candidate_idx++){
        let candidate = rayTriangleHit(ray, bvh[curr_node_idx].triangleOrChildIdx + candidate_idx);

        if (candidate.t > 0.0 && candidate.t < closest_t) {
          closest_t = candidate.t;
          found_hit = true;
          *hit = candidate;
          if(is_shadow){ // early stop if anything is blocking a shadow ray
            return true;
          }
        }
      }
    } else {
      let rightChild = bvh[curr_node_idx].triangleOrChildIdx;
      let leftChild  = curr_node_idx + 1;

      let rightChildHit_t = rayBoxHit(ray, bvh[rightChild]);
      let leftChildHit_t  = rayBoxHit(ray, bvh[leftChild]);

      if (leftChildHit_t < rightChildHit_t) {
        if (rightChildHit_t < closest_t) {
          stack[stack_ptr] = rightChild;
          stack_ptr += 1;
        }
        if (leftChildHit_t < closest_t) {
          stack[stack_ptr] = leftChild;
          stack_ptr += 1;
        }
      } else {
        if (leftChildHit_t < closest_t) {
          stack[stack_ptr] = leftChild;
          stack_ptr += 1;
        }
        if (rightChildHit_t < closest_t) {
          stack[stack_ptr] = rightChild;
          stack_ptr += 1;
        }
      }
    }
  }

  return found_hit;
}

@vertex
fn rayVertexMain(@builtin(vertex_index) vid: u32) -> RayVertexOutput {
  // Fullscreen quad, no vertex buffer needed
  let positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let pos = positions[vid];
  var output: RayVertexOutput;
  output.clip_pos   = vec4f(pos, 0.0, 1.0);
  output.screen_pos = pos;
  return output;
}

// ============================================================================
// Picking
// ============================================================================

@group(0) @binding(8) var<uniform>             pick_coords : vec2f;
@group(0) @binding(9) var<storage, read_write> pick_result : i32;

@compute @workgroup_size(1)
fn pick_main() {
  let ray = ray_at(pick_coords);
  var hit: Hit;
  if (rayTrace(ray, &hit, false, 1e30)) {
    pick_result = i32(objectIds[indices[hit.triIndex * 3u]]);
  } else {
    pick_result = -1;
  }
}

fn sampleSemiSphere(n : vec3f, seed: f32) -> vec3f{
  // Chain PCG in integer space to avoid correlated bitcast of adjacent floats
  let s0 = pcg_hash(bitcast<u32>(seed));
  let s1 = pcg_hash(s0);
  let phi       = f32(s0) / f32(0xFFFFFFFFu) * 6.28318530718;
  let cos_theta = f32(s1) / f32(0xFFFFFFFFu); // [0, 1] for hemisphere
  let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  // add semi sphere variation around the normal

  let up        = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let tangent   = normalize(cross(up, n));
  let bitangent = cross(n, tangent);

  return normalize(sin_theta * cos(phi) * tangent
                 + sin_theta * sin(phi) * bitangent
                 + cos_theta            * n);
}

@fragment
fn rayFragmentMain(input: RayVertexOutput) -> @location(0) vec4f {
  var output_color = vec3f(0.0);

  for(var sample = 0u; sample < u32(uniforms.spp); sample++){
    var ray = ray_at(input.screen_pos);
    var sample_output_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    for (var bounce = 1u; bounce <= 5; bounce++) {
      var hit_data: Hit;
      if (!rayTrace(ray, &hit_data, false, 1e30)) { break; }

      let sp = resolve_hit(hit_data);
      let viewDir = -ray.direction;

      sample_output_color += throughput * sp.material.emission;

      for (var i = 0u; i < u32(uniforms.nbLights); i++) {
        let lightVec = uniforms.lights[i].position - sp.world_pos;
        let lightDir = normalize(lightVec);

        var shadow_ray: Ray;
        shadow_ray.direction = lightDir;
        shadow_ray.origin = sp.world_pos + sp.normal * 0.1;

        var shadow_hit: Hit;
        let is_shadow = rayTrace(shadow_ray, &shadow_hit, true, length(lightVec));

        if (!is_shadow) {
          output_color += throughput * evaluateRadiance(
            uniforms.lights[i],
            sp.world_pos,
            sp.normal,
            viewDir,
            sp.material
          );
        }
      }

      let F = schlick_fresnel(sp.material, viewDir, sp.normal);
      //if (all(throughput * F < vec3f(0.001))) { break; }

      // white reflects more than black so we need a kd here too
      let kD = (1.0 - sp.material.metalness) * (vec3f(1.0) - F);
      throughput *= kD * sp.material.baseColor + F;

      let px = vec2u(input.clip_pos.xy);
      let seed_u = pcg_hash(px.x ^ pcg_hash(px.y ^ pcg_hash(sample ^ pcg_hash(bounce ^ u32(uniforms.frame_count)))));
      let seed = bitcast<f32>(seed_u);
      let random_ray_dir = sampleSemiSphere(sp.normal, seed);

      ray = Ray(sp.world_pos + sp.normal * 0.001, random_ray_dir);
      let rr_seed = pcg_hash(seed_u ^ 0xDEADBEEFu);
      if(all(throughput < vec3f(f32(rr_seed) / f32(0xFFFFFFFFu)))) {break;} // russian roulette
    }
    output_color += sample_output_color;
  }

  let coords = vec2u(input.clip_pos.xy);
  let idx = coords.y * u32(uniforms.canvas_width) + coords.x;
  let sample = output_color / uniforms.spp;
  if (uniforms.frame_count <= 1.0) {
    accum[idx] = vec4f(sample, 1.0);
  } else {
    accum[idx] += vec4f(sample, 0.0);
  }
  return vec4f(accum[idx].rgb / uniforms.frame_count, 1.0);
}
`,Zt=`fn pcg_hash(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random(t: vec2<f32>) -> f32 {
  let seed = pcg_hash(bitcast<u32>(t.x) ^ pcg_hash(bitcast<u32>(t.y)));
  return f32(seed) / f32(0xFFFFFFFFu);
}

fn random3D(t: vec3f) -> f32 {
  let seed = pcg_hash(bitcast<u32>(t.x) ^ pcg_hash(bitcast<u32>(t.y) ^ pcg_hash(bitcast<u32>(t.z))));
  return f32(seed) / f32(0xFFFFFFFFu);
}


fn valueNoise2D(x: vec2<f32>, freq: f32, amp: f32) -> f32{
  let pos_abs = x * freq;
  let pos_grid = floor(pos_abs);
  let pos_rel = pos_abs - pos_grid;

  let v00 = random(pos_grid);
  let v10 = random(pos_grid + vec2f(1.0, 0.0));
  let v01 = random(pos_grid + vec2f(0.0, 1.0));
  let v11 = random(pos_grid + vec2f(1.0, 1.0));

  let t = smoothstep(vec2f(0.0), vec2f(1.0), pos_rel);

  // interpolate both values on y axis
  let value_0x = mix(v00, v01, t.y);
  let value_1x = mix(v10, v11, t.y);

  // return interpolation on x axis
  return mix(value_0x, value_1x, t.x) * amp;
}

fn gradient2D(p: vec2f) -> vec2f {
  // sample from circle to get rid of noise
  let phi = random(p) * 6.28318530718;
  return vec2f(cos(phi), sin(phi));
}

fn gradient3D(p: vec3f) -> vec3f {
  // sample from sphere, similar to gradient 2d
  let phi       = random3D(p)              * 6.28318530718;
  let cos_theta = random3D(p + vec3f(1.0)) * 2.0 - 1.0;
  let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn perlinNoise2D(x: vec2f, freq: f32, amp: f32) -> f32 {
  let pos_abs = x * freq;
  let pos_grid = floor(pos_abs);
  let pos_rel = pos_abs - pos_grid;

  let g00 = gradient2D(pos_grid);
  let g10 = gradient2D(pos_grid + vec2f(1.0, 0.0));
  let g01 = gradient2D(pos_grid + vec2f(0.0, 1.0));
  let g11 = gradient2D(pos_grid + vec2f(1.0, 1.0));

  let v00 = dot(g00, pos_rel);
  let v10 = dot(g10, pos_rel - vec2f(1.0, 0.0));
  let v01 = dot(g01, pos_rel - vec2f(0.0, 1.0));
  let v11 = dot(g11, pos_rel - vec2f(1.0, 1.0));

  let t = pos_rel * pos_rel * pos_rel * (pos_rel * (pos_rel * 6.0 - 15.0) + 10.0);

  // interpolate both values on y axis
  let value_0x = mix(v00, v01, t.y);
  let value_1x = mix(v10, v11, t.y);

  // return interpolation on x axis
  return mix(value_0x, value_1x, t.x) * amp;
}

fn octavePerlin2D(x: vec2f, freq: f32, amp: f32, octaves: u32) -> f32 {
  var ret : f32 = 0.0;
  for(var i = 1u; i <= octaves; i++){
    ret += perlinNoise2D(x, freq*pow(f32(i), 2), amp/pow(f32(i), 2));
  }
  return ret;
}


fn octaveValue2D(x: vec2f, freq: f32, amp: f32, octaves: u32) -> f32 {
  var ret : f32 = 0.0;
  for(var i = 1u; i <= octaves; i++){
    ret += valueNoise2D(x, freq*pow(f32(i), 2), amp/pow(f32(i), 2));
  }
  return ret;
}


fn perlinNoise3D(x: vec3f, freq: f32, amp: f32) -> f32 {
  let pos_abs = x * freq;
  let pos_grid = floor(pos_abs);
  let pos_rel = pos_abs - pos_grid;

  let g000 = gradient3D(pos_grid);
  let g100 = gradient3D(pos_grid + vec3f(1.0, 0.0, 0.0));
  let g010 = gradient3D(pos_grid + vec3f(0.0, 1.0, 0.0));
  let g110 = gradient3D(pos_grid + vec3f(1.0, 1.0, 0.0));
  let g001 = gradient3D(pos_grid + vec3f(0.0, 0.0, 1.0));
  let g101 = gradient3D(pos_grid + vec3f(1.0, 0.0, 1.0));
  let g011 = gradient3D(pos_grid + vec3f(0.0, 1.0, 1.0));
  let g111 = gradient3D(pos_grid + vec3f(1.0, 1.0, 1.0));

  let v000 = dot(g000, pos_rel);
  let v100 = dot(g100, pos_rel - vec3f(1.0, 0.0, 0.0));
  let v010 = dot(g010, pos_rel - vec3f(0.0, 1.0, 0.0));
  let v110 = dot(g110, pos_rel - vec3f(1.0, 1.0, 0.0));
  let v001 = dot(g001, pos_rel - vec3f(0.0, 0.0, 1.0));
  let v101 = dot(g101, pos_rel - vec3f(1.0, 0.0, 1.0));
  let v011 = dot(g011, pos_rel - vec3f(0.0, 1.0, 1.0));
  let v111 = dot(g111, pos_rel - vec3f(1.0, 1.0, 1.0));

  let t = pos_rel * pos_rel * pos_rel * (pos_rel * (pos_rel * 6.0 - 15.0) + 10.0);

  let v00 = mix(v000, v001, t.z);
  let v10 = mix(v100, v101, t.z);

  let v01 = mix(v010, v011, t.z);
  let v11 = mix(v110, v111, t.z);


  // interpolate both values on y axis
  let value_0x = mix(v00, v01, t.y);
  let value_1x = mix(v10, v11, t.y);

  // return interpolation on x axis
  return mix(value_0x, value_1x, t.x) * amp;
}

fn octavePerlin3D(x: vec3f, freq: f32, amp: f32, octaves: u32) -> f32 {
  var ret : f32 = 0.0;
  for(var i = 1u; i <= octaves; i++){
    ret += perlinNoise3D(x, freq*pow(f32(i), 2), amp/pow(f32(i), 2));
  }
  return ret;
}
`,Kt=`// Only needs the MVP from the full uniform buffer
struct Uniforms {
  mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform>       uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> vertices : array<f32>;


// ============================================================================
// AABB wireframe
// ============================================================================

@vertex
fn aabbVertexMain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  let pos = vec3f(vertices[vid*3u], vertices[vid*3u+1u], vertices[vid*3u+2u]);
  return uniforms.mvp * vec4f(pos, 1.0);
}

@fragment
fn aabbFragmentMain() -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 0.0, 1.0);
}
`,$t={"Dragon (2k tri)":"dragon_2348.obj","Dragon (117k tri)":"dragon_117452.obj"};function Qt(t,e){const n=Rt(t,[278,273,-800],[278,273,-799],[0,1,0],1.0471666666666668,.1,2e3),r=[278,274,280],i=(y,b,_)=>{const g=Math.sqrt(y*y+b*b+_*_);return new Float32Array([y/g,b/g,_/g])},a=[{position:new Float32Array([71,412,140]),color:new Float32Array([1,1,1]),intensity:.8,direction:i(r[0]-71,r[1]-412,r[2]-140),angle:90},{position:new Float32Array([485,137,140]),color:new Float32Array([0,.2,.8]),intensity:.6,direction:i(r[0]-485,r[1]-137,r[2]-140),angle:360},{position:new Float32Array([71,137,70]),color:new Float32Array([.7,.3,0]),intensity:.2,direction:i(r[0]-71,r[1]-137,r[2]-70),angle:360},{position:new Float32Array([554,494,280]),color:new Float32Array([1,1,1]),intensity:.2,direction:i(r[0]-554,r[1]-494,r[2]-280),angle:360}],o={id:0,diffuseAlbedo:new Float32Array([1,1,1]),roughness:0,metalness:0,fresnel:new Float32Array([.05,.05,.05]),emission:0,basePerlinFreq:0,perlinOctaveAmp:0,perlinOctaveNb:0},c={id:1,diffuseAlbedo:new Float32Array([.65,.05,.05]),roughness:0,metalness:0,fresnel:new Float32Array([.05,.05,.05]),emission:0,basePerlinFreq:0,perlinOctaveAmp:0,perlinOctaveNb:0},d={id:2,diffuseAlbedo:new Float32Array([.12,.45,.15]),roughness:0,metalness:0,fresnel:new Float32Array([.05,.05,.05]),emission:0,basePerlinFreq:0,perlinOctaveAmp:0,perlinOctaveNb:0},p={id:3,diffuseAlbedo:new Float32Array([.8,0,0]),roughness:.5,metalness:1,fresnel:new Float32Array([1,.71,.29]),emission:0,basePerlinFreq:10,perlinOctaveAmp:.9,perlinOctaveNb:5},f={id:4,diffuseAlbedo:new Float32Array([1,1,1]),roughness:0,metalness:0,fresnel:new Float32Array([.05,.05,.05]),emission:3,basePerlinFreq:0,perlinOctaveAmp:0,perlinOctaveNb:0},u=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);return{objects:[{mesh:z([552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2],[1,1,1]),material:o,transform:u,label:"Floor"},{mesh:z([556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0],[1,1,1]),material:o,transform:u,label:"Ceiling"},{mesh:z([343,548,227],[343,548,332],[213,548,332],[213,548,227],[1,1,1]),material:f,transform:u,label:"Light"},{mesh:z([549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2],[1,1,1]),material:o,transform:u,label:"Back wall"},{mesh:z([0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2],[.12,.45,.15]),material:d,transform:u,label:"Right wall"},{mesh:z([552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0],[.65,.05,.05]),material:c,transform:u,label:"Left wall"},{mesh:z([556,548.8,0],[0,548.8,0],[0,0,0],[552.8,0,0],[1,1,1]),material:o,transform:u,label:"Front wall"},{mesh:qt(e,[.8,.2,.2]),material:p,transform:Tt({translation:[279,115,269],rotation:[0,Math.PI/4,0],scale:2}),label:"Dragon"}],lights:a,camera:n}}async function Jt(t,e){const{device:n,context:r,_adapter:i}=await Ot(t),a=ot(e),{positions:o,indices:c,uvs:d,objectIds:p,normals:f,materials:u}=a,x=n.createBuffer({label:"Vertices",size:o.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE}),y=n.createBuffer({label:"Vertex indices",size:c.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE}),b=n.createBuffer({label:"Object IDs",size:p.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE}),_=n.createBuffer({label:"Vertex normals",size:f.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE}),g=n.createBuffer({label:"Vertex UVs",size:d.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST|GPUBufferUsage.STORAGE}),B=8,M=performance.now(),U=Yt(e);console.log(`BVH build: ${(performance.now()-M).toFixed(1)}ms`);const{nodes:E,primitives:O}=Xt(U),S=new Uint32Array(c.length);for(let s=0;s<O.length;s++){const F=O[s]*3;S[s*3]=c[F],S[s*3+1]=c[F+1],S[s*3+2]=c[F+2]}n.queue.writeBuffer(x,0,o.buffer,o.byteOffset,o.byteLength),n.queue.writeBuffer(y,0,S.buffer,S.byteOffset,S.byteLength),n.queue.writeBuffer(b,0,p.buffer,p.byteOffset,p.byteLength),n.queue.writeBuffer(_,0,f.buffer,f.byteOffset,f.byteLength),n.queue.writeBuffer(g,0,d.buffer,d.byteOffset,d.byteLength);const X=new ArrayBuffer(E.length*B*4);{const s=new DataView(X);for(let F=0;F<E.length;F++){const v=E[F],l=F*B*4;s.setFloat32(l+0,v.minCorner[0],!0),s.setFloat32(l+4,v.minCorner[1],!0),s.setFloat32(l+8,v.minCorner[2],!0),v.isLeaf?s.setUint32(l+12,v.nbTris,!0):s.setUint32(l+12,0,!0),s.setFloat32(l+16,v.maxCorner[0],!0),s.setFloat32(l+20,v.maxCorner[1],!0),s.setFloat32(l+24,v.maxCorner[2],!0),v.isLeaf?s.setUint32(l+28,v.triangleIndex>=0?v.triangleIndex:0,!0):s.setUint32(l+28,v.rightChild>=0?v.rightChild:0,!0)}}const h=n.createBuffer({label:"BVH nodes",size:X.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(h,0,X);const m=4,D=16,C=12,G=16,I=4,T=m*12,R=4,A=12,H=G+I+T+R+D*C,Y=H+A;let W=1;const me=(s,F=0)=>{s.set(e.camera.position,16),s[19]=e.lights.length;for(let l=0;l<e.lights.length;l++)s.set(e.lights[l].position,20+l*12),s[23+l*12]=e.lights[l].intensity,s.set(e.lights[l].color,24+l*12),s.set(e.lights[l].direction,28+l*12),s[31+l*12]=e.lights[l].angle*Math.PI/180;const v=20+m*12;s[v]=u.length,s[v+1]=W,s[v+2]=F,s[v+3]=t.width;for(let l=0;l<u.length;l++){const P=v+4+l*C;s.set(u[l].diffuseAlbedo,P),s[P+3]=u[l].roughness??0,s.set(u[l].fresnel,P+4),s[P+7]=u[l].metalness??0,s[P+8]=u[l].emission,s[P+9]=u[l].basePerlinFreq,s[P+10]=u[l].perlinOctaveAmp,s[P+11]=u[l].perlinOctaveNb}},Z=(s=0,F=0)=>{const v=new Float32Array(Y);v.set(kt(e.camera),0),me(v,F);const l=Gt(e.camera),P=Math.tan(e.camera.fov/2);return v.set([...l.forward,P],H),v.set([...l.right,e.camera.aspect],H+4),v.set([...l.up,s],H+8),v},V=n.createBuffer({size:Y*4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});n.queue.writeBuffer(V,0,Z());const Q=n.createBuffer({size:t.width*t.height*4*4,usage:GPUBufferUsage.STORAGE}),ve=navigator.gpu.getPreferredCanvasFormat(),K=n.createShaderModule({label:"Shader",code:Zt.concat(Wt)}),Ve=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:4,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:6,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:7,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"storage"}}]}),Le=n.createBindGroup({layout:Ve,entries:[{binding:0,resource:{buffer:V}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:y}},{binding:3,resource:{buffer:b}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:g}},{binding:6,resource:{buffer:h}},{binding:7,resource:{buffer:Q}}]}),qe=n.createPipelineLayout({bindGroupLayouts:[Ve]}),Ne=n.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),ze=n.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),J=n.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),He=n.createComputePipeline({layout:"auto",compute:{module:K,entryPoint:"pick_main"}}),_t=n.createBindGroup({layout:He.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:V}},{binding:1,resource:{buffer:x}},{binding:2,resource:{buffer:y}},{binding:3,resource:{buffer:b}},{binding:6,resource:{buffer:h}},{binding:8,resource:{buffer:Ne}},{binding:9,resource:{buffer:ze}}]}),xt=n.createRenderPipeline({label:"Rasterizer pipeline",layout:qe,vertex:{module:K,entryPoint:"rastVertexMain"},fragment:{module:K,entryPoint:"rastFragmentMain",targets:[{format:ve}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),wt=n.createRenderPipeline({label:"Raytraced pipeline",layout:qe,vertex:{module:K,entryPoint:"rayVertexMain"},fragment:{module:K,entryPoint:"rayFragmentMain",targets:[{format:ve}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}}),At=n.createTexture({size:[t.width,t.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),je=12e5,Xe=n.createBuffer({label:"AABB vertices",size:je*8*3*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),Ye=n.createBuffer({label:"AABB indices",size:je*24*4,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),We=n.createShaderModule({label:"AABB shader",code:Kt}),Ze=n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),Ft=n.createBindGroup({layout:Ze,entries:[{binding:0,resource:{buffer:V}},{binding:1,resource:{buffer:Xe}}]}),Bt=n.createRenderPipeline({label:"AABB pipeline",layout:n.createPipelineLayout({bindGroupLayouts:[Ze]}),vertex:{module:We,entryPoint:"aabbVertexMain"},fragment:{module:We,entryPoint:"aabbFragmentMain",targets:[{format:ve}]},primitive:{topology:"line-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"less"}});let L=!1,ge=!1,he=!1,ye=0,be=0,$=1;const Ct=performance.now(),Pt=(s=0)=>{n.queue.writeBuffer(V,0,Z(s,$))},_e=s=>{if(L)return;const F=(s-Ct)/1e3;if(he){const P=F*1.25,q=278+220*Math.cos(P),N=280+220*Math.sin(P),$e=274+180*Math.sin(P);e.lights[3].position.set([q,$e,N]);const we=278-q,Ae=274-$e,Fe=280-N,Be=Math.sqrt(we*we+Ae*Ae+Fe*Fe);e.lights[3].direction.set([we/Be,Ae/Be,Fe/Be])}Pt(F);const v=n.createCommandEncoder(),l=v.beginRenderPass({colorAttachments:[{view:r.getCurrentTexture().createView(),loadOp:"clear",clearValue:{r:0,g:0,b:.4,a:1},storeOp:"store"}],depthStencilAttachment:{view:At.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});ge?(l.setPipeline(wt),l.setBindGroup(0,Le),l.draw(6)):(l.setPipeline(xt),l.setBindGroup(0,Le),l.draw(c.length)),!ge&&be>0&&(l.setPipeline(Bt),l.setBindGroup(0,Ft),l.setIndexBuffer(Ye,"uint32"),l.drawIndexed(be)),l.end(),n.queue.submit([v.finish()]),$++},Ke=s=>{L||(_e(s),ye=requestAnimationFrame(Ke))};ye=requestAnimationFrame(Ke);let ee=0,te=!1;const Et=s=>{xe(),te=!0;const F=()=>{if(L||!te)return;const v=$,l=performance.now();ee=window.setTimeout(()=>{if(L||!te)return;const q=(performance.now()-l)/1e3,N=($-v)/q;s(N),ee=window.setTimeout(F,0)},500)};ee=window.setTimeout(F,500)},xe=()=>{te=!1,clearTimeout(ee)};return{scene:e,bvhRoot:U,render(){L||_e(performance.now())},startAnimation(){he=!0},stopAnimation(){he=!1},setUseRaytracer(s){ge=s},setSpp(s){W=s},resetAccum(){$=1},async pickObject(s,F){if(L)return-1;n.queue.writeBuffer(Ne,0,new Float32Array([s,F]));const v=n.createCommandEncoder(),l=v.beginComputePass();l.setPipeline(He),l.setBindGroup(0,_t),l.dispatchWorkgroups(1),l.end(),v.copyBufferToBuffer(ze,0,J,0,4),n.queue.submit([v.finish()]),await J.mapAsync(GPUMapMode.READ);const P=new Int32Array(J.getMappedRange())[0];return J.unmap(),P},updateMaterial(s,F,v,l,P,q,N){s>=0&&s<e.objects.length&&(e.objects[s].material.diffuseAlbedo.set(F),v!==void 0&&(e.objects[s].material.roughness=v),l!==void 0&&(e.objects[s].material.metalness=l),P!==void 0&&(e.objects[s].material.basePerlinFreq=P),q!==void 0&&(e.objects[s].material.perlinOctaveAmp=q),N!==void 0&&(e.objects[s].material.perlinOctaveNb=N),L||_e(performance.now()))},setDebugAABBs(s){const{positions:F,indices:v}=Nt(s);n.queue.writeBuffer(Xe,0,F),n.queue.writeBuffer(Ye,0,v),be=v.length},startFPSMonitor:Et,stopFPSMonitor:xe,destroy(){L=!0,cancelAnimationFrame(ye),xe(),n.destroy()}}}const Ce=t=>t<=.04045?t/12.92:Math.pow((t+.055)/1.055,2.4),en=t=>{const e=Math.max(0,Math.min(1,t));return e<=.0031308?e*12.92:1.055*Math.pow(e,1/2.4)-.055},tn=t=>{const e=parseInt(t.slice(1,3),16)/255,n=parseInt(t.slice(3,5),16)/255,r=parseInt(t.slice(5,7),16)/255;return[Ce(e),Ce(n),Ce(r)]},nn=(t,e,n)=>{const r=i=>Math.round(en(i)*255).toString(16).padStart(2,"0");return`#${r(t)}${r(e)}${r(n)}`};function oe(t,e){return e===0||t.kind==="leaf"?[t.boundingBox]:[...oe(t.left,e-1),...oe(t.right,e-1)]}function Ue(t){return t.kind==="leaf"?0:1+Math.max(Ue(t.left),Ue(t.right))}const se=document.getElementById("canvas"),De=document.getElementById("no-webgpu"),Ie=document.getElementById("webgpu-content"),st=document.getElementById("performance-warning"),rt=document.getElementById("bvh-section"),lt=document.getElementById("material-section"),it=document.getElementById("model-select"),Me=document.getElementById("animate-checkbox"),Se=document.getElementById("raytraced-checkbox"),ae=document.getElementById("spp-input"),ct=document.getElementById("spp-value"),le=document.getElementById("bvh-depth-input"),ut=document.getElementById("bvh-depth-value"),rn=document.getElementById("material-object-name"),Te=document.getElementById("material-color"),ce=document.getElementById("roughness-input"),ft=document.getElementById("roughness-value"),ue=document.getElementById("metalness-input"),dt=document.getElementById("metalness-value"),fe=document.getElementById("perlin-freq-input"),pt=document.getElementById("perlin-freq-value"),de=document.getElementById("perlin-amp-input"),mt=document.getElementById("perlin-amp-value"),pe=document.getElementById("perlin-octaves-input"),vt=document.getElementById("perlin-octaves-value"),Re=document.getElementById("vulkan-summary"),gt=document.getElementById("vulkan-detail"),an=document.getElementById("vulkan-close");let w=null,ke=-1,Ge=null,Oe="Dragon (2k tri)";function on(){return{color:Te.value,roughness:parseFloat(ce.value),metalness:parseFloat(ue.value),basePerlinFreq:parseFloat(fe.value),perlinOctaveAmp:parseFloat(de.value),perlinOctaveNb:parseFloat(pe.value)}}function j(){if(!w||ke<0)return;const{color:t,roughness:e,metalness:n,basePerlinFreq:r,perlinOctaveAmp:i,perlinOctaveNb:a}=on(),[o,c,d]=tn(t);w.updateMaterial(ke,[o,c,d],e,n,r,i,a),w.resetAccum()}function ht(t){if(!w)return;ke=t;const e=w.scene.objects[t].material;Te.value=nn(e.diffuseAlbedo[0],e.diffuseAlbedo[1],e.diffuseAlbedo[2]),ce.value=String(e.roughness??.5),ft.textContent=(e.roughness??.5).toFixed(2),ue.value=String(e.metalness??0),dt.textContent=(e.metalness??0).toFixed(2),fe.value=String(e.basePerlinFreq),pt.textContent=e.basePerlinFreq.toFixed(1),de.value=String(e.perlinOctaveAmp),mt.textContent=e.perlinOctaveAmp.toFixed(2),pe.value=String(e.perlinOctaveNb),vt.textContent=String(e.perlinOctaveNb),rn.textContent=w.scene.objects[t].label??"",lt.style.display=""}async function yt(t){w?.destroy(),w=null,rt.style.display="none",lt.style.display="none",Me.checked=!1,Se.checked=!1,ae.value="1",ct.textContent="1";const e=await fetch(`/assets/${$t[t]}`).then(a=>a.text()),n=Qt(se,e);w=await Jt(se,n),Ge=w.bvhRoot;const r=Ue(w.bvhRoot);le.max=String(r),le.value="0",ut.textContent="0",rt.style.display="",w.setDebugAABBs(oe(w.bvhRoot,0));const i=n.objects.findIndex(a=>a.label);ht(i!==-1?i:0),w.startFPSMonitor(a=>{st.style.display=a<30?"":"none"})}Re.addEventListener("click",()=>{gt.style.display="",Re.style.display="none"});an.addEventListener("click",()=>{st.style.display="none",gt.style.display="none",Re.style.display=""});se.addEventListener("click",async t=>{if(!w)return;const e=se.getBoundingClientRect(),n=(t.clientX-e.left)/e.width*2-1,r=1-(t.clientY-e.top)/e.height*2,i=await w.pickObject(n,r);if(i<0)return;const a=w.scene.objects.findIndex(o=>o.material.id===i);a!==-1&&ht(a)});it.addEventListener("change",()=>{Oe=it.value,yt(Oe).catch(()=>{Ie.style.display="none",De.style.display=""})});Me.addEventListener("change",()=>{Me.checked?w?.startAnimation():w?.stopAnimation()});Se.addEventListener("change",()=>{w?.setUseRaytracer(Se.checked),w?.resetAccum()});ae.addEventListener("input",()=>{ct.textContent=ae.value,w?.setSpp(parseInt(ae.value)),w?.resetAccum()});le.addEventListener("input",()=>{const t=parseInt(le.value);ut.textContent=String(t),Ge&&w?.setDebugAABBs(oe(Ge,t))});ce.addEventListener("input",()=>{ft.textContent=parseFloat(ce.value).toFixed(2),j()});ue.addEventListener("input",()=>{dt.textContent=parseFloat(ue.value).toFixed(2),j()});fe.addEventListener("input",()=>{pt.textContent=parseFloat(fe.value).toFixed(1),j()});de.addEventListener("input",()=>{mt.textContent=parseFloat(de.value).toFixed(2),j()});pe.addEventListener("input",()=>{vt.textContent=pe.value,j()});Te.addEventListener("input",()=>{j()});const k=new Set,sn=200,ie=1.2;let at=performance.now();window.addEventListener("keydown",t=>{k.add(t.code),["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(t.code)&&t.preventDefault()});window.addEventListener("keyup",t=>k.delete(t.code));const bt=t=>{const e=Math.min((t-at)/1e3,.1);at=t;const n=w?.scene.camera;if(n){const r=sn*e;let i=!1;k.has("KeyW")&&(Qe(n,r),i=!0),k.has("KeyS")&&(Qe(n,-r),i=!0),k.has("KeyA")&&(re(n,-r,0),i=!0),k.has("KeyD")&&(re(n,r,0),i=!0),k.has("KeyR")&&(re(n,0,r),i=!0),k.has("KeyF")&&(re(n,0,-r),i=!0),k.has("ArrowLeft")&&(Je(n,ie*e),i=!0),k.has("ArrowRight")&&(Je(n,-ie*e),i=!0),k.has("ArrowUp")&&(et(n,ie*e),i=!0),k.has("ArrowDown")&&(et(n,-ie*e),i=!0),i&&w?.resetAccum()}requestAnimationFrame(bt)};requestAnimationFrame(bt);navigator.gpu?(Ie.style.display="",yt(Oe).catch(()=>{Ie.style.display="none",De.style.display=""})):De.style.display="";
