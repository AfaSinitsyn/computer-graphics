#version 450

layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=2) in vec3 vertexColor;
layout(location=0) out vec4 outColor;

layout(set=0,binding=0,std140) uniform SceneUniforms {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 lightDirection;
    vec4 lightColor;
    uvec4 lightCounts;
} scene;
layout(set=1,binding=0,std140) uniform ModelUniforms {
    mat4 matrix;
    vec4 albedo;
    vec4 specular;
    vec4 parameters;
} model;

struct PointLight {
    vec4 position;
    vec4 color; // rgb and intensity
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 cutoffs; // cos(inner), cos(outer)
};
// Unsized arrays in storage buffers: no fixed light count in the shader.
layout(set=0,binding=1,std430) readonly buffer PointLights {
    PointLight lights[];
} points;
layout(set=0,binding=2,std430) readonly buffer SpotLights {
    SpotLight lights[];
} spots;

vec3 safeNormalize(vec3 v) {
    return v*inversesqrt(max(dot(v,v),0.000001));
}

// All directions and positions used below are in world space.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diffuse=max(dot(N,L),0.0);
    vec3 H=safeNormalize(L+V);
    float specular=diffuse>0.0
        ? pow(max(dot(N,H),0.0),model.parameters.x) : 0.0;
    return radiance*(albedo*diffuse+model.specular.rgb*specular);
}

void main() {
    vec3 N=safeNormalize(worldNormal);
    vec3 V=safeNormalize(scene.cameraPosition.xyz-worldPosition);
    vec3 albedo=model.albedo.rgb;
    if (model.parameters.y>0.5) albedo*=vertexColor;

    // Ambient is added once for the whole scene.
    vec3 color=albedo*scene.ambient.rgb*scene.ambient.w;
    vec3 L=-safeNormalize(scene.lightDirection.xyz);
    color+=blinnPhong(N,V,L,scene.lightColor.rgb*scene.lightColor.w,albedo);

    for (uint i=0; i<scene.lightCounts.x; ++i) {
        vec3 toLight=points.lights[i].position.xyz-worldPosition;
        float distanceSquared=max(dot(toLight,toLight),0.0001);
        vec3 radiance=points.lights[i].color.rgb*points.lights[i].color.w/distanceSquared;
        color+=blinnPhong(N,V,safeNormalize(toLight),radiance,albedo);
    }
    for (uint i=0; i<scene.lightCounts.y; ++i) {
        vec3 toLight=spots.lights[i].position.xyz-worldPosition;
        L=safeNormalize(toLight);
        float distanceSquared=max(dot(toLight,toLight),0.0001);
        float theta=dot(-L,safeNormalize(spots.lights[i].direction.xyz));
        float inner=spots.lights[i].cutoffs.x;
        float outer=spots.lights[i].cutoffs.y;
        // Equal angles give a hard edge; different angles give a smooth transition.
        float cone=inner-outer>0.000001
            ? clamp((theta-outer)/(inner-outer),0.0,1.0)
            : step(outer,theta);
        vec3 radiance=spots.lights[i].color.rgb*spots.lights[i].color.w*cone/distanceSquared;
        color+=blinnPhong(N,V,L,radiance,albedo);
    }
    outColor=vec4(color,1.0);
}
