#version 450

layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inColor;
layout(location=2) in vec3 inNormal;

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

layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(location=2) out vec3 vertexColor;

void main() {
    vec4 position=model.matrix*vec4(inPosition,1.0);
    worldPosition=position.xyz;
    // The inverse transpose also works with non-uniform object scaling.
    worldNormal=transpose(inverse(mat3(model.matrix)))*inNormal;
    vertexColor=inColor;
    gl_Position=scene.viewProjection*position;
}
