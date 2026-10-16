#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0, std140) uniform GlobalUniforms {
    mat4 matrix;
    vec4 color;
} globalUniforms;

layout(location = 0) out vec3 vertexColor;

void main() {
    gl_Position = globalUniforms.matrix * vec4(inPosition, 1.0);
    vertexColor = inColor * globalUniforms.color.rgb;
}
