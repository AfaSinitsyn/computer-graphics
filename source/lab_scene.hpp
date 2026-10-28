#pragma once

#include "lab_math.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lab {

struct Vertex {
    float position[3];
    float color[3];
    float normal[3];
};
static_assert(sizeof(Vertex)==36 && std::is_standard_layout_v<Vertex>);

struct Mesh {
    std::array<Vertex,24> vertices{};
    std::array<uint16_t,36> indices{};
};

inline Mesh makeParallelepiped(Vec3 dimensions) {
    if (!(dimensions.x>0 && dimensions.y>0 && dimensions.z>0))
        throw std::runtime_error("Parallelepiped dimensions must be positive");
    constexpr int signs[8][3] = {
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}
    };
    // Four vertices per face: the same corner has different normals on its faces.
    constexpr int faces[6][4]={
        {0,3,2,1}, {4,5,6,7}, {0,4,7,3},
        {1,2,6,5}, {0,1,5,4}, {3,7,6,2}
    };
    std::array<Vec3,8> corners{};
    for (size_t i=0; i<corners.size(); ++i)
        corners[i]={0.5f*dimensions.x*signs[i][0],
                    0.5f*dimensions.y*signs[i][1],
                    0.5f*dimensions.z*signs[i][2]};
    Mesh result{};
    const float sizes[3]={dimensions.x,dimensions.y,dimensions.z};
    for (size_t face=0; face<6; ++face) {
        const Vec3 a=corners[faces[face][0]];
        const Vec3 b=corners[faces[face][1]];
        const Vec3 c=corners[faces[face][2]];
        const Vec3 n=normalize(cross(b-a,c-a));
        for (size_t j=0; j<4; ++j) {
            const Vec3 p=corners[faces[face][j]];
            auto& vertex=result.vertices[face*4+j];
            const float position[3]={p.x,p.y,p.z};
            const float normal[3]={n.x,n.y,n.z};
            for (int axis=0; axis<3; ++axis) {
                vertex.position[axis]=position[axis];
                vertex.normal[axis]=normal[axis];
                vertex.color[axis]=0.25f+0.75f*(position[axis]/sizes[axis]+0.5f);
            }
        }
        const uint16_t base=static_cast<uint16_t>(face*4);
        const uint16_t local_indices[6]={0,1,2,0,2,3};
        for (size_t j=0; j<6; ++j)
            result.indices[face*6+j]=base+local_indices[j];
    }
    return result;
}

struct ObjectState {
    float position[3]{};
    float rotation[3]{-15.0f,20.0f,0.0f}; // degrees
    float scale[3]{1.0f,1.0f,1.0f};
    float tint[3]{0.75f,0.50f,0.25f}; // material albedo
    float specular[3]{0.7f,0.7f,0.7f};
    float shininess=32.0f;
    bool vertex_colors=false;
    float path_radius=0.75f;
    float path_height=0.30f;
};

inline Mat4 modelMatrix(const ObjectState& object, double animation_time) {
    Vec3 position{object.position[0],object.position[1],object.position[2]};
    Vec3 angles{object.rotation[0],object.rotation[1],object.rotation[2]};
    const float t=static_cast<float>(animation_time);
    position=position+Vec3{
        object.path_radius*std::sin(t),
        object.path_height*std::sin(2.0f*t),
        object.path_radius*(std::cos(t)-1.0f)
    };
    angles.y+=30.0f*t;
    return translation(position)*rotationZ(radians(angles.z))*
           rotationY(radians(angles.y))*rotationX(radians(angles.x))*
           scaling({object.scale[0],object.scale[1],object.scale[2]});
}

struct Camera {
    float position[3]{0,0,-6};
    float rotation[3]{}; // degrees; looks along local +Z
    float fov=45.0f;
    float ortho_height=5.0f;
};

inline Mat4 viewMatrix(const Camera& camera) {
    // Inverse of T * Rz * Ry * Rx: reverse the order and negate each parameter.
    return rotationX(radians(-camera.rotation[0]))*
           rotationY(radians(-camera.rotation[1]))*
           rotationZ(radians(-camera.rotation[2]))*
           translation({-camera.position[0],-camera.position[1],-camera.position[2]});
}

struct DirectionalLight {
    float direction[3]{-0.5f,-1.0f,1.0f}; // direction in which rays travel
    float color[3]{1.0f,1.0f,1.0f};
    float intensity=0.9f;
};
struct PointLight {
    float position[3]{-2,2,-3};
    float color[3]{1.0f,0.65f,0.35f};
    float intensity=8.0f;
};
struct SpotLight {
    float position[3]{2,2,-3};
    float direction[3]{-2,-2,3};
    float color[3]{0.4f,0.65f,1.0f};
    float intensity=16.0f;
    float inner_angle=15.0f; // cone half-angles, degrees
    float outer_angle=25.0f;
};

// vec4 fields keep C++ and GLSL alignment simple (std140 / std430).
struct alignas(16) SceneUniforms {
    Mat4 view_projection;
    float camera_position[4];
    float ambient[4]; // rgb and intensity
    float light_direction[4];
    float light_color[4]; // rgb and intensity
    uint32_t light_counts[4]; // point count, spot count, padding
};
struct alignas(16) ModelUniforms {
    Mat4 model;
    float albedo[4];
    float specular[4];
    float parameters[4]; // shininess, use vertex colors, padding
};
struct alignas(16) PointLightGPU {
    float position[4];
    float color[4]; // rgb and intensity
};
struct alignas(16) SpotLightGPU {
    float position[4];
    float direction[4];
    float color[4];
    float cutoffs[4]; // cos(inner), cos(outer), padding
};
static_assert(sizeof(SceneUniforms)==144 && offsetof(SceneUniforms,light_counts)==128);
static_assert(sizeof(ModelUniforms)==112 && offsetof(ModelUniforms,parameters)==96);
static_assert(sizeof(PointLightGPU)==32 && sizeof(SpotLightGPU)==64);

} // namespace lab
