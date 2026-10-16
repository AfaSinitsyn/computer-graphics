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
};
static_assert(sizeof(Vertex)==24 && std::is_standard_layout_v<Vertex>);

struct Mesh {
    std::array<Vertex,8> vertices{};
    std::array<uint16_t,36> indices{};
};

inline Mesh makeParallelepiped(Vec3 dimensions) {
    if (!(dimensions.x>0 && dimensions.y>0 && dimensions.z>0))
        throw std::runtime_error("Parallelepiped dimensions must be positive");
    constexpr int signs[8][3] = {
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}
    };
    Mesh result{};
    const float sizes[3]={dimensions.x,dimensions.y,dimensions.z};
    for (size_t i=0; i<result.vertices.size(); ++i) {
        for (int axis=0; axis<3; ++axis) {
            const float coordinate=0.5f*sizes[axis]*static_cast<float>(signs[i][axis]);
            result.vertices[i].position[axis]=coordinate;
            // Each channel is a function of a local vertex coordinate.
            result.vertices[i].color[axis]=0.25f+0.75f*(coordinate/sizes[axis]+0.5f);
        }
    }
    result.indices={
        0,2,1, 0,3,2,  // -Z
        4,5,6, 4,6,7,  // +Z
        0,4,7, 0,7,3,  // -X
        1,2,6, 1,6,5,  // +X
        0,1,5, 0,5,4,  // -Y
        3,7,6, 3,6,2   // +Y
    };
    return result;
}

struct ObjectState {
    float position[3]{};
    float rotation[3]{-15.0f,20.0f,0.0f}; // degrees
    float scale[3]{1.0f,1.0f,1.0f};
    float tint[3]{1.0f,1.0f,1.0f};
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

// As in lecture 3: one transformation matrix in a uniform buffer.
// std140: mat4 at 0, vec4 at 64. Total: 80 bytes.
struct alignas(16) GlobalUniforms {
    Mat4 matrix;
    float color[4];
};
static_assert(sizeof(GlobalUniforms)==80);
static_assert(offsetof(GlobalUniforms,color)==64);

} // namespace lab
