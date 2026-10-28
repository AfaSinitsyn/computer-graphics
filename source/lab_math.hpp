#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace lab {

constexpr float pi = std::numbers::pi_v<float>;
constexpr float radians(float degrees) { return degrees * pi / 180.0f; }

struct Vec3 {
    float x{}, y{}, z{};
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator*(Vec3 a, float value) { return {a.x*value, a.y*value, a.z*value}; }
inline float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

inline Vec3 normalize(Vec3 v) {
    const float length=std::sqrt(dot(v,v));
    return length>0.00001f ? v*(1.0f/length) : Vec3{0,0,1};
}

// Column-major storage, matching a GLSL mat4. Vectors are columns.
struct alignas(16) Mat4 {
    std::array<float, 16> elements{};

    float& operator()(int row, int column) { return elements[column*4+row]; }
    float operator()(int row, int column) const { return elements[column*4+row]; }
};
static_assert(sizeof(Mat4) == 64 && alignof(Mat4) == 16);

inline Mat4 identity() {
    Mat4 result{};
    for (int i=0; i<4; ++i) result(i,i)=1.0f;
    return result;
}

inline Mat4 operator*(const Mat4& left, const Mat4& right) {
    Mat4 result{};
    for (int row=0; row<4; ++row)
        for (int column=0; column<4; ++column)
            for (int k=0; k<4; ++k)
                result(row,column) += left(row,k)*right(k,column);
    return result;
}

inline Mat4 translation(Vec3 position) {
    Mat4 result=identity();
    result(0,3)=position.x; result(1,3)=position.y; result(2,3)=position.z;
    return result;
}

inline Mat4 scaling(Vec3 scale) {
    Mat4 result=identity();
    result(0,0)=scale.x; result(1,1)=scale.y; result(2,2)=scale.z;
    return result;
}

inline Mat4 rotationX(float angle) {
    Mat4 result=identity();
    const float c=std::cos(angle), s=std::sin(angle);
    result(1,1)=c; result(1,2)=-s; result(2,1)=s; result(2,2)=c;
    return result;
}
inline Mat4 rotationY(float angle) {
    Mat4 result=identity();
    const float c=std::cos(angle), s=std::sin(angle);
    result(0,0)=c; result(0,2)=s; result(2,0)=-s; result(2,2)=c;
    return result;
}
inline Mat4 rotationZ(float angle) {
    Mat4 result=identity();
    const float c=std::cos(angle), s=std::sin(angle);
    result(0,0)=c; result(0,1)=-s; result(1,0)=s; result(1,1)=c;
    return result;
}

// Positive camera-space Z, as in lecture 1, Vulkan depth range [0,1].
// The Y sign accounts for
// a positive-height Vulkan viewport, whose screen Y axis points downward.
inline Mat4 perspective(float vertical_fov, float aspect, float near_plane, float far_plane) {
    if (!(aspect>0 && near_plane>0 && far_plane>near_plane && vertical_fov>0 && vertical_fov<pi))
        throw std::runtime_error("Invalid perspective projection parameters");
    const float focal=1.0f/std::tan(vertical_fov/2.0f);
    Mat4 result{};
    result(0,0)=focal/aspect;
    result(1,1)=-focal;
    result(2,2)=far_plane/(far_plane-near_plane);
    result(2,3)=-far_plane*near_plane/(far_plane-near_plane);
    result(3,2)=1.0f;
    return result;
}

inline Mat4 orthographic(float vertical_size, float aspect, float near_plane, float far_plane) {
    if (!(vertical_size>0 && aspect>0 && near_plane>0 && far_plane>near_plane))
        throw std::runtime_error("Invalid orthographic projection parameters");
    Mat4 result=identity();
    result(0,0)=2.0f/(vertical_size*aspect);
    result(1,1)=-2.0f/vertical_size;
    result(2,2)=1.0f/(far_plane-near_plane);
    result(2,3)=-near_plane/(far_plane-near_plane);
    return result;
}

} // namespace lab
