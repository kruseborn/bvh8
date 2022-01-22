#pragma once
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline uint64_t getMs(Time start, Time end) {
  return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

struct Vec3 {
  union {
    struct {
      float x, y, z;
    };
    float data[3];
  };
};

inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
  a.data[0] -= b.data[0];
  a.data[1] -= b.data[1];
  a.data[2] -= b.data[2];
  return a;
}

inline Vec3 vec3_add(Vec3 a, Vec3 b) {
  a.data[0] += b.data[0];
  a.data[1] += b.data[1];
  a.data[2] += b.data[2];
  return a;
}

inline Vec3 vec3_mul(Vec3 a, float s) {
  a.data[0] *= s;
  a.data[1] *= s;
  a.data[2] *= s;
  return a;
}

