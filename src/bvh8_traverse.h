#pragma once
#include <vector>
#include "bvh8_build.h"

struct Ray {
  Vec3 origin;
  Vec3 directionInv;
  Vec3 direction;
};

void rayBvh8Intersection(const Bvh8 &bvh, Ray ray, std::vector<Interval> *out);
void rayTriangleIntersection(const Vec3 orig, const Vec3 dir, Mesh &mesh, uint32_t start, uint32_t steps, float *tOut,
                              float *uOut, float *vOut);
