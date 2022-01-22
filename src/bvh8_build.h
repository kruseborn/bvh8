#pragma once

#include <memory_resource>
#include <immintrin.h>
#include <vector>
#include <cfloat>
#include "utils.h"
#include "simple_obj_parser.h"

struct LinearAllocator;

struct AABBs {
  float *xmin, *ymin, *zmin;
  float *xmax, *ymax, *zmax;
};

struct AABB {
  Vec3 min = {FLT_MAX, FLT_MAX, FLT_MAX};
  Vec3 max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
};

struct SSE256 {
  union {
    __m256 m256;
    float f[8];
  };
};

struct Interval {
  uint32_t start, end;
};

struct Node8 {
  __m256 minX, maxX, minY, maxY, minZ, maxZ;
  Interval internval[8];
  uint32_t children[8];
};

struct Bvh8 {
  Mesh *mesh;
  std::pmr::vector<Node8> node8;
};

Bvh8 buildBvh8(Mesh *mesh, LinearAllocator *allocator);
