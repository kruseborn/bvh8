#include "bvh8_traverse.h"

#include <cstring>
#include "bvh8_build.h"

// ray-aabb intersecion, slab method
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
static void rayAABB8Intersecion(Ray r, const Node8 &node, uint32_t out[8]) {
  static __m256 maxFloat = _mm256_set1_ps(FLT_MAX);
  static __m256 zero = _mm256_set1_ps(0.0f);

  __m256 rx = _mm256_set1_ps(r.origin.x);
  __m256 ry = _mm256_set1_ps(r.origin.y);
  __m256 rz = _mm256_set1_ps(r.origin.z);

  __m256 rdx = _mm256_set1_ps(r.directionInv.x);
  __m256 rdy = _mm256_set1_ps(r.directionInv.y);
  __m256 rdz = _mm256_set1_ps(r.directionInv.z);

  // x plane
  __m256 t1 = _mm256_mul_ps(_mm256_sub_ps(node.minX, rx), rdx);
  __m256 t2 = _mm256_mul_ps(_mm256_sub_ps(node.maxX, rx), rdx);

  __m256 tmin = _mm256_min_ps(t1, t2);
  __m256 tmax = _mm256_max_ps(t1, t2);

  // y plane
  t1 = _mm256_mul_ps(_mm256_sub_ps(node.minY, ry), rdy);
  t2 = _mm256_mul_ps(_mm256_sub_ps(node.maxY, ry), rdy);

  tmin = _mm256_max_ps(tmin, _mm256_min_ps(t1, t2));
  tmax = _mm256_min_ps(tmax, _mm256_max_ps(t1, t2));

  // z plane
  t1 = _mm256_mul_ps(_mm256_sub_ps(node.minZ, rz), rdz);
  t2 = _mm256_mul_ps(_mm256_sub_ps(node.maxZ, rz), rdz);

  tmin = _mm256_max_ps(tmin, _mm256_min_ps(t1, t2));
  tmax = _mm256_min_ps(tmax, _mm256_max_ps(t1, t2));

  // tmax >= tmin && tmax >= 0;
  __m256 maxGeMin = _mm256_cmp_ps(tmax, tmin, _CMP_GE_OQ);
  __m256 maxGeZero = _mm256_cmp_ps(tmax, zero, _CMP_GE_OQ);

  __m256 mask = _mm256_and_ps(maxGeMin, maxGeZero);
  __m256 maskValid = _mm256_cmp_ps(node.minY, maxFloat, _CMP_EQ_OQ);
  mask = _mm256_blendv_ps(mask, zero, maskValid);

  uint32_t *mask_ints = (uint32_t *)&mask;
  memcpy(out, mask_ints, sizeof(uint32_t) * 8);
}

void rayBvh8Intersection(const Bvh8 &bvh, Ray ray, std::vector<Interval> *out) {
  const Node8 *stack[64];
  const Node8 **stackPtr = stack;
  *stackPtr++ = nullptr; // push

  const Node8 *node = &bvh.node8[0];
  while (node != nullptr) {
    uint32_t hit[8];
    rayAABB8Intersecion(ray, *node, hit);
    for (int32_t i = 0; i < 8; i++) {
      if (hit[i] > 0) {
        if (node->children[i] == 0)
          out->push_back(node->internval[i]);
        else if (node->children[i] > 0)
          *stackPtr++ = &bvh.node8[node->children[i]]; // push
      }
    }
    node = *--stackPtr; // pop
  }
}

/// Moller-trumbore-ray-triangle-intersection
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
void rayTriangleIntersection(const Vec3 orig, const Vec3 dir, Mesh &mesh, uint32_t start, uint32_t steps, float *tOut,
                              float *uOut, float *vOut) {
  static const __m256 eps = _mm256_set1_ps(1e-8f);
  static const __m256 one = _mm256_set1_ps(1.0f);
  static const __m256 zero = _mm256_set1_ps(0.0f);
  static const __m256 negOne = _mm256_set1_ps(-1.0f);

  __m256 xdir = _mm256_set1_ps(dir.x);
  __m256 ydir = _mm256_set1_ps(dir.y);
  __m256 zdir = _mm256_set1_ps(dir.z);

  __m256 xorigin = _mm256_set1_ps(orig.x);
  __m256 yorigin = _mm256_set1_ps(orig.y);
  __m256 zorigin = _mm256_set1_ps(orig.z);

  for (uint32_t i = 0; i < steps; ++i) {
    uint32_t index = start + i * 8;
    __m256 xv0 = _mm256_loadu_ps(&mesh.xv0[index]);
    __m256 xv1 = _mm256_loadu_ps(&mesh.xv1[index]);
    __m256 xv2 = _mm256_loadu_ps(&mesh.xv2[index]);

    __m256 yv0 = _mm256_loadu_ps(&mesh.yv0[index]);
    __m256 yv1 = _mm256_loadu_ps(&mesh.yv1[index]);
    __m256 yv2 = _mm256_loadu_ps(&mesh.yv2[index]);

    __m256 zv0 = _mm256_loadu_ps(&mesh.zv0[index]);
    __m256 zv1 = _mm256_loadu_ps(&mesh.zv1[index]);
    __m256 zv2 = _mm256_loadu_ps(&mesh.zv2[index]);

    __m256 xv1v0 = _mm256_sub_ps(xv1, xv0);
    __m256 yv1v0 = _mm256_sub_ps(yv1, yv0);
    __m256 zv1v0 = _mm256_sub_ps(zv1, zv0);

    __m256 xv2v0 = _mm256_sub_ps(xv2, xv0);
    __m256 yv2v0 = _mm256_sub_ps(yv2, yv0);
    __m256 zv2v0 = _mm256_sub_ps(zv2, zv0);

    // vec = cross(dir, v2v0);
    __m256 xpvec = _mm256_sub_ps(_mm256_mul_ps(ydir, zv2v0), _mm256_mul_ps(zdir, yv2v0));
    __m256 ypvec = _mm256_sub_ps(_mm256_mul_ps(zdir, xv2v0), _mm256_mul_ps(xdir, zv2v0));
    __m256 zpvec = _mm256_sub_ps(_mm256_mul_ps(xdir, yv2v0), _mm256_mul_ps(ydir, xv2v0));

    // det = vec4_dot(v1v0, pvec);
    __m256 dx = _mm256_mul_ps(xv1v0, xpvec);
    __m256 dy = _mm256_mul_ps(yv1v0, ypvec);
    __m256 dz = _mm256_mul_ps(zv1v0, zpvec);

    __m256 dxdy = _mm256_add_ps(dx, dy);
    __m256 det = _mm256_add_ps(dxdy, dz);

    // invDet = 1 / det
    __m256 invDet = _mm256_div_ps(one, det);

    // tvec = vec4_sub(orig, v0);
    __m256 xtvec = _mm256_sub_ps(xorigin, xv0);
    __m256 ytvec = _mm256_sub_ps(yorigin, yv0);
    __m256 ztvec = _mm256_sub_ps(zorigin, zv0);

    // u = vec4_dot(tvec, pvec) * invDet;
    dx = _mm256_mul_ps(xtvec, xpvec);
    dy = _mm256_mul_ps(ytvec, ypvec);
    dz = _mm256_mul_ps(ztvec, zpvec);

    dxdy = _mm256_add_ps(dx, dy);
    __m256 u = _mm256_add_ps(dxdy, dz);
    u = _mm256_mul_ps(u, invDet);

    // qvec = vec4_cross(tvec, v1v0);
    __m256 xqvec = _mm256_sub_ps(_mm256_mul_ps(ytvec, zv1v0), _mm256_mul_ps(ztvec, yv1v0));
    __m256 yqvec = _mm256_sub_ps(_mm256_mul_ps(ztvec, xv1v0), _mm256_mul_ps(xtvec, zv1v0));
    __m256 zqvec = _mm256_sub_ps(_mm256_mul_ps(xtvec, yv1v0), _mm256_mul_ps(ytvec, xv1v0));

    // v = vec4_dot(dir, qvec) * invDet;
    dx = _mm256_mul_ps(xdir, xqvec);
    dy = _mm256_mul_ps(ydir, yqvec);
    dz = _mm256_mul_ps(zdir, zqvec);

    dxdy = _mm256_add_ps(dx, dy);
    __m256 v = _mm256_add_ps(dxdy, dz);
    v = _mm256_mul_ps(v, invDet);

    // t = dot(v2v0, qvec) * invDet;
    dx = _mm256_mul_ps(xv2v0, xqvec);
    dy = _mm256_mul_ps(yv2v0, yqvec);
    dz = _mm256_mul_ps(zv2v0, zqvec);

    dxdy = _mm256_add_ps(dx, dy);
    __m256 t = _mm256_add_ps(dxdy, dz);
    t = _mm256_mul_ps(t, invDet);

    // if (det < kEpsilon)
    __m256 detMask = _mm256_cmp_ps(det, eps, _CMP_LT_OS);

    // if (u < 0 || u > 1)
    __m256 uLTzero = _mm256_cmp_ps(u, zero, _CMP_LT_OQ);
    __m256 uGEone = _mm256_cmp_ps(u, one, _CMP_GT_OQ);
    __m256 uMask = _mm256_or_ps(uLTzero, uGEone);

    // if (v < 0 || u + v > 1)
    __m256 uv = _mm256_add_ps(u, v);
    __m256 vLTzero = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
    __m256 uvGEone = _mm256_cmp_ps(uv, one, _CMP_GT_OQ);
    __m256 uvMask = _mm256_or_ps(vLTzero, uvGEone);

    __m256 mask = _mm256_or_ps(detMask, uMask);
    mask = _mm256_or_ps(mask, uvMask);

    t = _mm256_blendv_ps(t, negOne, mask);

    _mm256_storeu_ps(tOut, t);
    _mm256_storeu_ps(uOut, u);
    _mm256_storeu_ps(vOut, v);
  }
}
