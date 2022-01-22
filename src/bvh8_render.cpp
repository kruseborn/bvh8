#include "bvh8_render.h"

#include <cassert>
#include <inttypes.h>
#include "bvh8_build.h"
#include "bvh8_traverse.h"
#include "utils.h"

static constexpr float aspectRatio = imageWidth / imageHeight;
static constexpr float focalLength = 1.0f;
static constexpr float viewportHeight = 2.0f;
static constexpr float viewportWidth = aspectRatio * viewportHeight;

static constexpr Vec3 horizontal = {viewportWidth, 0, 0};
static constexpr Vec3 vertical = {0, viewportHeight, 0};
static constexpr Vec3 origin = {0, 0, 0.7f};

inline Vec3 getDirection(int32_t x, int32_t y, const Vec3 lowerLeftCorner) {
  float xx = float(x) / float(imageWidth - 1);
  float yy = float(y) / float(imageHeight - 1);
  Vec3 hx = vec3_mul(horizontal, xx);
  Vec3 vy = vec3_mul(vertical, yy);
  return vec3_sub(vec3_add(lowerLeftCorner, vec3_add(hx, vy)), origin);
}

void renderBvh8(const Bvh8 &bvh8, std::vector<Vec3> &pixels) {

  Vec3 lowerLeftCorner = vec3_sub(vec3_sub(origin, vec3_mul(horizontal, 0.5f)),
                                  vec3_sub(vec3_mul(vertical, 0.5), Vec3{0, 0, -focalLength}));

  std::vector<Interval> triangles;
  for (int32_t y = imageHeight - 1; y >= 0; --y) {
    for (int32_t x = 0; x < int32_t(imageWidth); ++x) {
      Ray ray = {};
      ray.origin = origin;
      ray.direction = getDirection(x, y, lowerLeftCorner);
      ray.directionInv = {1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};

      Vec3 pixelColor = {};
      float minT = FLT_MAX;
      auto &mesh = *bvh8.mesh;
      triangles.clear();
      rayBvh8Intersection(bvh8, ray, &triangles);
      for (size_t i = 0; i < triangles.size(); i++) {
        uint32_t start = triangles[i].start;
        uint32_t end = triangles[i].end;

        uint32_t steps = (((end - start) + 8) / 8);
        uint32_t diff = start + steps * 8 - (end + 1);
        start = diff > start ? 0 : start - diff;

        float t[8], u[8], v[8];
        rayTriangleIntersection(ray.origin, ray.direction, mesh, start, steps, t, u, v);
        // _mm256_movemask_ps: Performs an extract operation of sign bits from eight single-precision floating point elements
        if (_mm256_movemask_ps(_mm256_loadu_ps(t)) == 255)
          continue;

        uint32_t diffStart = triangles[i].start - start;
        for (uint32_t j = triangles[i].start, k = 0; j <= triangles[i].end; j++, k++) {
          uint32_t index = k + diffStart;
          if (t[index] > 0 && t[index] < minT) {
            minT = t[index];
            pixelColor = Vec3{u[index], v[index], 1 - u[index] - v[index]};
          }
        }
      }
      pixels.push_back(pixelColor);
    }
  }
}

