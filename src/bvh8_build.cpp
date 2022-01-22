#include "bvh8_build.h"

#include <algorithm>
#include <cassert>
#include <vector>
#include <execution>
#include <future>

#include "allocator.h"
#include "utils.h"

struct Node {
  Node *children[8];
  Node *parent;
  AABB aabb;
  uint32_t start, end;
  std::atomic<uint32_t> childCount;
  std::atomic<bool> isUsed;
};

static inline void initNode(Node *node) {
  node->aabb = {};
  node->parent = nullptr;
  node->start = 0;
  node->end = 0;
  node->childCount = 0;
  node->isUsed = false;
}

static inline uint32_t clz(uint32_t value) {
#if _WIN32
  unsigned long leadingZeros = 0;
  leadingZeros = __lzcnt(value);
  return (leadingZeros);
#else
  return __builtin_clz(value);
#endif
}

inline AABB toAABB(const AABBs *aabbs, uint32_t index) {
  AABB aabb;
  aabb.min.x = aabbs->xmin[index];
  aabb.min.y = aabbs->ymin[index];
  aabb.min.z = aabbs->zmin[index];

  aabb.max.x = aabbs->xmax[index];
  aabb.max.y = aabbs->ymax[index];
  aabb.max.z = aabbs->zmax[index];

  return aabb;
}

static inline AABB Union(const AABB &a, const AABB &b) {
  AABB res;
  res.min.x = std::min(a.min.x, b.min.x);
  res.min.y = std::min(a.min.y, b.min.y);
  res.min.z = std::min(a.min.z, b.min.z);

  res.max.x = std::max(a.max.x, b.max.x);
  res.max.y = std::max(a.max.y, b.max.y);
  res.max.z = std::max(a.max.z, b.max.z);
  return res;
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
static inline __m256i expandBits(__m256i v) {
  static __m256i a1 = _mm256_set1_epi32(int32_t(0x00010001));
  static __m256i a2 = _mm256_set1_epi32(int32_t(0xFF0000FF));
  static __m256i b1 = _mm256_set1_epi32(int32_t(0x00000101));
  static __m256i b2 = _mm256_set1_epi32(int32_t(0x0F00F00F));
  static __m256i c1 = _mm256_set1_epi32(int32_t(0x00000011));
  static __m256i c2 = _mm256_set1_epi32(int32_t(0xC30C30C3));
  static __m256i d1 = _mm256_set1_epi32(int32_t(0x00000005));
  static __m256i d2 = _mm256_set1_epi32(int32_t(0x49249249));
  v = _mm256_mullo_epi32(v, a1);
  v = _mm256_and_si256(v, a2);
  v = _mm256_and_si256(_mm256_mullo_epi32(v, b1), b2);
  v = _mm256_and_si256(_mm256_mullo_epi32(v, c1), c2);
  v = _mm256_and_si256(_mm256_mullo_epi32(v, d1), d2);
  return v;
}

// Calculates a 30-bit Morton code for the
static inline __m256i morton3D(__m256 x, __m256 y, __m256 z) {
  __m256i xi = _mm256_cvttps_epi32(x);
  __m256i yi = _mm256_cvttps_epi32(y);
  __m256i zi = _mm256_cvttps_epi32(z);

  __m256i xx = expandBits(xi);
  __m256i yy = expandBits(yi);
  __m256i zz = expandBits(zi);

  static __m256i four = _mm256_set1_epi32(4);
  static __m256i two = _mm256_set1_epi32(2);

  xx = _mm256_mullo_epi32(xx, four);
  yy = _mm256_mullo_epi32(yy, two);

  return _mm256_add_epi32(zz, _mm256_add_epi32(xx, yy));
}

static void calculate64BitsMorton(AABB unionAABB, const AABBs *aabbs, size_t size, uint32_t *mortons) {
  static __m256 range = _mm256_set1_ps(1 << 10);
  static __m256 half = _mm256_set1_ps(0.5f);
  static __m256 one = _mm256_set1_ps(1.0f);


  __m256 minUnionx = _mm256_set1_ps(unionAABB.min.x);
  __m256 minUniony = _mm256_set1_ps(unionAABB.min.y);
  __m256 minUnionz = _mm256_set1_ps(unionAABB.min.z);

  __m256 invLengthx = _mm256_set1_ps(std::max(unionAABB.max.x - unionAABB.min.x, 1e-9f));
  __m256 invLengthy = _mm256_set1_ps(std::max(unionAABB.max.y - unionAABB.min.y, 1e-9f));
  __m256 invLengthz = _mm256_set1_ps(std::max(unionAABB.max.z - unionAABB.min.z, 1e-9f));

  invLengthx = _mm256_div_ps(one, invLengthx);
  invLengthy = _mm256_div_ps(one, invLengthy);
  invLengthz = _mm256_div_ps(one, invLengthz);

  for (size_t i = 0; i < size; i += 8) {
    __m256 xmin = _mm256_loadu_ps(&aabbs->xmin[i]);
    __m256 ymin = _mm256_loadu_ps(&aabbs->ymin[i]);
    __m256 zmin = _mm256_loadu_ps(&aabbs->zmin[i]);

    __m256 xmax = _mm256_loadu_ps(&aabbs->xmax[i]);
    __m256 ymax = _mm256_loadu_ps(&aabbs->ymax[i]);
    __m256 zmax = _mm256_loadu_ps(&aabbs->zmax[i]);

    __m256 centroidx = _mm256_mul_ps(_mm256_add_ps(xmin, xmax), half);
    __m256 centroidy = _mm256_mul_ps(_mm256_add_ps(ymin, ymax), half);
    __m256 centroidz = _mm256_mul_ps(_mm256_add_ps(zmin, zmax), half);

    __m256 offsetx = _mm256_mul_ps(_mm256_sub_ps(centroidx, minUnionx), invLengthx);
    __m256 offsety = _mm256_mul_ps(_mm256_sub_ps(centroidy, minUniony), invLengthy);
    __m256 offsetz = _mm256_mul_ps(_mm256_sub_ps(centroidz, minUnionz), invLengthz);

    __m256 scaledOffsetx = _mm256_mul_ps(offsetx, range);
    __m256 scaledOffsety = _mm256_mul_ps(offsety, range);
    __m256 scaledOffsetz = _mm256_mul_ps(offsetz, range);

    __m256i morton = morton3D(scaledOffsetx, scaledOffsety, scaledOffsetz);
    _mm256_storeu_si256((__m256i *)&mortons[i], morton);
  }
}

// Radix sort based on:
// https://www.geeksforgeeks.org/radix-sort
static void radixSort(LinearAllocator *allocator, uint32_t *mortonCodes, AABBs *aabbs, Mesh *mesh, size_t size) {
  uint32_t *temp_morton = (uint32_t *)allocator->allocate(sizeof(uint32_t) * size, alignof(uint32_t));
  float *temp_aabbs_xmin = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_aabbs_ymin = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_aabbs_zmin = (float *)allocator->allocate(sizeof(float) * size, alignof(float));

  float *temp_aabbs_xmax = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_aabbs_ymax = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_aabbs_zmax = (float *)allocator->allocate(sizeof(float) * size, alignof(float));

  float *temp_xv0 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_xv1 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_xv2 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));

  float *temp_yv0 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_yv1 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_yv2 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));

  float *temp_zv0 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_zv1 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));
  float *temp_zv2 = (float *)allocator->allocate(sizeof(float) * size, alignof(float));

  static uint32_t count[256];

  for (uint32_t s = 0; s < 32; s += 8) {
    memset((void *)count, 0, sizeof(count));
    for (uint32_t i = 0; i < size; i++) {
      ++count[(mortonCodes[i] >> s) & 0xff];
    }
    for (uint32_t i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }
    for (int32_t i = int32_t(size) - 1; i >= 0; i--) {
      uint32_t idx = (mortonCodes[i] >> s) & 0xff;
      temp_morton[--count[idx]] = mortonCodes[i];

      temp_aabbs_xmin[count[idx]] = aabbs->xmin[i];
      temp_aabbs_ymin[count[idx]] = aabbs->ymin[i];
      temp_aabbs_zmin[count[idx]] = aabbs->zmin[i];

      temp_aabbs_xmax[count[idx]] = aabbs->xmax[i];
      temp_aabbs_ymax[count[idx]] = aabbs->ymax[i];
      temp_aabbs_zmax[count[idx]] = aabbs->zmax[i];

      temp_xv0[count[idx]] = mesh->xv0[i];
      temp_xv1[count[idx]] = mesh->xv1[i];
      temp_xv2[count[idx]] = mesh->xv2[i];

      temp_yv0[count[idx]] = mesh->yv0[i];
      temp_yv1[count[idx]] = mesh->yv1[i];
      temp_yv2[count[idx]] = mesh->yv2[i];

      temp_zv0[count[idx]] = mesh->zv0[i];
      temp_zv1[count[idx]] = mesh->zv1[i];
      temp_zv2[count[idx]] = mesh->zv2[i];
    }
    std::swap(mortonCodes, temp_morton);
    std::swap(aabbs->xmin, temp_aabbs_xmin);
    std::swap(aabbs->ymin, temp_aabbs_ymin);
    std::swap(aabbs->zmin, temp_aabbs_zmin);

    std::swap(aabbs->xmax, temp_aabbs_xmax);
    std::swap(aabbs->ymax, temp_aabbs_ymax);
    std::swap(aabbs->zmax, temp_aabbs_zmax);

    std::swap(mesh->xv0, temp_xv0);
    std::swap(mesh->xv1, temp_xv1);
    std::swap(mesh->xv2, temp_xv2);

    std::swap(mesh->yv0, temp_yv0);
    std::swap(mesh->yv1, temp_yv1);
    std::swap(mesh->yv2, temp_yv2);

    std::swap(mesh->zv0, temp_zv0);
    std::swap(mesh->zv1, temp_zv1);
    std::swap(mesh->zv2, temp_zv2);
  }
}

static AABB trianglesToAABBs(Mesh *mesh, AABBs *aabbs) {
  __m256 union_xmin = _mm256_set1_ps(FLT_MAX);
  __m256 union_ymin = _mm256_set1_ps(FLT_MAX);
  __m256 union_zmin = _mm256_set1_ps(FLT_MAX);

  __m256 union_xmax = _mm256_set1_ps(-FLT_MAX);
  __m256 union_ymax = _mm256_set1_ps(-FLT_MAX);
  __m256 union_zmax = _mm256_set1_ps(-FLT_MAX);

  for (uint32_t i = 0; i < mesh->nrOfTriangles; i += 8) {
    __m256 xv0 = _mm256_loadu_ps(&mesh->xv0[i]);
    __m256 xv1 = _mm256_loadu_ps(&mesh->xv1[i]);
    __m256 xv2 = _mm256_loadu_ps(&mesh->xv2[i]);

    __m256 yv0 = _mm256_loadu_ps(&mesh->yv0[i]);
    __m256 yv1 = _mm256_loadu_ps(&mesh->yv1[i]);
    __m256 yv2 = _mm256_loadu_ps(&mesh->yv2[i]);

    __m256 zv0 = _mm256_loadu_ps(&mesh->zv0[i]);
    __m256 zv1 = _mm256_loadu_ps(&mesh->zv1[i]);
    __m256 zv2 = _mm256_loadu_ps(&mesh->zv2[i]);

    __m256 xmin = _mm256_min_ps(_mm256_min_ps(xv0, xv1), xv2);
    __m256 xmax = _mm256_max_ps(_mm256_max_ps(xv0, xv1), xv2);

    __m256 ymin = _mm256_min_ps(_mm256_min_ps(yv0, yv1), yv2);
    __m256 ymax = _mm256_max_ps(_mm256_max_ps(yv0, yv1), yv2);

    __m256 zmin = _mm256_min_ps(_mm256_min_ps(zv0, zv1), zv2);
    __m256 zmax = _mm256_max_ps(_mm256_max_ps(zv0, zv1), zv2);

    _mm256_storeu_ps(&aabbs->xmin[i], xmin);
    _mm256_storeu_ps(&aabbs->xmax[i], xmax);

    _mm256_storeu_ps(&aabbs->ymin[i], ymin);
    _mm256_storeu_ps(&aabbs->ymax[i], ymax);

    _mm256_storeu_ps(&aabbs->zmin[i], zmin);
    _mm256_storeu_ps(&aabbs->zmax[i], zmax);

    union_xmin = _mm256_min_ps(union_xmin, xmin);
    union_xmax = _mm256_max_ps(union_xmax, xmax);

    union_ymin = _mm256_min_ps(union_ymin, ymin);
    union_ymax = _mm256_max_ps(union_ymax, ymax);

    union_zmin = _mm256_min_ps(union_zmin, zmin);
    union_zmax = _mm256_max_ps(union_zmax, zmax);
  }
  AABB unionAABB = {};
  for (size_t i = 0; i < 8; ++i) {
    unionAABB.min.x = std::min(((float *)&union_xmin)[i], unionAABB.min.x);
    unionAABB.min.y = std::min(((float *)&union_ymin)[i], unionAABB.min.y);
    unionAABB.min.z = std::min(((float *)&union_zmin)[i], unionAABB.min.z);

    unionAABB.max.x = std::max(((float *)&union_xmax)[i], unionAABB.max.x);
    unionAABB.max.y = std::max(((float *)&union_ymax)[i], unionAABB.max.y);
    unionAABB.max.z = std::max(((float *)&union_zmax)[i], unionAABB.max.z);
  }
  return unionAABB;
}

static uint32_t findSplit(uint32_t *sortedMortonCodes, uint32_t first, uint32_t last) {
  // Identical Morton codes => split the range in the middle.
  uint32_t firstCode = sortedMortonCodes[first];
  uint32_t lastCode = sortedMortonCodes[last];

  if (firstCode == lastCode)
    return (first + last) >> 1;

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.
  uint32_t commonPrefix = clz(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.
  uint32_t split = first; // initial guess
  uint32_t step = last - first;

  do {
    step = (step + 1) >> 1;           // exponential decrease
    uint32_t newSplit = split + step; // proposed new position

    if (newSplit < last) {
      uint32_t splitCode = sortedMortonCodes[newSplit];
      uint32_t splitPrefix = clz(firstCode ^ splitCode);
      if (splitPrefix > commonPrefix)
        split = newSplit; // accept proposal
    }
  } while (step > 1);

  return split;
}

static inline Node *createLeaf(uint32_t first, uint32_t last, AABBs *aabbs, LinearAllocator *allocator, Node **leafs,
                               std::atomic<uint32_t> *leafIndex) {
  Node *node = (Node *)allocator->allocate(sizeof(Node), alignof(Node));
  initNode(node);
  node->aabb = toAABB(aabbs, first);
  for (uint32_t i = first + 1; i <= last; i++) {
    node->aabb = Union(node->aabb, toAABB(aabbs, i));
  }
  node->start = first;
  node->end = last;
  leafs[++*leafIndex - 1] = node;
  return node;
}

static Node *generateBvhSynchronous(uint32_t *sortedMortonCodes, AABBs *aabbs, uint32_t first, uint32_t last,
                                    LinearAllocator *allocator, Node **leafs, std::atomic<uint32_t> *leafIndex) {
  if (last - first < 8) {
    return createLeaf(first, last, aabbs, allocator, leafs, leafIndex);
  }
  uint32_t split = findSplit(sortedMortonCodes, first, last);

  Node *childA = generateBvhSynchronous(sortedMortonCodes, aabbs, first, split, allocator, leafs, leafIndex);
  Node *childB = generateBvhSynchronous(sortedMortonCodes, aabbs, split + 1, last, allocator, leafs, leafIndex);

  Node *node = (Node *)allocator->allocate(sizeof(Node), alignof(Node));
  initNode(node);
  node->aabb = Union(childA->aabb, childB->aabb);

  childA->parent = node;
  childB->parent = node;

  return node;
}
static Node *generateBvh(uint32_t *sortedMortonCodes, AABBs *aabbs, uint32_t first, uint32_t last, uint32_t depth,
                         LinearAllocator *allocator, Node **leafs, std::atomic<uint32_t> *leafIndex) {
  if (last - first < 8) {
    return createLeaf(first, last, aabbs, allocator, leafs, leafIndex);
  }
  if (depth >= 4)
    return generateBvhSynchronous(sortedMortonCodes, aabbs, first, last, allocator, leafs, leafIndex);

  uint32_t split = findSplit(sortedMortonCodes, first, last);

  auto f1 = std::async(std::launch::async, generateBvh, sortedMortonCodes, aabbs, first, split, depth + 1, allocator,
                       leafs, leafIndex);

  auto f2 = std::async(std::launch::async, generateBvh, sortedMortonCodes, aabbs, split + 1, last, depth + 1, allocator,
                       leafs, leafIndex);

  Node *node = (Node *)allocator->allocate(sizeof(Node), alignof(Node));
  initNode(node);
  Node *childA = f1.get();
  Node *childB = f2.get();

  node->aabb = Union(childA->aabb, childB->aabb);
  childA->parent = node;
  childB->parent = node;
  return node;
}

static inline void clearNode(Node8 *node) {
  assert(node);
  node->minX = _mm256_set1_ps(FLT_MAX);
  node->maxX = _mm256_set1_ps(-FLT_MAX);
  node->minY = _mm256_set1_ps(FLT_MAX);
  node->maxY = _mm256_set1_ps(-FLT_MAX);
  node->minZ = _mm256_set1_ps(FLT_MAX);
  node->maxZ = _mm256_set1_ps(-FLT_MAX);

  memset(node->children, 0, sizeof(int32_t) * 8);
  memset(node->internval, 0, sizeof(int32_t) * 8);
}

static inline void setAABB8(Node8 *node, const AABB aabb, uint32_t index) {
  ((float *)(&node->minX))[index] = aabb.min.x;
  ((float *)(&node->maxX))[index] = aabb.max.x;
  ((float *)(&node->minY))[index] = aabb.min.y;
  ((float *)(&node->maxY))[index] = aabb.max.y;
  ((float *)(&node->minZ))[index] = aabb.min.z;
  ((float *)(&node->maxZ))[index] = aabb.max.z;
}

static void updateAABBFromBottom(Node *leafNode) {
  assert(leafNode);
  assert(leafNode->parent);

  Node *stack[32] = {};
  uint32_t top = 0;
  stack[top++] = leafNode;
  leafNode->isUsed = true;
  AABB aabb = leafNode->aabb;
  while (top > 0) {
    Node *node = stack[--top];
    node->aabb = Union(node->aabb, aabb);
    aabb = node->aabb;
    if (node->isUsed.exchange(true, std::memory_order_acq_rel) == false)
      break;
    if (node->parent != nullptr) {
      stack[top++] = node->parent;
      if (node->childCount == 0 || node->childCount > 4) {
        node->parent->children[node->parent->childCount++] = node;
      } else {
        for (uint32_t i = 0; i < node->childCount; i++) {
          node->parent->children[node->parent->childCount++] = node->children[i];
        }
      }
    }
  }
}

static inline uint32_t bvh2ToBhv8(Node *node, Node8 *nodes4, uint32_t *offset) {
  uint32_t value = *offset;
  ++*offset;
  Node8 *node8 = &nodes4[value];
  clearNode(node8);

  for (uint32_t i = 0; i < node->childCount; i++) {
    setAABB8(node8, node->children[i]->aabb, i);
    node8->internval[i] = {node->children[i]->start, node->children[i]->end};
  }
  for (uint32_t i = 0; i < node->childCount; i++) {
    if (node->children[i]->childCount != 0) {
      node8->children[i] = bvh2ToBhv8(node->children[i], nodes4, offset);
    } else
      node8->children[i] = 0;
  }
  return value;
}

Bvh8 buildBvh8(Mesh *mesh, LinearAllocator *allocator) {
  Bvh8 bvh8 = {};
  bvh8.mesh = mesh;

  size_t count = mesh->nrOfTriangles;
  AABBs aabbs = {};
  aabbs.xmin = (float *)allocator->allocate(count * sizeof(float), alignof(float));
  aabbs.ymin = (float *)allocator->allocate(count * sizeof(float), alignof(float));
  aabbs.zmin = (float *)allocator->allocate(count * sizeof(float), alignof(float));

  aabbs.xmax = (float *)allocator->allocate(count * sizeof(float), alignof(float));
  aabbs.ymax = (float *)allocator->allocate(count * sizeof(float), alignof(float));
  aabbs.zmax = (float *)allocator->allocate(count * sizeof(float), alignof(float));
   
  uint32_t *mortons = (uint32_t *)allocator->allocate(sizeof(uint32_t) * count, alignof(uint32_t));

  AABB unionAABB = trianglesToAABBs(mesh, &aabbs);
  calculate64BitsMorton(unionAABB, &aabbs, count, mortons);

  radixSort(allocator, mortons, &aabbs, mesh, count);

  std::pmr::vector<Node *> leafs(allocator);
  leafs.resize(count/4);
  
  std::atomic<uint32_t> leafIndex = 0;
  //Node *node = generateBvhSynchronous(mortons, &aabbs, 0, uint32_t(count) - 1, allocator, leafs.data(), &leafIndex);
  Node *node = generateBvh(mortons, &aabbs, 0, uint32_t(count) - 1, 0, allocator, leafs.data(), &leafIndex);

  std::for_each_n(std::execution::par_unseq, std::begin(leafs), leafIndex.load(),
                  [](Node *node) { updateAABBFromBottom(node); });

  
  std::pmr::vector<Node8> nodes8(allocator);
  nodes8.resize(leafs.size());
  uint32_t offset = 0;

  bvh2ToBhv8(node, nodes8.data(), &offset);
  bvh8.node8 = nodes8;

  return bvh8;
}
