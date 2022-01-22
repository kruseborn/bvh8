#include "allocator.h"
#include "bvh8_build.h"
#include "bvh8_render.h"
#include "bvh8_traverse.h"
#include "simple_obj_parser.h"
#include "utils.h"

#if defined(WIN32)
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free _aligned_free
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free free
#endif

// Bvh AVX implementation
// Based of https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
int main() { 
  // faster iostreams
  std::ios::sync_with_stdio(false);

  const size_t memorySize = 1024 * 1024 * 256;
  void *data = aligned_malloc(memorySize, alignof(__m256));
  LinearAllocator allocator(data, memorySize);

  Mesh mesh = {};
  Bvh8 bvh8 = {};

  {
    std::cout << "Parsing obj file...\n";
    auto start = Clock::now();
    mesh = createMeshFromObj("c:/git/bvh8/data/buddha.obj");
    //mesh = createMeshFromObj("/mnt/c/git/bvh8/data/buddha.obj");
    auto end = Clock::now();
    std::cout << "Obj file parsed: " << getMs(start, end) << " ms\n";
  }

  std::cout << "Building Bvh...\n";
  {
    auto start = Clock::now();
    bvh8 = buildBvh8(&mesh, &allocator);
    auto end = Clock::now();
    std::cout << "Bvh building done: " << getMs(start, end) << " ms\n";
    std::cout << "Used memory for Bvh8: " << allocator._offset / 1024 / 1024 << " mb\n";
  }

  {
    std::vector<Vec3> pixels;
    pixels.reserve(imageWidth * imageHeight);

    auto start = Clock::now();
    renderBvh8(bvh8, pixels);
    auto end = Clock::now();

    std::cout << "Render bvh done: " << getMs(start, end) << " ms\n";

    FILE *file = fopen("image.ppm", "w");
    std::stringstream stream;
    stream << "P3\n" << imageWidth << " " << imageHeight << "\n225\n";
    for (size_t i = 0, size = pixels.size(); i < size; i++) {
      uint32_t ir = uint32_t(255.999f * pixels[i].x);
      uint32_t ig = uint32_t(255.999f * pixels[i].y);
      uint32_t ib = uint32_t(255.999f * pixels[i].z);
      stream << ir << " " << ig << " " << ib << "\n";
    }
    fwrite(stream.str().c_str(), 1, stream.str().size(), file);
    fclose(file);
    std::cout << "image.ppm stored to disc\n";
  }

  aligned_free(data);

  return 0;
}
