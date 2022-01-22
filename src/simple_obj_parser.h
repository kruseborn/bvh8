#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Super basic obj parser, it is not written for performance
// Only supporting triangle faces and positive face indices
// Vertices for the mesh are layed out in memory to support simd
struct Mesh {
  float *xv0, *xv1, *xv2;
  float *yv0, *yv1, *yv2;
  float *zv0, *zv1, *zv2;

  size_t nrOfTriangles;
  std::vector<float> data;
};

inline Mesh createMeshFromObj(const char *filename) {
  assert(filename);

  std::ios_base::sync_with_stdio(false);
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file: " << filename << std::endl;
    exit(1);
  }

  struct Vertex {
    float x, y, z;
  };
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::string buffer;
  while (std::getline(file, buffer)) {
    if (buffer[0] == 'v' && buffer[1] == ' ') {
      std::stringstream ss(buffer);
      float x, y, z;
      char c;
      ss >> c >> x >> y >> z;
      vertices.push_back({x, y, z});
    }
    if (buffer[0] == 'f') {
      std::stringstream ss(buffer);
      char c;
      uint32_t v0, v1, v2;
      uint32_t t0, t1, t2;
      uint32_t n0, n1, n2;
      if (buffer.find("//") != std::string::npos) {
        ss >> c >> v0 >> c >> c >> n0 >> v1 >> c >> c >> n1 >> v2 >> c >> c >> n2;
      } else {
        ss >> c >> v0 >> c >> t0 >> c >> n0 >> v1 >> c >> t1 >> c >> n1 >> v2 >> c >> t2 >> c >> n2;
      }
      indices.push_back(v0 - 1);
      indices.push_back(v1 - 1);
      indices.push_back(v2 - 1);
    }
  }
  // Add indices until nr of triangles is dividible by 8
  // Else the avx instructions may read outside memory
  while ((indices.size() / 3) % 8 != 0) {
    indices.push_back(indices[0]);
    indices.push_back(indices[1]);
    indices.push_back(indices[2]);
  }

  Mesh mesh = {};
  size_t size = indices.size() / 3;
  mesh.data.resize(size * 9);
  mesh.xv0 = mesh.data.data();
  mesh.xv1 = mesh.xv0 + size;
  mesh.xv2 = mesh.xv1 + size;

  mesh.yv0 = mesh.xv2 + size;
  mesh.yv1 = mesh.yv0 + size;
  mesh.yv2 = mesh.yv1 + size;

  mesh.zv0 = mesh.yv2 + size;
  mesh.zv1 = mesh.zv0 + size;
  mesh.zv2 = mesh.zv1 + size;

  mesh.nrOfTriangles = size;

  for (size_t i = 0, j = 0; i < indices.size(); i += 3, j++) {
    mesh.xv0[j] = vertices[indices[i]].x;
    mesh.xv1[j] = vertices[indices[i + 1]].x;
    mesh.xv2[j] = vertices[indices[i + 2]].x;

    mesh.yv0[j] = vertices[indices[i]].y;
    mesh.yv1[j] = vertices[indices[i + 1]].y;
    mesh.yv2[j] = vertices[indices[i + 2]].y;

    mesh.zv0[j] = vertices[indices[i]].z;
    mesh.zv1[j] = vertices[indices[i + 1]].z;
    mesh.zv2[j] = vertices[indices[i + 2]].z;
  }
  return mesh;
}
