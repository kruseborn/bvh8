#pragma once
#include <vector>
#include "utils.h"

static constexpr uint32_t imageWidth = 1024;
static constexpr uint32_t imageHeight = 1024;

struct Bvh8;
void renderBvh8(const Bvh8 &bvh8, std::vector<Vec3> &pixels);
