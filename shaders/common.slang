#pragma once

#include "nexion.slang"

struct Particle {
  float mass;
  float3 vel;
  float3 pos;
  float radius;
};

static const uint GROUP_SIZE = 256;
static const uint NO_OF_LODS = 4;

static const uint LOD_0_BINDING = 0;
static const uint LOD_1_BINDING = 1;
static const uint LOD_2_BINDING = 2;
static const uint LOD_3_BINDING = 3;

static const uint PARTICLES_BINDING = 4;
