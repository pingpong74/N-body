#include "common.slang"

// Simulation params
struct ForceParams {
  uint cell_size; // size of LOD0 cells in world space
  uint lod_0_res;
  float dt;
  float lodFalloff;
};

[[vk::push_constant]]
ForceParams params;

float3 compute_force_from_lod(float3 pos, uint lod_binding, uint lod_num) {
  let lod_cell_size = params.cell_size << lod_num;
  let lod_res = params.lod_0_res >> lod_num;

  let lod_buffer = nexion::ReadOnlyBuffer<float>::get_buffer(lod_binding);

  let sampling_window : int = 8;

  let cell_index = int3(pos / lod_cell_size);
  var force = float3(0);

  for (int x = -sampling_window; x <= sampling_window; x++) {
    for (int y = -sampling_window; y <= sampling_window; y++) {
      for (int z = -sampling_window; z <= sampling_window; z++) {

        int3 c = cell_index + int3(x, y, z);
        if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= lod_res || c.y >= lod_res ||
            c.z >= lod_res)
          continue;
        let mass = lod_buffer[c.x + c.y * lod_res + c.z * lod_res * lod_res];
        let d = float3(c) * lod_cell_size + lod_cell_size * 0.5 - pos;
        force += (mass / (dot(d, d) + 0.001)) * d;
      }
    }
  }

  return force;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint id: SV_DispatchThreadID) {
  var particles =
      nexion::ReadWriteBuffer<Particle>::get_buffer(PARTICLES_BINDING);
  Particle p = particles.load(id.x);

  // Compute gravitational acceleration from LOD grids
  float3 accel = float3(0);
  accel += 0.8 * compute_force_from_lod(p.pos, LOD_0_BINDING, 0);
  accel += 0.87 * compute_force_from_lod(p.pos, LOD_1_BINDING, 1);
  accel += 0.93 * compute_force_from_lod(p.pos, LOD_2_BINDING, 2);
  accel += 1.5 * compute_force_from_lod(p.pos, LOD_3_BINDING, 3);

  // === Integrate ===
  p.vel += accel * params.dt;
  p.pos += p.vel * params.dt;

  float3 min_bound = float3(0.0, 0.0, 0.0);
  let width = params.cell_size * params.lod_0_res;
  float3 max_bound = float3(width, width, width);
  float bounce = 0.8;

  // reflect in each axis
  for (int axis = 0; axis < 3; axis++) {
    if (p.pos[axis] < min_bound[axis]) {
      p.pos[axis] = min_bound[axis];
      p.vel[axis] = -p.vel[axis] * bounce;
    }
    if (p.pos[axis] > max_bound[axis]) {
      p.pos[axis] = max_bound[axis];
      p.vel[axis] = -p.vel[axis] * bounce;
    }
  }

  particles.store(id, p);
}
