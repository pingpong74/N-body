#include "common.slang"

struct ScatterParams {
  uint cell_size;
  uint grid_res;
  uint num_particles;
};

[[vk_push_constant]]
ScatterParams sp;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
  if (tid.x >= sp.num_particles)
    return;

  let lod_0_buffer = nexion::ReadWriteBuffer<float>::get_buffer(LOD_0_BINDING);
  let particles =
      nexion::ReadOnlyBuffer<Particle>::get_buffer(PARTICLES_BINDING);
  Particle p = particles[tid.x];

  // Convert world position → grid index
  float3 rel = p.pos / float(sp.cell_size);
  int3 cell = int3(rel);

  // Bounds check
  if (cell.x < 0 || cell.y < 0 || cell.z < 0)
    return;
  if (cell.x >= sp.grid_res || cell.y >= sp.grid_res || cell.z >= sp.grid_res)
    return;

  // Compute 1D index
  uint idx = cell.x + cell.y * sp.grid_res + cell.z * sp.grid_res * sp.grid_res;

  // ATOMIC scatter add
  lod_0_buffer.atomic_add_f32(idx, p.mass);
  // lod_0_buffer.store(idx, 10000);
}
