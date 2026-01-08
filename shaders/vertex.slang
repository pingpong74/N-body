#include "common.slang"

struct DrawParams {
  float4x4 view_proj;
  float3 cam_pos;
  float focal_lenght;
};

[[vk_push_constant]]
DrawParams draw_prams;

struct VsOut {
  float4 position : SV_Position;
  float size : SV_PointSize;
};

[shader("vertex")]
VsOut main(uint id: SV_VertexID) {
  VsOut vs;

  let particle_buffer =
      nexion::ReadOnlyBuffer<Particle>::get_buffer(PARTICLES_BINDING);

  let particle = particle_buffer[id];

  vs.position = mul(draw_prams.view_proj, float4(particle.pos, 1.0));
  vs.size =
      particle.radius / (length(draw_prams.cam_pos - particle.pos) * 0.01);

  return vs;
}
