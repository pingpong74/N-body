import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint particle_buffer_id;
  uint bounds_buffer_id;
  uint num_of_particles;
};

[vk_push_constant]
PushConstants pc;

groupshared float3 local_min[256];
groupshared float3 local_max[256];

uint float_to_sortable(float f) {
  uint bits = asuint(f);
  uint mask = (bits >> 31) != 0 ? 0xFFFFFFFFu : 0x80000000u;
  return bits ^ mask;
}

float sortable_to_float(uint u) {
  uint mask = (u >> 31) != 0 ? 0x80000000u : 0xFFFFFFFFu;
  return asfloat(u ^ mask);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID,
          uint3 grp_thread_id: SV_GroupThreadID) {

  uint i = dispatch_id.x;
  let particles = get_buffer<Particle>(pc.particle_buffer_id);

  float3 pos = i < pc.num_of_particles ? particles[i].position() : float3(0.0);

  local_min[grp_thread_id.x] = i < pc.num_of_particles ? pos : float3(1e30);
  local_max[grp_thread_id.x] = i < pc.num_of_particles ? pos : float3(-1e30);

  GroupMemoryBarrierWithGroupSync();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (grp_thread_id.x < stride) {
      local_min[grp_thread_id.x] =
          min(local_min[grp_thread_id.x], local_min[grp_thread_id.x + stride]);
      local_max[grp_thread_id.x] =
          max(local_max[grp_thread_id.x], local_max[grp_thread_id.x + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (grp_thread_id.x == 0) {
    var bounds = get_rw_buffer<uint>(pc.bounds_buffer_id);
    bounds.interlocked_max(0 * 4, float_to_sortable(local_max[0].x));
    bounds.interlocked_max(1 * 4, float_to_sortable(local_max[0].y));
    bounds.interlocked_max(2 * 4, float_to_sortable(local_max[0].z));
    bounds.interlocked_min(4 * 4, float_to_sortable(local_min[0].x));
    bounds.interlocked_min(5 * 4, float_to_sortable(local_min[0].y));
    bounds.interlocked_min(6 * 4, float_to_sortable(local_min[0].z));
  }
}
