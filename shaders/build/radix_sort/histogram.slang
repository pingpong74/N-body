import "../../sgpu.slang";

struct PushConstants {
  uint count;
  uint shift;
  uint workgroups;
  uint blocks_per_workgroup;

  uint in_buffer_id;
  uint histogram_buffer_id;
};

[vk_push_constant]
PushConstants pc;

groupshared uint histogram[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 grp_thread_id: SV_GroupThreadID, uint3 grp_id: SV_GroupID) {

  if (grp_thread_id.x < 256) {
    histogram[grp_thread_id.x] = 0;
  }

  GroupMemoryBarrierWithGroupSync();

  let values = get_buffer<uint>(pc.in_buffer_id);

  for (uint index = 0; index < pc.blocks_per_workgroup; index++) {
    uint element_id = grp_id.x * pc.blocks_per_workgroup * 256 + index * 256 +
                      grp_thread_id.x;

    if (element_id < pc.count) {
      const uint bin = uint(values[element_id] >> pc.shift) & uint(256 - 1);
      InterlockedAdd(histogram[bin], 1U);
    }
  }

  GroupMemoryBarrierWithGroupSync();

  var histogram_buffer = get_rw_buffer<uint>(pc.histogram_buffer_id);

  if (grp_thread_id.x < 256) {
    histogram_buffer[256 * grp_id.x + grp_thread_id.x] =
        histogram[grp_thread_id.x];
  }
}
