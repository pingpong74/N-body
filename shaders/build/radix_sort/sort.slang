import "../../sgpu.slang";

struct PushConstants {
  uint count;
  uint shift;
  uint workgroups;
  uint blocks_per_workgroup;

  uint in_buffer_id;
  uint out_buffer_id;
  uint in_index_id;
  uint out_index_id;
  uint histogram_buffer_id;
};

[vk_push_constant]
PushConstants pc;

// IMP!!
// set to 32 for nvidia
// set 64 for amd
static const uint SUBGROUP_COUNT = 32;

groupshared uint sums[256 / SUBGROUP_COUNT];
groupshared uint global_offsets[256];

struct BinFlags {
  uint flags[256 / 32];
};

groupshared BinFlags bin_flags[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 grp_thread_id: SV_GroupThreadID, uint3 grp_id: SV_GroupID) {
  let histogram_buffer = get_buffer<uint>(pc.histogram_buffer_id);

  uint local_histogram = 0;
  uint histogram_count = 0;
  if (grp_thread_id.x < 256) {
    for (uint j = 0; j < pc.workgroups; j++) {
      // read histogram counts of all workgroups for our bin
      const uint t = histogram_buffer[256 * j + grp_thread_id.x];
      local_histogram = (j == grp_id.x) ? histogram_count : local_histogram;
      // sum them up to get the global histogram count for our bin
      histogram_count += t;
    }
  }

  const uint sum = WaveActiveSum(histogram_count);
  const uint prefix_sum = WavePrefixSum(histogram_count);
  if (WaveIsFirstLane()) {
    // only the elected thread of a subgroup writes to the array
    if (WaveGetWaveIndex() < (256 / SUBGROUP_COUNT)) {
      // check if the subgroup ID is within the sums array bounds
      sums[WaveGetWaveIndex()] = sum;
    }
  }

  GroupMemoryBarrierWithGroupSync();

  if (grp_thread_id.x < 256) {
    uint sums_prefix_sum = 0;
    for (uint i = 0; i < WaveGetWaveIndex(); i++) {
      sums_prefix_sum += sums[i];
    }
    global_offsets[grp_thread_id.x] =
        sums_prefix_sum + prefix_sum + local_histogram;
  }

  // meow meow

  const uint flags_bin = grp_thread_id.x / 32;
  const uint flags_bit = 1 << (grp_thread_id.x % 32);

  let in_buffer = get_buffer<uint>(pc.in_buffer_id);
  var out_buffer = get_rw_buffer<uint>(pc.out_buffer_id);

  let in_index_buffer = get_buffer<uint>(pc.in_index_id);
  var out_index_buffer = get_rw_buffer<uint>(pc.out_index_id);

  for (uint index = 0; index < pc.blocks_per_workgroup; index++) {
    uint elementId = grp_id.x * pc.blocks_per_workgroup * 256 + index * 256 +
                     grp_thread_id.x;

    // initialize bin flags
    if (grp_thread_id.x < 256) {
      for (int i = 0; i < 256 / 32; i++) {
        bin_flags[grp_thread_id.x].flags[i] = 0U; // init all bin flags to 0
      }
    }

    GroupMemoryBarrierWithGroupSync();

    uint element_in = 0;
    uint binID = 0;
    uint binOffset = 0;

    if (elementId < pc.count) {
      element_in = in_buffer[elementId];
      binID = uint(element_in >> pc.shift) & uint(256 - 1);
      // offset for group
      binOffset = global_offsets[binID];
      // add bit to flag
      InterlockedAdd(bin_flags[binID].flags[flags_bin], flags_bit);
    }

    GroupMemoryBarrierWithGroupSync();

    if (elementId < pc.count) {
      // calculate output index of element
      uint prefix = 0;
      uint count = 0;
      for (uint i = 0; i < 256 / 32; i++) {
        const uint bits = bin_flags[binID].flags[i];
        const uint full_count = countbits(bits);
        const uint partial_count = countbits(bits & (flags_bit - 1));
        prefix += (i < flags_bin) ? full_count : 0U;
        prefix += (i == flags_bin) ? partial_count : 0U;
        count += full_count;
      }

      out_buffer[binOffset + prefix] = element_in;
      out_index_buffer[binOffset + prefix] = in_index_buffer[elementId];

      if (prefix == count - 1) {
        InterlockedAdd(global_offsets[binID], count);
      }
    }

    GroupMemoryBarrierWithGroupSync();
  }
}
