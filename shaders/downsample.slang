#include "common.slang"

struct DownsampleParams {
  uint src_res;
  uint src_binding;
  uint dst_res;
  uint dst_binding;
};

[[vk_push_constant]]
DownsampleParams dsp;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 gid: SV_DispatchThreadID) {
  // Bounds — restrict to lower LOD space
  if (gid.x >= dsp.dst_res || gid.y >= dsp.dst_res || gid.z >= dsp.dst_res)
    return;

  let src = nexion::ReadOnlyBuffer<float>::get_buffer(dsp.src_binding);
  let dst = nexion::ReadWriteBuffer<float>::get_buffer(dsp.dst_binding);

  // Map output cell → 2x2x2 block inside parent grid
  uint3 base = gid * 2; // (x,y,z) at LOD(N)
  uint src_res = dsp.src_res;

  float accum = 0.0;

  for (uint dx = 0; dx < 2; dx++) {
    for (uint dy = 0; dy < 2; dy++) {
      for (uint dz = 0; dz < 2; dz++) {

        uint3 c = uint3(base.x + dx, base.y + dy, base.z + dz);

        // BOUNDS CHECK
        if (c.x >= src_res || c.y >= src_res || c.z >= src_res)
          continue;

        accum += src[c.x + c.y * src_res + c.z * src_res * src_res];

        // accum += 10;
      }
    }
  }

  // Write average into lower LOD
  uint dstIdx = gid.x + gid.y * dsp.dst_res + gid.z * dsp.dst_res * dsp.dst_res;

  dst.store(dstIdx, accum);
}
