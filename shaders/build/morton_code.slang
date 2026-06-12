import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint particle_buffer_id;
  uint morton_buffer_id;
  uint num_of_particles;
  Bounds bound;
};

[vk_push_constant]
PushConstants pc;

uint expandBits(uint v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  uint i = dispatch_id.x;

  if (i >= pc.num_of_particles) {
    return;
  }

  let morton_buffer = get_rw_buffer<uint>(pc.morton_buffer_id);

  let particles = get_buffer<Particle>(pc.particle_buffer_id);
  let position = particles[i].position();

  let extent = pc.bound.max - pc.bound.min;
  let n_pos = (position - pc.bound.min) / extent;

  uint x = (uint)(n_pos.x * 1023.0);
  uint y = (uint)(n_pos.y * 1023.0);
  uint z = (uint)(n_pos.z * 1023.0);

  uint morton = expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);

  morton_buffer.store_indexed(morton, i);
}
