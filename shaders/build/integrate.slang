import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint particle_buffer_id;
  uint indices_buffer_id;
  uint node_buffer_id;
  uint num_of_particles;
  float dt;
  float theta;
  float epsilon;
};

[vk_push_constant]
PushConstants pc;

// groupshared uint stack[256][24];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID,
          uint3 grp_thread_id: SV_GroupThreadID) {
  uint i = dispatch_id.x;
  uint n = pc.num_of_particles;
  if (i >= n) {
    return;
  }

  var particles = get_rw_buffer<Particle>(pc.particle_buffer_id);
  let nodes = get_buffer<Node>(pc.node_buffer_id);
  let indices = get_buffer<uint>(pc.indices_buffer_id);

  float3 pos = particles[i].position();
  float3 acc = float3(0.0);

  // iterative traversal starting from root
  uint stack_ptr = 0;
  uint stack[64];
  stack[stack_ptr++] = 0;

  while (stack_ptr > 0) {

    if (stack_ptr >= 64) {
      acc = float3(1000, 0, 0); // or write debug buffer
      break;
    }

    uint node_idx = stack[--stack_ptr];
    Node node = nodes[node_idx];

    if (node.is_leaf == 1) {
      uint particle_idx = node.left;
      if (particle_idx == indices[i])
        continue;

      float3 r = node.com.xyz / node.com.w - pos;
      float dist = length(r) + pc.epsilon;
      acc += (1000 * node.com.w / (dist * dist * dist)) * r;
      continue;
    }

    float3 size = node.aabb_max.xyz - node.aabb_min.xyz;
    float s = max(size.x, max(size.y, size.z));
    float3 com = node.com.xyz / node.com.w;
    float d = length(com - pos);

    if (s / (d + pc.epsilon) < pc.theta) {
      float3 r = com - pos;
      float dist = length(r) + pc.epsilon;
      acc += (1000 * node.com.w / (dist * dist * dist)) * r;
    } else {
      stack[stack_ptr++] = node.left;
      stack[stack_ptr++] = node.right;
    }
  }

  // euler integration
  float3 vel = particles[i].velocity();
  vel += acc * pc.dt;
  pos += vel * pc.dt;

  particles[i].set_velocity(vel);
  particles[i].set_position(pos);
}
