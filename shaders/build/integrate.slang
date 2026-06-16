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

uint is_leaf(uint node_idx) { return node_idx >= (pc.num_of_particles - 1); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  uint n = pc.num_of_particles;
  if (dispatch_id.x >= n) {
    return;
  }

  var particles = get_rw_buffer<Particle>(pc.particle_buffer_id);
  let nodes = get_buffer<Node>(pc.node_buffer_id);
  let indices = get_buffer<uint>(pc.indices_buffer_id);

  uint i = indices[dispatch_id.x];
  float3 pos = particles[i].position();
  float3 acc = float3(0.0);

  uint node_idx = 0;

  uint iterations = 0;
  while (node_idx != 0xFFFFFFFFu) {

    if (++iterations > 2000) {
      break;
    }

    Node node = nodes[node_idx];

    if (node_idx >= n - 1) {
      if (node.left != i) {
        float3 r = node.com.xyz / node.com.w - pos;
        float dist = length(r) + pc.epsilon;
        acc += (node.com.w / (dist * dist * dist)) * r;
      }
      node_idx = node.skip_index;
      continue;
    }

    float3 size = node.aabb_max.xyz - node.aabb_min.xyz;
    float s = max(size.x, max(size.y, size.z));
    float3 com = node.com.xyz / node.com.w;
    float3 r = com - pos;
    float d = length(r);

    if (s / (d + pc.epsilon) < pc.theta) {
      float dist = d + pc.epsilon;
      acc += (node.com.w / (dist * dist * dist)) * r;
      node_idx = node.skip_index;
    } else {
      node_idx = node.left;
    }
  }

  float3 vel = particles[i].velocity();
  vel += acc * pc.dt;
  pos += vel * pc.dt;

  particles[i].set_velocity(vel);
  particles[i].set_position(pos);
}
