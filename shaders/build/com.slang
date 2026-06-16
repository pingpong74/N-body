import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint particle_buffer_id;
  uint indices_buffer_id;
  uint node_buffer_id;
  uint counter_buffer_id;
  uint num_of_particles;
};

[vk_push_constant]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  uint n = pc.num_of_particles;
  if (dispatch_id.x >= n) {
    return;
  }

  let particles = get_buffer<Particle>(pc.particle_buffer_id);
  let indices = get_buffer<uint>(pc.indices_buffer_id);
  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);
  var counters = get_rw_buffer<uint>(pc.counter_buffer_id);

  // initialize this leaf
  uint leaf = dispatch_id.x + n - 1;
  uint particle_idx = indices[dispatch_id.x];
  Particle p = particles[particle_idx];
  float3 pos = p.position();

  nodes[leaf].aabb_min = float4(pos, 0.0);
  nodes[leaf].aabb_max = float4(pos, 0.0);
  nodes[leaf].com = float4(pos * p.mass(), p.mass()); // com.w = mass
  nodes[leaf].left = particle_idx; // store particle index for traversal later
  nodes[leaf].right = particle_idx;

  // walk up the tree
  uint node = nodes[leaf].parent;
  while (true) {
    uint prev = counters.interlocked_add(node * 4, 1);
    if (prev == 0) {
      return;
    }

    Node node_l = nodes[nodes[node].left];
    Node node_r = nodes[nodes[node].right];

    float3 lmin = node_l.aabb_min.xyz;
    float3 lmax = node_l.aabb_max.xyz;
    float3 rmin = node_r.aabb_min.xyz;
    float3 rmax = node_r.aabb_max.xyz;

    float lmass = node_l.com.w;
    float rmass = node_r.com.w;
    float3 lcom = node_l.com.xyz;
    float3 rcom = node_r.com.xyz;

    nodes[node].aabb_min = float4(min(lmin, rmin), 0.0);
    nodes[node].aabb_max = float4(max(lmax, rmax), 0.0);
    nodes[node].com = float4((lcom + rcom), lmass + rmass);

    if (node == 0) {
      break;
    }

    node = nodes[node].parent;
  }
}
