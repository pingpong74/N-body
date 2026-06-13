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
  if (dispatch_id.x >= n)
    return;

  uint i = (n - 1) - dispatch_id.x;

  let particles = get_buffer<Particle>(pc.particle_buffer_id);
  let indices = get_buffer<uint>(pc.indices_buffer_id);
  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);
  var counters = get_rw_buffer<uint>(pc.counter_buffer_id);

  // initialize this leaf
  uint leaf = i + n - 1;
  uint particle_idx = indices[i];
  Particle p = particles[particle_idx];
  float3 pos = p.position();

  nodes[leaf].aabb_min = float4(pos, 0.0);
  nodes[leaf].aabb_max = float4(pos, 0.0);
  nodes[leaf].com = float4(pos * p.mass(), p.mass()); // com.w = mass
  nodes[leaf].is_leaf = 1;
  nodes[leaf].left = particle_idx; // store particle index for traversal later
  nodes[leaf].right = particle_idx;

  // walk up the tree
  uint node = nodes[leaf].parent;
  while (true) {
    uint prev = counters.interlocked_add(node * 4, 1);
    if (prev == 0)
      break;

    uint L = nodes[node].left;
    uint R = nodes[node].right;

    float3 lmin = nodes[L].aabb_min.xyz;
    float3 lmax = nodes[L].aabb_max.xyz;
    float3 rmin = nodes[R].aabb_min.xyz;
    float3 rmax = nodes[R].aabb_max.xyz;

    float lmass = nodes[L].com.w;
    float rmass = nodes[R].com.w;
    float3 lcom = nodes[L].com.xyz;
    float3 rcom = nodes[R].com.xyz;

    nodes[node].aabb_min = float4(min(lmin, rmin), 0.0);
    nodes[node].aabb_max = float4(max(lmax, rmax), 0.0);
    nodes[node].com = float4(lcom + rcom, lmass + rmass);
    nodes[node].is_leaf = 0;

    if (node == 0)
      break; // reached root
    node = nodes[node].parent;
  }
}
