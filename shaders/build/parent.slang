import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint node_buffer_id;
  uint num_of_particles;
};

[vk_push_constant]
PushConstants pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  uint n = pc.num_of_particles;
  if (dispatch_id.x >= n - 1)
    return;

  uint i = dispatch_id.x;
  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);

  uint l = nodes[i].left;
  uint r = nodes[i].right;

  nodes[l].parent = i;
  nodes[r].parent = i;

  if (i == 0) {
    nodes[0].parent = 0xFFFFFFFFu;
  }
}
