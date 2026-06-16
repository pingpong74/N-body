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
  uint i = dispatch_id.x;

  if (i >= 2 * pc.num_of_particles - 1) {
    return;
  }

  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);

  uint par = nodes[i].parent;

  if (par == 0xFFFFFFFFu) {
    nodes[i].skip_index = 0xFFFFFFFFu;
    return;
  }

  uint cur = i;
  while (true) {
    uint p = nodes[cur].parent;
    if (p == 0xFFFFFFFFu) {
      nodes[i].skip_index = 0xFFFFFFFFu;
      break;
    }
    if (nodes[p].left == cur) {
      nodes[i].skip_index = nodes[p].right;
      break;
    }

    cur = p;
  }
}
