import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  uint morton_buffer_id;
  uint indices_buffer_id;
  uint node_buffer_id;
  uint num_of_particles;
};

[vk_push_constant]
PushConstants pc;

int common_prefix(int i, int j, ReadOnlyBuffer<uint> codes, uint n) {
  if (j < 0 || uint(j) >= n) {
    return -1;
  }

  uint a = codes[i];
  uint b = codes[j];

  if (a == b) {
    return 32 + firstbithigh(~(uint(i) ^ uint(j)));
  }

  return firstbithigh(~(a ^ b));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  int i = int(dispatch_id.x);
  int n = int(pc.num_of_particles);
  if (i >= n - 1)
    return;

  let codes = get_buffer<uint>(pc.morton_buffer_id);
  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);

  int d = sign(common_prefix(i, i + 1, codes, pc.num_of_particles) -
               common_prefix(i, i - 1, codes, pc.num_of_particles));

  int min_prefix = common_prefix(i, i - d, codes, pc.num_of_particles);
  int l_max = 2;
  while (common_prefix(i, i + l_max * d, codes, pc.num_of_particles) >
         min_prefix)
    l_max *= 2;

  int l = 0;
  for (int t = l_max >> 1; t >= 1; t >>= 1) {
    if (common_prefix(i, i + (l + t) * d, codes, pc.num_of_particles) >
        min_prefix)
      l += t;
  }
  int j = i + l * d;

  int node_prefix = common_prefix(i, j, codes, pc.num_of_particles);
  int s = 0;
  for (int t = (l + 1) >> 1; t >= 1; t = (t + 1) >> 1) {
    if (common_prefix(i, i + (s + t) * d, codes, pc.num_of_particles) >
        node_prefix)
      s += t;
    if (t == 1)
      break;
  }
  int split = i + s * d + min(d, 0);

  uint left = (min(i, j) == split) ? uint(split + n - 1) : uint(split);
  uint right =
      (max(i, j) == split + 1) ? uint(split + 1 + n - 1) : uint(split + 1);

  nodes[i].left = left;
  nodes[i].right = right;
  // nodes[i].parent = 0;
  nodes[i].is_leaf = 0;

  nodes[left].parent = uint(i);
  nodes[right].parent = uint(i);
}
