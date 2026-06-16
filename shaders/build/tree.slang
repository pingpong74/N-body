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

uint clz(uint x) {
  int msb = firstbithigh(x);
  return msb < 0 ? 32 : uint(31 - msb);
}

int delta(int i, int j, ReadOnlyBuffer<uint> codes, uint n) {
  if (j < 0 || uint(j) >= n) {
    return -1;
  }

  uint ki = codes[i];
  uint kj = codes[j];

  if (ki != kj) {
    return int(clz(ki ^ kj));
  }

  return 32 + int(clz(i ^ j));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id: SV_DispatchThreadID) {
  uint i = dispatch_id.x;
  uint n = pc.num_of_particles;
  if (i >= n - 1)
    return;

  let codes = get_buffer<uint>(pc.morton_buffer_id);
  var nodes = get_rw_buffer<Node>(pc.node_buffer_id);

  // Direction determination
  int d = -1;
  int delta_fwd = delta((int)i, (int)i + 1, codes, n);
  int delta_bwd = delta((int)i, (int)i - 1, codes, n);
  if (delta_fwd >= delta_bwd)
    d = 1; // tie goes to +1

  // δmin = δ(i, i - d)
  int delta_min = delta((int)i, (int)i - d, codes, n);

  // Exponential search for upper bound
  uint l_max = 2;
  while (true) {
    int idx = (int)i + (int)l_max * d;
    if (idx < 0 || (uint)idx >= n)
      break;
    if (delta((int)i, idx, codes, n) <= delta_min)
      break;
    l_max <<= 1;
    if (l_max > n)
      l_max = n;
  }

  // Binary search to find j (the other end)
  uint l = 0;
  for (uint t = l_max >> 1; t >= 1; t >>= 1) {
    int idx = (int)i + (int)(l + t) * d;
    if (idx >= 0 && (uint)idx < n && delta((int)i, idx, codes, n) > delta_min) {
      l += t;
    }
  }
  int j = (int)i + (int)l * d;

  int delta_node = delta((int)i, j, codes, n);
  uint s = 0;
  uint ts = l_max >> 1;
  while (ts >= 1) {
    if (s + ts <= l) {
      int idx = (int)i + (int)(s + ts) * d;
      if (idx >= 0 && (uint)idx < n &&
          delta((int)i, idx, codes, n) > delta_node) {
        s += ts;
      }
    }
    ts >>= 1;
  }

  int gamma = (int)i + (int)s * d + min(d, 0);

  uint left, right;
  uint min_ij = min((uint)i, (uint)j);
  uint max_ij = max((uint)i, (uint)j);
  uint gamma_u = (uint)gamma;

  if (min_ij == gamma_u) {
    left = (n - 1) + gamma_u; // leaf
  } else {
    left = gamma_u; // internal
  }

  if (max_ij == gamma_u + 1) {
    right = (n - 1) + gamma_u + 1; // leaf
  } else {
    right = gamma_u + 1; // internal
  }

  nodes[i].left = left;
  nodes[i].right = right;

  if (i == 0) {
    nodes[0].parent = 0xFFFFFFFFu;
    nodes[0].skip_index = 0xFFFFFFFFu;
  }
}
