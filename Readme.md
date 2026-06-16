# N-Body Simulation

A GPU-accelerated Barnes–Hut N-body simulation built with Vulkan compute shaders.

## Features

1. Barnes–Hut gravitational approximation
2. Linear Bounding Volume Hierarchy (LBVH) construction using Morton codes
3. Skip-pointer based BVH traversal (stackless)
4. Fully GPU-driven simulation pipeline
5. Real time for 1 million particles (requires aggressive approximation in barns hut algorithm)

## Algorithm

1. Generate Morton codes for all particles.
2. Sort particles by Morton code.
3. Construct an LBVH using Karras' algorithm.
4. Compute center of mass and total mass for each node.
5. Traverse the BVH using skip pointers to approximate gravitational forces.
6. Integrate particle positions and velocities.
