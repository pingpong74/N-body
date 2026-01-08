# Multi‑Level LOD Particle Simulation (Compute + Graphics)

This project implements a GPU‑accelerated particle simulation using **multi‑level LOD (Level of Detail)** grids, fully driven by Vulkan compute shaders. Rendering is performed in a separate graphics pass, synchronized with compute each frame. The system is designed for extremely large particle counts while maintaining stable and scalable performance.

---

## Overview

The core idea is to maintain **multiple downsampled grids** (LOD0 → LODn), where each level stores aggregated particle information at a lower resolution. Higher LODs respond to coarse, large‑scale behavior, while LOD0 handles fine detail used for rendering.

This architecture allows efficient spatial queries, collision detection, density estimation, and neighbor lookups across different scales without expensive full‑resolution computations.

---

## 🧩 System Architecture

### **1. Compute Pass (Simulation)**

The simulation runs in a dedicated compute command buffer. It is recorded once (if parameters are static) and executed every frame.

**Steps:**

1. **Scattering / Grid Write (LOD0)**

   * Particle positions are written into the highest‑resolution grid.
   * Each particle determines its cell index.
2. **LOD Pyramid Build (LOD1 → LODn)**

   * Each subsequent LOD waits only for the LOD *below* it.
   * Each level downsamples the previous one, reducing resolution by a factor of 2.
   * No need to wait for all levels globally — the pipeline is sequential.
3. **Final Barrier**

   * Ensures compute writes are visible to graphics.

A semaphore signals completion to the graphics queue.

---

### **2. Graphics Pass (Rendering)**

Rendering uses the particle buffer directly in the vertex shader.

**Steps:**

1. Wait on swapchain image availability.
2. Wait on compute completion semaphore.
3. Fetch particle positions and draw as points.

Projection × View (a single **view_proj** matrix) transforms the particles into clip space.

---

### **3. Synchronization Scheme**

Two command buffers are used:

* **Compute command buffer** → simulation
* **Graphics command buffer** → rendering

Synchronization:

* Compute → Graphics: semaphore + buffer memory barrier
* Swapchain → Graphics: `image_acquired_semaphore`
* Graphics → Present: `present_semaphore`

This ensures that rendering never reads partially updated particle data.

---

## 🏗️ Multi‑Level LOD Algorithm Explained

The project uses a **hierarchical LOD grid system** similar to a mip‑map pyramid but for simulation data.

### **LOD0 (Highest Resolution)**

* Cell size is smallest.
* Particles scatter into this grid.
* Contains the most accurate density and occupancy information.

### **LOD1, LOD2, ... LODn (Coarse Levels)**

* Each level is half the resolution of the previous one:

```
LOD0: 256 × 256 × 256
LOD1: 128 × 128 × 128
LOD2:  64 ×  64 ×  64
...
```

### **Building LODn from LOD0**

LOD generation is sequential:

* LOD1 = downsample(LOD0)
* LOD2 = downsample(LOD1)
* …

At each level, a compute shader:

* Reads 8 child cells (2×2×2 block)
* Aggregates

  * density
  * velocity
  * occupancy
* Writes result to the parent cell

This creates a **pyramid of spatial information**.

---

## 🌋 Why Multi‑Level LOD Helps

### ✔ **Fast neighbor queries**

Instead of scanning high‑res neighborhoods, particles can query coarser LODs for large‑scale context.

### ✔ **Efficient long‑range effects**

Low‑resolution grids are cheap to process and give broad density estimates.

### ✔ **Stable performance**

Each level is 1/8th the cost of the previous.

### ✔ **Parallel friendly**

Each LOD depends only on the level directly below it.

This avoids global sync and keeps GPU occupancy high.

---

## ⏱️ Dispatch Dimensions

Each compute pass uses a fixed number of threads per workgroup:

```
[ 256, 1, 1 ] threads
```

Dispatch count per LOD is computed from the grid resolution:

```
dispatch_x = ceil(grid_res.x / 256)
dispatch_y = grid_res.y
dispatch_z = grid_res.z
```

---

## 📦 Rendering

Particles are rendered as points using a simple pipeline:

* Vertex shader: read buffer, apply `view_proj`, output `gl_Position`
* Fragment shader: shade a point

---

## 🔗 Pipeline Summary

```
[ Compute: Scatter ]
        ↓
[ Compute: LOD1 ]
        ↓
[ Compute: LOD2 ]
        ↓
    ...
        ↓
[ Compute → Graphics Semaphore ]
        ↓
[ Graphics: Draw Particles ]
        ↓
[ Present ]
```

---

## 🧹 Command Buffer Reuse

Because no frame‑dependent data is recorded inside the compute command buffer:

* It can be recorded once at initialization
* Reused every frame

This avoids CPU overhead.

---

## 📌 Summary

This project demonstrates:

* A clean separation of compute and graphics pipelines
* Correct Vulkan synchronization patterns
* A multi‑LOD grid system for scalable particle simulation
* Full GPU‑driven simulation and rendering

Perfect for large particle counts, fluid‑like interactions, or volumetric effects.

If you want, I can extend the README with diagrams, equations, or add a **"How it works step‑by‑step"** section.
