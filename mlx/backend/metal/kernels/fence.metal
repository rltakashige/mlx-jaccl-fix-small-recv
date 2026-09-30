// Copyright © 2024 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// One threadgroup: make input visible to the system, then update timestamp.
// Words go in rounds of 16 per thread; the loads of a round go first so that
// they overlap.
[[kernel]] void input_coherent_fence_update(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    volatile coherent(system) device uint* timestamp [[buffer(2)]],
    constant uint& value [[buffer(3)]],
    uint index [[thread_index_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
  constexpr uint N = 16;
  for (uint base = 0; base < size; base += N * nthreads) {
    uint vals[N];
    for (uint j = 0; j < N; j++) {
      uint i = base + index + j * nthreads;
      if (i < size) {
        vals[j] = input[i];
      }
    }
    for (uint j = 0; j < N; j++) {
      uint i = base + index + j * nthreads;
      if (i < size) {
        input[i] = vals[j];
      }
    }
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
  metal::threadgroup_barrier(metal::mem_flags::mem_device);
  if (index == 0) {
    timestamp[0] = value;
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
  }
}

// single thread kernel to update timestamp
[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to spin wait for timestamp value. The volatile read alone
// can keep returning a stale line; a system-scope atomic load every 4096 spins
// refreshes it from the system level cache.
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    for (uint i = 0; i < 4096; i++) {
      metal::atomic_thread_fence(
          metal::mem_flags::mem_device,
          metal::memory_order_seq_cst,
          metal::thread_scope_system);
      if (timestamp[0] >= value) {
        return;
      }
    }
    uint cur = __metal_atomic_load_explicit(
        timestamp, int(metal::memory_order_relaxed), __METAL_MEMORY_SCOPE_SYSTEM__);
    if (cur >= value) {
      return;
    }
  }
}
