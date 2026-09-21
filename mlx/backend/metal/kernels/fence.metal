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

// single thread kernel to spin wait for timestamp value
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  for (uint i = 0; i < 30; i++) {
    // Fast path: volatile reads through GPU cache + system-scope fence
    for (uint i = 0; i < 1000000; i++) {
      metal::atomic_thread_fence(
          metal::mem_flags::mem_device,
          metal::memory_order_seq_cst,
          metal::thread_scope_system);
      if (timestamp[0] >= value) {
        return;
      }
    }
    // Slow path: force a GPU cache refresh from the SLC before re-checking.
    //
    // This used to call the internal builtin __metal_atomic_load_explicit()
    // directly. That builtin lives behind `#pragma METAL internals : enable`,
    // is undocumented, and its arity changed in the Metal 4.1 compiler
    // (Xcode 27 / macOS 27 SDK), which rejects the 3-argument form with
    // "too few arguments to function call, expected 4, have 3".
    //
    // The public API below is what upstream mlx uses in this same kernel and
    // is already proven to compile here: a system-scope seq_cst device fence
    // followed by a read through the `volatile coherent(system)` pointer.
    // Ordering is strictly stronger than the previous relaxed atomic load,
    // and the read is coherent at system scope, so the observable behaviour
    // (see a value published by another agent) is preserved.
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    uint cur = timestamp[0];
    if (cur >= value) {
      return;
    }
  }
}
