// Copyright © 2026 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

// Control word indices. Keep equal to jaccl::GpuChannel.
constant constexpr uint kSendSeq = 0;
constant constexpr uint kRecvSeq = 16;
constant constexpr uint kConsumed = 32;
constant constexpr uint kError = 48;
constant constexpr uint kGpuError = 64;

inline void system_fence() {
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// One threadgroup: copy n elements of x into the send slot with system
// coherent stores, then publish seq.
template <typename U>
[[kernel]] void jaccl_send_copy(
    device const U* in [[buffer(0)]],
    volatile coherent(system) device U* slot [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    volatile coherent(system) device uint* ctrl [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]]) {
  constexpr uint N = 16;
  for (uint base = 0; base < n; base += N * nt) {
    U v[N];
    for (uint j = 0; j < N; j++) {
      uint i = base + tid + j * nt;
      if (i < n) {
        v[j] = in[i];
      }
    }
    for (uint j = 0; j < N; j++) {
      uint i = base + tid + j * nt;
      if (i < n) {
        slot[i] = v[j];
      }
    }
  }
  system_fence();
  metal::threadgroup_barrier(metal::mem_flags::mem_device);
  if (tid == 0) {
    ctrl[kSendSeq] = seq;
    system_fence();
  }
}

// One threadgroup: thread 0 waits until the peer data of seq is in the recv
// slot (or the host reports an error), then all threads add in float and
// round once, then thread 0 frees the slot. On error the output is not
// written and kGpuError holds seq.
template <typename T>
[[kernel]] void jaccl_recv_add(
    device const T* in [[buffer(0)]],
    volatile coherent(system) device T* peer [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    volatile coherent(system) device uint* ctrl [[buffer(4)]],
    constant uint& seq [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint nt [[threads_per_threadgroup]]) {
  threadgroup uint status;
  if (tid == 0) {
    uint st = 0;
    for (uint it = 1;; it++) {
      system_fence();
      if (ctrl[kRecvSeq] >= seq) {
        break;
      }
      if (ctrl[kError] != 0) {
        st = 1;
        break;
      }
      // As in fence_wait: a system-scope load now and then refreshes a
      // stale line.
      if ((it & 4095) == 0) {
        uint cur = __metal_atomic_load_explicit(
            &ctrl[kRecvSeq],
            int(metal::memory_order_relaxed),
            __METAL_MEMORY_SCOPE_SYSTEM__);
        if (cur >= seq) {
          break;
        }
        uint err = __metal_atomic_load_explicit(
            &ctrl[kError],
            int(metal::memory_order_relaxed),
            __METAL_MEMORY_SCOPE_SYSTEM__);
        if (err != 0) {
          st = 1;
          break;
        }
      }
    }
    status = st;
  }
  metal::threadgroup_barrier(
      metal::mem_flags::mem_threadgroup | metal::mem_flags::mem_device);
  if (status != 0) {
    if (tid == 0) {
      ctrl[kGpuError] = seq;
      system_fence();
    }
    return;
  }
  system_fence();

  constexpr uint N = 8;
  for (uint base = 0; base < n; base += N * nt) {
    T a[N];
    T b[N];
    for (uint j = 0; j < N; j++) {
      uint i = base + tid + j * nt;
      if (i < n) {
        a[j] = in[i];
        b[j] = peer[i];
      }
    }
    for (uint j = 0; j < N; j++) {
      uint i = base + tid + j * nt;
      if (i < n) {
        out[i] = static_cast<T>(static_cast<float>(a[j]) +
                                static_cast<float>(b[j]));
      }
    }
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_device);
  if (tid == 0) {
    ctrl[kConsumed] = seq;
    system_fence();
  }
}

instantiate_kernel("jaccl_send_copy_uint32", jaccl_send_copy, uint32_t)
instantiate_kernel("jaccl_send_copy_uint16", jaccl_send_copy, uint16_t)
instantiate_kernel("jaccl_recv_add_bfloat16", jaccl_recv_add, bfloat16_t)
instantiate_kernel("jaccl_recv_add_float16", jaccl_recv_add, float16_t)
instantiate_kernel("jaccl_recv_add_float32", jaccl_recv_add, float)
