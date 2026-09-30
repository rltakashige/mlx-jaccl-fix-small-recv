// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace jaccl {

/**
 * A GPU-visible region for a two-rank all_sum done by GPU kernels. The region
 * is page aligned. Offsets are in bytes from base. The control words are
 * uint32 at ctrl_offset + 4 * GpuChannel::k*.
 */
struct GpuChannel {
  // Control word indices (each on its own 64-byte line).
  static constexpr int kSendSeq = 0; // GPU: data for this seq is in the slot
  static constexpr int kRecvSeq = 16; // CPU: peer data for this seq arrived
  static constexpr int kConsumed = 32; // GPU: recv slot of this seq is free
  static constexpr int kError = 48; // CPU: nonzero stops the GPU wait
  static constexpr int kGpuError = 64; // GPU: seq whose wait stopped on error

  void* base{nullptr};
  size_t bytes{0};
  size_t slot_bytes{0};
  size_t ctrl_offset{0};
  size_t send_offset[2]{0, 0};
  size_t recv_offset[2]{0, 0};
};

/**
 * Abstract base class for a JACCL communication group.
 */
class Group {
 public:
  virtual ~Group() {}

  virtual int rank() = 0;
  virtual int size() = 0;

  virtual void
  all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  /**
   * Reduce scatter with a sum reduction.
   *
   * The input holds size() contiguous chunks of n_bytes each (total
   * size() * n_bytes bytes). After the call, output (n_bytes bytes) on rank r
   * contains the elementwise sum over all ranks of the r-th input chunk.
   */
  virtual void
  sum_scatter(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
  virtual void barrier() = 0;

  /**
   * The GPU all_sum channel, or nullptr when there is none. It exists for a
   * two-rank mesh started with MLX_JACCL_GPU_ALLREDUCE=1.
   */
  virtual const GpuChannel* gpu_channel() {
    return nullptr;
  }

  /**
   * Tell the channel that a kernel will publish seq with n_bytes in send
   * slot seq % 2. Calls must come in seq order, starting at 1.
   */
  virtual void gpu_channel_post(uint32_t seq, size_t n_bytes) {}

  /**
   * Text of the channel error, empty when there is none.
   */
  virtual std::string gpu_channel_error() {
    return {};
  }
};

/**
 * Type IDs for dispatch in the standalone JACCL library.
 *
 * Users pass one of these to all_sum/all_max/all_min so JACCL knows how to
 * interpret the data for typed reduction operations.
 */
enum Dtype {
  Bool = 0,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  BFloat16,
  Float32,
  Float64,
  Complex64,
};

} // namespace jaccl
