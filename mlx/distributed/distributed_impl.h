// Copyright © 2024 Apple Inc.

#pragma once

#include <functional>
#include <optional>
#include <string>

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

/**
 * A two-rank all_sum done by GPU kernels over a GPU-visible region that the
 * group sends from and receives into. See jaccl/gpu_channel.h.
 */
struct GpuAllSumChannel {
  void* base{nullptr};
  size_t bytes{0};
  size_t slot_bytes{0};
  size_t ctrl_offset{0};
  size_t send_offset[2]{0, 0};
  size_t recv_offset[2]{0, 0};
  std::function<void(uint32_t, size_t)> post;
  std::function<std::string()> error;
  // All GPU all_sums use one stream, so seq order is the GPU execution order.
  std::optional<Stream> stream;
  // Set by the GPU backend.
  uint32_t next_seq{0};
  void* gpu_buffer{nullptr};
};

/**
 * Abstract base class of a distributed group implementation.
 */
class GroupImpl {
 public:
  virtual ~GroupImpl() {}

  // Choose the stream this communication group can operate on
  virtual Stream communication_stream(StreamOrDevice s = {}) = 0;

  // Group operations
  virtual int rank() = 0;
  virtual int size() = 0;
  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) = 0;

  // Actual communication operations
  virtual void all_sum(const array& input, array& output, Stream stream) = 0;
  virtual void all_gather(const array& input, array& output, Stream stream) = 0;
  virtual void send(const array& input, int dst, Stream stream) = 0;
  virtual void recv(array& out, int src, Stream stream) = 0;
  virtual void all_max(const array& input, array& output, Stream stream) = 0;
  virtual void all_min(const array& input, array& output, Stream stream) = 0;
  virtual void
  sum_scatter(const array& input, array& output, Stream stream) = 0;

  // The GPU all_sum channel, or nullptr.
  virtual GpuAllSumChannel* gpu_all_sum_channel() {
    return nullptr;
  }
};

/* Define the MLX stream that the communication should happen in. */
Stream communication_stream(Group group, StreamOrDevice s = {});

/* Perform an all reduce sum operation */
void all_sum(Group group, const array& input, array& output, Stream stream);

/* Perform an all gather operation */
void all_gather(Group group, const array& input, array& output, Stream stream);

/** Send an array to the dst rank */
void send(Group group, const array& input, int dst, Stream stream);

/** Recv an array from the src rank */
void recv(Group group, array& out, int src, Stream stream);

/** Max reduction */
void all_max(Group group, const array& input, array& output, Stream stream);

/** Min reduction */
void all_min(Group group, const array& input, array& output, Stream stream);

/** Reduce scatter with average operation */
void sum_scatter(Group group, const array& input, array& output, Stream stream);

} // namespace mlx::core::distributed::detail
