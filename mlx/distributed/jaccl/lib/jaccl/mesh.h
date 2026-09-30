// Copyright © 2026 Apple Inc.

#pragma once

#include "jaccl/gpu_channel.h"
#include "jaccl/group.h"
#include "jaccl/mesh_impl.h"
#include "jaccl/rdma.h"

namespace jaccl {

/**
 * The JACCL communication group for a fully connected mesh. We expect one
 * connection per peer and it should be the lowest latency communication group
 * for small to medium size messages.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class MeshGroup : public Group {
 public:
  MeshGroup(
      int rank,
      const std::vector<std::string>& device_names,
      SideChannel sc);

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void all_gather(const void* input, void* output, size_t n_bytes) override;

  void sum_scatter(const void* input, void* output, size_t n_bytes, int dtype)
      override;

  void send(const void* input, size_t n_bytes, int dst) override;
  void recv(void* output, size_t n_bytes, int src) override;

  void barrier() override;

  const GpuChannel* gpu_channel() override {
    return gpu_channel_ ? &gpu_channel_->info() : nullptr;
  }

  void gpu_channel_post(uint32_t seq, size_t n_bytes) override {
    gpu_channel_->post(seq, n_bytes);
  }

  std::string gpu_channel_error() override {
    return gpu_channel_ ? gpu_channel_->error_message() : std::string();
  }

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  template <typename T, typename ReduceOp>
  void reduce_scatter(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state and all buffers
   * should have been allocated.
   */
  void initialize();

  /**
   * Allocate all the buffers that we will use in the communication group.
   */
  void allocate_buffers();

  /**
   * Open a second set of queue pairs to the peers, in RTS state. Collective:
   * every rank must call it at the same point.
   */
  std::vector<Connection> connect_extra(
      const std::vector<std::string>& device_names);

  /**
   * MLX_JACCL_RECV_RING=1: set up the ring queue pairs and post their
   * receives.
   */
  void initialize_recv_ring(const std::vector<std::string>& device_names);

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
  std::vector<SharedBuffer> scatter_buffers_;
  std::vector<Connection> ring_connections_;
  std::vector<SharedBuffer> ring_send_;
  std::vector<SharedBuffer> ring_recv_;
  std::unique_ptr<GpuChannelImpl> gpu_channel_;

  MeshImpl mesh_;
};

} // namespace jaccl
