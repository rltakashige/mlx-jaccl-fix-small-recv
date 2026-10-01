// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "jaccl/group.h"
#include "jaccl/rdma.h"

namespace jaccl {

/**
 * Two-rank channel for an all_sum done by GPU kernels.
 *
 * Per collective seq (1, 2, ...), slot b = seq % 2:
 *   GPU kernel K1 copies x into send slot b, then stores kSendSeq = seq.
 *   The progress thread sees it, posts the receive for seq + 1 once the GPU
 *   has freed that slot (kConsumed >= seq - 1), sends n_bytes rounded up to
 *   MLX_JACCL_GPU_ALLREDUCE_SEND_ALIGN (default 4096), waits
 *   for the peer's seq in recv slot b and stores kRecvSeq = seq.
 *   GPU kernel K2 waits for kRecvSeq >= seq, adds rank 0 + rank 1, then
 *   stores kConsumed = seq.
 *
 * A receive is always posted before the peer can send into it: the peer
 * sends seq + 1 only after its K2 of seq, which needs our seq, and we post
 * the receive for seq + 1 before we send seq.
 *
 * Waits on the peer are bounded by MLX_JACCL_GPU_ALLREDUCE_TIMEOUT_S
 * (default 60). On a timeout or a failed completion the thread stores
 * kError (so the GPU wait stops), prints the message and aborts the process.
 * A result whose seq did not arrive is never returned.
 */
class GpuChannelImpl {
 public:
  GpuChannelImpl(int rank, std::vector<Connection> conns, size_t slot_bytes);

  /**
   * Test channel with no network. The peer data of seq is our data of
   * seq - 1 (zeros for seq 1), so the two operands differ. All calls must
   * have the same size. With drop_from > 0, seq >= drop_from never arrives
   * (to test the timeout).
   */
  GpuChannelImpl(size_t slot_bytes, uint32_t drop_from);
  ~GpuChannelImpl();

  GpuChannelImpl(const GpuChannelImpl&) = delete;
  GpuChannelImpl& operator=(const GpuChannelImpl&) = delete;

  const GpuChannel& info() const {
    return info_;
  }
  void post(uint32_t seq, size_t n_bytes);
  std::string error_message();

 private:
  std::atomic<uint32_t>& word(int k) {
    return *reinterpret_cast<std::atomic<uint32_t>*>(
        static_cast<char*>(info_.base) + info_.ctrl_offset + 4 * k);
  }
  void setup(size_t slot_bytes);
  void post_recv(uint32_t seq);
  void fail(const std::string& msg);
  void run();

  int rank_;
  int peer_;
  std::vector<Connection> conns_;
  GpuChannel info_;
  ibv_mr* mr_{nullptr};
  double timeout_s_;
  // MLX_JACCL_GPU_ALLREDUCE_SEND_ALIGN, default 4096.
  size_t send_align_{4096};
  bool loopback_{false};
  uint32_t drop_from_{0};
  std::vector<char> prev_;

  std::mutex mtx_;
  std::condition_variable cv_;
  std::deque<std::pair<uint32_t, size_t>> queue_;
  bool stop_{false};
  std::string error_;
  std::thread thread_;
};

} // namespace jaccl
