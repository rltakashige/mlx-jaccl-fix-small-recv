// Copyright © 2024 Apple Inc.

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

// Two-rank all_sum through the group's GPU channel (MLX_JACCL_GPU_ALLREDUCE).
// K1 copies x into the send slot and publishes seq; the group's progress
// thread sends it and marks the peer's data; K2 waits for it and adds rank 0
// and rank 1 in float with one rounding, as the CPU path does.
void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto ch = group().raw_group()->gpu_all_sum_channel();
  if (reduce_type_ != Sum || ch == nullptr) {
    throw std::runtime_error(
        "[AllReduce::eval_gpu] has no GPU implementation.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& enc = metal::get_command_encoder(s);

  array in = inputs[0];
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_gpu(in, s);
    enc.add_temporary(in);
  }
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  size_t nbytes = in.nbytes();
  if (nbytes > ch->slot_bytes) {
    throw std::runtime_error("[AllReduce::eval_gpu] Array too large.");
  }

  if (ch->gpu_buffer == nullptr) {
    auto b = allocator::make_buffer(ch->base, ch->bytes);
    if (b.ptr() == nullptr) {
      throw std::runtime_error(
          "[AllReduce::eval_gpu] Could not map the channel region.");
    }
    ch->gpu_buffer = b.ptr();
  }
  auto buf = static_cast<MTL::Buffer*>(ch->gpu_buffer);
  uint32_t seq = ++ch->next_seq;
  int slot = seq % 2;

  // K1. The channel buffer is tracked as an output, so K1 and K2 of all
  // collectives stay in order.
  bool words = nbytes % 4 == 0;
  uint32_t n = words ? nbytes / 4 : nbytes / 2;
  auto k1 =
      d.get_kernel(words ? "jaccl_send_copy_uint32" : "jaccl_send_copy_uint16");
  size_t tg1 = std::min<size_t>(1024, k1->maxTotalThreadsPerThreadgroup());
  enc.set_compute_pipeline_state(k1);
  enc.set_input_array(in, 0);
  enc.set_output_buffer(buf, 1, ch->send_offset[slot]);
  enc.set_bytes(n, 2);
  enc.set_output_buffer(buf, 3, ch->ctrl_offset);
  enc.set_bytes(seq, 4);
  enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(tg1, 1, 1));
  ch->post(seq, nbytes);

  // K2
  uint32_t nelem = in.size();
  auto k2 = d.get_kernel("jaccl_recv_add_" + type_to_name(out));
  size_t tg2 = std::min<size_t>(1024, k2->maxTotalThreadsPerThreadgroup());
  enc.set_compute_pipeline_state(k2);
  enc.set_input_array(in, 0);
  enc.set_output_buffer(buf, 1, ch->recv_offset[slot]);
  enc.set_output_array(out, 2);
  enc.set_bytes(nelem, 3);
  enc.set_output_buffer(buf, 4, ch->ctrl_offset);
  enc.set_bytes(seq, 5);
  enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(tg2, 1, 1));

  // A stopped wait leaves `out` unwritten. Stop the process rather than let
  // the result be used.
  auto ctrl = static_cast<std::atomic<uint32_t>*>(
      static_cast<void*>(static_cast<char*>(ch->base) + ch->ctrl_offset));
  enc.get_command_buffer()->addCompletedHandler(
      [ctrl, ch](MTL::CommandBuffer*) {
        uint32_t bad = ctrl[64].load(std::memory_order_acquire);
        if (bad != 0) {
          std::cerr << "[AllReduce::eval_gpu] all_sum " << bad
                    << " did not complete: " << ch->error() << std::endl;
          std::abort();
        }
      });
}

void AllGather::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllGather::eval_gpu] has no GPU implementation.");
}

void Send::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Send::eval_gpu] has no GPU implementation.");
}

void Recv::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Recv::eval_gpu] has no GPU implementation.");
}

void ReduceScatter::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[ReduceScatter::eval_gpu] has no GPU implementation.");
}

} // namespace mlx::core::distributed
