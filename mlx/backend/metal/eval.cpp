// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  // record_gpu_time/record_buffer_ops write into a process-wide stats table
  // keyed by stream index, so they are safe to call from Metal's completion-
  // handler thread without touching the per-thread encoder map.
  int idx = s.index;
  if (encoder.needs_commit()) {
    int ops_in_buffer = encoder.buffer_ops();
    metal::CommandEncoder::record_buffer_ops(idx, ops_in_buffer);
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, idx, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          metal::CommandEncoder::record_gpu_time(
              idx, cbuf->GPUEndTime() - cbuf->GPUStartTime());
          scheduler::notify_task_completion(s);
          check_error(cbuf);
        });
    encoder.commit();
  } else {
    command_buffer->addCompletedHandler(
        [idx, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          metal::CommandEncoder::record_gpu_time(
              idx, cbuf->GPUEndTime() - cbuf->GPUStartTime());
          check_error(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  int idx = s.index;
  int ops_in_buffer = encoder.buffer_ops();
  metal::CommandEncoder::record_buffer_ops(idx, ops_in_buffer);
  encoder.end_encoding();
  cb->addCompletedHandler([idx](MTL::CommandBuffer* cbuf) {
    metal::CommandEncoder::record_gpu_time(
        idx, cbuf->GPUEndTime() - cbuf->GPUStartTime());
    check_error(cbuf);
  });
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
}

} // namespace mlx::core::gpu
