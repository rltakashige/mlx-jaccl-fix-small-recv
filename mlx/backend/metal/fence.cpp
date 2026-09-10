// Copyright © 2024 Apple Inc.
#include "mlx/fence.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <arm_acle.h>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace mlx::core {

struct FenceImpl {
  FenceImpl(Stream stream) : producer(stream) {
    auto d = metal::device(Device::gpu).mtl_device();
    if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
      use_fast = false;
    } else if (__builtin_available(macOS 15, iOS 18, *)) {
      use_fast = env::metal_fast_synch();
    }

    if (!use_fast) {
      auto p = metal::new_scoped_memory_pool();
      fence = static_cast<void*>(d->newSharedEvent());
    } else {
      auto buf = allocator::malloc(sizeof(uint32_t)).ptr();
      fence = static_cast<void*>(buf);
      cpu_value()[0] = 0;
      // Same-device cross-stream handoffs (one GPU queue consuming what
      // another GPU queue produced) are ordered with this shared event
      // instead of a fence_wait spin kernel. A spin kernel parked on the
      // consumer queue is not guaranteed to run concurrently with the
      // producer queue's command buffer; when it does not, nothing
      // progresses until the kernel's iteration bound. The spin path stays
      // for CPU<->GPU handoffs, which is what it exists for.
      auto p = metal::new_scoped_memory_pool();
      gpu_event = d->newSharedEvent();
    }
  }

  ~FenceImpl() {
    if (!use_fast) {
      // Wraps Metal SharedEvent
      auto p = metal::new_scoped_memory_pool();
      static_cast<MTL::SharedEvent*>(fence)->release();
    } else {
      if (gpu_event) {
        auto p = metal::new_scoped_memory_pool();
        gpu_event->release();
      }
      allocator::free(allocator::Buffer{static_cast<MTL::Buffer*>(fence)});
    }
  }
  bool use_fast{false};
  Stream producer; // the stream whose arrays this fence publishes
  // Updates published through the fence word (spin path) and through
  // gpu_event (same-device cross-stream path) are counted separately.
  uint32_t count{0};
  uint32_t event_count{0};
  void* fence;
  MTL::SharedEvent* gpu_event{nullptr};
  std::mutex mu;
  std::condition_variable cv;
  // Fast path only: for each array handed across streams, whether its
  // update went through gpu_event and the value a consumer waits for.
  // Only touched on the scheduling thread.
  std::unordered_map<uintptr_t, std::pair<bool, uint32_t>> targets;

  std::atomic_uint* cpu_value() {
    return static_cast<std::atomic_uint*>(
        static_cast<MTL::Buffer*>(fence)->contents());
  }
};

Fence::Fence(Stream stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{stream}, dtor);
}

void Fence::wait(Stream stream, const array& x) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  bool via_event = false;
  uint32_t value = f.count;
  if (f.use_fast) {
    if (auto it = f.targets.find(x.id()); it != f.targets.end()) {
      via_event = it->second.first;
      value = it->second.second;
    }
  }

  if (stream.device == Device::cpu) {
    if (via_event) {
      // x was published through the event because every consumer looked
      // like a GPU stream when it was produced; wait on the event.
      scheduler::enqueue(stream, [fence_ = fence_, value]() mutable {
        auto& f = *static_cast<FenceImpl*>(fence_.get());
        if (!f.gpu_event->waitUntilSignaledValue(value, -1)) {
          throw std::runtime_error("[Fence::wait] Timed out");
        }
      });
      return;
    }
    scheduler::enqueue(stream, [fence_ = fence_, count = value]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      if (!f.use_fast) {
        if (!static_cast<MTL::SharedEvent*>(f.fence)->waitUntilSignaledValue(
                count, -1)) {
          throw std::runtime_error("[Fence::wait] Timed out");
        }
        return;
      }
      while (f.cpu_value()[0] < count) {
      }
    });
    return;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  if (!f.use_fast) {
    compute_encoder.end_encoding();
    auto* command_buffer = compute_encoder.get_command_buffer();
    command_buffer->encodeWait(static_cast<MTL::Event*>(f.fence), f.count);
    command_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    return;
  }

  if (via_event) {
    // Another GPU queue produced x: wait on the event in hardware. Every
    // command encoded after this point in the buffer waits behind it, so
    // there is nothing to register.
    compute_encoder.end_encoding();
    auto* command_buffer = compute_encoder.get_command_buffer();
    command_buffer->encodeWait(f.gpu_event, value);
    command_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    return;
  }

  if (f.producer.device == Device::gpu) {
    // x was published through the fence word because a CPU stream also
    // consumes it, but this consumer is another GPU queue, and a spin
    // kernel here would poll a word the producer queue writes (see above).
    // Signal the event behind the producer queue's work encoded so far and
    // wait on that instead; later GPU consumers of x reuse the signal.
    auto& producer_encoder = metal::get_command_encoder(f.producer);
    producer_encoder.end_encoding();
    auto* producer_buffer = producer_encoder.get_command_buffer();
    f.event_count++;
    producer_buffer->encodeSignalEvent(f.gpu_event, f.event_count);
    producer_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    f.targets[x.id()] = {true, f.event_count};
    compute_encoder.end_encoding();
    auto* command_buffer = compute_encoder.get_command_buffer();
    command_buffer->encodeWait(f.gpu_event, f.event_count);
    command_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    return;
  }

  // Wait CPU-side for the producer to publish the value before encoding the
  // GPU spin kernel. Without this, the fence_wait kernel can run through its
  // entire iteration bound (~7-8s on Apple Silicon) while the producer is
  // blocked on slow CPU work (e.g. a distributed recv waiting on a slow
  // peer). The kernel then returns silently — having neither been satisfied
  // nor signaled failure — and the dependent compute kernel encoded right
  // after it runs on stale/uninitialized data. On some hardware Metal's 5s
  // IOGPU watchdog also fires. CPU-gating ensures the kernel sees the
  // atomic satisfied on iteration 0 and exits in microseconds.
  auto* cpu_val = f.cpu_value();
  for (int i = 0; i < 4096; ++i) {
    if (cpu_val->load(std::memory_order_acquire) >= value) {
      break;
    }
  }
  if (cpu_val->load(std::memory_order_acquire) < value) {
    std::unique_lock<std::mutex> lk(f.mu);
    f.cv.wait(lk, [&]() {
      return cpu_val->load(std::memory_order_acquire) >= value;
    });
  }

  // Register outputs to ensure that no kernels which depends on the
  // output starts before this one is done
  compute_encoder.register_output_array(x);

  auto kernel = d.get_kernel("fence_wait");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(value, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
}

void Fence::update(Stream stream, const array& x, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (stream.device == Device::cpu) {
    f.count++;
    if (f.use_fast) {
      f.targets[x.id()] = {false, f.count};
    }
    scheduler::enqueue(stream, [fence_ = fence_, count = f.count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      if (!f.use_fast) {
        static_cast<MTL::SharedEvent*>(f.fence)->setSignaledValue(count);
        return;
      }

      // seq_cst compiles to DMB ISH (inner-shareable) on ARM64,
      // which is not visible to the GPU. Use DSB SY (full system)
      // to force the store to the point of coherence.
      f.cpu_value()->store(count, std::memory_order_seq_cst);
      __dsb(0xF);
      // Wake any GPU-stream waiter that is CPU-blocking on this fence
      // (see the corresponding cv.wait() in Fence::wait above). The
      // empty locked region establishes happens-before between the store
      // and any waiter observing `count` under the mutex.
      {
        std::lock_guard<std::mutex> lk(f.mu);
      }
      f.cv.notify_all();
    });
    return;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  if (!f.use_fast) {
    f.count++;
    compute_encoder.end_encoding();
    auto* command_buffer = compute_encoder.get_command_buffer();
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(f.fence), f.count);
    command_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    return;
  }

  if (!cross_device) {
    // Every consumer of x is another GPU stream: signal the shared event
    // behind what this queue has encoded so far. The consumer queue waits
    // on it in hardware (see Fence::wait); no spin kernel is involved.
    f.event_count++;
    f.targets[x.id()] = {true, f.event_count};
    compute_encoder.end_encoding();
    auto* command_buffer = compute_encoder.get_command_buffer();
    command_buffer->encodeSignalEvent(f.gpu_event, f.event_count);
    command_buffer->addCompletedHandler(
        [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
    return;
  }

  // A CPU consumer is involved: make x visible across the device boundary
  // and publish the value with a kernel; the CPU polls the fence word.
  f.count++;
  f.targets[x.id()] = {false, f.count};
  {
    auto kernel = d.get_kernel("input_coherent");
    uint32_t nthreads = (x.data_size() * x.itemsize() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    MTL::Size group_dims = MTL::Size(1024, 1, 1);
    MTL::Size grid_dims = MTL::Size((nthreads + 1024 - 1) / 1024, 1, 1);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_bytes(nthreads, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Barrier on previous kernels
  compute_encoder.barrier();

  // Launch value update kernel
  auto kernel = d.get_kernel("fence_update");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  auto buf = static_cast<MTL::Buffer*>(f.fence);
  compute_encoder.set_buffer(buf, 0);
  compute_encoder.set_bytes(f.count, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {
        auto& f = *static_cast<FenceImpl*>(fence_.get());
        {
          std::lock_guard<std::mutex> lk(f.mu);
        }
        f.cv.notify_all();
      });
}

} // namespace mlx::core
