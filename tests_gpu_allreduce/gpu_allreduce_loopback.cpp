// Copyright © 2026 Apple Inc.
//
// Single-box test of the GPU all_sum path (AllReduce::eval_gpu, its kernels,
// and the GPU channel protocol) with the loopback channel: the peer data of
// call k is our data of call k - 1. Checks bits against a CPU-stream add.
//
//   gpu_allreduce_loopback bits   -> prints PASS/FAIL per case, exit 0 if all
//   pass gpu_allreduce_loopback drop   -> call 5 never arrives; must abort with
//   a message
//                                    (run with
//                                    MLX_JACCL_GPU_ALLREDUCE_TIMEOUT_S=2)
//
// Build against an unpacked wheel W (W/mlx/include, W/mlx/lib):
//   clang++ -std=c++20 -O2 -I W/mlx/include -I W/mlx/include/metal_cpp \
//     gpu_allreduce_loopback.cpp -L W/mlx/lib -lmlx -ljaccl \
//     -Wl,-rpath,<installed mlx/lib> -o gpu_allreduce_loopback

#include <cstring>
#include <iostream>

#include "jaccl/gpu_channel.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;
namespace dist = mlx::core::distributed;

class LoopGroup : public dist::detail::GroupImpl {
 public:
  LoopGroup(size_t slot, uint32_t drop) : chan_(slot, drop) {
    auto& i = chan_.info();
    ch_.base = i.base;
    ch_.bytes = i.bytes;
    ch_.slot_bytes = i.slot_bytes;
    ch_.ctrl_offset = i.ctrl_offset;
    for (int b = 0; b < 2; b++) {
      ch_.send_offset[b] = i.send_offset[b];
      ch_.recv_offset[b] = i.recv_offset[b];
    }
    ch_.post = [this](uint32_t s, size_t n) { chan_.post(s, n); };
    ch_.error = [this]() { return chan_.error_message(); };
  }
  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }
  int rank() override {
    return 0;
  }
  int size() override {
    return 2;
  }
  std::shared_ptr<GroupImpl> split(int, int) override {
    throw std::runtime_error("no split");
  }
  void all_sum(const array&, array&, Stream) override {
    throw std::runtime_error("CPU path used");
  }
  void all_gather(const array&, array&, Stream) override {
    throw std::runtime_error("no");
  }
  void send(const array&, int, Stream) override {
    throw std::runtime_error("no");
  }
  void recv(array&, int, Stream) override {
    throw std::runtime_error("no");
  }
  void all_max(const array&, array&, Stream) override {
    throw std::runtime_error("no");
  }
  void all_min(const array&, array&, Stream) override {
    throw std::runtime_error("no");
  }
  void sum_scatter(const array&, array&, Stream) override {
    throw std::runtime_error("no");
  }
  dist::detail::GpuAllSumChannel* gpu_all_sum_channel() override {
    return &ch_;
  }

 private:
  jaccl::GpuChannelImpl chan_;
  dist::detail::GpuAllSumChannel ch_;
};

bool same_bits(const array& a, const array& b) {
  auto ua = a.dtype() == float32 ? view(a, uint32) : view(a, uint16);
  auto ub = b.dtype() == float32 ? view(b, uint32) : view(b, uint16);
  return array_equal(ua, ub).item<bool>();
}

// One case: `iters` chains of `links` all_sums. Link j: x_j = f(v) on the GPU,
// v = all_sum(x_j) = x_j + x_{j-1}. The reference uses a CPU-stream add of
// the same x arrays.
bool run_case(Shape shape, Dtype dt, int links, int iters, bool async) {
  size_t nbytes = shape[0] * shape[1] * size_of(dt);
  dist::Group g(std::make_shared<LoopGroup>(1024 * 1024, 0));
  auto w = astype(
      random::normal({1, shape[1]}, float32, 0.0, 1.0, random::key(3)), dt);
  bool ok = true;
  array prev_x = zeros(shape, dt);
  eval(prev_x);
  for (int it = 0; it < iters && ok; it++) {
    auto v = astype(
        random::normal(shape, float32, 0.0, 3.0, random::key(100 + it)), dt);
    eval(v);
    std::vector<array> xs, ys;
    for (int j = 0; j < links; j++) {
      // A GPU kernel writes x right before the all_sum.
      auto x = multiply(add(v, w), array(0.75f, dt));
      auto y = dist::all_sum(x, g);
      if (y.primitive().stream().device != Device::gpu) {
        std::cout << "all_sum not on the GPU stream" << std::endl;
        return false;
      }
      xs.push_back(x);
      ys.push_back(y);
      v = y;
    }
    if (async) {
      async_eval(ys);
    }
    eval(ys);
    for (int j = 0; j < links; j++) {
      auto ref = add(xs[j], j == 0 ? prev_x : xs[j - 1], Device::cpu);
      eval(ref);
      if (!same_bits(ys[j], ref)) {
        std::cout << "mismatch iter " << it << " link " << j << std::endl;
        ok = false;
        break;
      }
    }
    prev_x = xs.back();
  }
  std::cout << (ok ? "PASS " : "FAIL ") << shape[0] << "x" << shape[1] << " "
            << dt << " bytes=" << nbytes << " links=" << links
            << " iters=" << iters << (async ? " async" : "") << std::endl;
  return ok;
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "bits";
  if (mode == "drop") {
    dist::Group g(std::make_shared<LoopGroup>(64 * 1024, 5));
    auto v = astype(random::normal({1, 5120}), bfloat16);
    for (int j = 0; j < 10; j++) {
      v = dist::all_sum(multiply(v, array(0.5f, bfloat16)), g);
    }
    eval(v);
    std::cout << "NOT ABORTED" << std::endl;
    return 1;
  }
  bool ok = true;
  for (auto dt : {bfloat16, float16, float32}) {
    ok &= run_case({1, 5120}, dt, 130, 20, false);
    ok &= run_case({32, 5120}, dt, 130, 5, true);
    ok &= run_case({3, 7}, dt, 16, 5, false);
  }
  ok &= run_case({1, 5121}, bfloat16, 64, 5, false); // odd byte count / 4
  std::cout << (ok ? "ALL PASS" : "SOME FAIL") << std::endl;
  return ok ? 0 : 1;
}
