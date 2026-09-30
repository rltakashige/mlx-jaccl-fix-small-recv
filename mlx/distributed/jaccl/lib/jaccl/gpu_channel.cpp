// Copyright © 2026 Apple Inc.

#include "jaccl/gpu_channel.h"

#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace jaccl {

namespace {

constexpr size_t kPage = 16384;

size_t round_up(size_t n) {
  return (n + kPage - 1) / kPage * kPage;
}

} // namespace

GpuChannelImpl::GpuChannelImpl(
    int rank,
    std::vector<Connection> conns,
    size_t slot_bytes)
    : rank_(rank), peer_(1 - rank), conns_(std::move(conns)) {
  setup(slot_bytes);
  mr_ = ibv().reg_mr(
      conns_[peer_].protection_domain,
      info_.base,
      info_.bytes,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE);
  if (mr_ == nullptr) {
    throw std::runtime_error("[jaccl] GPU channel memory registration failed");
  }

  // Receives for seq 1 and 2 go in before any peer can send.
  post_recv(1);
  post_recv(2);

  thread_ = std::thread([this] { run(); });
}

GpuChannelImpl::GpuChannelImpl(size_t slot_bytes, uint32_t drop_from)
    : rank_(0), peer_(1), loopback_(true), drop_from_(drop_from) {
  setup(slot_bytes);
  thread_ = std::thread([this] { run(); });
}

void GpuChannelImpl::setup(size_t slot_bytes) {
  const char* t = std::getenv("MLX_JACCL_GPU_ALLREDUCE_TIMEOUT_S");
  timeout_s_ = t ? std::atof(t) : 60.0;

  slot_bytes = round_up(slot_bytes);
  info_.slot_bytes = slot_bytes;
  info_.ctrl_offset = 0;
  for (int b = 0; b < 2; b++) {
    info_.send_offset[b] = kPage + b * slot_bytes;
    info_.recv_offset[b] = kPage + (2 + b) * slot_bytes;
  }
  info_.bytes = kPage + 4 * slot_bytes;
  size_t page = std::max<size_t>(kPage, sysconf(_SC_PAGESIZE));
  if (posix_memalign(&info_.base, page, info_.bytes)) {
    throw std::runtime_error("[jaccl] GPU channel allocation failed");
  }
  std::memset(info_.base, 0, info_.bytes);
}

GpuChannelImpl::~GpuChannelImpl() {
  {
    std::lock_guard lk(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  if (mr_ != nullptr) {
    ibv().dereg_mr(mr_);
  }
  conns_.clear();
  std::free(info_.base);
}

void GpuChannelImpl::post(uint32_t seq, size_t n_bytes) {
  {
    std::lock_guard lk(mtx_);
    queue_.emplace_back(seq, n_bytes);
  }
  cv_.notify_one();
}

std::string GpuChannelImpl::error_message() {
  std::lock_guard lk(mtx_);
  return error_;
}

void GpuChannelImpl::post_recv(uint32_t seq) {
  ibv_sge sge;
  sge.addr = reinterpret_cast<uintptr_t>(
      static_cast<char*>(info_.base) + info_.recv_offset[seq % 2]);
  sge.length = info_.slot_bytes;
  sge.lkey = mr_->lkey;
  ibv_recv_wr wr = {}, *bad = nullptr;
  wr.wr_id = RECV_WR << 16 | (seq % 2) << 8 | peer_;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (int status = ibv_post_recv(conns_[peer_].queue_pair, &wr, &bad)) {
    std::ostringstream msg;
    msg << "[jaccl] GPU channel recv post failed with error " << status;
    throw std::runtime_error(msg.str());
  }
}

void GpuChannelImpl::fail(const std::string& msg) {
  {
    std::lock_guard lk(mtx_);
    if (error_.empty()) {
      error_ = msg;
    }
  }
  word(GpuChannel::kError).store(1, std::memory_order_release);
}

void GpuChannelImpl::run() {
  using clk = std::chrono::steady_clock;
  uint64_t recv_done = 0;
  uint32_t recv_len[4] = {0, 0, 0, 0};
  int sends_pending = 0;
  auto stopped = [&] {
    std::lock_guard lk(mtx_);
    return stop_;
  };
  auto reap = [&]() {
    if (loopback_) {
      return;
    }
    ibv_wc wc[8];
    int n = conns_[peer_].poll(8, wc);
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        std::ostringstream msg;
        msg << "[jaccl] GPU channel completion failed with status "
            << wc[i].status;
        throw std::runtime_error(msg.str());
      }
      if ((wc[i].wr_id >> 16) == SEND_WR) {
        sends_pending--;
      } else {
        recv_len[recv_done % 4] = wc[i].byte_len;
        recv_done++;
      }
    }
  };

  try {
    while (true) {
      uint32_t seq;
      size_t n;
      {
        std::unique_lock lk(mtx_);
        cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
        if (stop_) {
          return;
        }
        std::tie(seq, n) = queue_.front();
        queue_.pop_front();
      }

      // Our own GPU publishes seq. No bound: it depends only on local work.
      uint64_t spins = 0;
      while (word(GpuChannel::kSendSeq).load(std::memory_order_acquire) < seq) {
        if ((++spins & 0xFFFFF) == 0 && stopped()) {
          return;
        }
      }

      // Free the recv slot of seq + 1 (it held seq - 1) and post it.
      if (seq >= 2) {
        auto end = clk::now() + std::chrono::duration<double>(timeout_s_);
        while (word(GpuChannel::kConsumed).load(std::memory_order_acquire) <
               seq - 1) {
          if (clk::now() > end) {
            throw std::runtime_error(
                "[jaccl] GPU channel: the GPU did not free a recv slot");
          }
        }
        if (!loopback_) {
          post_recv(seq + 1);
        }
      }

      // Send exactly n bytes from send slot seq % 2.
      char* base = static_cast<char*>(info_.base);
      if (loopback_) {
        if (drop_from_ == 0 || seq < drop_from_) {
          prev_.resize(n, 0);
          std::memcpy(base + info_.recv_offset[seq % 2], prev_.data(), n);
          std::memcpy(prev_.data(), base + info_.send_offset[seq % 2], n);
          recv_len[recv_done % 4] = n;
          recv_done++;
        }
      } else {
        ibv_sge sge;
        sge.addr =
            reinterpret_cast<uintptr_t>(base + info_.send_offset[seq % 2]);
        sge.length = n;
        sge.lkey = mr_->lkey;
        ibv_send_wr wr = {}, *bad = nullptr;
        wr.wr_id = SEND_WR << 16 | (seq % 2) << 8 | peer_;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        if (int status = ibv_post_send(conns_[peer_].queue_pair, &wr, &bad)) {
          std::ostringstream msg;
          msg << "[jaccl] GPU channel send failed with error " << status;
          throw std::runtime_error(msg.str());
        }
        sends_pending++;
      }

      // Wait for the peer's seq. Bounded: it depends on the peer.
      auto end = clk::now() + std::chrono::duration<double>(timeout_s_);
      while (recv_done < seq) {
        reap();
        if (recv_done < seq && clk::now() > end) {
          std::ostringstream msg;
          msg << "[jaccl] GPU channel: no data from rank " << peer_
              << " for all_sum " << seq << " within " << timeout_s_ << " s";
          throw std::runtime_error(msg.str());
        }
      }
      if (recv_len[(seq - 1) % 4] != n) {
        std::ostringstream msg;
        msg << "[jaccl] GPU channel: all_sum " << seq << " received "
            << recv_len[(seq - 1) % 4] << " bytes, expected " << n;
        throw std::runtime_error(msg.str());
      }
      word(GpuChannel::kRecvSeq).store(seq, std::memory_order_release);

      // Keep the send queue short.
      while (sends_pending > MAX_SEND_WR / 2) {
        reap();
      }
    }
  } catch (std::exception& e) {
    // Stop here: the host may already wait on a result that can not be
    // right, and nothing else reports the failure in time.
    fail(e.what());
    std::cerr << e.what() << std::endl;
    std::abort();
  }
}

} // namespace jaccl
