// Copyright © 2026 Apple Inc.
//
// Two-rank probe of the RDMA provider behaviour that the GPU all_reduce and
// the exact-length sends rely on. Every wait has a deadline, so no probe
// hangs.
//
//   jaccl_provider_probe <rank> <coordinator ip:port> <device to the peer>
//
// Probes:
//   send_case   one SEND of len bytes into a receive of rcv bytes, each case
//               on a fresh link (stock JACCL only uses len == rcv == 4K << k)
//   second_qp   open the device again and bring a second queue pair to RTS
//   rdma_write  IBV_WR_RDMA_WRITE of 4096 bytes, then a 4-byte flag write
//   reg_twice   register one page-aligned region with two protection domains

#include <jaccl/rdma.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

using namespace jaccl;
using clk = std::chrono::steady_clock;

namespace {

struct Link {
  std::vector<Connection> conns;
  int peer;
  Connection& c() {
    return conns[peer];
  }
};

Link connect(SideChannel& sc, int rank, const std::string& dev) {
  Link l;
  l.peer = 1 - rank;
  std::vector<std::string> names(2);
  names[l.peer] = dev;
  l.conns = create_connections(names);
  auto& c = l.c();
  c.allocate_protection_domain();
  c.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
  c.create_queue_pair();
  c.queue_pair_init();
  auto all = sc.all_gather(c.info());
  c.queue_pair_rtr(all[l.peer]);
  c.queue_pair_rts();
  return l;
}

// Poll until `want` completions arrive or the deadline passes.
int poll_n(Connection& c, int want, ibv_wc* out, double seconds) {
  int got = 0;
  auto end = clk::now() + std::chrono::duration<double>(seconds);
  while (got < want && clk::now() < end) {
    got += c.poll(want - got, out + got);
  }
  return got;
}

void report(const char* name, bool ok, const std::string& detail) {
  std::cout << "{\"probe\": \"" << name
            << "\", \"ok\": " << (ok ? "true" : "false") << ", \"detail\": \""
            << detail << "\"}" << std::endl;
}

// One SEND of `len` bytes into a posted receive of `rcv` bytes, on its own
// fresh link so that a failed case can not leave state for the next one.
bool send_case(
    SideChannel& sc,
    int rank,
    const std::string& dev,
    size_t rcv,
    size_t len) {
  auto l = connect(sc, rank, dev);
  size_t cap = std::max(rcv, len);
  SharedBuffer rb(std::max<size_t>(cap, 16384)),
      sb(std::max<size_t>(cap, 16384));
  rb.register_to_protection_domain(l.c().protection_domain);
  sb.register_to_protection_domain(l.c().protection_domain);
  std::memset(rb.data<char>(), 0xEE, rb.size());
  for (size_t i = 0; i < sb.size(); i++) {
    sb.data<unsigned char>()[i] = static_cast<unsigned char>(i * 7 + rank);
  }
  {
    ibv_sge sge;
    sge.addr = reinterpret_cast<uintptr_t>(rb.data<char>());
    sge.length = rcv;
    sge.lkey = rb.local_key(l.c().protection_domain);
    ibv_recv_wr wr = {}, *bad = nullptr;
    wr.wr_id = RECV_WR << 16;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    int st = ibv_post_recv(l.c().queue_pair, &wr, &bad);
    if (st != 0) {
      std::ostringstream d;
      d << "rcv=" << rcv << " len=" << len << " post_recv=" << st;
      report("send_case", false, d.str());
      sc.barrier();
      return false;
    }
  }
  sc.barrier();
  l.c().post_send(sb, SEND_WR << 16, len);
  ibv_wc wc[4];
  int n = poll_n(l.c(), 2, wc, 3.0);
  bool ok = n == 2;
  uint32_t byte_len = 0;
  int recv_status = -1, send_status = -1;
  for (int i = 0; i < n; i++) {
    ok = ok && wc[i].status == IBV_WC_SUCCESS;
    if ((wc[i].wr_id >> 16) == RECV_WR) {
      byte_len = wc[i].byte_len;
      recv_status = wc[i].status;
    } else {
      send_status = wc[i].status;
    }
  }
  size_t first_bad = len;
  for (size_t i = 0; i < len; i++) {
    if (rb.data<unsigned char>()[i] !=
        static_cast<unsigned char>(i * 7 + l.peer)) {
      first_bad = i;
      break;
    }
  }
  bool data_ok = first_bad == len;
  std::ostringstream d;
  d << "rcv=" << rcv << " len=" << len << " completions=" << n
    << " send_status=" << send_status << " recv_status=" << recv_status
    << " byte_len=" << byte_len << " data_ok=" << data_ok
    << " first_bad_byte=" << first_bad;
  report("send_case", ok && data_ok && byte_len == len, d.str());
  sc.barrier();
  return ok && data_ok;
}

void rdma_write(SideChannel& sc, Link& l, int rank) {
  constexpr size_t SZ = 65536;
  void* mem = nullptr;
  posix_memalign(&mem, 16384, SZ);
  std::memset(mem, 0, SZ);
  auto mr = ibv().reg_mr(
      l.c().protection_domain,
      mem,
      SZ,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE);
  if (!mr) {
    report("rdma_write", false, "reg_mr failed");
    return;
  }
  struct Key {
    uint64_t addr;
    uint32_t rkey;
  } mine{reinterpret_cast<uint64_t>(mem), mr->rkey};
  auto keys = sc.all_gather(mine);
  auto peer = keys[l.peer];

  // Source data sits in the upper half of our region.
  auto src = static_cast<unsigned char*>(mem) + SZ / 2;
  for (int i = 0; i < 4096; i++) {
    src[i] = static_cast<unsigned char>(i * 3 + rank);
  }
  uint32_t flag = 0xC0DE0000u | rank;
  std::memcpy(src + 8192, &flag, 4);
  sc.barrier();

  auto write = [&](size_t src_off, size_t len, uint64_t dst_off, uint64_t id) {
    ibv_sge sge;
    sge.addr = reinterpret_cast<uintptr_t>(src + src_off);
    sge.length = len;
    sge.lkey = mr->lkey;
    ibv_send_wr wr = {}, *bad = nullptr;
    wr.wr_id = id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = peer.addr + dst_off;
    wr.wr.rdma.rkey = peer.rkey;
    return ibv_post_send(l.c().queue_pair, &wr, &bad);
  };
  int s1 = write(0, 4096, 0, 1);
  int s2 = write(8192, 4, 16384 - 4, 2);
  ibv_wc wc[2];
  int n = (s1 == 0 && s2 == 0) ? poll_n(l.c(), 2, wc, 5.0) : 0;
  bool cq_ok = n == 2;
  for (int i = 0; i < n; i++) {
    cq_ok = cq_ok && wc[i].status == IBV_WC_SUCCESS;
  }
  // Wait for the peer's flag in our region.
  volatile uint32_t* fl =
      reinterpret_cast<volatile uint32_t*>(static_cast<char*>(mem) + 16384 - 4);
  auto end = clk::now() + std::chrono::seconds(5);
  uint32_t want = 0xC0DE0000u | l.peer;
  while (*fl != want && clk::now() < end) {
  }
  bool flag_ok = *fl == want;
  bool data_ok = flag_ok;
  for (int i = 0; flag_ok && i < 4096; i++) {
    if (static_cast<unsigned char*>(mem)[i] !=
        static_cast<unsigned char>(i * 3 + l.peer)) {
      data_ok = false;
      break;
    }
  }
  std::ostringstream d;
  d << "post=" << s1 << "," << s2 << " completions=" << n
    << " flag_seen=" << flag_ok << " data_ok=" << data_ok;
  for (int i = 0; i < n; i++) {
    d << " status" << i << "=" << wc[i].status;
  }
  report("rdma_write", cq_ok && flag_ok && data_ok, d.str());
  sc.barrier();
  ibv().dereg_mr(mr);
  free(mem);
}

void reg_twice(Link& a, Link& b) {
  constexpr size_t SZ = 5 * 65536;
  void* mem = nullptr;
  posix_memalign(&mem, 16384, SZ);
  int acc =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  auto m1 = ibv().reg_mr(a.c().protection_domain, mem, SZ, acc);
  auto m2 = ibv().reg_mr(b.c().protection_domain, mem, SZ, acc);
  std::ostringstream d;
  d << "first=" << (m1 != nullptr) << " second=" << (m2 != nullptr);
  report("reg_twice", m1 && m2, d.str());
  if (m1) {
    ibv().dereg_mr(m1);
  }
  if (m2) {
    ibv().dereg_mr(m2);
  }
  free(mem);
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: jaccl_provider_probe <rank> <ip:port> <device>\n";
    return 2;
  }
  int rank = std::atoi(argv[1]);
  TCPAllGather agf(rank, 2, argv[2]);
  SideChannel sc(
      rank, 2, [&](const char* s, char* d, size_t n) { agf(s, d, n); });
  std::string dev = argv[3];
  // Control first: the shape stock JACCL uses (send = recv = 4 KB << k).
  size_t cases[][2] = {
      {16384, 16384},
      {4096, 4096},
      {8192, 8192},
      {16384, 8192},
      {16384, 12288},
      {16384, 10240},
      {8192, 2048},
      {10240, 10240},
      {2048, 2048},
      {65536, 65536},
      {65536, 40960},
      {65536, 10240},
  };
  for (auto& c : cases) {
    try {
      send_case(sc, rank, dev, c[0], c[1]);
    } catch (std::exception& e) {
      report("send_case", false, e.what());
    }
  }

  auto l = connect(sc, rank, dev);
  try {
    auto l2 = connect(sc, rank, dev);
    report("second_qp", true, "second device open + queue pair to RTS");
    reg_twice(l, l2);
  } catch (std::exception& e) {
    report("second_qp", false, e.what());
  }
  sc.barrier();
  rdma_write(sc, l, rank);
  sc.barrier();
  return 0;
}
