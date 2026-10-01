// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <cstdlib>

namespace jaccl {

// Opt-in single exchange all_sum for a 2-rank mesh (MLX_JACCL_FAST_ALLSUM=1).
struct FastAllSumConfig {
  bool enabled = false;
  int64_t chunk0 = 4096; // first work request, so the wire starts at once
  int64_t chunk = 65536; // later work requests
  int64_t max_bytes = 0; // larger messages use the stock path

  static constexpr int MAX_CHUNKS = 16; // fits MAX_SEND_WR and MAX_RECV_WR

  static const FastAllSumConfig& get() {
    static const FastAllSumConfig cfg = [] {
      FastAllSumConfig c;
      const char* on = std::getenv("MLX_JACCL_FAST_ALLSUM");
      c.enabled = on != nullptr && on[0] == '1';
      if (const char* v = std::getenv("MLX_JACCL_FAST_CHUNK0")) {
        c.chunk0 = std::strtoll(v, nullptr, 10);
      }
      if (const char* v = std::getenv("MLX_JACCL_FAST_CHUNK")) {
        c.chunk = std::strtoll(v, nullptr, 10);
      }
      // Chunk lengths stay multiples of 4 KiB so every element and every
      // receive packet boundary lines up on both ranks.
      if (c.chunk0 < 4096 || c.chunk0 % 4096 || c.chunk < 4096 ||
          c.chunk % 4096) {
        c.enabled = false;
      }
      c.max_bytes = c.chunk0 + (MAX_CHUNKS - 1) * c.chunk;
      if (const char* v = std::getenv("MLX_JACCL_FAST_MAX")) {
        int64_t m = std::strtoll(v, nullptr, 10);
        if (m < c.max_bytes) {
          c.max_bytes = m;
        }
      }
      return c;
    }();
    return cfg;
  }
};

} // namespace jaccl
