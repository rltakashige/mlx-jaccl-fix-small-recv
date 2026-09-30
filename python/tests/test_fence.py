# Copyright © 2026 Apple Inc.

import os
import subprocess
import sys
import unittest

import mlx.core as mx
import mlx_tests

# Runs in a child process because MLX reads MLX_METAL_FAST_SYNCH once.
CHILD = r"""
import sys
import numpy as np
import mlx.core as mx

iters = int(sys.argv[1])
# Word counts around the one-threadgroup limit (16 * 1024 words).
sizes = [1, 2, 2560, 10240, 16 * 1024 - 1, 16 * 1024, 16 * 1024 + 1, 81920]
w = mx.random.normal((256, 256)).astype(mx.float32)

def producer(i, n):
    # A GPU kernel writes x just before the fence update.
    base = (mx.arange(n, dtype=mx.uint32) * 2654435761 + i).astype(mx.uint32)
    busy = mx.sum(w @ w) * 0
    return base + busy.astype(mx.uint32)

def expect(i, n):
    return (np.arange(n, dtype=np.uint64) * 2654435761 + i).astype(np.uint32)

bad = 0
for it in range(iters):
    # Many independent GPU -> CPU crossings in one eval: fence values must
    # stay in order, and each CPU read must see the GPU data.
    xs = [producer(it * 131 + k, n) for k, n in enumerate(sizes)]
    ys = [mx.add(x, mx.array(0, mx.uint32), stream=mx.cpu) for x in xs]
    mx.eval(ys)
    for k, (n, y) in enumerate(zip(sizes, ys)):
        if not np.array_equal(np.array(y), expect(it * 131 + k, n)):
            bad += 1
    # A chain GPU -> CPU -> GPU -> CPU, as in a collective chain.
    n = 2560
    x = producer(it, n)
    for j in range(8):
        y = mx.add(x, mx.array(1, mx.uint32), stream=mx.cpu)
        x = mx.add(y, mx.array(1, mx.uint32), stream=mx.gpu)
    mx.eval(x)
    if not np.array_equal(np.array(x), expect(it, n) + np.uint32(16)):
        bad += 1
    # The same chain with async_eval and bf16 data.
    xb = (mx.arange(5120) * (it + 1) / 7.0).astype(mx.bfloat16)
    ref = np.array(xb.astype(mx.float32))
    for j in range(4):
        yb = mx.multiply(xb, mx.array(1.0, mx.bfloat16), stream=mx.cpu)
        xb = mx.multiply(yb, mx.array(1.0, mx.bfloat16), stream=mx.gpu)
    mx.async_eval(xb)
    mx.eval(xb)
    if not np.array_equal(np.array(xb.astype(mx.float32)), ref):
        bad += 1
print("BAD", bad)
"""


class TestFence(mlx_tests.MLXTestCase):
    def run_child(self, fold):
        env = dict(os.environ)
        env["MLX_METAL_FAST_SYNCH"] = "1"
        env["MLX_METAL_FENCE_FOLD_MAX_KB"] = fold
        iters = os.environ.get("MLX_TEST_FENCE_ITERS", "50")
        out = subprocess.run(
            [sys.executable, "-c", CHILD, iters],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("BAD 0", out.stdout)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_gpu_to_cpu_fast_synch_fold(self):
        self.run_child("64")

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_gpu_to_cpu_fast_synch_fold_large(self):
        self.run_child("320")

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_gpu_to_cpu_fast_synch_no_fold(self):
        self.run_child("0")


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
