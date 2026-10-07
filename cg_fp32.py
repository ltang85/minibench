#!/usr/bin/env python3
"""FP32 dense CG with the same vectorized algorithm for PyTorch and TTNN.

Examples:
  python cg_fp32.py --backend torch --device cpu --sizes 512 1024 --threads 4
  python cg_fp32.py --backend torch --device cuda:0 --sizes 1024 2048 4096
  python cg_fp32.py --backend ttnn --device-id 0 --sizes 1024 2048
  python cg_fp32.py --backend ttnn --sizes 1024 --iterations 50 --iteration-details

Requires torch for host problem generation and the PyTorch backend; ttnn is
imported only for --backend ttnn. No autograd, torch.compile, or preconditioner.
Only the CG iteration loop is used inside the solver. Benchmark repetitions
and the size sweep have ordinary orchestration loops outside the solver.

All solver floating tensors are float32. This does NOT guarantee bitwise
equivalence: TTNN matmul fidelity/reduction implementation may differ from
PyTorch FP32 arithmetic even with FP32 destination accumulation enabled.

Total timing: synchronize at solve boundaries only, excludes data generation,
transfers and validation, includes CG initialization, Python dispatch and
temporary allocation. A separate profiling solve synchronizes every stage;
its timings include synchronization overhead and must not be compared directly
with the uninterrupted solve time. Warmups compile/cache the exact shape.
"""

import argparse
import statistics
import time


class TorchOps:
    def __init__(self, torch):
        self.t = torch

    def matmul(self, a, b):
        return self.t.matmul(a, b)

    def add(self, a, b):
        return self.t.add(a, b)

    def subtract(self, a, b):
        return self.t.subtract(a, b)

    def multiply(self, a, b):
        return self.t.multiply(a, b)

    def divide(self, a, b):
        return self.t.divide(a, b)

    def gt(self, a, b):
        return self.t.gt(a, b)

    def where(self, predicate, yes, no):
        return self.t.where(predicate, yes, no)

    def dot(self, a, b):
        return self.t.sum(self.t.multiply(a, b), dim=(-2, -1), keepdim=True)


class TtnnOps:
    def __init__(self, ttnn):
        self.t = ttnn
        self.config = ttnn.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def matmul(self, a, b):
        return self.t.matmul(a, b, dtype=self.t.float32,
                             compute_kernel_config=self.config)

    def add(self, a, b):
        return self.t.add(a, b)

    def subtract(self, a, b):
        return self.t.subtract(a, b)

    def multiply(self, a, b):
        return self.t.multiply(a, b)

    def divide(self, a, b):
        return self.t.divide(a, b)

    def gt(self, a, b):
        return self.t.gt(a, b)

    def where(self, predicate, yes, no):
        return self.t.where(predicate, yes, no)

    def dot(self, a, b):
        return self.t.sum(self.t.multiply(a, b), dim=(-2, -1), keepdim=True,
                          compute_kernel_config=self.config)


class StageTimer:
    def __init__(self, synchronize):
        self.sync = synchronize
        self.rows = []

    def call(self, iteration, stage, operation):
        self.sync()
        start = time.perf_counter()
        value = operation()
        self.sync()
        self.rows.append((iteration, stage, time.perf_counter() - start))
        return value


def cg(a, b, zero, ops, iterations, rtol, timer=None):
    """Fixed-count CG with device-side convergence masking.

    A: [1,1,N,N], vectors: [1,1,N,1], reductions: [1,1,1,1].
    A must be symmetric positive definite. The shared sequence also guards
    nonpositive denominators; final validation exposes numerical breakdown.
    Once recursive residual <= rtol*initial residual, x and r are frozen,
    but all remaining iterations still execute the same tensor operations.
    No scalar is copied to the host inside the iteration loop.
    """
    def call(k, name, function):
        return function() if timer is None else timer.call(k, name, function)

    # x0=0, so r0=b: no initial matvec is necessary.
    x = zero
    r = b
    p = r
    rho = call(-1, "initial_dot", lambda: ops.dot(r, r))
    threshold = call(-1, "threshold", lambda: ops.multiply(rho, rtol * rtol))

    for k in range(iterations):
        q = call(k, "matvec", lambda: ops.matmul(a, p))
        denominator = call(k, "p_dot_q", lambda: ops.dot(p, q))

        def step_length():
            positive = ops.gt(denominator, 0.0)
            active = ops.gt(rho, threshold)
            safe_denominator = ops.where(positive, denominator, 1.0)
            numerator = ops.where(active, rho, 0.0)
            numerator = ops.where(positive, numerator, 0.0)
            return ops.divide(numerator, safe_denominator)

        alpha = call(k, "alpha_and_guards", step_length)
        x = call(k, "solution_update", lambda: ops.add(x, ops.multiply(alpha, p)))
        r = call(k, "residual_update", lambda: ops.subtract(r, ops.multiply(alpha, q)))
        rho_new = call(k, "r_dot_r", lambda: ops.dot(r, r))

        def direction_coefficient():
            safe_rho = ops.where(ops.gt(rho, 0.0), rho, 1.0)
            numerator = ops.where(ops.gt(rho_new, threshold), rho_new, 0.0)
            return ops.divide(numerator, safe_rho)

        beta = call(k, "beta_and_guards", direction_coefficient)
        p = call(k, "direction_update", lambda: ops.add(r, ops.multiply(beta, p)))
        rho = rho_new

    return x, rho


def make_problem(torch, n, seed):
    """O(N^2) setup: symmetric strictly diagonally dominant SPD matrix.

    Varying the diagonal scale avoids a nearly scalar identity problem.
    Store dense A even though CG itself requires only matrix application.
    Identical host inputs are used for either backend with the same seed/size.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed + n)
    a = torch.randn((n, n), generator=generator, dtype=torch.float32)
    a = (a + a.T) * (0.5 / n)
    a.diagonal().zero_()
    scale = torch.linspace(1.0, 10.0, n, dtype=torch.float32)
    diagonal = (torch.sum(torch.abs(a), dim=1) + 1.0) * scale
    a.diagonal().copy_(diagonal)
    exact = torch.randn((n, 1), generator=generator, dtype=torch.float32)
    b = torch.matmul(a, exact)
    return a.reshape(1, 1, n, n), b.reshape(1, 1, n, 1), exact


def report_profile(timer, detailed):
    names = list(dict.fromkeys(row[1] for row in timer.rows))
    print("  Separate synchronized stage profile (one additional solve):")
    print("  Stage                 calls    total_ms     mean_ms      min_ms      max_ms")
    for name in names:
        values = [v * 1000 for _, stage, v in timer.rows if stage == name]
        print(f"  {name:22s} {len(values):5d} {sum(values):11.3f} "
              f"{statistics.mean(values):11.3f} {min(values):11.3f} {max(values):11.3f}")
    if detailed:
        print("  Per-iteration stage times (ms; -1 is initialization):")
        for iteration, stage, seconds in timer.rows:
            print(f"    iteration={iteration:4d} stage={stage:22s} ms={seconds * 1000:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["torch", "ttnn"], required=True)
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu or cuda:0")
    parser.add_argument("--device-id", type=int, default=0, help="Blackhole device ID")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512], help="Dense matrix orders N")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3, help="Measured full solves per size")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured full solves per size")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Recursive relative residual threshold")
    parser.add_argument("--validation-tol", type=float, default=1e-4,
                        help="Allowed final true relative residual")
    parser.add_argument("--threads", type=int, help="PyTorch CPU intra-op threads")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-profile", action="store_true", help="Skip separate stage profiling solve")
    parser.add_argument("--iteration-details", action="store_true", help="Print each profile stage in each iteration")
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.iterations < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("sizes, iterations and runs must be positive; warmup must be nonnegative")
    if not 0 < args.rtol < 1 or not 0 < args.validation_tol < 1:
        parser.error("rtol and validation-tol must be between zero and one")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if args.iteration_details and args.no_profile:
        parser.error("iteration-details requires profiling")
    return args


def main():
    args = parse_args()
    import torch

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    # Disable TF32 for the PyTorch CUDA baseline.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    device = None
    if args.backend == "torch":
        target = torch.device(args.device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA PyTorch devices are supported")
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        ops = TorchOps(torch)
        sync = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else (lambda: None)
        upload = lambda x: x.to(device=target, dtype=torch.float32)
        download = lambda x: x.detach().cpu()
        description = f"PyTorch {torch.__version__}, {target}, CPU threads={torch.get_num_threads()}"
    else:
        import ttnn
        device = ttnn.open_device(device_id=args.device_id)
        try:
            ops = TtnnOps(ttnn)
        except Exception:
            ttnn.close_device(device)
            raise
        sync = lambda: ttnn.synchronize_device(device)
        upload = lambda x: ttnn.from_torch(x, dtype=ttnn.float32,
                                         layout=ttnn.TILE_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        download = lambda x: ttnn.to_torch(x)
        description = f"TTNN {getattr(ttnn, '__version__', 'unknown')}, Blackhole device={args.device_id}, HiFi4/FP32 accumulation"

    print(description)
    print(f"FP32 dense CG | iterations={args.iterations}, warmup={args.warmup}, runs={args.runs}, rtol={args.rtol:g}")
    print("Fixed iteration count; device guards freeze updates after recursive convergence.")
    print("Total and stage timings include dispatch/allocation; stage profile adds synchronization overhead.")
    print("The same A and b are reused across repetitions; this is not a cold-data benchmark.")
    failed = False
    try:
        for n in args.sizes:
            start = time.perf_counter()
            a_host, b_host, exact = make_problem(torch, n, args.seed)
            setup_ms = (time.perf_counter() - start) * 1000
            sync()
            start = time.perf_counter()
            a = upload(a_host)
            b = upload(b_host)
            zero = upload(torch.zeros_like(b_host))
            sync()
            transfer_ms = (time.perf_counter() - start) * 1000
            # Drop previous run results before each solve; no in-place input mutation.
            x = rho = None
            for _ in range(args.warmup):
                x = rho = None
                x, rho = cg(a, b, zero, ops, args.iterations, args.rtol)
                sync()

            times = []
            for _ in range(args.runs):
                x = rho = None
                sync()
                start = time.perf_counter()
                x, rho = cg(a, b, zero, ops, args.iterations, args.rtol)
                sync()
                times.append((time.perf_counter() - start) * 1000)

            print(f"\nN={n}: A={4*n*n / (1024**2):.2f} MiB logical FP32 storage")
            print(f"  Host generation: {setup_ms:.3f} ms | input placement: {transfer_ms:.3f} ms")
            print("  Full solve runs (ms): " + ", ".join(f"{v:.3f}" for v in times))
            print(f"  Full solve mean/median/min/max: {statistics.mean(times):.3f} / "
                  f"{statistics.median(times):.3f} / {min(times):.3f} / {max(times):.3f} ms")
            print(f"  Mean per scheduled iteration: {statistics.mean(times)/args.iterations:.3f} ms")

            if not args.no_profile:
                x = rho = None
                timer = StageTimer(sync)
                x, rho = cg(a, b, zero, ops, args.iterations, args.rtol, timer)
                report_profile(timer, args.iteration_details)

            start = time.perf_counter()
            solution = download(x).reshape(n, 1).to(torch.float64)
            recursive_rho = float(download(rho).reshape(-1)[0])
            a64 = a_host.reshape(n, n).to(torch.float64)
            b64 = b_host.reshape(n, 1).to(torch.float64)
            norm_b = torch.linalg.vector_norm(b64)
            true_residual = float(torch.linalg.vector_norm(b64 - a64 @ solution) / norm_b)
            recursive_residual = (max(recursive_rho, 0.0) ** 0.5) / float(norm_b)
            solution_error = float(torch.linalg.vector_norm(solution - exact.to(torch.float64)) /
                                   torch.linalg.vector_norm(exact.to(torch.float64)))
            passed = bool(torch.isfinite(solution).all()) and true_residual <= args.validation_tol
            validation_ms = (time.perf_counter() - start) * 1000
            print(f"  Recursive relative residual: {recursive_residual:.6e}")
            print(f"  True relative residual:      {true_residual:.6e}")
            print(f"  Relative solution error:     {solution_error:.6e}")
            print(f"  Validation: {'PASS' if passed else 'FAIL'} (true residual <= {args.validation_tol:g}); {validation_ms:.3f} ms outside timing")
            failed |= not passed
            del a, b, zero, x, rho, a64, b64, a_host, b_host, solution, exact
    finally:
        if device is not None:
            ttnn.close_device(device)
    if failed:
        print("One or more solves failed validation: increase iterations or inspect backend arithmetic accuracy.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
