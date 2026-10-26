# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check actual device iterations, including conditional graph replay."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm.solve_rheology import _run_solver_loop
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@wp.kernel(enable_backward=False)
def _increment_iteration(count: wp.array[int]):
    count[0] += 1


@wp.kernel(enable_backward=False)
def _iteration_residual(count: wp.array[int], convergence_after: wp.array[int], residual: wp.array2d[float]):
    batch = wp.tid()
    value = wp.where(count[0] >= convergence_after[batch], 0.0, 1.0)
    residual[0, batch] = value
    residual[1, batch] = value


class _CountingSolver:
    def __init__(self, device, convergence_after, granularity):
        self.device = device
        self.name = "Counting rheology"
        self.solve_granularity = granularity
        self.count = wp.zeros(1, dtype=int, device=device)
        self.convergence_after = wp.array(convergence_after, dtype=int, device=device)
        self.residual = wp.empty((2, len(convergence_after)), dtype=float, device=device)
        self.rheology = SimpleNamespace(strain_environment_offsets=None)
        self.launch = wp.launch(_increment_iteration, dim=1, inputs=[self.count], device=device, record_cmd=True)

    def solve(self):
        self.launch.launch()

    def eval_residual(self):
        wp.launch(
            _iteration_residual,
            dim=len(self.convergence_after),
            inputs=[self.count, self.convergence_after],
            outputs=[self.residual],
            device=self.device,
        )
        return self.residual


def _check_iterations(test, device, use_graph, batched, enclosing_capture=False):
    if use_graph and not wp.is_conditional_graph_supported():
        test.skipTest("Conditional CUDA graphs are unavailable")
    with wp.ScopedDevice(device):
        store = fem.TemporaryStore()
        for granularity in (25, 50):
            for budget in (0, 1, 2, 5, 6, 9, 11, 26, 51):
                with test.subTest(budget=budget, granularity=granularity):
                    rheology = _CountingSolver(device, [1000, 1000] if batched else [1000], granularity)
                    contact = _CountingSolver(device, [1000], granularity)
                    scale = wp.ones(2, dtype=float, device=device) if batched else 1.0
                    if batched:
                        rheology.rheology.strain_environment_offsets = wp.array([0, 1, 2], dtype=int, device=device)
                    if enclosing_capture:
                        with wp.ScopedCapture(device=device) as capture:
                            _run_solver_loop(rheology, contact, budget, 0.1, scale, use_graph, False, store)
                        graph = capture.graph
                        wp.capture_launch(graph)
                    else:
                        graph = _run_solver_loop(rheology, contact, budget, 0.1, scale, use_graph, False, store)
                    test.assertEqual(int(rheology.count.numpy()[0]), budget)
                    test.assertEqual(int(contact.count.numpy()[0]), budget)
                    if graph is not None:
                        # Every replay must reset the loop counter and enforce the same budget.
                        wp.capture_launch(graph)
                        test.assertEqual(int(rheology.count.numpy()[0]), 2 * budget)
                        test.assertEqual(int(contact.count.numpy()[0]), 2 * budget)

            group = 5 if use_graph else granularity
            for budget, convergence_after in ((2, 1), (12, 2), (12, 6), (52, 26)):
                with test.subTest(budget=budget, convergence_after=convergence_after):
                    targets = [1, convergence_after] if batched else [convergence_after]
                    rheology = _CountingSolver(device, targets, granularity)
                    contact = _CountingSolver(device, [1000], granularity)
                    scale = wp.ones(2, dtype=float, device=device) if batched else 1.0
                    if batched:
                        rheology.rheology.strain_environment_offsets = wp.array([0, 1, 2], dtype=int, device=device)
                    _run_solver_loop(rheology, contact, budget, 0.1, scale, use_graph, False, store)
                    expected = min(budget, group * int(np.ceil(convergence_after / group)))
                    test.assertEqual(int(rheology.count.numpy()[0]), expected)
                    test.assertEqual(int(contact.count.numpy()[0]), expected)


def test_iteration_budget(test, device):
    """Honor small and nondivisible budgets without graph capture."""
    _check_iterations(test, device, use_graph=False, batched=False)


def test_iteration_budget_batched(test, device):
    """Stop only when every world converges without exceeding the budget."""
    _check_iterations(test, device, use_graph=False, batched=True)


def test_iteration_budget_graph(test, device):
    """Honor iteration budgets on repeated conditional CUDA graph launches."""
    _check_iterations(test, device, use_graph=True, batched=False)


def test_iteration_budget_batched_graph(test, device):
    """Honor conditional graph budgets while checking every world's residual."""
    _check_iterations(test, device, use_graph=True, batched=True)


def test_iteration_budget_enclosing_graph(test, device):
    """Reset iteration conditions when the loop is inside a captured simulation step."""
    _check_iterations(test, device, use_graph=True, batched=False, enclosing_capture=True)


def test_iteration_budget_batched_enclosing_graph(test, device):
    """Enforce per-world convergence and budgets inside a captured simulation step."""
    _check_iterations(test, device, use_graph=True, batched=True, enclosing_capture=True)


class TestImplicitMPMIterations(unittest.TestCase):
    pass


add_function_test(TestImplicitMPMIterations, "test_iteration_budget", test_iteration_budget, devices=get_test_devices())
add_function_test(
    TestImplicitMPMIterations,
    "test_iteration_budget_batched",
    test_iteration_budget_batched,
    devices=get_test_devices(),
)
add_function_test(
    TestImplicitMPMIterations,
    "test_iteration_budget_graph",
    test_iteration_budget_graph,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestImplicitMPMIterations,
    "test_iteration_budget_batched_graph",
    test_iteration_budget_batched_graph,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestImplicitMPMIterations,
    "test_iteration_budget_enclosing_graph",
    test_iteration_budget_enclosing_graph,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestImplicitMPMIterations,
    "test_iteration_budget_batched_enclosing_graph",
    test_iteration_budget_batched_enclosing_graph,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
