#!/usr/bin/env python3
"""Octree AMR + marching cubes + safeguarded Newton edge refinement.

Run: python heart_amr_newton.py --depths 4 5 6 7
Requires numpy and scikit-image, plus heart_amr_marching_cubes.py and
heart_safeguarded_newton.py in the same directory.

Each finest-level surface cell is meshed independently. Newton changes only
edge-intersection positions, not triangle connectivity. FLOPs are not counted.
"""
import argparse
import time
from functools import lru_cache
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area
from heart_amr_marching_cubes import BOUNDS, REF_AREA, REF_MASS, heart_f, heart_interval
from heart_safeguarded_newton import refine_vertices


def compute(depth, max_iterations=12, tolerance=1e-12):
    n = 1 << depth
    low = np.array([a for a, _ in BOUNDS], dtype=np.float64)
    high = np.array([b for _, b in BOUNDS], dtype=np.float64)
    spacing = (high - low) / n
    voxel_volume = float(np.prod(spacing))
    stats = dict(interval_tests=0, leaf_cells=0, mc_cells=0,
                 triangles=0, vertices=0, newton=0, bisection=0,
                 unbracketed=0, inside_cells=0, outside_cells=0)
    mc_area = newton_area = volume = 0.0

    @lru_cache(maxsize=None)
    def sample(i, j, k):
        point = low + spacing * np.array([i, j, k], dtype=np.float64)
        return float(heart_f(*point))

    def visit(i, j, k, width):
        nonlocal mc_area, newton_area, volume
        origin = low + spacing * np.array([i, j, k], dtype=np.float64)
        far = origin + spacing * width
        bounds = tuple((float(a), float(b)) for a, b in zip(origin, far))
        f_interval = heart_interval(bounds)
        stats['interval_tests'] += 1
        cell_volume = voxel_volume * width**3

        if f_interval[0] > 0:
            stats['outside_cells'] += 1
            return
        if f_interval[1] < 0:
            stats['inside_cells'] += 1
            volume += cell_volume
            return
        if width > 1:
            half = width // 2
            for di in (0, half):
                for dj in (0, half):
                    for dk in (0, half):
                        visit(i + di, j + dj, k + dk, half)
            return

        stats['leaf_cells'] += 1
        if heart_f(*(origin + 0.5 * spacing)) < 0:
            volume += voxel_volume

        values = np.empty((2, 2, 2), dtype=np.float64)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    values[di, dj, dk] = sample(i + di, j + dj, k + dk)
        if not (values.min() < 0 < values.max()):
            return

        verts, faces, _, _ = marching_cubes(
            values, level=0, spacing=tuple(spacing),
            allow_degenerate=False, method='lewiner')
        mc_area += float(mesh_surface_area(verts, faces))
        # The refiner expects coordinates relative to this cell's origin.
        axes = [np.array([origin[d], origin[d] + spacing[d]]) for d in range(3)]
        refined, newton_stats = refine_vertices(
            verts, axes, spacing, max_iterations=max_iterations,
            f_tolerance=tolerance)
        newton_area += float(mesh_surface_area(refined, faces))
        stats['mc_cells'] += 1
        stats['triangles'] += len(faces)
        stats['vertices'] += len(verts)  # counts duplicates between cells
        for key in ('newton', 'bisection', 'unbracketed'):
            stats[key] += newton_stats[key]

    t0 = time.perf_counter()
    visit(0, 0, 0, n)
    elapsed = time.perf_counter() - t0
    stats['cached_vertex_evaluations'] = sample.cache_info().currsize
    stats['center_evaluations'] = stats['leaf_cells']
    return mc_area, newton_area, 0.5 * volume, stats, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args()
    print('N   MC area    Newton area   MC err% Newton err%  mass      mass err%  MC_cells  F_evals*  interval_tests  Newton steps  fallback  seconds')
    for depth in args.depths:
        if not 1 <= depth <= 10:
            parser.error('depths must be between 1 and 10')
        mc, refined, mass, s, sec = compute(depth, args.max_iterations, args.tolerance)
        fevals = s['cached_vertex_evaluations'] + s['center_evaluations']
        print(f'{1<<depth:<4} {mc:10.6f} {refined:12.6f} '
              f'{100*abs(mc-REF_AREA)/REF_AREA:8.4f} '
              f'{100*abs(refined-REF_AREA)/REF_AREA:11.4f} '
              f'{mass:10.6f} {100*abs(mass-REF_MASS)/REF_MASS:9.4f} '
              f'{s["mc_cells"]:9d} {fevals:9d} {s["interval_tests"]:15d} '
              f'{s["newton"]:13d} {s["bisection"]:9d} {sec:8.3f}')
        if s['unbracketed']:
            print(f'  Note: {s["unbracketed"]} intersections were not bracketed and remained unchanged.')
    print('Reference: area=11.5247, mass=1.66427')
    print('*F_evals excludes Newton function/gradient calls and interval arithmetic; NOT a FLOP count.')


if __name__ == '__main__':
    main()
