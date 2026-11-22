#!/usr/bin/env python3
"""Adaptive octree + standard marching cubes for the implicit heart.

Usage:
  python heart_amr_marching_cubes.py --depths 4 5 6

Only numpy and scikit-image are required. Each surface cell at the finest
level has the same size, so neighboring extracted surface patches align.
"""
import argparse
import time
from functools import lru_cache

import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area

BOUNDS = ((-1.13903, 1.13903), (-0.67285, 0.67285), (-1.0, 1.23666))
REF_AREA = 11.5247
REF_MASS = 1.66427


def heart_f(x, y, z):
    a = x*x + 2.25*y*y + z*z - 1.0
    b = x*x + 0.045*y*y
    return a*a*a - b*z*z*z


def add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def sub(a, b):
    return (a[0]-b[1], a[1]-b[0])


def mul(a, b):
    vals = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return (min(vals), max(vals))


def scale_pos(a, c):
    return (c*a[0], c*a[1])


def square(a):
    lo, hi = a
    return (0.0 if lo <= 0 <= hi else min(lo*lo, hi*hi), max(lo*lo, hi*hi))


def cube(a):
    return (a[0]**3, a[1]**3)


def heart_interval(box):
    """Conservative interval enclosure of F over a rectangular box."""
    x, y, z = box
    x2, y2, z2 = square(x), square(y), square(z)
    a = sub(add(add(x2, scale_pos(y2, 2.25)), z2), (1.0, 1.0))
    b = add(x2, scale_pos(y2, 0.045))
    return sub(cube(a), mul(b, cube(z)))


def compute(depth):
    n = 1 << depth
    low = np.array([a for a, b in BOUNDS], dtype=np.float64)
    high = np.array([b for a, b in BOUNDS], dtype=np.float64)
    spacing = (high-low)/n
    voxel_volume = float(np.prod(spacing))
    stats = dict(interval_tests=0, inside_cells=0, outside_cells=0,
                 leaf_cells=0, mc_cells=0, triangles=0, surface_vertices=0)
    area = 0.0
    volume = 0.0

    @lru_cache(maxsize=None)
    def sample(i, j, k):
        x, y, z = low + spacing*np.array([i, j, k], dtype=np.float64)
        return float(heart_f(x, y, z))

    def visit(i, j, k, width):
        nonlocal area, volume
        xyz0 = low + spacing*np.array([i, j, k], dtype=np.float64)
        xyz1 = xyz0 + spacing*width
        interval = heart_interval(tuple((float(a), float(b)) for a, b in zip(xyz0, xyz1)))
        stats['interval_tests'] += 1
        cell_vol = voxel_volume*width**3

        if interval[0] > 0.0:
            stats['outside_cells'] += 1
            return
        if interval[1] < 0.0:
            stats['inside_cells'] += 1
            volume += cell_vol
            return

        if width > 1:
            half = width // 2
            for di in (0, half):
                for dj in (0, half):
                    for dk in (0, half):
                        visit(i+di, j+dj, k+dk, half)
            return

        stats['leaf_cells'] += 1
        # Midpoint integration for the remaining (boundary/uncertain) cells.
        center = xyz0 + spacing*0.5
        if heart_f(*center) < 0.0:
            volume += voxel_volume

        values = np.empty((2, 2, 2), dtype=np.float64)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    values[di, dj, dk] = sample(i+di, j+dj, k+dk)
        if not (values.min() < 0.0 < values.max()):
            return
        verts, faces, _, _ = marching_cubes(
            values, level=0.0, spacing=tuple(spacing),
            allow_degenerate=False, method='lewiner')
        area += float(mesh_surface_area(verts, faces))
        stats['mc_cells'] += 1
        stats['triangles'] += len(faces)
        stats['surface_vertices'] += len(verts)

    start = time.perf_counter()
    visit(0, 0, 0, n)
    elapsed = time.perf_counter()-start
    stats['cached_vertex_evaluations'] = sample.cache_info().currsize
    stats['center_evaluations'] = stats['leaf_cells']
    return area, 0.5*volume, stats, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths', nargs='+', type=int, default=[4, 5, 6],
                        help='Octree depths; effective grid resolution N=2**depth')
    args = parser.parse_args()
    print('N    area        area_err%   mass        mass_err%  MC_cells  '
          'F_evals  interval_tests  triangles  seconds')
    for depth in args.depths:
        if not 1 <= depth <= 10:
            parser.error('depths must be between 1 and 10')
        area, mass, s, sec = compute(depth)
        evaluations = s['cached_vertex_evaluations']+s['center_evaluations']
        print(f'{1<<depth:<4} {area:11.6f} {100*abs(area-REF_AREA)/REF_AREA:10.4f} '
              f'{mass:11.6f} {100*abs(mass-REF_MASS)/REF_MASS:10.4f} '
              f'{s["mc_cells"]:9d} {evaluations:8d} '
              f'{s["interval_tests"]:14d} {s["triangles"]:10d} {sec:8.3f}')
    print('Reference: area=11.5247, mass=1.66427')
    print('NOTE: F_evals excludes interval arithmetic; these are NOT FLOP counts.')


if __name__ == '__main__':
    main()
