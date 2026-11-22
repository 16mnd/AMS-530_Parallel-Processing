#!/usr/bin/env python3
"""Heart surface: marching cubes with safeguarded Newton edge intersections.

Run: python heart_safeguarded_newton.py --resolutions 32 64 128
Requires: numpy, scikit-image

The triangulation is unchanged; only edge-crossing vertex positions are refined.
This is a numerical experiment, not a validated FLOP-minimal implementation.
"""
import argparse
import time
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area

BOUNDS = ((-1.13903, 1.13903), (-0.67285, 0.67285), (-1.0, 1.23666))
REFERENCE_AREA = 11.5247
REFERENCE_MASS = 1.66427


def heart_f(x, y, z):
    q = x*x + 2.25*y*y + z*z - 1.0
    return q*q*q - (x*x + 0.045*y*y)*z*z*z


def heart_gradient(x, y, z):
    q = x*x + 2.25*y*y + z*z - 1.0
    q2 = q*q
    z2 = z*z
    z3 = z2*z
    r = x*x + 0.045*y*y
    return (6.0*x*q2 - 2.0*x*z3,
            13.5*y*q2 - 0.09*y*z3,
            6.0*z*q2 - 3.0*r*z2)


def refine_vertices(vertices, axes, spacing, max_iterations=12,
                    s_tolerance=1e-12, f_tolerance=1e-12):
    """Refine MC vertices along original grid edges, using bracketed Newton.

    MC vertices are given in coordinates relative to the grid's minimum corner.
    Return refined vertices in that same coordinate system.
    """
    refined = np.asarray(vertices, dtype=np.float64).copy()
    n = len(axes[0]) - 1
    grid_coords = refined / np.asarray(spacing)
    rounded = np.rint(grid_coords)
    fractional = np.abs(grid_coords - rounded)
    varying_axis = np.argmax(fractional, axis=1)
    row = np.arange(len(refined))
    # A vertex exactly at a grid corner needs no refinement.
    active = fractional[row, varying_axis] > 2e-5
    selected = np.flatnonzero(active)
    if not len(selected):
        return refined, dict(refined=0, newton=0, bisection=0, iterations=0, unbracketed=0)

    edge_axis = varying_axis[selected]
    grid = np.rint(grid_coords[selected]).astype(np.int64)
    lower = np.floor(grid_coords[selected, edge_axis]).astype(np.int64)
    lower = np.clip(lower, 0, n - 1)
    grid[row[:len(selected)], edge_axis] = lower
    # For the two constant axes, the rounded grid coordinates locate the edge.
    grid = np.clip(grid, 0, n)
    starts = np.column_stack([axes[d][grid[:, d]] for d in range(3)])
    edge_lengths = np.asarray(spacing)[edge_axis]
    ends = starts.copy()
    ends[np.arange(len(selected)), edge_axis] += edge_lengths

    f_left = heart_f(starts[:, 0], starts[:, 1], starts[:, 2])
    f_right = heart_f(ends[:, 0], ends[:, 1], ends[:, 2])
    bracketed = (f_left * f_right < 0.0)
    # If float32 marching-cubes values indicated a crossing but double-precision
    # values don't, leave that vertex untouched rather than invent a root.
    unbracketed = int(np.count_nonzero(~bracketed))
    selected = selected[bracketed]
    starts = starts[bracketed]
    edge_axis = edge_axis[bracketed]
    edge_lengths = edge_lengths[bracketed]
    f_left = f_left[bracketed]
    f_right = f_right[bracketed]
    if not len(selected):
        return refined, dict(refined=0, newton=0, bisection=0, iterations=0,
                             unbracketed=unbracketed)

    count = len(selected)
    lo = np.zeros(count)
    hi = np.ones(count)
    flo = f_left.copy()
    # Initial guess from MC's linear interpolation, clipped to the bracket.
    s = np.clip(-f_left / (f_right - f_left), 0.0, 1.0)
    done = np.zeros(count, dtype=bool)
    newton_steps = 0
    bisection_steps = 0
    iterations = 0
    rows = np.arange(count)

    for _ in range(max_iterations):
        work = ~done
        if not np.any(work):
            break
        iterations += 1
        ids = rows[work]
        pos = starts[work].copy()
        pos[np.arange(len(ids)), edge_axis[work]] += s[work] * edge_lengths[work]
        value = heart_f(pos[:, 0], pos[:, 1], pos[:, 2])
        converged = (np.abs(value) <= f_tolerance) | ((hi[work] - lo[work]) <= s_tolerance)
        done[ids[converged]] = True
        ids = ids[~converged]
        if not len(ids):
            continue
        value = value[~converged]
        pos = pos[~converged]
        gx, gy, gz = heart_gradient(pos[:, 0], pos[:, 1], pos[:, 2])
        gradients = np.stack((gx, gy, gz), axis=1)
        slope = gradients[np.arange(len(ids)), edge_axis[ids]] * edge_lengths[ids]

        # Shrink the root bracket before deciding whether Newton is safe.
        same_sign_as_left = np.signbit(value) == np.signbit(flo[ids])
        left_ids = ids[same_sign_as_left]
        right_ids = ids[~same_sign_as_left]
        lo[left_ids] = s[left_ids]
        flo[left_ids] = value[same_sign_as_left]
        hi[right_ids] = s[right_ids]

        midpoint = 0.5 * (lo[ids] + hi[ids])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            candidate = s[ids] - value / slope
        safe = (np.isfinite(candidate) & (np.abs(slope) > 1e-14)
                & (candidate > lo[ids]) & (candidate < hi[ids]))
        s[ids] = np.where(safe, candidate, midpoint)
        newton_steps += int(np.count_nonzero(safe))
        bisection_steps += int(np.count_nonzero(~safe))

    # If iteration limit was reached, the bracket midpoint is the safer estimate.
    s[~done] = 0.5 * (lo[~done] + hi[~done])
    result = starts.copy()
    result[rows, edge_axis] += s * edge_lengths
    refined[selected] = result - np.array([a[0] for a in axes])
    return refined, dict(refined=count, newton=newton_steps,
                         bisection=bisection_steps, iterations=iterations,
                         unbracketed=unbracketed)


def compute(n, max_iterations, tolerance):
    if n < 4:
        raise ValueError('Resolution must be >= 4')
    start_time = time.perf_counter()
    axes = [np.linspace(lo, hi, n + 1, dtype=np.float64) for lo, hi in BOUNDS]
    spacing = tuple(float(a[1] - a[0]) for a in axes)
    x, y, z = np.meshgrid(*axes, indexing='ij', sparse=True)
    # Float32 matches the original baseline's marching-cubes input.
    field = heart_f(x, y, z).astype(np.float32)
    vertices, faces, _, _ = marching_cubes(field, level=0.0, spacing=spacing,
                                           allow_degenerate=False)
    original_area = float(mesh_surface_area(vertices, faces))
    refined_vertices, stats = refine_vertices(
        vertices, axes, spacing, max_iterations=max_iterations,
        f_tolerance=tolerance)
    refined_area = float(mesh_surface_area(refined_vertices, faces))

    centers = [(a[:-1] + a[1:]) * 0.5 for a in axes]
    cx, cy, cz = np.meshgrid(*centers, indexing='ij', sparse=True)
    inside = int(np.count_nonzero(heart_f(cx, cy, cz) < 0))
    volume = inside * np.prod(spacing)
    mass = 0.5 * volume
    elapsed = time.perf_counter() - start_time
    return original_area, refined_area, mass, len(vertices), len(faces), stats, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--max-iterations', type=int, default=12)
    parser.add_argument('--tolerance', type=float, default=1e-12,
                        help='Absolute heart F residual tolerance')
    args = parser.parse_args()
    print('P=1 | Marching cubes + safeguarded Newton edge refinement')
    print('N    MC area     Newton area  MC err%  Newton err%  mass       mass err%   vertices  triangles   Newton steps  fallback steps  time(s)')
    for n in args.resolutions:
        original, refined, mass, verts, tris, stats, elapsed = compute(
            n, args.max_iterations, args.tolerance)
        print(f'{n:<4} {original:11.6f} {refined:11.6f}'
              f' {abs(original-REFERENCE_AREA)/REFERENCE_AREA*100:8.4f}'
              f' {abs(refined-REFERENCE_AREA)/REFERENCE_AREA*100:11.4f}'
              f' {mass:10.6f} {abs(mass-REFERENCE_MASS)/REFERENCE_MASS*100:10.4f}'
              f' {verts:9d} {tris:10d} {stats["newton"]:13d}'
              f' {stats["bisection"]:15d} {elapsed:8.3f}')
        if stats['unbracketed']:
            print(f'  Note: {stats["unbracketed"]} vertices skipped (no double-precision sign bracket).')
    print('Reference: area=11.5247, mass=1.66427')
    print('Note: Newton/fallback counts are iterations, NOT FLOP counts.')


if __name__ == '__main__':
    main()
