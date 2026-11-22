#!/usr/bin/env python3
"""Uniform marching cubes + safeguarded Newton + quadratic curved triangles.

Run: python heart_uniform_curved.py --resolutions 16 32 64
Requires numpy, scikit-image, heart_safeguarded_newton.py, heart_amr_curved.py.
No AMR. Curved patch area uses 6-node quadratic triangles and 7-point quadrature.
Mass uses the same voxel-center midpoint rule as the baseline.
"""
import argparse
import time
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area
from heart_safeguarded_newton import (BOUNDS, REFERENCE_AREA, REFERENCE_MASS,
                                      heart_f, refine_vertices)
from heart_amr_curved import curved_triangle_area


def compute(n, newton_iterations=12, projection_iterations=5):
    if n < 4:
        raise ValueError('resolution must be >= 4')
    t0 = time.perf_counter()
    axes = [np.linspace(lo, hi, n+1, dtype=np.float64) for lo,hi in BOUNDS]
    spacing = tuple(float(a[1]-a[0]) for a in axes)
    x,y,z = np.meshgrid(*axes, indexing='ij', sparse=True)
    field = heart_f(x,y,z).astype(np.float32)
    vertices, faces, _, _ = marching_cubes(field, level=0.0,
                                          spacing=spacing, allow_degenerate=False)
    mc_area = float(mesh_surface_area(vertices, faces))
    refined, stats = refine_vertices(vertices, axes, spacing,
                                     max_iterations=newton_iterations)
    planar_area = float(mesh_surface_area(refined, faces))
    # Physical translation does not affect area, but use absolute coordinates
    # for the heart function when projecting quadratic triangle midpoints.
    origin = np.array([lo for lo,_ in BOUNDS])
    physical = refined + origin
    curved_area = 0.0
    midpoint_steps = 0
    for face in faces:
        area, steps = curved_triangle_area(physical[face],
                                          projection_iterations=projection_iterations)
        curved_area += area
        midpoint_steps += steps
    centers = [(a[:-1]+a[1:])*0.5 for a in axes]
    cx,cy,cz = np.meshgrid(*centers, indexing='ij', sparse=True)
    inside = np.count_nonzero(heart_f(cx,cy,cz) < 0)
    mass = 0.5 * float(inside) * float(np.prod(spacing))
    return mc_area, planar_area, curved_area, mass, len(faces), stats, midpoint_steps, time.perf_counter()-t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[16,32,64])
    parser.add_argument('--newton-iterations', type=int, default=12)
    parser.add_argument('--projection-iterations', type=int, default=5)
    args = parser.parse_args()
    print('N  MC area   Newton area  Curved area  MC err% Newton err% Curved err%  mass     mass err%  triangles  Newton steps  midpoint steps  seconds')
    for n in args.resolutions:
        mc, flat, curved, mass, tris, stats, mids, seconds = compute(
            n, args.newton_iterations, args.projection_iterations)
        err = lambda a: 100*abs(a-REFERENCE_AREA)/REFERENCE_AREA
        print(f'{n:<3} {mc:9.6f} {flat:11.6f} {curved:12.6f} '
              f'{err(mc):8.4f} {err(flat):11.4f} {err(curved):11.4f} '
              f'{mass:9.6f} {100*abs(mass-REFERENCE_MASS)/REFERENCE_MASS:10.4f} '
              f'{tris:10d} {stats["newton"]:13d} {mids:15d} {seconds:8.3f}')
    print('Reference: area=11.5247, mass=1.66427')
    print('NOTE: Iteration counts and wall time are NOT FLOP counts.')

if __name__ == '__main__':
    main()
