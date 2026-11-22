#!/usr/bin/env python3
"""Problem 4.1, P=1: uniform-grid heart area (marching cubes) and mass.

Requires numpy and scikit-image. Run: python heart_baseline.py --resolutions 32 64 128
The bounding box is from the assignment; reference answers are checks only.
"""
import argparse
import time
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area

BOUNDS = ((-1.13903, 1.13903), (-0.67285, 0.67285), (-1.0, 1.23666))
REFERENCE_AREA = 11.5247
REFERENCE_MASS = 1.66427


def heart_f(x, y, z):
    """Negative inside, zero on surface, positive outside."""
    return (x*x + 2.25*y*y + z*z - 1.0)**3 - (x*x + 0.045*y*y)*z**3


def compute(n):
    if n < 4:
        raise ValueError('resolution must be >= 4')
    start = time.perf_counter()
    axes = [np.linspace(lo, hi, n+1, dtype=np.float64) for lo, hi in BOUNDS]
    dx, dy, dz = [a[1]-a[0] for a in axes]
    # Values at grid vertices: marching cubes interpolates surface crossings.
    # Shape n+1 in x, y, z order; indexing='ij' preserves coordinate order.
    x, y, z = np.meshgrid(*axes, indexing='ij', sparse=True)
    field = heart_f(x, y, z).astype(np.float32)
    vertices, faces, _, _ = marching_cubes(field, level=0.0,
                                            spacing=(dx, dy, dz),
                                            allow_degenerate=False)
    area = float(mesh_surface_area(vertices, faces))
    # Midpoint-rule volume integral: every voxel contributes if center is inside.
    centers = [(a[:-1]+a[1:])*0.5 for a in axes]
    cx, cy, cz = np.meshgrid(*centers, indexing='ij', sparse=True)
    inside = np.count_nonzero(heart_f(cx, cy, cz) < 0)
    volume = inside * dx * dy * dz
    density_t1 = 0.5  # rho0=1, f=1 Hz, t=1 second
    mass = density_t1 * volume
    elapsed = time.perf_counter() - start
    return area, mass, volume, inside, len(faces), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[32, 64, 128],
                        help='number of voxels along EACH axis')
    args = parser.parse_args()
    print('P=1 | Uniform grid + marching cubes | density(t=1)=0.5')
    print('N    area        area err %   mass        mass err %   triangles    time (s)')
    for n in args.resolutions:
        area, mass, volume, inside, triangles, elapsed = compute(n)
        print(f'{n:<4} {area:11.6f} {abs(area-REFERENCE_AREA)/REFERENCE_AREA*100:11.4f}'
              f' {mass:11.6f} {abs(mass-REFERENCE_MASS)/REFERENCE_MASS*100:11.4f}'
              f' {triangles:11d} {elapsed:10.3f}')
    print('Reference: area=11.5247, mass=1.66427')
    print('Note: FLOP counting is NOT implemented; elapsed time is not a FLOP count.')


if __name__ == '__main__':
    main()
