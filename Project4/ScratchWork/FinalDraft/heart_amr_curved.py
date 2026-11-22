#!/usr/bin/env python3
"""AMR + marching cubes + quadratic curved triangles for heart area.

Run: python heart_amr_curved.py --depths 4 5 6
Requires numpy, scikit-image, and heart_amr_marching_cubes.py plus
heart_safeguarded_newton.py in the same directory.

Each MC triangle becomes a six-node quadratic patch: three corners are
safeguarded-Newton refined along voxel edges, and three edge midpoints are
projected onto F=0 along the local gradient. Area uses 7-point Gaussian
quadrature of the quadratic surface Jacobian. This is an experimental
approximation, not a certified error bound. Mass uses the existing AMR
midpoint/interval method, unchanged.
"""
import argparse
import time
from functools import lru_cache
import numpy as np
from skimage.measure import marching_cubes
from heart_amr_marching_cubes import BOUNDS, REF_AREA, REF_MASS, heart_f, heart_interval
from heart_safeguarded_newton import heart_gradient, refine_vertices

# Seven-point degree-5 Dunavant quadrature on reference triangle (area 1/2).
_QUAD = [
    (1/3, 1/3, 1/3, 0.225 / 2),
]
for a,b,w in [
    (0.059715871789770, 0.470142064105115, 0.132394152788506 / 2),
    (0.797426985353087, 0.101286507323456, 0.125939180544827 / 2),
]:
    _QUAD.extend([(a,b,b,w), (b,a,b,w), (b,b,a,w)])


def project_midpoints(points, iterations=5, tol=1e-12):
    """Project physical xyz points onto F=0 along gradient (damped Newton)."""
    p = np.array(points, dtype=np.float64, copy=True)
    steps = 0
    for _ in range(iterations):
        f = heart_f(p[:,0], p[:,1], p[:,2])
        active = np.abs(f) > tol
        if not np.any(active):
            break
        g = np.column_stack(heart_gradient(p[:,0], p[:,1], p[:,2]))
        norm2 = np.einsum('ij,ij->i', g, g)
        valid = active & (norm2 > 1e-22)
        if not np.any(valid):
            break
        step = (f[valid] / norm2[valid])[:,None] * g[valid]
        # Limit jumps to avoid large excursions near singular gradients.
        max_step = 0.5 * np.linalg.norm(points[1]-points[0]) if len(points)>1 else np.inf
        # The caller's midpoint array may include unrelated triangles; use a
        # fixed per-call characteristic scale passed as optional argument below.
        p[valid] -= step
        steps += int(np.count_nonzero(valid))
    return p, steps


def curved_triangle_area(tri, projection_iterations=5, projection_tol=1e-12):
    """Area of a 6-node isoparametric quadratic triangle.

    tri: (3,3) physical coordinates. Midpoints are projected onto F=0.
    """
    a,b,c = tri
    mids = np.array([(a+b)/2, (b+c)/2, (c+a)/2])
    scale = max(np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c))
    steps = 0
    # Gradient projection with a trust-radius and backtracking on |F|.
    for _ in range(projection_iterations):
        f = heart_f(mids[:,0], mids[:,1], mids[:,2])
        if np.max(np.abs(f)) <= projection_tol:
            break
        g = np.column_stack(heart_gradient(mids[:,0], mids[:,1], mids[:,2]))
        norm2 = np.einsum('ij,ij->i',g,g)
        for i in range(3):
            if abs(f[i]) <= projection_tol or norm2[i] < 1e-22:
                continue
            delta = f[i] * g[i] / norm2[i]
            length = np.linalg.norm(delta)
            if length > 0.5*scale:
                delta *= (0.5*scale/length)
            alpha = 1.0
            for _ in range(10):
                q = mids[i]-alpha*delta
                if abs(heart_f(*q)) < abs(f[i]):
                    mids[i] = q
                    break
                alpha *= 0.5
            steps += 1
    nodes = np.vstack((a,b,c,mids))
    area = 0.0
    for L1,L2,L3,w in _QUAD:
        # Quadratic Lagrange shape functions: N1..N3 corners; N4..N6 edges.
        # Coordinates on reference triangle: u=L2, v=L3.
        ddu = np.array([-(4*L1-1), 4*L2-1, 0,
                        4*(L1-L2), 4*L3, -4*L3])
        ddv = np.array([-(4*L1-1), 0, 4*L3-1,
                        -4*L2, 4*L2, 4*(L1-L3)])
        du = ddu @ nodes
        dv = ddv @ nodes
        area += w * np.linalg.norm(np.cross(du,dv))
    return area, steps


def compute(depth, newton_iterations=12, projection_iterations=5):
    n = 1 << depth
    low = np.array([a for a,_ in BOUNDS], dtype=float)
    high = np.array([b for _,b in BOUNDS], dtype=float)
    spacing = (high-low)/n
    voxel_volume = float(np.prod(spacing))
    stats = dict(interval_tests=0, leaf_cells=0, mc_cells=0,
                 triangles=0, newton_steps=0, midpoint_steps=0,
                 unbracketed=0, inside_cells=0, outside_cells=0)
    planar_area = curved_area = volume = 0.0

    @lru_cache(maxsize=None)
    def sample(i,j,k):
        p = low + spacing*np.array([i,j,k],dtype=float)
        return float(heart_f(*p))

    def visit(i,j,k,width):
        nonlocal planar_area,curved_area,volume
        origin = low + spacing*np.array([i,j,k],dtype=float)
        far = origin + spacing*width
        interval = heart_interval(tuple((float(a),float(b)) for a,b in zip(origin,far)))
        stats['interval_tests'] += 1
        if interval[0] > 0:
            stats['outside_cells'] += 1
            return
        if interval[1] < 0:
            stats['inside_cells'] += 1
            volume += voxel_volume*width**3
            return
        if width > 1:
            h=width//2
            for di in (0,h):
                for dj in (0,h):
                    for dk in (0,h):
                        visit(i+di,j+dj,k+dk,h)
            return
        stats['leaf_cells'] += 1
        if heart_f(*(origin+0.5*spacing)) < 0:
            volume += voxel_volume
        values = np.empty((2,2,2),dtype=float)
        for di in (0,1):
            for dj in (0,1):
                for dk in (0,1):
                    values[di,dj,dk] = sample(i+di,j+dj,k+dk)
        if not (values.min()<0<values.max()):
            return
        verts, faces, _, _ = marching_cubes(values, level=0.0,
                     spacing=tuple(spacing),allow_degenerate=False,method='lewiner')
        axes=[np.array([origin[d],origin[d]+spacing[d]]) for d in range(3)]
        refined, ns=refine_vertices(verts,axes,spacing,max_iterations=newton_iterations)
        stats['newton_steps'] += ns['newton']
        stats['unbracketed'] += ns['unbracketed']
        pts=refined+origin
        for face in faces:
            tri=pts[face]
            planar_area += 0.5*np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
            a,steps=curved_triangle_area(tri,projection_iterations)
            curved_area += a
            stats['midpoint_steps'] += steps
        stats['mc_cells'] += 1
        stats['triangles'] += len(faces)

    t0=time.perf_counter()
    visit(0,0,0,n)
    sec=time.perf_counter()-t0
    stats['vertex_evals']=sample.cache_info().currsize
    return planar_area,curved_area,0.5*volume,stats,sec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths',type=int,nargs='+',default=[4,5,6])
    parser.add_argument('--newton-iterations',type=int,default=12)
    parser.add_argument('--projection-iterations',type=int,default=5)
    args=parser.parse_args()
    print('N   planar area  curved area   planar err% curved err%  mass       mass err%   triangles  interval tests  seconds')
    for d in args.depths:
        if not 1<=d<=9:
            parser.error('depths must be between 1 and 9')
        p,c,m,s,t=compute(d,args.newton_iterations,args.projection_iterations)
        print(f'{1<<d:<4} {p:11.6f} {c:12.6f} {100*abs(p-REF_AREA)/REF_AREA:11.4f} '
              f'{100*abs(c-REF_AREA)/REF_AREA:11.4f} {m:10.6f} '
              f'{100*abs(m-REF_MASS)/REF_MASS:10.4f} {s["triangles"]:11d} '
              f'{s["interval_tests"]:15d} {t:8.3f}')
        print(f'     Newton steps={s["newton_steps"]}, midpoint steps={s["midpoint_steps"]}, '
              f'cached vertex F calls={s["vertex_evals"]}, leaf center F calls={s["leaf_cells"]}')
    print('Reference: area=11.5247, mass=1.66427')
    print('NOTE: steps/evaluations are NOT FLOP counts; curved patches are independent per cell.')

if __name__=='__main__':
    main()
