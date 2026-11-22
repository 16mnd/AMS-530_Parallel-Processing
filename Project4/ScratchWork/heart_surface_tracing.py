#!/usr/bin/env python3
"""Surface-frontier tracing for the implicit heart: uniform or hierarchical search.

Uniform: seed a sign-changing voxel, flood to neighboring surface voxels.
Adaptive: same frontier, but uses conservative interval bounds on larger
blocks to reject empty neighborhoods before visiting their finest voxels.
Both use identical finest-size flat marching-cubes patches; this isolates
search overhead, NOT curvature-adaptive triangulation. Mass is computed by
independent uniform midpoint integration (so mass work is NOT reduced).

Usage: python heart_surface_tracing.py --resolutions 16 32 64 --mode both
"""
import argparse
from collections import deque
from functools import lru_cache
from time import perf_counter
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area
from heart_amr_marching_cubes import BOUNDS, REF_AREA, REF_MASS, heart_f, heart_interval

OFFSETS = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1)
           for c in (-1,0,1) if (a,b,c)!=(0,0,0)]


def run(n, mode='uniform'):
    low = np.array([b[0] for b in BOUNDS], dtype=float)
    high = np.array([b[1] for b in BOUNDS], dtype=float)
    step = (high-low)/n
    counts = dict(cells=0, triangles=0, vertex_F=0, interval_tests=0,
                  rejected_blocks=0, queue_visits=0, components=0)

    @lru_cache(None)
    def vertex(i,j,k):
        counts['vertex_F'] += 1
        return float(heart_f(*(low + step*np.array([i,j,k]))))

    @lru_cache(None)
    def block_possible(i,j,k,w):
        # A conservative interval test; True means MAY contain F=0.
        counts['interval_tests'] += 1
        lo = low + step*np.array([i,j,k]); hi = lo + w*step
        v = heart_interval(tuple((float(a),float(b)) for a,b in zip(lo,hi)))
        possible = v[0] <= 0 <= v[1]
        if not possible:
            counts['rejected_blocks'] += 1
        return possible

    @lru_cache(None)
    def possible(i,j,k):
        if mode != 'adaptive':
            return True
        # Query dyadic ancestors, largest first, to prune whole blocks.
        for depth in range(n.bit_length()-1, -1, -1):
            w = 1 << depth
            if w > n: continue
            key = (i//w*w, j//w*w, k//w*w, w)
            if not block_possible(*key):
                return False
        return True

    def cell_values(i,j,k):
        vals = np.empty((2,2,2),dtype=float)
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    vals[a,b,c] = vertex(i+a,j+b,k+c)
        return vals

    # Seed: a vertical line through a central x,y column. This is cheap,
    # but assumes the heart is intersected by that line.
    ix = min(n-1,max(0,int((0-low[0])/step[0])))
    iy = min(n-1,max(0,int((0-low[1])/step[1])))
    seed = None
    for k in range(n):
        v = cell_values(ix,iy,k)
        if v.min() < 0 < v.max():
            seed=(ix,iy,k); break
    if seed is None:
        raise RuntimeError('Could not find a seed crossing in central column')

    # Flood the surface, not the full volume. Use 26-connectivity so
    # surface patches touching only along corners are not skipped.
    queue = deque([seed]); seen = {seed}
    area=0.
    while queue:
        i,j,k=queue.popleft(); counts['queue_visits']+=1
        if not possible(i,j,k):
            continue
        vals = cell_values(i,j,k)
        if not (vals.min() < 0 < vals.max()):
            continue
        counts['cells']+=1
        verts, faces, _, _ = marching_cubes(vals, level=0,
                        spacing=tuple(step), allow_degenerate=False)
        area += float(mesh_surface_area(verts,faces))
        counts['triangles']+=len(faces)
        for di,dj,dk in OFFSETS:
            nxt=(i+di,j+dj,k+dk)
            if any(q<0 or q>=n for q in nxt) or nxt in seen: continue
            seen.add(nxt)
            queue.append(nxt)

    # Mass is separate: uniform midpoint integration, vectorized in slices.
    xs=low[0]+(np.arange(n)+.5)*step[0]
    ys=low[1]+(np.arange(n)+.5)*step[1]
    zs=low[2]+(np.arange(n)+.5)*step[2]
    inside=0
    for x in xs:
        inside += int(np.count_nonzero(heart_f(x,ys[:,None],zs[None,:])<0))
    mass=.5*inside*float(np.prod(step))
    counts['mass_F']=n**3
    return area,mass,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resolutions',nargs='+',type=int,default=[16,32,64])
    p.add_argument('--mode',choices=['uniform','adaptive','both'],default='both')
    a=p.parse_args()
    print('mode       N   area       area_err% mass       mass_err%  surface_cells vertex_F interval_tests triangles seconds')
    for n in a.resolutions:
        if n<4 or n>256 or n&(n-1): p.error('N must be a power of two from 4 to 256')
        for mode in (['uniform','adaptive'] if a.mode=='both' else [a.mode]):
            t=perf_counter(); area,mass,s=run(n,mode); sec=perf_counter()-t
            print(f'{mode:9s} {n:<3} {area:10.6f} {100*abs(area-REF_AREA)/REF_AREA:9.4f} '
                  f'{mass:10.6f} {100*abs(mass-REF_MASS)/REF_MASS:9.4f} '
                  f'{s["cells"]:13d} {s["vertex_F"]:8d} {s["interval_tests"]:14d} '
                  f'{s["triangles"]:9d} {sec:7.3f}')
    print('Mass evaluates F at N^3 voxel centers in BOTH modes.')
    print('Counts are not FLOPs. Single seed may miss disconnected components.')
    print('Adaptive mode changes hierarchical SEARCH, not triangle resolution.')

if __name__=='__main__': main()
