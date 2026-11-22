/* Heart hybrid: N=16 curved area, N=64 midpoint mass, MPI workload sharing.
   FLOP convention: each +,-,*,/,sqrt counts as one; comparisons, casts,
   indexing, MPI, memory operations and transcendental calls are not counted.
   Counters are explicit arithmetic-model counts, NOT hardware FP instructions. */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mc_table.h"

#define NA 16
#define NM 64
#define REF_AREA 11.5247
#define REF_MASS 1.66427
#define MAX_NEWTON 12
#define MAX_PROJ 5
#define TOL 1e-12

typedef struct { double x,y,z; } Vec;
typedef struct { double area; long long flops, triangles, newton, projections; } AreaResult;
static const double low[3]={-1.13903,-0.67285,-1.0};
static const double high[3]={1.13903,0.67285,1.23666};
static const int corner[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
static const int edges[12][2]={{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
static const double quad[7][4]={
 {1./3,1./3,1./3,0.1125},
 {0.059715871789770,0.470142064105115,0.470142064105115,0.066197076394253},
 {0.470142064105115,0.059715871789770,0.470142064105115,0.066197076394253},
 {0.470142064105115,0.470142064105115,0.059715871789770,0.066197076394253},
 {0.797426985353087,0.101286507323456,0.101286507323456,0.062969590272414},
 {0.101286507323456,0.797426985353087,0.101286507323456,0.062969590272414},
 {0.101286507323456,0.101286507323456,0.797426985353087,0.062969590272414}
};

static double F(Vec p, long long *ops) {
    double x2=p.x*p.x, y2=p.y*p.y, z2=p.z*p.z;
    double q=x2+2.25*y2+z2-1.0;
    double r=x2+0.045*y2;
    double ans=q*q*q-r*z2*p.z;
    *ops+=14; /* count each explicit arithmetic operation */
    return ans;
}
static Vec grad(Vec p,long long *ops) {
    double x2=p.x*p.x,y2=p.y*p.y,z2=p.z*p.z;
    double q=x2+2.25*y2+z2-1.0, q2=q*q, z3=z2*p.z;
    double r=x2+0.045*y2;
    Vec g={6*p.x*q2-2*p.x*z3,
           13.5*p.y*q2-0.09*p.y*z3,
           6*p.z*q2-3*r*z2};
    *ops+=26; /* count each explicit arithmetic operation */
    return g;
}
static Vec add(Vec a,Vec b,long long *ops) { Vec v={a.x+b.x,a.y+b.y,a.z+b.z}; *ops+=3; return v; }
static Vec sub(Vec a,Vec b,long long *ops) { Vec v={a.x-b.x,a.y-b.y,a.z-b.z}; *ops+=3; return v; }
static Vec mul(Vec a,double s,long long *ops) { Vec v={a.x*s,a.y*s,a.z*s}; *ops+=3; return v; }
static double dot(Vec a,Vec b,long long *ops) { *ops+=5; return a.x*b.x+a.y*b.y+a.z*b.z; }
static double norm(Vec a,long long *ops) { double d=dot(a,a,ops); *ops+=1; return sqrt(d); }
static Vec cross(Vec a,Vec b,long long *ops) {
    Vec v={a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; *ops+=9; return v;
}
static Vec point(int i,int j,int k,int n,long long *ops) {
    Vec v={low[0]+(high[0]-low[0])*i/n,
           low[1]+(high[1]-low[1])*j/n,
           low[2]+(high[2]-low[2])*k/n};
    *ops+=12; return v;
}
static double coordinate(Vec v,int d) { return d==0?v.x:(d==1?v.y:v.z); }

/* A safeguarded Newton root on one original grid edge. */
static Vec edge_root(Vec a,Vec b,long long *ops,long long *steps) {
    double fa=F(a,ops), fb=F(b,ops);
    if(fa==0) return a;
    if(fb==0) return b;
    double lo=0,hi=1,flo=fa;
    double s=-fa/(fb-fa); *ops+=3;
    if(s<0 || s>1) s=0.5;
    Vec delta=sub(b,a,ops);
    int axis=fabs(delta.x)>0?0:(fabs(delta.y)>0?1:2);
    for(int it=0;it<MAX_NEWTON;it++) {
        Vec p=add(a,mul(delta,s,ops),ops);
        double val=F(p,ops);
        if(fabs(val)<=TOL || hi-lo<=1e-12) { *ops+=1; return p; }
        Vec g=grad(p,ops);
        double slope=coordinate(g,axis)*coordinate(delta,axis); *ops+=1;
        if(signbit(val)==signbit(flo)) { lo=s; flo=val; }
        else hi=s;
        double candidate=s-val/slope; *ops+=2;
        if(fabs(slope)<1e-14 || !isfinite(candidate) || candidate<=lo || candidate>=hi) {
            candidate=0.5*(lo+hi); *ops+=2;
        }
        s=candidate;
        (*steps)++;
    }
    s=0.5*(lo+hi); *ops+=2;
    return add(a,mul(delta,s,ops),ops);
}

static Vec project(Vec p,double scale,long long *ops,long long *steps) {
    for(int it=0;it<MAX_PROJ;it++) {
        double f=F(p,ops);
        if(fabs(f)<=TOL) break;
        Vec g=grad(p,ops);
        double gg=dot(g,g,ops);
        if(gg<1e-22) break;
        Vec delta=mul(g,f/gg,ops); *ops+=1;
        double length=norm(delta,ops);
        if(length>0.5*scale) { delta=mul(delta,0.5*scale/length,ops); *ops+=2; }
        double alpha=1;
        for(int k=0;k<10;k++) {
            Vec q=sub(p,mul(delta,alpha,ops),ops);
            if(fabs(F(q,ops))<fabs(f)) { p=q; break; }
            alpha*=0.5; *ops+=1;
        }
        (*steps)++;
    }
    return p;
}

static double triangle_area(Vec a,Vec b,Vec c,long long *ops,long long *steps) {
    Vec n[6]={a,b,c};
    n[3]=mul(add(a,b,ops),0.5,ops);
    n[4]=mul(add(b,c,ops),0.5,ops);
    n[5]=mul(add(c,a,ops),0.5,ops);
    double scale=norm(sub(b,a,ops),ops);
    double t=norm(sub(c,b,ops),ops); if(t>scale) scale=t;
    t=norm(sub(a,c,ops),ops); if(t>scale) scale=t;
    for(int i=3;i<6;i++) n[i]=project(n[i],scale,ops,steps);
    double area=0;
    for(int k=0;k<7;k++) {
        double L1=quad[k][0],L2=quad[k][1],L3=quad[k][2];
        double du[6]={-(4*L1-1),4*L2-1,0,4*(L1-L2),4*L3,-4*L3};
        double dv[6]={-(4*L1-1),0,4*L3-1,-4*L2,4*L2,4*(L1-L3)};
        *ops+=16;
        Vec u={0,0,0},v={0,0,0};
        for(int j=0;j<6;j++) {
            u=add(u,mul(n[j],du[j],ops),ops);
            v=add(v,mul(n[j],dv[j],ops),ops);
        }
        Vec cr=cross(u,v,ops);
        area+=quad[k][3]*norm(cr,ops); *ops+=2;
    }
    return area;
}

/* Local edge cache: shared edge roots computed once per rank. */
#define EDGE_CACHE_SIZE (3*(NA+1)*(NA+1)*(NA+1))
static Vec cache[EDGE_CACHE_SIZE];
static unsigned char cache_valid[EDGE_CACHE_SIZE];
static int edge_key(int i,int j,int k,int axis) {
    return 3*((i*(NA+1)+j)*(NA+1)+k)+axis;
}
static AreaResult area_cell(int id) {
    AreaResult out={0};
    int i=id/(NA*NA),j=(id/NA)%NA,k=id%NA;
    Vec p[8]; double f[8]; int mask=0;
    for(int c=0;c<8;c++) {
        p[c]=point(i+corner[c][0],j+corner[c][1],k+corner[c][2],NA,&out.flops);
        f[c]=F(p[c],&out.flops);
        if(f[c]<0) mask|=1<<c;
    }
    if(mask==0 || mask==255) return out;
    Vec verts[12];
    unsigned used=0;
    for(int t=0;t<16 && tri_table[mask][t]>=0;t++) used|=1u<<tri_table[mask][t];
    for(int e=0;e<12;e++) if(used&(1u<<e)) {
        int a=edges[e][0],b=edges[e][1];
        int gi=i+(corner[a][0]<corner[b][0]?corner[a][0]:corner[b][0]);
        int gj=j+(corner[a][1]<corner[b][1]?corner[a][1]:corner[b][1]);
        int gk=k+(corner[a][2]<corner[b][2]?corner[a][2]:corner[b][2]);
        int axis=corner[a][0]!=corner[b][0]?0:(corner[a][1]!=corner[b][1]?1:2);
        int key=edge_key(gi,gj,gk,axis);
        if(!cache_valid[key]) {
            cache[key]=edge_root(p[a],p[b],&out.flops,&out.newton);
            cache_valid[key]=1;
        }
        verts[e]=cache[key];
    }
    for(int t=0;t<16 && tri_table[mask][t]>=0;t+=3) {
        int a=tri_table[mask][t],b=tri_table[mask][t+1],c=tri_table[mask][t+2];
        out.area+=triangle_area(verts[a],verts[b],verts[c],&out.flops,&out.projections);
        out.flops++; /* area accumulation */
        out.triangles++;
    }
    return out;
}

static int *surface_ids(int *count,long long *flops,int **weights) {
    double f[(NA+1)*(NA+1)*(NA+1)];
    int *ids=malloc(NA*NA*NA*sizeof(int));
    *weights=malloc(NA*NA*NA*sizeof(int));
    for(int i=0;i<=NA;i++) for(int j=0;j<=NA;j++) for(int k=0;k<=NA;k++) {
        Vec p=point(i,j,k,NA,flops);
        f[(i*(NA+1)+j)*(NA+1)+k]=F(p,flops);
    }
    *count=0;
    for(int i=0;i<NA;i++) for(int j=0;j<NA;j++) for(int k=0;k<NA;k++) {
        int neg=0, mask=0;
        for(int c=0;c<8;c++) {
            int x=i+corner[c][0],y=j+corner[c][1],z=k+corner[c][2];
            if(f[(x*(NA+1)+y)*(NA+1)+z]<0) { neg++; mask|=1<<c; }
        }
        if(neg>0 && neg<8) {
            int tris=0;
            while(tris*3<16 && tri_table[mask][tris*3]>=0) tris++;
            ids[*count]=(i*NA+j)*NA+k;
            (*weights)[*count]=200+1200*tris; /* approximate curved-triangle work */
            (*count)++;
        }
    }
    return ids;
}

static void log_result(FILE *fp,int ranks,double area,double mass,
                       long long *area_rank,long long *mass_rank,
                       long long tris,long long roots,long long projections,
                       double seconds) {
    long long area_total=0,mass_total=0;
    long long area_max=0,mass_max=0,total_max=0;
    for(int r=0;r<ranks;r++) {
        long long total=area_rank[r]+mass_rank[r];
        area_total+=area_rank[r];
        mass_total+=mass_rank[r];
        if(area_rank[r]>area_max) area_max=area_rank[r];
        if(mass_rank[r]>mass_max) mass_max=mass_rank[r];
        if(total>total_max) total_max=total;
    }
    double ea=100*fabs(area-REF_AREA)/REF_AREA;
    double em=100*fabs(mass-REF_MASS)/REF_MASS;
    fprintf(fp,"============================================================\n");
    fprintf(fp,"P=%d  |  curved area N=%d  |  voxel mass N=%d\n",ranks,NA,NM);
    fprintf(fp,"Surface area  %.9f  reference %.5f  error %.5f%%  %s\n",
            area,REF_AREA,ea,ea<0.05?"PASS":"FAIL");
    fprintf(fp,"Mass          %.9f  reference %.5f  error %.5f%%  %s\n",
            mass,REF_MASS,em,em<0.05?"PASS":"FAIL");
    fprintf(fp,"Triangles=%lld  Newton steps=%lld  midpoint steps=%lld\n",
            tris,roots,projections);
    fprintf(fp,"\nACTUAL COUNT PER MPI RANK (source-level arithmetic model)\n");
    fprintf(fp,"Rank       Area FLOPs       Mass FLOPs      Total FLOPs\n");
    for(int r=0;r<ranks;r++) {
        fprintf(fp,"%4d %16lld %16lld %16lld\n",r,
                area_rank[r],mass_rank[r],area_rank[r]+mass_rank[r]);
    }
    fprintf(fp,"\nPROFESSOR'S REQUIRED TABLE: AVERAGE FLOPs PER CORE\n");
    fprintf(fp,"                 P=%d\n",ranks);
    fprintf(fp,"Surface area     %.2f\n",(double)area_total/ranks);
    fprintf(fp,"Mass at t=1      %.2f\n",(double)mass_total/ranks);
    fprintf(fp,"\nTotal arithmetic: area=%lld mass=%lld combined=%lld\n",
            area_total,mass_total,area_total+mass_total);
    fprintf(fp,"Maximum rank FLOPs: area=%lld mass=%lld combined=%lld\n",
            area_max,mass_max,total_max);
    fprintf(fp,"Combined load imbalance (max/average): %.3f\n",
            (double)total_max/((double)(area_total+mass_total)/ranks));
    fprintf(fp,"Elapsed time (slowest rank): %.6f s\n",seconds);
    fprintf(fp,"Count convention: +,-,*,/,sqrt each 1; integer work, comparisons, MPI and memory excluded.\n");
    fprintf(fp,"These are instrumented arithmetic counts, not CPU hardware counters.\n\n");
    fflush(fp);
}

int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=1 && size!=4 && size!=16) {
        if(rank==0) fprintf(stderr,"Run with 1, 4 or 16 ranks.\n");
        MPI_Finalize(); return 1;
    }
    const char *logname=argc>1?argv[1]:"heart_hybrid.log";
    double start=MPI_Wtime();
    int n_surface=0,*ids=NULL,*weights=NULL;
    long long discovery_flops=0;
    if(rank==0) ids=surface_ids(&n_surface,&discovery_flops,&weights);
    MPI_Bcast(&n_surface,1,MPI_INT,0,MPI_COMM_WORLD);
    int *counts=calloc(size,sizeof(int)),*offsets=calloc(size,sizeof(int));
    int *packed=NULL;
    if(rank==0) {
        /* Greedy assignment: balance estimated curved-triangle work, including
           the surface discovery work performed by rank 0. */
        int *owner=malloc(n_surface*sizeof(int));
        long long *load=calloc(size,sizeof(long long));
        load[0]=discovery_flops;
        for(int t=0;t<n_surface;t++) {
            int best=0;
            for(int r=1;r<size;r++) if(load[r]<load[best]) best=r;
            owner[t]=best;
            counts[best]++;
            load[best]+=weights[t];
        }
        for(int r=1;r<size;r++) offsets[r]=offsets[r-1]+counts[r-1];
        int *next=malloc(size*sizeof(int));
        memcpy(next,offsets,size*sizeof(int));
        packed=malloc(n_surface*sizeof(int));
        for(int t=0;t<n_surface;t++) packed[next[owner[t]]++]=ids[t];
        free(next); free(owner); free(load); free(ids); free(weights);
    }
    MPI_Bcast(counts,size,MPI_INT,0,MPI_COMM_WORLD);
    if(rank!=0) for(int r=1;r<size;r++) offsets[r]=offsets[r-1]+counts[r-1];
    int *my_ids=malloc((counts[rank]+1)*sizeof(int));
    MPI_Scatterv(packed,counts,offsets,MPI_INT,my_ids,counts[rank],MPI_INT,0,MPI_COMM_WORLD);
    free(packed);
    AreaResult local={0};
    for(int t=0;t<counts[rank];t++) {
        AreaResult x=area_cell(my_ids[t]);
        local.area+=x.area;
        local.flops+=x.flops+1; /* local area sum */
        local.triangles+=x.triangles;
        local.newton+=x.newton;
        local.projections+=x.projections;
    }
    free(my_ids); free(counts); free(offsets);
    if(rank==0) local.flops+=discovery_flops;
    /* Flat index partition: each mass voxel belongs to exactly one rank. */
    long long inside=0,mass_flops=0;
    int total=NM*NM*NM;
    int begin=(long long)total*rank/size,end=(long long)total*(rank+1)/size;
    for(int id=begin;id<end;id++) {
        int i=id/(NM*NM),j=(id/NM)%NM,k=id%NM;
        Vec p=point(i,j,k,NM,&mass_flops);
        p.x+=(high[0]-low[0])/(2*NM);
        p.y+=(high[1]-low[1])/(2*NM);
        p.z+=(high[2]-low[2])/(2*NM);
        mass_flops+=9; /* three half-step calculations per coordinate */
        if(F(p,&mass_flops)<0) inside++;
    }
    if(rank==0) mass_flops+=10; /* final voxel widths and mass product */
    double area=0,seconds=MPI_Wtime()-start,elapsed=0;
    long long total_inside=0,totals[3]={0},local_stats[3]={local.triangles,local.newton,local.projections};
    MPI_Reduce(&local.area,&area,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&inside,&total_inside,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(local_stats,totals,3,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&seconds,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    long long *area_per=NULL,*mass_per=NULL;
    if(rank==0) { area_per=malloc(size*sizeof(long long)); mass_per=malloc(size*sizeof(long long)); }
    MPI_Gather(&local.flops,1,MPI_LONG_LONG,area_per,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Gather(&mass_flops,1,MPI_LONG_LONG,mass_per,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);
    if(rank==0) {
        area_per[0]+=size-1; /* model additions for MPI area reduction */
        double dx=(high[0]-low[0])/NM,dy=(high[1]-low[1])/NM,dz=(high[2]-low[2])/NM;
        double mass=0.5*total_inside*dx*dy*dz;
        FILE *fp=fopen(logname,"a");
        if(!fp) { perror(logname); fp=stderr; }
        log_result(fp,size,area,mass,area_per,mass_per,
                   totals[0],totals[1],totals[2],elapsed);
        if(fp!=stderr) fclose(fp);
        printf("P=%d area=%.8f mass=%.8f  details: %s\n",size,area,mass,logname);
        free(area_per); free(mass_per);
    }
    MPI_Finalize(); return 0;
}
