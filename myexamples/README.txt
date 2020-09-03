USE_SPIRAL=TRUE means compare with the Spiral-generated code.
In order to run it, also need USE_PSATD=TRUE.

USE_FULL_SPIRAL=TRUE means do the full step with Spiral-generated code.
In order to run it, also need USE_SPIRAL=TRUE and USE_PSATD=TRUE.
Not working yet.

In directory
Source/FieldSolver/SpectralSolver/SpectralAlgorithms
if USE_SPIRAL=TRUE then you need the Spiral-generated:
warpx-symbol_rho_80.c
warpx-symbol_norho_80.c
warpx-fullstep_rho_80.c

To build, from WarpX directory:
make DIM=3 USE_PSATD=TRUE COMP=gcc USE_MPI=TRUE USE_OMP=FALSE USE_GPU=FALSE USE_SPIRAL=TRUE

Input files take 20 steps of PSATD:
inputs_3d_multi_rt_rho_20  updating with rho
inputs_3d_multi_rt_norho_20  updating without rho

To run, from this directory:
../Bin/main3d.gnu.TPROF.MPI.PSATD.ex inputs_3d_multi_rt_rho_20
../Bin/main3d.gnu.TPROF.MPI.PSATD.ex inputs_3d_multi_rt_norho_20
