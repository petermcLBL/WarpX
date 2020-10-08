/* Copyright 2019 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralKSpace.H"
#include "SpectralSolver.H"
#include "SpectralAlgorithms/PsatdAlgorithm.H"
#include "SpectralAlgorithms/GalileanAlgorithm.H"
#include "SpectralAlgorithms/AvgGalileanAlgorithm.H"
#include "SpectralAlgorithms/PMLPsatdAlgorithm.H"
#include "WarpX.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Utils/WarpXUtil.H"

#if WARPX_USE_PSATD

/* \brief Initialize the spectral Maxwell solver
 *
 * This function selects the spectral algorithm to be used, allocates the
 * corresponding coefficients for the discretized field update equation,
 * and prepares the structures that store the fields in spectral space.
 *
 * \param norder_x Order of accuracy of the spatial derivatives along x
 * \param norder_y Order of accuracy of the spatial derivatives along y
 * \param norder_z Order of accuracy of the spatial derivatives along z
 * \param nodal    Whether the solver is applied to a nodal or staggered grid
 * \param dx       Cell size along each dimension
 * \param dt       Time step
 * \param pml      Whether the boxes in which the solver is applied are PML boxes
 * \param periodic_single_box Whether the full simulation domain consists of a single periodic box (i.e. the global domain is not MPI parallelized)
 */
SpectralSolver::SpectralSolver(
                const amrex::BoxArray& realspace_ba,
                const amrex::DistributionMapping& dm,
                const int norder_x, const int norder_y,
                const int norder_z, const bool nodal,
                const amrex::Array<amrex::Real,3>& v_galilean,
                const amrex::RealVect dx, const amrex::Real dt,
                const bool pml, const bool periodic_single_box,
                const bool update_with_rho,
                const bool fft_do_time_averaging) {
#if WARPX_USE_SPIRAL
  std::cout << "SpectralSolver nodal = " << nodal << std::endl;
#endif
    // Initialize all structures using the same distribution mapping dm

    // - Initialize k space object (Contains info about the size of
    // the spectral space corresponding to each box in `realspace_ba`,
    // as well as the value of the corresponding k coordinates)
    const SpectralKSpace k_space= SpectralKSpace(realspace_ba, dm, dx);

    // - Select the algorithm depending on the input parameters
    //   Initialize the corresponding coefficients over k space

    if (pml) {
        algorithm = std::unique_ptr<PMLPsatdAlgorithm>( new PMLPsatdAlgorithm(
            k_space, dm, norder_x, norder_y, norder_z, nodal, dt ) );
    }
    else {
        if (fft_do_time_averaging){
            algorithm = std::unique_ptr<AvgGalileanAlgorithm>( new AvgGalileanAlgorithm(
                k_space, dm, norder_x, norder_y, norder_z, nodal, v_galilean, dt ) );
        }
        else {
            if ((v_galilean[0]==0) && (v_galilean[1]==0) && (v_galilean[2]==0)){
                // v_galilean is 0: use standard PSATD algorithm
                algorithm = std::unique_ptr<PsatdAlgorithm>( new PsatdAlgorithm(
                   k_space, dm, norder_x, norder_y, norder_z, nodal, dt, update_with_rho ) );
            }
            else {
                algorithm = std::unique_ptr<GalileanAlgorithm>( new GalileanAlgorithm(
                    k_space, dm, norder_x, norder_y, norder_z, nodal, v_galilean, dt, update_with_rho ) );
            }
        }
    }

    // - Initialize arrays for fields in spectral space + FFT plans
    field_data = SpectralFieldData( realspace_ba, k_space, dm,
            algorithm->getRequiredNumberOfFields(), periodic_single_box );

}

void
SpectralSolver::ForwardTransform( const amrex::MultiFab& mf,
                                  const int field_index,
                                  const int i_comp )
{
    WARPX_PROFILE("SpectralSolver::ForwardTransform");
    field_data.ForwardTransform( mf, field_index, i_comp );
}

#if WARPX_USE_SPIRAL
void
SpectralSolver::allForwardTransform(
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield,
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& current,
    std::unique_ptr<amrex::MultiFab>& rho )
{
  WARPX_PROFILE("SpectralSolver::allForwardTransform");
  field_data.allForwardTransform(Efield, Bfield, current, rho);
}

void
SpectralSolver::compareSpiralForwardStep()
{
  field_data.compareSpiralForwardStep();
}

void
SpectralSolver::scaleSpiralForward()
{
  field_data.scaleSpiralForward();
}

void
SpectralSolver::setCopySpectralFieldBackward()
{
  field_data.setCopySpectralFieldBackward();
}

void
SpectralSolver::scaleSpiralBackward()
{
  field_data.scaleSpiralBackward();
}

void
SpectralSolver::allBackwardTransform(
                                     std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
                                     std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield)
{
  WARPX_PROFILE("SpectralSolver::allBackwardTransform");
  field_data.allBackwardTransform(Efield, Bfield);
}

void
SpectralSolver::compareSpiralBackwardStep(
                                          std::array<std::unique_ptr< amrex::MultiFab >, 3>& EfieldBack,
                                          std::array<std::unique_ptr< amrex::MultiFab >, 3>& BfieldBack,
                                          std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
                                          std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield)
{
  field_data.compareSpiralBackwardStep(EfieldBack, BfieldBack, Efield, Bfield);
}

#endif

void
SpectralSolver::BackwardTransform( amrex::MultiFab& mf,
                                   const int field_index,
                                   const int i_comp )
{
    WARPX_PROFILE("SpectralSolver::BackwardTransform");
    field_data.BackwardTransform( mf, field_index, i_comp );
}

void
SpectralSolver::pushSpectralFields(){
    WARPX_PROFILE("SpectralSolver::pushSpectralFields");
    // Virtual function: the actual function used here depends
    // on the sub-class of `SpectralBaseAlgorithm` that was
    // initialized in the constructor of `SpectralSolver`
    algorithm->pushSpectralFields( field_data );
}

#if WARPX_USE_FULL_SPIRAL
void
SpectralSolver::stepSpiral(std::array<std::unique_ptr<amrex::MultiFab>,3>& EfieldNew,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& BfieldNew,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield_avg,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield_avg,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& current,
                           std::unique_ptr<amrex::MultiFab>& rho )
{
  WARPX_PROFILE("SpectralSolver::stepSpiral");
  // Virtual function: the actual function used here depends
  // on the sub-class of `SpectralBaseAlgorithm` that was
  // initialized in the constructor of `SpectralSolver`
  algorithm->stepSpiral(EfieldNew, BfieldNew,
                        Efield, Bfield, Efield_avg, Bfield_avg, current, rho);
}
#endif

#endif // WARPX_USE_PSATD
