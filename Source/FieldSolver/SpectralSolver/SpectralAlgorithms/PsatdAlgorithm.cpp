/* Copyright 2019 Remi Lehe, Revathi Jambunathan, Edoardo Zoni
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "PsatdAlgorithm.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <cmath>


#if WARPX_USE_PSATD
using namespace amrex;

#if WARPX_USE_SPIRAL
#include "warpx-symbol_rho_80.c"
#include "warpx-symbol_norho_80.c"
#include "warpx-fullstep_rho_80.c"
#include "psatd.fftx.codegen.hpp"
#endif

/**
 * \brief Constructor
 */
PsatdAlgorithm::PsatdAlgorithm(const SpectralKSpace& spectral_kspace,
                         const DistributionMapping& dm,
                         const int norder_x, const int norder_y,
                         const int norder_z, const bool nodal, const Real dt,
                         const bool update_with_rho)
    // Initialize members of base class
    : m_dt( dt ),
      m_update_with_rho( update_with_rho ),
      SpectralBaseAlgorithm( spectral_kspace, dm, norder_x, norder_y, norder_z, nodal )
{
    const BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Allocate the arrays of coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X1_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X2_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X3_coef = SpectralRealCoefficients(ba, dm, 1, 0);

    // Initialize coefficients for update equations
    InitializeSpectralCoefficients(spectral_kspace, dm, dt);
#if WARPX_USE_FULL_SPIRAL
    psatd::init();
#endif
}

/**
 * \brief Advance E and B fields in spectral space (stored in `f`) over one time step
 */
void
PsatdAlgorithm::pushSpectralFields(SpectralFieldData& f) const{

    const bool update_with_rho = m_update_with_rho;

    // Loop over boxes
    for (MFIter mfi(f.fields); mfi.isValid(); ++mfi){

        const Box& bx = f.fields[mfi].box();

#if WARPX_USE_SPIRAL
        std::cout << "PsatdAlgorithm::pushSpectralFields"
                  << " update_with_rho=" << update_with_rho
                  << " box " << bx
                  << " c = " << PhysConst::c
                  << " ep0 = " << PhysConst::ep0
                  << std::endl;
#endif
        // Extract arrays for the fields to be updated
        Array4<Complex> fields = f.fields[mfi].array();
        // Extract arrays for the coefficients
        Array4<const Real> C_arr = C_coef[mfi].array();
        Array4<const Real> S_ck_arr = S_ck_coef[mfi].array();
        Array4<const Real> X1_arr = X1_coef[mfi].array();
        Array4<const Real> X2_arr = X2_coef[mfi].array();
        Array4<const Real> X3_arr = X3_coef[mfi].array();
        // Extract pointers for the k vectors
        const Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

#if WARPX_USE_SPIRAL
        BaseFab<Complex> spiral_new(bx, 6);
        // Array4<Complex> spiral_new4 = spiral_new.array();
        // We are assuming fields in the SpectralFieldIndex order:
        // Ex=0, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho_old, rho_new.
        double** sym_for_psatd = new double*[8];
        int xdim = bx.length(0);
        int ydim = bx.length(1);
        int zdim = bx.length(2);
        int npts = bx.numPts();
        std::cout << "Dimensions "
                  << xdim << " " << ydim << " " << zdim
                  << " Points " << npts
                  << std::endl;
        sym_for_psatd[0] = (double*) modified_kx_arr;
        sym_for_psatd[1] = (double*) modified_ky_arr;
        sym_for_psatd[2] = (double*) modified_kz_arr;
        sym_for_psatd[3] = (double*) C_arr.dataPtr();
        sym_for_psatd[4] = (double*) S_ck_arr.dataPtr();
        sym_for_psatd[5] = (double*) X1_arr.dataPtr();
        sym_for_psatd[6] = (double*) X2_arr.dataPtr();
        sym_for_psatd[7] = (double*) X3_arr.dataPtr();
        if (update_with_rho) {
          init_warpxsym_rho_80();
          warpxsym_rho_80((double*) spiral_new.dataPtr(),
                          (double*) fields.dataPtr(),
                          sym_for_psatd);
        } else {
          init_warpxsym_norho_80();
          warpxsym_norho_80((double*) spiral_new.dataPtr(),
                            (double*) fields.dataPtr(),
                            sym_for_psatd);
        }
#endif
        // Loop over indices within one box
        ParallelFor( bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            using Idx = SpectralFieldIndex;

            const Complex Ex_old = fields(i,j,k,Idx::Ex);
            const Complex Ey_old = fields(i,j,k,Idx::Ey);
            const Complex Ez_old = fields(i,j,k,Idx::Ez);
            const Complex Bx_old = fields(i,j,k,Idx::Bx);
            const Complex By_old = fields(i,j,k,Idx::By);
            const Complex Bz_old = fields(i,j,k,Idx::Bz);

            // Shortcut for the values of J and rho
            const Complex Jx = fields(i,j,k,Idx::Jx);
            const Complex Jy = fields(i,j,k,Idx::Jy);
            const Complex Jz = fields(i,j,k,Idx::Jz);
            const Complex rho_old = fields(i,j,k,Idx::rho_old);
            const Complex rho_new = fields(i,j,k,Idx::rho_new);

            // k vector values, and coefficients
            const Real kx = modified_kx_arr[i];
#if (AMREX_SPACEDIM==3)
            const Real ky = modified_ky_arr[j];
            const Real kz = modified_kz_arr[k];
#else
            constexpr Real ky = 0;
            const Real kz = modified_kz_arr[j];
#endif
            constexpr Real c2 = PhysConst::c*PhysConst::c;
            constexpr Real inv_eps0 = 1.0_rt/PhysConst::ep0;

            const Complex I = Complex{0,1};

            const Real C = C_arr(i,j,k);
            const Real S_ck = S_ck_arr(i,j,k);
            const Real X1 = X1_arr(i,j,k);
            const Real X2 = X2_arr(i,j,k);
            const Real X3 = X3_arr(i,j,k);

            // Update E (see WarpX online documentation: theory section)

            if (update_with_rho) {

                fields(i,j,k,Idx::Ex) = C*Ex_old + S_ck*(c2*I*(ky*Bz_old-kz*By_old)-inv_eps0*Jx)
                                        - I*(X2*rho_new-X3*rho_old)*kx;

                fields(i,j,k,Idx::Ey) = C*Ey_old + S_ck*(c2*I*(kz*Bx_old-kx*Bz_old)-inv_eps0*Jy)
                                        - I*(X2*rho_new-X3*rho_old)*ky;

                fields(i,j,k,Idx::Ez) = C*Ez_old + S_ck*(c2*I*(kx*By_old-ky*Bx_old)-inv_eps0*Jz)
                                        - I*(X2*rho_new-X3*rho_old)*kz;
            } else {

                Complex k_dot_J = kx*Jx + ky*Jy + kz*Jz;
                Complex k_dot_E = kx*Ex_old + ky*Ey_old + kz*Ez_old;

                fields(i,j,k,Idx::Ex) = C*Ex_old + S_ck*(c2*I*(ky*Bz_old-kz*By_old)-inv_eps0*Jx)
                                        + X2*k_dot_E*kx + X3*inv_eps0*k_dot_J*kx;

                fields(i,j,k,Idx::Ey) = C*Ey_old + S_ck*(c2*I*(kz*Bx_old-kx*Bz_old)-inv_eps0*Jy)
                                        + X2*k_dot_E*ky + X3*inv_eps0*k_dot_J*ky;

                fields(i,j,k,Idx::Ez) = C*Ez_old + S_ck*(c2*I*(kx*By_old-ky*Bx_old)-inv_eps0*Jz)
                                        + X2*k_dot_E*kz + X3*inv_eps0*k_dot_J*kz;
            }

            // Update B (see WarpX online documentation: theory section)

            fields(i,j,k,Idx::Bx) = C*Bx_old - S_ck*I*(ky*Ez_old-kz*Ey_old) + X1*I*(ky*Jz-kz*Jy);

            fields(i,j,k,Idx::By) = C*By_old - S_ck*I*(kz*Ex_old-kx*Ez_old) + X1*I*(kz*Jx-kx*Jz);

            fields(i,j,k,Idx::Bz) = C*Bz_old - S_ck*I*(kx*Ey_old-ky*Ex_old) + X1*I*(kx*Jy-ky*Jx);
        } );

#if WARPX_USE_SPIRAL
        BaseFab<Complex> diff_new(bx, 6);
        Array4<Complex> diff_new4 = diff_new.array();
        const BaseFab<Complex> fields_new(fields);
        // components: source, dest, number
        diff_new.copy(fields_new, 0, 0, 6);
        diff_new.minus(spiral_new, 0, 0, 6);
        for (int icomp = 0; icomp < 6; icomp++)
          {
            // .norm(1, icomp) does not work.
            Real soln_sum1 = 0.; // fields_new.norm(1, icomp);
            Real diff_sum1 = 0.; // diff_new.norm(1, icomp);
            Real soln_max = 0.;
            Real diff_max = 0.;
            // I don't want to use the GPU version, for race conditions.
            const Dim3 lo = amrex::lbound(bx);
            const Dim3 hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
              for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                  {
                    Real fields_new_abs = abs(fields(i, j, k, icomp));
                    Real diff_new_abs = abs(diff_new4(i, j, k, icomp));
                    soln_sum1 += fields_new_abs;
                    diff_sum1 += diff_new_abs;
                    if (fields_new_abs > soln_max)
                      {
                        soln_max = fields_new_abs;
                      }
                    if (diff_new_abs > diff_max)
                      {
                        diff_max = diff_new_abs;
                      }
                  }
            // Real soln_avg = soln_sum1 / (npts * 1.);
            std::cout << "Component " << icomp
                      << " |diff| <= " << diff_max
                      << " |solution| <= " << soln_max
                      << " relative " << (diff_max/soln_max)
                      << std::endl;
          }
        delete[] sym_for_psatd;
#endif
    }
};

#if WARPX_USE_FULL_SPIRAL

fftx::array_t<3, double> alias(amrex::BaseFab<double>& fab)
{
  const amrex::Box& b = fab.box();
  amrex::IntVect lo=b.smallEnd(), hi=b.bigEnd();
  fftx::box_t<3> b2({{lo[0],lo[1],lo[2]}},{{hi[0],hi[1],hi[2]}});
  return fftx::array_t<3,double>(fftx::global_ptr<double>(fab.dataPtr(), 0,0),b2);
}
fftx::array_t<3, double> alias(amrex::PODVector<double>& v)
{

  fftx::box_t<3> b2({{1,1,1}},{{static_cast<int>(v.size()),1,1}});
  return fftx::array_t<3,double>(fftx::global_ptr<double>(v.dataPtr(), 0,0),b2);
}

/**
 * \brief Advance E and B fields in spectral space (stored in `f`) over one time step
 */
void
PsatdAlgorithm::stepSpiral(std::array<std::unique_ptr<amrex::MultiFab>,3>& EfieldNew,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& BfieldNew,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield_avg,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield_avg,
                           std::array<std::unique_ptr<amrex::MultiFab>,3>& current,
                           std::unique_ptr<amrex::MultiFab>& rho) {
  const bool update_with_rho = m_update_with_rho;

  // Loop over boxes
  for (MFIter mfi(*rho); mfi.isValid(); ++mfi){

    // Extract arrays for the coefficients
    Array4<const Real> C_arr = C_coef[mfi].array();
    Array4<const Real> S_ck_arr = S_ck_coef[mfi].array();
    Array4<const Real> X1_arr = X1_coef[mfi].array();
    Array4<const Real> X2_arr = X2_coef[mfi].array();
    Array4<const Real> X3_arr = X3_coef[mfi].array();
    // Extract pointers for the k vectors
    const Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
    const Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
    const Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

    // Extract arrays for the fields to be updated
    // Array4<Complex> fields = f.fields[mfi].array();
    FArrayBox& ExFab = (*Efield[0])[mfi];
    FArrayBox& EyFab = (*Efield[1])[mfi];
    FArrayBox& EzFab = (*Efield[2])[mfi];
    FArrayBox& BxFab = (*Bfield[0])[mfi];
    FArrayBox& ByFab = (*Bfield[1])[mfi];
    FArrayBox& BzFab = (*Bfield[2])[mfi];
    FArrayBox& JxFab = (*current[0])[mfi];
    FArrayBox& JyFab = (*current[1])[mfi];
    FArrayBox& JzFab = (*current[2])[mfi];
    FArrayBox& rhoFab = (*rho)[mfi];
    FArrayBox rhoOldFab(rhoFab, amrex::make_alias, 0, 1); // first comp, number
    FArrayBox rhoNewFab(rhoFab, amrex::make_alias, 1, 1); // first comp, number

    FArrayBox& ExNewFab = (*EfieldNew[0])[mfi];
    FArrayBox& EyNewFab = (*EfieldNew[1])[mfi];
    FArrayBox& EzNewFab = (*EfieldNew[2])[mfi];
    FArrayBox& BxNewFab = (*BfieldNew[0])[mfi];
    FArrayBox& ByNewFab = (*BfieldNew[1])[mfi];
    FArrayBox& BzNewFab = (*BfieldNew[2])[mfi];

    std::cout << "ExFab on " << ExFab.box() << std::endl;
    std::cout << "EyFab on " << EyFab.box() << std::endl;
    std::cout << "EzFab on " << EzFab.box() << std::endl;
    std::cout << "BxFab on " << BxFab.box() << std::endl;
    std::cout << "ByFab on " << ByFab.box() << std::endl;
    std::cout << "BzFab on " << BzFab.box() << std::endl;

    std::cout << "PsatdAlgorithm::stepSpiral allocating inputDataPtr" << std::endl;
    double** inputDataPtr = new double*[11];
    
    std::array<fftx::array_t<3,double>,11> inputArray = {alias(ExFab),alias(EyFab),alias(EzFab),
                                                         alias(BxFab),alias(ByFab),alias(BzFab),
                                                         alias(JxFab),alias(JyFab),alias(JzFab),
                                                         alias(rhoOldFab),alias(rhoNewFab)};
    inputDataPtr[0] = (double*) ExFab.dataPtr(); 
    inputDataPtr[1] = (double*) EyFab.dataPtr();
    inputDataPtr[2] = (double*) EzFab.dataPtr();
    inputDataPtr[3] = (double*) BxFab.dataPtr();
    inputDataPtr[4] = (double*) ByFab.dataPtr();
    inputDataPtr[5] = (double*) BzFab.dataPtr();
    inputDataPtr[6] = (double*) JxFab.dataPtr();
    inputDataPtr[7] = (double*) JyFab.dataPtr();
    inputDataPtr[8] = (double*) JzFab.dataPtr();
    inputDataPtr[9] = (double*) rhoOldFab.dataPtr();
    inputDataPtr[10] = (double*) rhoNewFab.dataPtr();

    std::cout << "PsatdAlgorithm::stepSpiral allocating outputDataPtr" << std::endl;
    double** outputDataPtr = new double*[6];
    std::array<fftx::array_t<3,double>,6> outputArray = {alias(ExNewFab),alias(EyNewFab),alias(EzNewFab),
                                                         alias(BxNewFab),alias(ByNewFab),alias(BzNewFab)};
                                                        
    outputDataPtr[0] = (double*) ExNewFab.dataPtr();
    outputDataPtr[1] = (double*) EyNewFab.dataPtr();
    outputDataPtr[2] = (double*) EzNewFab.dataPtr();
    outputDataPtr[3] = (double*) BxNewFab.dataPtr();
    outputDataPtr[4] = (double*) ByNewFab.dataPtr();
    outputDataPtr[5] = (double*) BzNewFab.dataPtr();
        
    std::cout << "PsatdAlgorithm::stepSpiral allocating sym_for_psatd" << std::endl;
    double** sym_for_psatd = new double*[8];
    std::array<fftx::array_t<3,double>,8> symArray = {alias(modified_kx_vec[mfi]),
                                                      alias(modified_ky_vec[mfi]),
                                                      alias(modified_kz_vec[mfi]),
                                                      alias(C_coef[mfi]),
                                                      alias(S_ck_coef[mfi]),
                                                      alias(X1_coef[mfi]),
                                                      alias(X2_coef[mfi]),
                                                      alias(X3_coef[mfi])};
    sym_for_psatd[0] = (double*) modified_kx_arr;
    sym_for_psatd[1] = (double*) modified_ky_arr;
    sym_for_psatd[2] = (double*) modified_kz_arr;
    sym_for_psatd[3] = (double*) C_arr.dataPtr();
    sym_for_psatd[4] = (double*) S_ck_arr.dataPtr();
    sym_for_psatd[5] = (double*) X1_arr.dataPtr();
    sym_for_psatd[6] = (double*) X2_arr.dataPtr();
    sym_for_psatd[7] = (double*) X3_arr.dataPtr();

    
    // Call the Spiral-generated C function here.
    if (update_with_rho) {
      /*
      std::cout << "Calling init_warpxfull_rho_80" << std::endl;
      init_warpxfull_rho_80();
      std::cout << "Calling warpxfull_rho_80" << std::endl;
      warpxfull_rho_80(outputDataPtr, inputDataPtr, sym_for_psatd);
      std::cout << "Called warpxfull_rho_80" << std::endl;
      */
      psatd::transform(inputArray, outputArray, symArray);
    } else {
    }

    delete[] sym_for_psatd;
    delete[] inputDataPtr;
    delete[] outputDataPtr;
  }
}
#endif


/**
 * \brief Initialize coefficients for update equations
 */
void PsatdAlgorithm::InitializeSpectralCoefficients(const SpectralKSpace& spectral_kspace,
                                    const amrex::DistributionMapping& dm,
                                    const amrex::Real dt)
{
    const bool update_with_rho = m_update_with_rho;

    const BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Loop over boxes and allocate the corresponding coefficients
    // for each box owned by the local MPI proc
    for (MFIter mfi(ba, dm); mfi.isValid(); ++mfi){

        const Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const Real* modified_kx = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const Real* modified_ky = modified_ky_vec[mfi].dataPtr();
#endif
        const Real* modified_kz = modified_kz_vec[mfi].dataPtr();

        // Extract arrays for the coefficients
        Array4<Real> C = C_coef[mfi].array();
        Array4<Real> S_ck = S_ck_coef[mfi].array();
        Array4<Real> X1 = X1_coef[mfi].array();
        Array4<Real> X2 = X2_coef[mfi].array();
        Array4<Real> X3 = X3_coef[mfi].array();

        // Loop over indices within one box
        ParallelFor( bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Calculate norm of vector
            const Real k_norm = std::sqrt(
                std::pow(modified_kx[i],2) +
#if (AMREX_SPACEDIM==3)
                std::pow(modified_ky[j],2) + std::pow(modified_kz[k],2));
#else
                std::pow(modified_kz[j],2));
#endif
            // Calculate coefficients
            constexpr Real c = PhysConst::c;
            constexpr Real eps0 = PhysConst::ep0;

            if (k_norm != 0) {
                C(i,j,k)    = std::cos(c*k_norm*dt);
                S_ck(i,j,k) = std::sin(c*k_norm*dt)/(c*k_norm);
                X1(i,j,k) = (1.0_rt-C(i,j,k))/(eps0*c*c*k_norm*k_norm);
                if (update_with_rho) {
                    X2(i,j,k) = (1.0_rt-S_ck(i,j,k)/dt)/(eps0*k_norm*k_norm);
                    X3(i,j,k) = (C(i,j,k)-S_ck(i,j,k)/dt)/(eps0*k_norm*k_norm);
                } else {
                    X2(i,j,k) = (1.0_rt-C(i,j,k))/(k_norm*k_norm);
                    X3(i,j,k) = (S_ck(i,j,k)-dt)/(k_norm*k_norm);
                }
            } else { // Handle k_norm = 0 with analytical limit
                C(i,j,k) = 1.0_rt;
                S_ck(i,j,k) = dt;
                X1(i,j,k) = 0.5_rt*dt*dt/eps0;
                if (update_with_rho) {
                    X2(i,j,k) = c*c*dt*dt/(6.0_rt*eps0);
                    X3(i,j,k) = -c*c*dt*dt/(3.0_rt*eps0);
                } else {
                    X2(i,j,k) = 0.5_rt*dt*dt*c*c;
                    X3(i,j,k) = -c*c*dt*dt*dt/6.0_rt;
                }
            }
        } );
     }
}

void
PsatdAlgorithm::CurrentCorrection (SpectralFieldData& field_data,
                                   std::array<std::unique_ptr<amrex::MultiFab>,3>& current,
                                   const std::unique_ptr<amrex::MultiFab>& rho) {
    // Profiling
    WARPX_PROFILE( "PsatdAlgorithm::CurrentCorrection" );

    using Idx = SpectralFieldIndex;

    // Forward Fourier transform of J and rho
    field_data.ForwardTransform( *current[0], Idx::Jx, 0 );
    field_data.ForwardTransform( *current[1], Idx::Jy, 0 );
    field_data.ForwardTransform( *current[2], Idx::Jz, 0 );
    field_data.ForwardTransform( *rho, Idx::rho_old, 0 );
    field_data.ForwardTransform( *rho, Idx::rho_new, 1 );

    // Loop over boxes
    for (MFIter mfi(field_data.fields); mfi.isValid(); ++mfi){

        const Box& bx = field_data.fields[mfi].box();

        // Extract arrays for the fields to be updated
        Array4<Complex> fields = field_data.fields[mfi].array();

        // Extract pointers for the k vectors
        const Real* const modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const Real* const modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const Real* const modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Local copy of member variables before GPU loop
        const Real dt = m_dt;

        // Loop over indices within one box
        ParallelFor( bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            using Idx = SpectralFieldIndex;

            // Shortcuts for the values of J and rho
            const Complex Jx = fields(i,j,k,Idx::Jx);
            const Complex Jy = fields(i,j,k,Idx::Jy);
            const Complex Jz = fields(i,j,k,Idx::Jz);
            const Complex rho_old = fields(i,j,k,Idx::rho_old);
            const Complex rho_new = fields(i,j,k,Idx::rho_new);

            // k vector values, and coefficients
            const Real kx = modified_kx_arr[i];
#if (AMREX_SPACEDIM==3)
            const Real ky = modified_ky_arr[j];
            const Real kz = modified_kz_arr[k];
#else
            constexpr Real ky = 0;
            const Real kz = modified_kz_arr[j];
#endif
            const Real k_norm = std::sqrt( kx*kx + ky*ky + kz*kz );

            constexpr Complex I = Complex{0,1};

            // div(J) in Fourier space
            const Complex k_dot_J = kx*Jx + ky*Jy + kz*Jz;

            // Correct J
            if ( k_norm != 0 )
            {
                fields(i,j,k,Idx::Jx) = Jx - (k_dot_J-I*(rho_new-rho_old)/dt)*kx/(k_norm*k_norm);
                fields(i,j,k,Idx::Jy) = Jy - (k_dot_J-I*(rho_new-rho_old)/dt)*ky/(k_norm*k_norm);
                fields(i,j,k,Idx::Jz) = Jz - (k_dot_J-I*(rho_new-rho_old)/dt)*kz/(k_norm*k_norm);
            }
        } );
    }

    // Backward Fourier transform of J
    field_data.BackwardTransform( *current[0], Idx::Jx, 0 );
    field_data.BackwardTransform( *current[1], Idx::Jy, 0 );
    field_data.BackwardTransform( *current[2], Idx::Jz, 0 );
}

void
PsatdAlgorithm::VayDeposition (SpectralFieldData& field_data,
                               std::array<std::unique_ptr<amrex::MultiFab>,3>& current) {
    // Profiling
    WARPX_PROFILE("PsatdAlgorithm::VayDeposition");

    using Idx = SpectralFieldIndex;

    // Forward Fourier transform of D (temporarily stored in current):
    // D is nodal and does not match the staggering of J, therefore we pass the
    // actual staggering of D (IntVect(1)) to the ForwardTransform function
    field_data.ForwardTransform(*current[0], Idx::Jx, 0, IntVect(1));
    field_data.ForwardTransform(*current[1], Idx::Jy, 0, IntVect(1));
    field_data.ForwardTransform(*current[2], Idx::Jz, 0, IntVect(1));

    // Loop over boxes
    for (amrex::MFIter mfi(field_data.fields); mfi.isValid(); ++mfi) {

        const amrex::Box& bx = field_data.fields[mfi].box();

        // Extract arrays for the fields to be updated
        amrex::Array4<Complex> fields = field_data.fields[mfi].array();

        // Extract pointers for the modified k vectors
        const amrex::Real* const modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const amrex::Real* const modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* const modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Loop over indices within one box
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            using Idx = SpectralFieldIndex;

            // Shortcuts for the values of D
            const Complex Dx = fields(i,j,k,Idx::Jx);
            const Complex Dy = fields(i,j,k,Idx::Jy);
            const Complex Dz = fields(i,j,k,Idx::Jz);

            // Imaginary unit
            constexpr Complex I = Complex{0._rt, 1._rt};

            // Modified k vector values
            const amrex::Real kx_mod = modified_kx_arr[i];
#if (AMREX_SPACEDIM==3)
            const amrex::Real ky_mod = modified_ky_arr[j];
            const amrex::Real kz_mod = modified_kz_arr[k];
#else
            constexpr amrex::Real ky_mod = 0._rt;
            const     amrex::Real kz_mod = modified_kz_arr[j];
#endif

            // Compute Jx
            if (kx_mod != 0._rt) fields(i,j,k,Idx::Jx) = I * Dx / kx_mod;
            else                 fields(i,j,k,Idx::Jx) = 0._rt;

#if (AMREX_SPACEDIM==3)
            // Compute Jy
            if (ky_mod != 0._rt) fields(i,j,k,Idx::Jy) = I * Dy / ky_mod;
            else                 fields(i,j,k,Idx::Jy) = 0._rt;
#endif

            // Compute Jz
            if (kz_mod != 0._rt) fields(i,j,k,Idx::Jz) = I * Dz / kz_mod;
            else                 fields(i,j,k,Idx::Jz) = 0._rt;

        });
    }

    // Backward Fourier transform of J
    field_data.BackwardTransform(*current[0], Idx::Jx, 0);
    field_data.BackwardTransform(*current[1], Idx::Jy, 0);
    field_data.BackwardTransform(*current[2], Idx::Jz, 0);
}
#endif // WARPX_USE_PSATD
