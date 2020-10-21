/* Copyright 2019 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include "BoundaryConditions/WarpX_PML_kernels.H"
#include "BoundaryConditions/PML_current.H"
#include "WarpX_FDTD.H"
#ifdef WARPX_USE_PY
#   include "Python/WarpX_py.H"
#endif

#ifdef BL_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif

#include <AMReX_Math.H>
#include <limits>


using namespace amrex;

#ifdef WARPX_USE_PSATD
namespace {
    void
    PushPSATDSinglePatch (
#ifdef WARPX_DIM_RZ
        SpectralSolverRZ& solver,
#else
        SpectralSolver& solver,
#endif
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield_avg,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield_avg,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& current,
        std::unique_ptr<amrex::MultiFab>& rho ) {

        using Idx = SpectralAvgFieldIndex;

#if WARPX_USE_FULL_SPIRAL
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > EfieldNew;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > BfieldNew;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > diffE;
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > diffB;
        
        for (int idir = 0; idir < 3; idir++)
          {
            EfieldNew[idir].reset( new MultiFab(Efield[idir]->boxArray(),
                                                Efield[idir]->DistributionMap(),
                                                Efield[idir]->nComp(),
                                                Efield[idir]->nGrow()) );
            BfieldNew[idir].reset( new MultiFab(Bfield[idir]->boxArray(),
                                                Bfield[idir]->DistributionMap(),
                                                Bfield[idir]->nComp(),
                                                Bfield[idir]->nGrow()) );
            diffE[idir].reset( new MultiFab(Efield[idir]->boxArray(),
                                                Efield[idir]->DistributionMap(),
                                                Efield[idir]->nComp(),
                                            -4 ));
            diffB[idir].reset( new MultiFab(Bfield[idir]->boxArray(),
                                                Bfield[idir]->DistributionMap(),
                                                Bfield[idir]->nComp(),
                                            -4 ));
          }
        solver.stepSpiral(EfieldNew, BfieldNew,
                          Efield, Bfield, Efield_avg, Bfield_avg, current, rho);
#endif
        // Perform forward Fourier transform
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(*Efield[0], Idx::Ex,
                                *Efield[1], Idx::Ey);
#else
        solver.ForwardTransform(*Efield[0], Idx::Ex);
        solver.ForwardTransform(*Efield[1], Idx::Ey);
#endif
        solver.ForwardTransform(*Efield[2], Idx::Ez);
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(*Bfield[0], Idx::Bx,
                                *Bfield[1], Idx::By);
#else
        solver.ForwardTransform(*Bfield[0], Idx::Bx);
        solver.ForwardTransform(*Bfield[1], Idx::By);
#endif
        solver.ForwardTransform(*Bfield[2], Idx::Bz);
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(*current[0], Idx::Jx,
                                *current[1], Idx::Jy);
#else
        solver.ForwardTransform(*current[0], Idx::Jx);
        solver.ForwardTransform(*current[1], Idx::Jy);
#endif
        solver.ForwardTransform(*current[2], Idx::Jz);
        solver.ForwardTransform(*rho, Idx::rho_old, 0);
        solver.ForwardTransform(*rho, Idx::rho_new, 1);
#ifdef WARPX_DIM_RZ
        if (WarpX::use_kspace_filter) {
            solver.ApplyFilter(Idx::rho_old);
            solver.ApplyFilter(Idx::rho_new);
            solver.ApplyFilter(Idx::Jx, Idx::Jy, Idx::Jz);
        }
#endif

        // Advance fields in spectral space
        solver.pushSpectralFields();

        // Perform backward Fourier Transform
#ifdef WARPX_DIM_RZ
        solver.BackwardTransform(*Efield[0], Idx::Ex,
                                 *Efield[1], Idx::Ey);
#else
        solver.BackwardTransform(*Efield[0], Idx::Ex);
        solver.BackwardTransform(*Efield[1], Idx::Ey);
#endif
        solver.BackwardTransform(*Efield[2], Idx::Ez);
#ifdef WARPX_DIM_RZ
        solver.BackwardTransform(*Bfield[0], Idx::Bx,
                                 *Bfield[1], Idx::By);
#else
        solver.BackwardTransform(*Bfield[0], Idx::Bx);
        solver.BackwardTransform(*Bfield[1], Idx::By);
#endif
        solver.BackwardTransform(*Bfield[2], Idx::Bz);

#ifndef WARPX_DIM_RZ
        if (WarpX::fft_do_time_averaging){
            solver.BackwardTransform(*Efield_avg[0], Idx::Ex_avg);
            solver.BackwardTransform(*Efield_avg[1], Idx::Ey_avg);
            solver.BackwardTransform(*Efield_avg[2], Idx::Ez_avg);

            solver.BackwardTransform(*Bfield_avg[0], Idx::Bx_avg);
            solver.BackwardTransform(*Bfield_avg[1], Idx::By_avg);
            solver.BackwardTransform(*Bfield_avg[2], Idx::Bz_avg);
        }
#endif


#if WARPX_USE_FULL_SPIRAL
        /*
        //auto G = WarpX::GetInstance().Geom(0);
        //Geometry G = Geometry(Box({0,0,0},{63,64,64}),RealBox({-1,-1,-1},{1,1,1}),
        //                      CoordSys::cartesian, Array<int,3>({0,0,0}));
        for(int i=0; i< 3; i++)
          {
            MultiFab::Copy(*diffE[i], *EfieldNew[i],0,0,1,0);
            diffE[i]->minus(*Efield[i],0,1,0);
            double normE = diffE[i]->norminf(0,0);
            std::cout<<"E"<<i<<" Old Norm: "<<Efield[i]->norminf(0,0)<<"\n";
            std::cout<<"E"<<i<<" FFTX Norm: "<<EfieldNew[i]->norminf(0,0)<<"\n";
            auto& FAB1 = Efield[i]->get(0);
            auto& FAB2 = EfieldNew[i]->get(0);
            auto& FAB3 = diffE[i]->get(0);
            IntVect m = FAB1.maxIndex();
            std::cout<<"maxIndex:"<<m<<" Warp="<<FAB1(m)<<" FFTX="
                     <<FAB2(m)<<" diff="<<FAB3(m)
                     <<"\n";
                       std::cout<<"E"<<i<<" norm Warp vs FFTX = "<<normE<<"  maxIndex: "<<FAB3.maxIndex() <<" val: "<<FAB3(FAB3.maxIndex()) <<"\n";
          }
 
        static int istep = 0;
        // WriteSingleLevelPlotfile(amrex::Concatenate("pltEx",istep),
        //                          *diffE[0], {{"Ex"}}, G, istep, istep);

        istep++;
        */
        // Find differences between Efield and EfieldNew.
        Real amrexE2max = 0.;
        Real spiralE2max = 0.;
        Real diffE2max = 0.;
        for (int idir = 0; idir < 3; idir++)
          {
            std::unique_ptr< amrex::MultiFab >& EcompNew = EfieldNew[idir];
            std::unique_ptr< amrex::MultiFab >& Ecomp = Efield[idir];
            for ( MFIter mfi(*Ecomp); mfi.isValid(); ++mfi ){
              // compare EcompNew (from Spiral) and Ecomp (from WarpX)
              BaseFab<Real>& EcompNewFab = (*EcompNew)[mfi];
              BaseFab<Real>& EcompFab = (*Ecomp)[mfi];
              const Box& bx = mfi.validbox();
              FArrayBox diffFab(bx, 1);
              // components: source, dest, count
              diffFab.copy(EcompFab, 0, 0, 1);
              diffFab.minus(EcompNewFab, 0, 0, 1);
              // 0 for max norm; CPU will ignore RunOn::Device
              Real amrex_max = EcompFab.norm<RunOn::Device>(bx, 0);
              Real spiral_max = EcompNewFab.norm<RunOn::Device>(bx, 0); 
              Real diff_max = diffFab.norm<RunOn::Device>(bx, 0);
              amrexE2max += amrex_max * amrex_max;
              spiralE2max += spiral_max * spiral_max;
              diffE2max += diff_max * diff_max;
              std::cout << "Full step |diff(E" << idir << ")| <= "
                        << diff_max
                        << " |solution| <= " << amrex_max
                        << " relative " << (diff_max/amrex_max)
                        << std::endl;
            }
          }
        
        // Find differences between Bfield and BfieldNew.
        Real amrexB2max = 0.;
        Real spiralB2max = 0.;
        Real diffB2max = 0.;
        for (int idir = 0; idir < 3; idir++)
          {
            std::unique_ptr< amrex::MultiFab >& Bcomp = Bfield[idir];
            std::unique_ptr< amrex::MultiFab >& BcompNew = BfieldNew[idir];
            for ( MFIter mfi(*Bcomp); mfi.isValid(); ++mfi ){
              // compare BcompNew (from Spiral) and Bcomp (from WarpX)
              BaseFab<Real>& BcompNewFab = (*BcompNew)[mfi];
              BaseFab<Real>& BcompFab = (*Bcomp)[mfi];
              const Box& bx = mfi.validbox();
              FArrayBox diffFab(bx, 1);
              // components: source, dest, count
              diffFab.copy(BcompFab, 0, 0, 1);
              diffFab.minus(BcompNewFab, 0, 0, 1);
              // 0 for max norm; CPU will ignore RunOn::Device
              Real amrex_max = BcompFab.norm<RunOn::Device>(bx, 0);
              Real spiral_max = BcompNewFab.norm<RunOn::Device>(bx, 0); 
              Real diff_max = diffFab.norm<RunOn::Device>(bx, 0);
              amrexB2max += amrex_max * amrex_max;
              spiralB2max += spiral_max * spiral_max;
              diffB2max += diff_max * diff_max;
              std::cout << "Full step |diff(B" << idir << ")| <= "
                        << diff_max
                        << " |solution| <= " << amrex_max
                        << " relative " << (diff_max/amrex_max)
                        << std::endl;
            }
          }

        Real diffEmax = std::sqrt(diffE2max);
        Real diffBmax = std::sqrt(diffB2max);
        Real Emax = std::sqrt(amrexE2max);
        Real Bmax = std::sqrt(amrexB2max);
        Real const c2 = PhysConst::c * PhysConst::c;
        Real compositeEmax = std::sqrt(amrexE2max + c2*amrexB2max);
        Real compositeBmax = std::sqrt(amrexE2max/c2 + amrexB2max);

        std::cout << "Full Step ||diff(E)|| <= " << diffEmax
                  << " of ||E|| <= " << Emax
                  << " relative " << (diffEmax / Emax)
                  << std::endl;
        std::cout << "Full Step ||diff(E)|| <= " << diffEmax
                  << " of sqrt(||E||^2 + c^2*||B||^2) <= " << compositeEmax
                  << " relative " << (diffEmax / compositeEmax)
                  << std::endl;

        std::cout << "Full Step ||diff(B)|| <= " << diffBmax
                  << " of ||B|| <= " << Bmax
                  << " relative " << (diffBmax / Bmax)
                  << std::endl;
        std::cout << "Full Step ||diff(B)|| <= " << diffBmax
                  << " of sqrt(||E||^2/c^2 + ||B||^2) <= " << compositeBmax
                  << " relative " << (diffBmax / compositeBmax)
                  << std::endl;
#endif
    }
}

void
WarpX::PushPSATD (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt[lev] == a_dt, "dt must be consistent");
        PushPSATD(lev, a_dt);

        // Evolve the fields in the PML boxes
        if (do_pml && pml[lev]->ok()) {
            pml[lev]->PushPSATD();
        }
    }
}

void
WarpX::PushPSATD (int lev, amrex::Real /* dt */)
{
    // Update the fields on the fine and coarse patch
  std::cout << "WarpX::PushPSATD calling PushPSATDSinglePatch on fp" << std::endl;
    PushPSATDSinglePatch( *spectral_solver_fp[lev],
        Efield_fp[lev], Bfield_fp[lev], Efield_avg_fp[lev], Bfield_avg_fp[lev], current_fp[lev], rho_fp[lev] );
  std::cout << "WarpX::PushPSATD called PushPSATDSinglePatch on fp" << std::endl;
    if (spectral_solver_cp[lev]) {
      std::cout << "WarpX::PushPSATD calling PushPSATDSinglePatch on cp" << std::endl;
        PushPSATDSinglePatch( *spectral_solver_cp[lev],
             Efield_cp[lev], Bfield_cp[lev], Efield_avg_cp[lev], Bfield_avg_cp[lev], current_cp[lev], rho_cp[lev] );
      std::cout << "WarpX::PushPSATD called PushPSATDSinglePatch on cp" << std::endl;
    }
}
#endif

void
WarpX::EvolveB (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveB(lev, a_dt);
    }
}

void
WarpX::EvolveB (int lev, amrex::Real a_dt)
{
    WARPX_PROFILE("WarpX::EvolveB()");
    EvolveB(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        EvolveB(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::EvolveB (int lev, PatchType patch_type, amrex::Real a_dt)
{

    // Evolve B field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveB( Bfield_fp[lev], Efield_fp[lev], a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveB( Bfield_cp[lev], Efield_cp[lev], a_dt );
    }

    // Evolve B field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveBPML(
                pml[lev]->GetB_fp(), pml[lev]->GetE_fp(), a_dt );
        } else {
            m_fdtd_solver_cp[lev]->EvolveBPML(
                pml[lev]->GetB_cp(), pml[lev]->GetE_cp(), a_dt );
        }
    }

}

void
WarpX::EvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveE(lev, a_dt);
    }
}

void
WarpX::EvolveE (int lev, amrex::Real a_dt)
{
    WARPX_PROFILE("WarpX::EvolveE()");
    EvolveE(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        EvolveE(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::EvolveE (int lev, PatchType patch_type, amrex::Real a_dt)
{
    // Evolve E field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveE( Efield_fp[lev], Bfield_fp[lev],
                                      current_fp[lev], F_fp[lev], a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveE( Efield_cp[lev], Bfield_cp[lev],
                                      current_cp[lev], F_cp[lev], a_dt );
    }

    // Evolve E field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveEPML(
                pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
                pml[lev]->Getj_fp(), pml[lev]->GetF_fp(),
                pml[lev]->GetMultiSigmaBox_fp(),
                a_dt, pml_has_particles );
        } else {
            m_fdtd_solver_cp[lev]->EvolveEPML(
                pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
                pml[lev]->Getj_cp(), pml[lev]->GetF_cp(),
                pml[lev]->GetMultiSigmaBox_cp(),
                a_dt, pml_has_particles );
        }
    }
}


void
WarpX::EvolveF (amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveF(lev, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveF (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    EvolveF(lev, PatchType::fine, a_dt, a_dt_type);
    if (lev > 0) EvolveF(lev, PatchType::coarse, a_dt, a_dt_type);
}

void
WarpX::EvolveF (int lev, PatchType patch_type, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    WARPX_PROFILE("WarpX::EvolveF()");

    const int rhocomp = (a_dt_type == DtType::FirstHalf) ? 0 : 1;

    // Evolve F field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveF( F_fp[lev], Efield_fp[lev],
                                        rho_fp[lev], rhocomp, a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveF( F_cp[lev], Efield_cp[lev],
                                        rho_cp[lev], rhocomp, a_dt );
    }

    // Evolve F field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveFPML(
                pml[lev]->GetF_fp(), pml[lev]->GetE_fp(), a_dt );
        } else {
            m_fdtd_solver_cp[lev]->EvolveFPML(
                pml[lev]->GetF_cp(), pml[lev]->GetE_cp(), a_dt );
        }
    }

}

void
WarpX::MacroscopicEvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev ) {
        MacroscopicEvolveE(lev, a_dt);
    }
}

void
WarpX::MacroscopicEvolveE (int lev, amrex::Real a_dt) {

    WARPX_PROFILE("WarpX::MacroscopicEvolveE()");
    MacroscopicEvolveE(lev, PatchType::fine, a_dt);
    if (lev > 0) {
        amrex::Abort("Macroscopic EvolveE is not implemented for lev>0, yet.");
    }
}

void
WarpX::MacroscopicEvolveE (int lev, PatchType patch_type, amrex::Real a_dt) {
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->MacroscopicEvolveE( Efield_fp[lev], Bfield_fp[lev],
                                             current_fp[lev], a_dt,
                                             m_macroscopic_properties);
    }
    else {
        amrex::Abort("Macroscopic EvolveE is not implemented for lev > 0, yet.");
    }
    if (do_pml) {
        amrex::Abort("Macroscopic EvolveE is not implemented for pml boundary condition, yet");
    }
}

#ifdef WARPX_DIM_RZ
// This scales the current by the inverse volume and wraps around the depostion at negative radius.
// It is faster to apply this on the grid than to do it particle by particle.
// It is put here since there isn't another nice place for it.
void
WarpX::ApplyInverseVolumeScalingToCurrentDensity (MultiFab* Jx, MultiFab* Jy, MultiFab* Jz, int lev)
{
    const long ngJ = Jx->nGrow();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Jx->ixType().toIntVect()[0] != NODE,
        "Jr should never node-centered in r");


    for ( MFIter mfi(*Jx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Jr_arr = Jx->array(mfi);
        Array4<Real> const& Jt_arr = Jy->array(mfi);
        Array4<Real> const& Jz_arr = Jz->array(mfi);

        Box const & tilebox = mfi.tilebox();
        Box tbr = convert( tilebox, Jx->ixType().toIntVect() );
        Box tbt = convert( tilebox, Jy->ixType().toIntVect() );
        Box tbz = convert( tilebox, Jz->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        std::array<amrex::Real,3> galilean_shift = {0,0,0};
        const std::array<Real, 3>& xyzmin = WarpX::LowerCorner(tilebox, galilean_shift, lev);
        const Real rmin  = xyzmin[0];
        const Real rminr = xyzmin[0] + (tbr.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Real rmint = xyzmin[0] + (tbt.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Real rminz = xyzmin[0] + (tbz.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Dim3 lo = lbound(tilebox);
        const int irmin = lo.x;

        // For ishift, 1 means cell centered, 0 means node centered
        int const ishift_t = (rmint > rmin ? 1 : 0);
        int const ishift_z = (rminz > rmin ? 1 : 0);

        const long nmodes = n_rz_azimuthal_modes;

        // Grow the tileboxes to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0.) {
           tbr.growLo(0, ngJ);
           tbt.growLo(0, ngJ);
           tbz.growLo(0, ngJ);
        }
        tbr.growHi(0, ngJ);
        tbt.growHi(0, ngJ);
        tbz.growHi(0, ngJ);
        tbr.grow(1, ngJ);
        tbt.grow(1, ngJ);
        tbz.grow(1, ngJ);

        // Rescale current in r-z mode since the inverse volume factor was not
        // included in the current deposition.
        amrex::ParallelFor(tbr, tbt, tbz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // Note that Jr(i==0) is at 1/2 dr.
            if (rmin == 0. && 0 <= i && i < ngJ) {
                Jr_arr(i,j,0,0) -= Jr_arr(-1-i,j,0,0);
            }
            // Apply the inverse volume scaling
            // Since Jr is never node centered in r, no need for distinction
            // between on axis and off-axis factors
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            Jr_arr(i,j,0,0) /= (2.*MathConst::pi*r);

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                // Note that Jr(i==0) is at 1/2 dr.
                if (rmin == 0. && 0 <= i && i < ngJ) {
                    Jr_arr(i,j,0,2*imode-1) -= Jr_arr(-1-i,j,0,2*imode-1);
                    Jr_arr(i,j,0,2*imode) -= Jr_arr(-1-i,j,0,2*imode);
                }
                // Apply the inverse volume scaling
                // Since Jr is never node centered in r, no need for distinction
                // between on axis and off-axis factors
                Jr_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                Jr_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jt is node centered, Jt[0] is located on the boundary.
            // If Jt is cell centered, Jt[0] is at 1/2 dr.
            if (rmin == 0. && 1-ishift_t <= i && i <= ngJ-ishift_t) {
                Jt_arr(i,j,0,0) -= Jt_arr(-ishift_t-i,j,0,0);
            }

            // Apply the inverse volume scaling
            // Jt is forced to zero on axis.
            const amrex::Real r = amrex::Math::abs(rmint + (i - irmin)*dr);
            if (r == 0.) {
                Jt_arr(i,j,0,0) = 0.;
            } else {
                Jt_arr(i,j,0,0) /= (2.*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0. && 1-ishift_t <= i && i <= ngJ-ishift_t) {
                    Jt_arr(i,j,0,2*imode-1) -= Jt_arr(-ishift_t-i,j,0,2*imode-1);
                    Jt_arr(i,j,0,2*imode) -= Jt_arr(-ishift_t-i,j,0,2*imode);
                }

                // Apply the inverse volume scaling
                // Jt is forced to zero on axis.
                if (r == 0.) {
                    Jt_arr(i,j,0,2*imode-1) = 0.;
                    Jt_arr(i,j,0,2*imode) = 0.;
                } else {
                    Jt_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                    Jt_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
                }
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jz is node centered, Jt[0] is located on the boundary.
            // If Jz is cell centered, Jt[0] is at 1/2 dr.
            if (rmin == 0. && 1-ishift_z <= i && i <= ngJ-ishift_z) {
                Jz_arr(i,j,0,0) -= Jz_arr(-ishift_z-i,j,0,0);
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminz + (i - irmin)*dr);
            if (r == 0.) {
                // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                Jz_arr(i,j,0,0) /= (MathConst::pi*dr/3.);
            } else {
                Jz_arr(i,j,0,0) /= (2.*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0. && 1-ishift_z <= i && i <= ngJ-ishift_z) {
                    Jz_arr(i,j,0,2*imode-1) -= Jz_arr(-ishift_z-i,j,0,2*imode-1);
                    Jz_arr(i,j,0,2*imode) -= Jz_arr(-ishift_z-i,j,0,2*imode);
                }

                // Apply the inverse volume scaling
                if (r == 0.) {
                    // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                    Jz_arr(i,j,0,2*imode-1) /= (MathConst::pi*dr/3.);
                    Jz_arr(i,j,0,2*imode) /= (MathConst::pi*dr/3.);
                } else {
                    Jz_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                    Jz_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
                }
            }

        });
    }
}

void
WarpX::ApplyInverseVolumeScalingToChargeDensity (MultiFab* Rho, int lev)
{
    const long ngRho = Rho->nGrow();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;

    Box tilebox;

    for ( MFIter mfi(*Rho, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Rho_arr = Rho->array(mfi);

        tilebox = mfi.tilebox();
        Box tb = convert( tilebox, Rho->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        std::array<amrex::Real,3> galilean_shift = {0,0,0};
        const std::array<Real, 3>& xyzmin = WarpX::LowerCorner(tilebox, galilean_shift, lev);
        const Dim3 lo = lbound(tilebox);
        const Real rmin = xyzmin[0];
        const Real rminr = xyzmin[0] + (tb.type(0) == NODE ? 0. : 0.5*dx[0]);
        const int irmin = lo.x;
        int ishift = (rminr > rmin ? 1 : 0);

        // Grow the tilebox to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0.) {
           tb.growLo(0, ngRho);
        }
        tb.growHi(0, ngRho);
        tb.grow(1, ngRho);

        // Rescale charge in r-z mode since the inverse volume factor was not
        // included in the charge deposition.
        // Note that the loop is also over ncomps, which takes care of the RZ modes,
        // as well as the old and new rho.
        amrex::ParallelFor(tb, Rho->nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
        {
            // Wrap the charge density deposited in the guard cells around
            // to the cells above the axis.
            // Rho is located on the boundary
            if (rmin == 0. && 1-ishift <= i && i <= ngRho-ishift) {
                Rho_arr(i,j,0,icomp) -= Rho_arr(-ishift-i,j,0,icomp);
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            if (r == 0.) {
                // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                Rho_arr(i,j,0,icomp) /= (MathConst::pi*dr/3.);
            } else {
                Rho_arr(i,j,0,icomp) /= (2.*MathConst::pi*r);
            }
        });
    }
}
#endif
