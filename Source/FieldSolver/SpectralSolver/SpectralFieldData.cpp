/* Copyright 2019 Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralFieldData.H"

#include <map>

#if WARPX_USE_PSATD

using namespace amrex;

#if WARPX_USE_SPIRAL
//extern "C"
//{
//  void init_warpxforward_only_80();
//  void warpxforward_only_80(double  *Yptr, double  *Xptr);
//
//  void init_warpxbackward_only_80();
//  void warpxbackward_only_80(double  *Yptr, double  *Xptr);
//}
#include "warpx-forward-only_80.c"
#include "warpx-backward-only_80.c"
#include "warpx-forward-step_80.c"
#include "warpx-backward-step_80.c"
#include "warpx-scale-forward_80.c"
#include "warpx-scale-backward_80.c"
#endif

/* \brief Initialize fields in spectral space, and FFT plans */
SpectralFieldData::SpectralFieldData( const amrex::BoxArray& realspace_ba,
                                      const SpectralKSpace& k_space,
                                      const amrex::DistributionMapping& dm,
                                      const int n_field_required,
                                      const bool periodic_single_box )
{
    m_periodic_single_box = periodic_single_box;

    const BoxArray& spectralspace_ba = k_space.spectralspace_ba;

    // Allocate the arrays that contain the fields in spectral space
    // (one component per field)
    fields = SpectralField(spectralspace_ba, dm, n_field_required, 0);
    
    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    tmpRealField = MultiFab(realspace_ba, dm, 1, 0);
    tmpSpectralField = SpectralField(spectralspace_ba, dm, 1, 0);
#if WARPX_USE_SPIRAL
    // Output of allForwardTransform: 11 complex components
    std::cout << "SpectralFieldData constructor n_field_required="
              << n_field_required << std::endl;
    fieldsForward = SpectralField(spectralspace_ba, dm, n_field_required, 0);
    // Input of allBackwardTransform: 6 complex components
    fieldsBackward = SpectralField(spectralspace_ba, dm, 2*AMREX_SPACEDIM, 0);
    // Need this because the backward FFT overwrites input tmpSpectralField.
    // This is for just one component.
    copySpectralField = SpectralField(spectralspace_ba, dm, 1, 0);
    // Result of Spiral's 3D R2C FFT.
    spiralFieldHatForward = SpectralField(spectralspace_ba, dm, n_field_required, 0);
    // tmpSpectralField as it enters BackwardTransform.
    copySpectralFieldBackward = SpectralField(spectralspace_ba, dm, 2*AMREX_SPACEDIM, 0);
#endif

    // By default, we assume the FFT is done from/to a nodal grid in real space
    // It the FFT is performed from/to a cell-centered grid in real space,
    // a correcting "shift" factor must be applied in spectral space.
    xshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformFromCellCentered);
    xshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformToCellCentered);
#if (AMREX_SPACEDIM == 3)
    yshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    yshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);
    zshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformFromCellCentered);
    zshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformToCellCentered);
#else
    zshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    zshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);
#endif

    // Allocate and initialize the FFT plans
    forward_plan = AnyFFT::FFTplans(spectralspace_ba, dm);
    backward_plan = AnyFFT::FFTplans(spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        IntVect fft_size = realspace_ba[mfi].length();

        std::cout << "SpectralFieldData constructor defining plans on box " << tmpRealField[mfi].box() << " size " << fft_size << std::endl;
        forward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::R2C, AMREX_SPACEDIM);

        backward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::C2R, AMREX_SPACEDIM);
    }
}


SpectralFieldData::~SpectralFieldData()
{
    if (tmpRealField.size() > 0){
        for ( MFIter mfi(tmpRealField); mfi.isValid(); ++mfi ){
            AnyFFT::DestroyPlan(forward_plan[mfi]);
            AnyFFT::DestroyPlan(backward_plan[mfi]);
        }
    }
}

#if WARPX_USE_SPIRAL
void
SpectralFieldData::allForwardTransform (
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield,
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& current,
    std::unique_ptr<amrex::MultiFab>& rho )
{
  // Loop over boxes
  for ( MFIter mfi(*rho); mfi.isValid(); ++mfi ){
    // These are inputs.
    BaseFab<Real>& ExFab = (*Efield[0])[mfi];
    BaseFab<Real>& EyFab = (*Efield[1])[mfi];
    BaseFab<Real>& EzFab = (*Efield[2])[mfi];
    BaseFab<Real>& BxFab = (*Bfield[0])[mfi];
    BaseFab<Real>& ByFab = (*Bfield[1])[mfi];
    BaseFab<Real>& BzFab = (*Bfield[2])[mfi];
    BaseFab<Real>& JxFab = (*current[0])[mfi];
    BaseFab<Real>& JyFab = (*current[1])[mfi];
    BaseFab<Real>& JzFab = (*current[2])[mfi];
    BaseFab<Real>& rhoFab = (*rho)[mfi];
    BaseFab<Real> rhoOldFab(rhoFab, amrex::make_alias, 0, 1); // start, count
    BaseFab<Real> rhoNewFab(rhoFab, amrex::make_alias, 1, 1); // start, count

    std::cout << "allForwardTransform Ex on " << ExFab.box() << std::endl;
    std::cout << "allForwardTransform Ey on " << EyFab.box() << std::endl;
    std::cout << "allForwardTransform Ez on " << EzFab.box() << std::endl;
    std::cout << "allForwardTransform Bx on " << BxFab.box() << std::endl;
    std::cout << "allForwardTransform By on " << ByFab.box() << std::endl;
    std::cout << "allForwardTransform Bz on " << BzFab.box() << std::endl;
    std::cout << "allForwardTransform Jx on " << JxFab.box() << std::endl;
    std::cout << "allForwardTransform Jy on " << JyFab.box() << std::endl;
    std::cout << "allForwardTransform Jz on " << JzFab.box() << std::endl;
    std::cout << "allForwardTransform rhoOld on " << rhoOldFab.box() << std::endl;
    std::cout << "allForwardTransform rhoNew on " << rhoNewFab.box() << std::endl;
    
    double** inPtr = new double*[11];
    // If these FABs are all set to constants, then Spiral gives same answer.
    inPtr[SpectralFieldIndex::Ex] = ExFab.dataPtr();
    inPtr[SpectralFieldIndex::Ey] = EyFab.dataPtr();
    inPtr[SpectralFieldIndex::Ez] = EzFab.dataPtr();
    inPtr[SpectralFieldIndex::Bx] = BxFab.dataPtr();
    inPtr[SpectralFieldIndex::By] = ByFab.dataPtr();
    inPtr[SpectralFieldIndex::Bz] = BzFab.dataPtr();
    inPtr[SpectralFieldIndex::Jx] = JxFab.dataPtr();
    inPtr[SpectralFieldIndex::Jy] = JyFab.dataPtr();
    inPtr[SpectralFieldIndex::Jz] = JzFab.dataPtr();
    inPtr[SpectralFieldIndex::rho_old] = rhoOldFab.dataPtr();
    inPtr[SpectralFieldIndex::rho_new] = rhoNewFab.dataPtr();

    {
      Array4<Real> rhoNewArray = rhoNewFab.array();
      FILE* fp_spiral = fopen("spiralin_10.out", "w");
      const Dim3 realLo = amrex::lbound(rhoNewFab.box());
      const Dim3 realHi = amrex::ubound(rhoNewFab.box());
      for (int k = realLo.z; k <= realHi.z - 1; ++k) // NOTE -1: prune
        for (int j = realLo.y; j <= realHi.y - 1; ++j) // NOTE -1: prune
          for (int i = realLo.x; i <= realHi.x - 1; ++i) // NOTE -1: prune
            {
              Real spiral_val = rhoNewArray(i, j, k);
              fprintf(fp_spiral, " %3d%3d%3d%20.9e\n", i, j, k,
                      spiral_val);
            }
      fclose(fp_spiral);
    }
    
    // FIXME
    // SpectralFieldData has
    // void ForwardTransform (const amrex::MultiFab& mf, const int field_index,
    //                        const int i_comp, const amrex::IntVect& stag);
    // does
    // ForwardTransform(mf, field_index, i_comp, mf.ixType().toIntVect());
    // But I do not need stag.
    // field_index is used only in the final copy into field_arr.

    double* outPtr = (double*) (fieldsForward[mfi].dataPtr());

    init_warpxforward_step_80();
    std::cout << "SpectralFieldData::allForwardTransform calling warpxforward_step_80" << std::endl;
    warpxforward_step_80(outPtr, inPtr);
    destroy_warpxforward_step_80();

    delete[] inPtr;

    {
      amrex::BaseFab<Complex>& fieldsForwardFab = fieldsForward[mfi];
      // first comp, count
      BaseFab<Complex> rhoNewHatFab(fieldsForwardFab, amrex::make_alias, 10, 1);
      Array4<Complex> rhoNewHatArray = rhoNewHatFab.array();
      FILE* fp_spiral = fopen("spiralhat_10.out", "w");
      const Dim3 spectralLo = amrex::lbound(rhoNewHatFab.box());
      const Dim3 spectralHi = amrex::ubound(rhoNewHatFab.box());
      for (int k = spectralLo.z; k <= spectralHi.z; ++k)
        for (int j = spectralLo.y; j <= spectralHi.y; ++j)
          for (int i = spectralLo.x; i <= spectralHi.x; ++i)
            {
              Complex spiral_val = rhoNewHatArray(i, j, k);
              fprintf(fp_spiral, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                      spiral_val.real(), spiral_val.imag());
            }
      fclose(fp_spiral);
    }
    
  }
}

void
SpectralFieldData::allBackwardTransform (
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
    std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield)
{
  // Loop over boxes
  for ( MFIter mfi(*Efield[0]); mfi.isValid(); ++mfi ){
    // These are outputs.
    BaseFab<Real>& ExFab = (*Efield[0])[mfi];
    BaseFab<Real>& EyFab = (*Efield[1])[mfi];
    BaseFab<Real>& EzFab = (*Efield[2])[mfi];
    BaseFab<Real>& BxFab = (*Bfield[0])[mfi];
    BaseFab<Real>& ByFab = (*Bfield[1])[mfi];
    BaseFab<Real>& BzFab = (*Bfield[2])[mfi];

    std::cout << "allBackwardTransform Ex on " << ExFab.box() << std::endl;
    std::cout << "allBackwardTransform Ey on " << EyFab.box() << std::endl;
    std::cout << "allBackwardTransform Ez on " << EzFab.box() << std::endl;
    std::cout << "allBackwardTransform Bx on " << BxFab.box() << std::endl;
    std::cout << "allBackwardTransform By on " << ByFab.box() << std::endl;
    std::cout << "allBackwardTransform Bz on " << BzFab.box() << std::endl;
    
    double** outPtr = new double*[6];
    outPtr[SpectralFieldIndex::Ex] = ExFab.dataPtr();
    outPtr[SpectralFieldIndex::Ey] = EyFab.dataPtr();
    outPtr[SpectralFieldIndex::Ez] = EzFab.dataPtr();
    outPtr[SpectralFieldIndex::Bx] = BxFab.dataPtr();
    outPtr[SpectralFieldIndex::By] = ByFab.dataPtr();
    outPtr[SpectralFieldIndex::Bz] = BzFab.dataPtr();

    // double* inPtr = (double*) (fieldsBackward[mfi].dataPtr());
    double* inPtr = (double*) (copySpectralFieldBackward[mfi].dataPtr());

    init_warpxbackward_step_80();
    std::cout << "SpectralFieldData::allBackwardTransform calling warpxbackward_step_80" << std::endl;
    warpxbackward_step_80(outPtr, inPtr);
    destroy_warpxbackward_step_80();
    
    delete[] outPtr;
  }
}

void
SpectralFieldData::scaleSpiralForward()
{
  for ( MFIter mfi(spiralFieldHatForward); mfi.isValid(); ++mfi ){
    // spiralFieldHatForward is the result of Spiral's 3D R2C FFT.
    BaseFab<Complex>& spiralFieldHatForwardFab =
      spiralFieldHatForward[mfi];
    const Box& bx = spiralFieldHatForwardFab.box();
    BaseFab<Complex> outSpectralForwardFab(bx, 11);
    double** sym = new double*[3];
    sym[0] = (double*) xshift_FFTfromCell[mfi].dataPtr();
    sym[1] = (double*) yshift_FFTfromCell[mfi].dataPtr();
    sym[2] = (double*) zshift_FFTfromCell[mfi].dataPtr();
    std::cout << "Scaling forward x:" << xshift_FFTfromCell[mfi].size()
              << " y:" << yshift_FFTfromCell[mfi].size()
              << " z:" << zshift_FFTfromCell[mfi].size()
              << std::endl;
    init_warpxscale_forward_80();
    warpxscale_forward_80((double*) outSpectralForwardFab.dataPtr(),
                          (double*) spiralFieldHatForwardFab.dataPtr(),
                          sym);
    destroy_warpxscale_forward_80();
    delete[] sym;

    // Now compare outSpectralForwardFab with SpectralFieldData::fields[mfi].
    BaseFab<Complex>& spectralFieldsFab = SpectralFieldData::fields[mfi];
    int ncomps = spectralFieldsFab.nComp();
    for (int field_index = 0; field_index < ncomps; field_index++)
      {
        BaseFab<Complex> amrexFab(spectralFieldsFab,
                                  amrex::make_alias, field_index, 1);
        BaseFab<Complex> spiralFab(outSpectralForwardFab,
                                   amrex::make_alias, field_index, 1);
        Array4<Complex> amrex_arr = amrexFab.array();
        Array4<Complex> spiral_arr = spiralFab.array();
        BaseFab<Complex> diffFab(bx, 1);
        Array4<Complex> diffArray = diffFab.array();
        // components: source, dest, count
        diffFab.copy(amrexFab, 0, 0, 1);
        diffFab.minus(spiralFab, 0, 0, 1);
        Real amrex_max = 0.;
        Real spiral_max = 0.;
        Real diff_max = 0.;
        Real amrex2_sum = 0.;
        Real spiral2_sum = 0.;
        IntVect amrex_biggest = IntVect(0, 0, 0);
        IntVect spiral_biggest = IntVect(0, 0, 0);
        const Dim3 spectralLo = amrex::lbound(bx);
        const Dim3 spectralHi = amrex::ubound(bx);
        for (int k = spectralLo.z; k <= spectralHi.z; ++k)
          for (int j = spectralLo.y; j <= spectralHi.y; ++j)
            for (int i = spectralLo.x; i <= spectralHi.x; ++i)
              {
                Complex amrex_val = amrex_arr(i, j, k);
                Complex spiral_val = spiral_arr(i, j, k);
                Real amrex_abs = abs(amrex_val);
                Real spiral_abs = abs(spiral_val);
                amrex2_sum += amrex_abs * amrex_abs;
                spiral2_sum += spiral_abs * spiral_abs;
                if (amrex_abs > amrex_max)
                  {
                    amrex_max = amrex_abs;
                    amrex_biggest = IntVect(i, j, k);
                  }
                if (spiral_abs > spiral_max)
                  {
                    spiral_max = spiral_abs;
                    spiral_biggest = IntVect(i, j, k);
                  }
                Real diff_abs = abs(diffArray(i, j, k));
                if (diff_abs > diff_max)
                  {
                    diff_max = diff_abs;
                  }
              }
        std::cout << "Scaling forward [" << field_index << "] "
                  << " |diff| <= " << diff_max
                  << " |solution| <= " << amrex_max
                  << " relative " << (diff_max/amrex_max)
                  << std::endl;
        
      }
  }
}

void
SpectralFieldData::scaleSpiralBackward()
{
  std::cout << "In scaleSpiralBackward()" << std::endl;
  for ( MFIter mfi(copySpectralFieldBackward); mfi.isValid(); ++mfi ){
    BaseFab<Complex>& copySpectralFieldBackwardFab =
      copySpectralFieldBackward[mfi];
    const Box& bx = copySpectralFieldBackwardFab.box();
    BaseFab<Complex> outSpectralBackwardFab(bx, 2*AMREX_SPACEDIM);
    double** sym = new double*[3];
    sym[0] = (double*) xshift_FFTtoCell[mfi].dataPtr();
    sym[1] = (double*) yshift_FFTtoCell[mfi].dataPtr();
    sym[2] = (double*) zshift_FFTtoCell[mfi].dataPtr();
    std::cout << "Scaling backward x:" << xshift_FFTtoCell[mfi].size()
              << " y:" << yshift_FFTtoCell[mfi].size()
              << " z:" << zshift_FFTtoCell[mfi].size()
              << std::endl;
    init_warpxscale_backward_80();
    warpxscale_backward_80((double*) outSpectralBackwardFab.dataPtr(),
                           (double*) copySpectralFieldBackwardFab.dataPtr(),
                           sym);
    destroy_warpxscale_backward_80();
    delete[] sym;

    // Now compare outSpectralBackwardFab with fieldsBackward[mfi].
    BaseFab<Complex>& fieldsBackwardFab = fieldsBackward[mfi];
    for (int field_index = 0; field_index < 2*AMREX_SPACEDIM; field_index++)
      {
        // alias:  start, count
        BaseFab<Complex> amrexFab(fieldsBackwardFab,
                                  amrex::make_alias, field_index, 1);
        BaseFab<Complex> spiralFab(outSpectralBackwardFab,
                                   amrex::make_alias, field_index, 1);
        Array4<Complex> amrex_arr = amrexFab.array();
        Array4<Complex> spiral_arr = spiralFab.array();
        BaseFab<Complex> diffFab(bx, 1);
        Array4<Complex> diffArray = diffFab.array();
        // components: source, dest, count
        diffFab.copy(amrexFab, 0, 0, 1);
        diffFab.minus(spiralFab, 0, 0, 1);
        Real amrex_max = 0.;
        Real spiral_max = 0.;
        Real diff_max = 0.;
        Real amrex2_sum = 0.;
        Real spiral2_sum = 0.;
        IntVect amrex_biggest = IntVect(0, 0, 0);
        IntVect spiral_biggest = IntVect(0, 0, 0);
        const Dim3 spectralLo = amrex::lbound(bx);
        const Dim3 spectralHi = amrex::ubound(bx);
        for (int k = spectralLo.z; k <= spectralHi.z; ++k)
          for (int j = spectralLo.y; j <= spectralHi.y; ++j)
            for (int i = spectralLo.x; i <= spectralHi.x; ++i)
              {
                Complex amrex_val = amrex_arr(i, j, k);
                Complex spiral_val = spiral_arr(i, j, k);
                Real amrex_abs = abs(amrex_val);
                Real spiral_abs = abs(spiral_val);
                amrex2_sum += amrex_abs * amrex_abs;
                spiral2_sum += spiral_abs * spiral_abs;
                if (amrex_abs > amrex_max)
                  {
                    amrex_max = amrex_abs;
                    amrex_biggest = IntVect(i, j, k);
                  }
                if (spiral_abs > spiral_max)
                  {
                    spiral_max = spiral_abs;
                    spiral_biggest = IntVect(i, j, k);
                  }
                Real diff_abs = abs(diffArray(i, j, k));
                if (diff_abs > diff_max)
                  {
                    diff_max = diff_abs;
                  }
              }
        /*
        std::cout << "Scaling backward [" << field_index
                  << "] sum(|amrex|^2) = " << amrex2_sum
                  << " sum(|spiral|^2) = " << spiral2_sum
                  << " ratio " << (spiral2_sum/amrex2_sum)
                  << std::endl;
        */
        std::cout << "Scaling backward [" << field_index << "]"
                  << " |diff| <= " << diff_max
                  << " |solution| <= " << amrex_max
                  << " relative " << (diff_max/amrex_max)
                  << std::endl;
        /*
        std::cout << "Location of biggest"
                  << " amrex " << amrex_biggest
                  << " spiral " << spiral_biggest
                  << " within " << bx
                  << std::endl;
        */
      }
  }
}

void
SpectralFieldData::compareSpiralForwardStep()
{
  // Compare fields vs. fieldsForward.
  for ( MFIter mfi(fieldsForward); mfi.isValid(); ++mfi ){
    amrex::BaseFab<Complex>& fieldsForwardFab = fieldsForward[mfi];
    amrex::BaseFab<Complex>& fieldsFab = fields[mfi];
    const Box& spectralBox = fieldsFab.box();
    int ncomps = fieldsForwardFab.nComp();
    for (int field_index = 0; field_index < ncomps; field_index++)
      {
        //        std::cout << "compareSpiralForwardStep on component "
        //                  << field_index << std::endl;
        //        char amrexstr[30];
        //        sprintf(amrexstr, "fwdamrex_%d.out", field_index);
        //        char spiralstr[30];
        //        sprintf(spiralstr, "fwdspiral_%d.out", field_index);

        //        FILE* fp_amrex = fopen(amrexstr, "w");
        //        FILE* fp_spiral = fopen(spiralstr, "w");

        BaseFab<Complex> diffFab(spectralBox, 1);
        // first comp, count
        BaseFab<Complex> spiralFab(fieldsForwardFab,
                                   amrex::make_alias, field_index, 1);
        BaseFab<Complex> amrexFab(fieldsFab,
                                  amrex::make_alias, field_index, 1);
        Array4<Complex> spiralArray = spiralFab.array();
        Array4<Complex> amrexArray = amrexFab.array();
        Array4<Complex> diffArray = diffFab.array();
        // components: source, dest, count
        diffFab.copy(fieldsFab, field_index, 0, 1);
        diffFab.minus(fieldsForwardFab, field_index, 0, 1);
        Real amrex_max = 0.;
        Real spiral_max = 0.;
        Real diff_max = 0.;
        Real amrex2_sum = 0.;
        Real spiral2_sum = 0.;
        IntVect amrex_biggest = IntVect(0, 0, 0);
        IntVect spiral_biggest = IntVect(0, 0, 0);
        const Dim3 spectralLo = amrex::lbound(spectralBox);
        const Dim3 spectralHi = amrex::ubound(spectralBox);
        for (int k = spectralLo.z; k <= spectralHi.z; ++k)
          for (int j = spectralLo.y; j <= spectralHi.y; ++j)
            for (int i = spectralLo.x; i <= spectralHi.x; ++i)
              {
                Complex amrex_val = amrexArray(i, j, k);
                Complex spiral_val = spiralArray(i, j, k);
                // fprintf(fp_amrex, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                // amrex_val.real(), amrex_val.imag());
                // fprintf(fp_spiral, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                // spiral_val.real(), spiral_val.imag());
                Real amrex_abs = abs(amrex_val);
                Real spiral_abs = abs(spiral_val);
                amrex2_sum += amrex_abs * amrex_abs;
                spiral2_sum += spiral_abs * spiral_abs;
                if (amrex_abs > amrex_max)
                  {
                    amrex_max = amrex_abs;
                    amrex_biggest = IntVect(i, j, k);
                  }
                if (spiral_abs > spiral_max)
                  {
                    spiral_max = spiral_abs;
                    spiral_biggest = IntVect(i, j, k);
                  }
                Real diff_abs = abs(diffArray(i, j, k));
                if (diff_abs > diff_max)
                  {
                    diff_max = diff_abs;
                  }
              }
        // fclose(fp_amrex);
        // fclose(fp_spiral);
        std::cout << "Step Forward [" << field_index
                  << "] 3DFFT sum(|amrex|^2) = " << amrex2_sum
                  << " sum(|spiral|^2) = " << spiral2_sum
                  << " ratio " << (spiral2_sum/amrex2_sum)
                  << std::endl;
        std::cout << "Step Forward [" << field_index
                  << "] 3DFFT |diff| <= " << diff_max
                  << " |solution| <= " << amrex_max
                  << " relative " << (diff_max/amrex_max)
                  << std::endl;
        std::cout << "Location of biggest"
                  << " amrex " << amrex_biggest
                  << " spiral " << spiral_biggest
                  << " within " << spectralBox
                  << std::endl;
      }
  }
}

void
SpectralFieldData::compareSpiralBackwardStep(
                                             std::array<std::unique_ptr< amrex::MultiFab >, 3>& EfieldBack,
                                             std::array<std::unique_ptr< amrex::MultiFab >, 3>& BfieldBack,
                                             std::array<std::unique_ptr< amrex::MultiFab >, 3>& Efield,
                                             std::array<std::unique_ptr< amrex::MultiFab >, 3>& Bfield)
                                             
{
  Real finalScaling = 1.0 / (80.0 * 80.0 * 80.0);
  for (int idir = 0; idir < 3; idir++)
    {
      std::unique_ptr< amrex::MultiFab >& EcompBack = EfieldBack[idir];
      std::unique_ptr< amrex::MultiFab >& Ecomp = Efield[idir];
      for ( MFIter mfi(*Ecomp); mfi.isValid(); ++mfi ){
        // compare EcompBack (from Spiral) and Ecomp (from WarpX)
        std::cout << "compareSpiralBackwardStep on E" << idir << std::endl;
        BaseFab<Real>& EcompBackFab = (*EcompBack)[mfi];
        EcompBackFab.mult(finalScaling); // FIXME
        BaseFab<Real>& EcompFab = (*Ecomp)[mfi];
        // char amrexstr[30];
        // sprintf(amrexstr, "backamrexE%d.out", idir);
        // char spiralstr[30];
        // sprintf(spiralstr, "backspiralE%d.out", idir);
        const Box& bx = mfi.validbox();
        FArrayBox diffFab(bx, 1);
        // components: source, dest, count
        diffFab.copy(EcompFab, 0, 0, 1);
        diffFab.minus(EcompBackFab, 0, 0, 1);
        Array4<Real> amrexArray = EcompFab.array();
        Array4<Real> spiralArray = EcompBackFab.array();
        Array4<Real> diffArray = diffFab.array();
        Real amrex_max = 0.;
        Real spiral_max = 0.;
        Real diff_max = 0.;
        Real amrex2_sum = 0.;
        Real spiral2_sum = 0.;
        IntVect amrex_biggest = IntVect(0, 0, 0);
        IntVect spiral_biggest = IntVect(0, 0, 0);
        const Dim3 bxLo = amrex::lbound(bx);
        const Dim3 bxHi = amrex::ubound(bx);          
        // FILE* fp_amrex = fopen(amrexstr, "w");
        // FILE* fp_spiral = fopen(spiralstr, "w");
        for (int k = bxLo.z; k <= bxHi.z; ++k)
          for (int j = bxLo.y; j <= bxHi.y; ++j)
            for (int i = bxLo.x; i <= bxHi.x; ++i)
              {
                Real amrex_val = amrexArray(i, j, k);
                Real spiral_val = spiralArray(i, j, k);
                // fprintf(fp_amrex, " %3d%3d%3d%20.9e\n", i, j, k, amrex_val);
                // fprintf(fp_spiral, " %3d%3d%3d%20.9e\n", i, j, k, spiral_val);
                Real amrex_abs = amrex::Math::abs(amrex_val);
                Real spiral_abs = amrex::Math::abs(spiral_val);
                amrex2_sum += amrex_abs * amrex_abs;
                spiral2_sum += spiral_abs * spiral_abs;
                if (amrex_abs > amrex_max)
                  {
                    amrex_max = amrex_abs;
                    amrex_biggest = IntVect(i, j, k);
                  }
                if (spiral_abs > spiral_max)
                  {
                    spiral_max = spiral_abs;
                    spiral_biggest = IntVect(i, j, k);
                  }
                Real diff_abs = amrex::Math::abs(diffArray(i, j, k));
                if (diff_abs > diff_max)
                  {
                    diff_max = diff_abs;
                  }
              }
        // fclose(fp_amrex);
        // fclose(fp_spiral);
        std::cout << "Step Backward 3DFFT sum(|E" << idir << "|^2) = " << amrex2_sum
                  << " sum(|spiral|^2) = " << spiral2_sum
                  << " ratio " << (spiral2_sum/amrex2_sum)
                  << std::endl;
        std::cout << "Step Backward 3DFFT |diff(E" << idir << ")| <= " << diff_max
                  << " |solution| <= " << amrex_max
                  << " relative " << (diff_max/amrex_max)
                  << std::endl;
        std::cout << "Location of biggest"
                  << " amrex " << amrex_biggest
                  << " spiral " << spiral_biggest
                  << " within " << bx
                  << std::endl;
      }
    }

  for (int idir = 0; idir < 3; idir++)
    {
      std::unique_ptr< amrex::MultiFab >& Bcomp = Bfield[idir];
      std::unique_ptr< amrex::MultiFab >& BcompBack = BfieldBack[idir];
      for ( MFIter mfi(*Bcomp); mfi.isValid(); ++mfi ){
        // compare BcompBack (from Spiral) and Bcomp (from WarpX)
        std::cout << "compareSpiralBackwardStep on B" << idir << std::endl;
        BaseFab<Real>& BcompBackFab = (*BcompBack)[mfi];
        BcompBackFab.mult(finalScaling); // FIXME
        BaseFab<Real>& BcompFab = (*Bcomp)[mfi];
        // char amrexstr[30];
        // sprintf(amrexstr, "backamrexB%d.out", idir);
        // char spiralstr[30];
        // sprintf(spiralstr, "backspiralB%d.out", idir);
        const Box& bx = mfi.validbox();
        FArrayBox diffFab(bx, 1);
        // components: source, dest, count
        diffFab.copy(BcompFab, 0, 0, 1);
        diffFab.minus(BcompBackFab, 0, 0, 1);
        Array4<Real> amrexArray = BcompFab.array();
        Array4<Real> spiralArray = BcompBackFab.array();
        Array4<Real> diffArray = diffFab.array();
        Real amrex_max = 0.;
        Real spiral_max = 0.;
        Real diff_max = 0.;
        Real amrex2_sum = 0.;
        Real spiral2_sum = 0.;
        IntVect amrex_biggest = IntVect(0, 0, 0);
        IntVect spiral_biggest = IntVect(0, 0, 0);
        const Dim3 bxLo = amrex::lbound(bx);
        const Dim3 bxHi = amrex::ubound(bx);          
        // FILE* fp_amrex = fopen(amrexstr, "w");
        // FILE* fp_spiral = fopen(spiralstr, "w");
        for (int k = bxLo.z; k <= bxHi.z; ++k)
          for (int j = bxLo.y; j <= bxHi.y; ++j)
            for (int i = bxLo.x; i <= bxHi.x; ++i)
              {
                Real amrex_val = amrexArray(i, j, k);
                Real spiral_val = spiralArray(i, j, k);
                // fprintf(fp_amrex, " %3d%3d%3d%20.9e\n", i, j, k, amrex_val);
                // fprintf(fp_spiral, " %3d%3d%3d%20.9e\n", i, j, k, spiral_val);
                Real amrex_abs = amrex::Math::abs(amrex_val);
                Real spiral_abs = amrex::Math::abs(spiral_val);
                amrex2_sum += amrex_abs * amrex_abs;
                spiral2_sum += spiral_abs * spiral_abs;
                if (amrex_abs > amrex_max)
                  {
                    amrex_max = amrex_abs;
                    amrex_biggest = IntVect(i, j, k);
                  }
                if (spiral_abs > spiral_max)
                  {
                    spiral_max = spiral_abs;
                    spiral_biggest = IntVect(i, j, k);
                  }
                Real diff_abs = amrex::Math::abs(diffArray(i, j, k));
                if (diff_abs > diff_max)
                  {
                    diff_max = diff_abs;
                  }
              }
        // fclose(fp_amrex);
        // fclose(fp_spiral);
        std::cout << "Step Backward 3DFFT sum(|B" << idir << "|^2) = " << amrex2_sum
                  << " sum(|spiral|^2) = " << spiral2_sum
                  << " ratio " << (spiral2_sum/amrex2_sum)
                  << std::endl;
        std::cout << "Step Backward 3DFFT |diff(B" << idir << ")| <= " << diff_max
                  << " |solution| <= " << amrex_max
                  << " relative " << (diff_max/amrex_max)
                  << std::endl;
        std::cout << "Location of biggest"
                  << " amrex " << amrex_biggest
                  << " spiral " << spiral_biggest
                  << " within " << bx
                  << std::endl;
      }
    }
}

#endif

/* \brief Transform the component `i_comp` of MultiFab `mf`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral field specified by `field_index`) */
void
SpectralFieldData::ForwardTransform (const MultiFab& mf, const int field_index,
                                     const int i_comp, const IntVect& stag)
{
    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_x = (stag[0] == amrex::IndexType::NODE) ? true : false;
#if (AMREX_SPACEDIM == 3)
    const bool is_nodal_y = (stag[1] == amrex::IndexType::NODE) ? true : false;
    const bool is_nodal_z = (stag[2] == amrex::IndexType::NODE) ? true : false;
#else
    const bool is_nodal_z = (stag[1] == amrex::IndexType::NODE) ? true : false;
#endif

    // Loop over boxes
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){

        // Copy the real-space field `mf` to the temporary field `tmpRealField`
        // This ensures that all fields have the same number of points
        // before the Fourier transform.
        // As a consequence, the copy discards the *last* point of `mf`
        // in any direction that has *nodal* index type.
        {
            Box realspace_bx;
            if (m_periodic_single_box) {
                realspace_bx = mfi.validbox(); // Discard guard cells
            } else {
                realspace_bx = mf[mfi].box(); // Keep guard cells
            }
            realspace_bx.enclosedCells(); // Discard last point in nodal direction
            // added by petermc
            std::cout << "SpectralField::ForwardTransform[" << i_comp
                      << "] filling tmpRealField[mfi].box()="
                      << tmpRealField[mfi].box()
                      << std::endl;
            AMREX_ALWAYS_ASSERT( realspace_bx.contains(tmpRealField[mfi].box()) );
            Array4<const Real> mf_arr = mf[mfi].array();
            Array4<Real> tmp_arr = tmpRealField[mfi].array();
            ParallelFor( tmpRealField[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_arr(i,j,k) = mf_arr(i,j,k,i_comp);
            });
#if WARPX_USE_SPIRAL
            /*
            if (field_index == SpectralFieldIndex::rho_new)
              {
                FILE* fp_amrex = fopen("amrexin_10.out", "w");
                const Dim3 realLo = amrex::lbound(realspace_bx);
                const Dim3 realHi = amrex::ubound(realspace_bx);
                for (int k = realLo.z; k <= realHi.z; ++k)
                  for (int j = realLo.y; j <= realHi.y; ++j)
                    for (int i = realLo.x; i <= realHi.x; ++i)
                      {
                        Real amrex_val = tmp_arr(i, j, k);
                        fprintf(fp_amrex, " %3d%3d%3d%20.9e\n", i, j, k,
                                amrex_val);
                      }
                fclose(fp_amrex);
              }
            */
#endif            
        }

#if WARPX_USE_SPIRAL
        std::cout << "is_nodal_x=" << is_nodal_x 
                  << " is_nodal_y=" << is_nodal_y
                  << " is_nodal_z=" << is_nodal_z
                  << std::endl;
        // result of calling forward transform in Spiral.
        // Array4<Complex> spiralSpectralField(realspace_bx, ...);
#endif
        // Perform Fourier transform from `tmpRealField` to `tmpSpectralField`
        AnyFFT::Execute(forward_plan[mfi]);

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // index of the FabArray `fields` (specified by `field_index`)
        // and apply correcting shift factor if the real space data comes
        // from a cell-centered grid in real space instead of a nodal grid.
        {
            Array4<Complex> fields_arr = SpectralFieldData::fields[mfi].array();
            Array4<const Complex> tmp_arr = tmpSpectralField[mfi].array();
#if WARPX_USE_SPIRAL
            { // Compare tmpSpectralField with Spiral FFT on tmpRealField.
              const BaseFab<Real>& inputRealFab = tmpRealField[mfi];
              const Box& inputBox = inputRealFab.box();
              std::cout << "inputBox = " << inputBox
                        << std::endl;
              BaseFab<Complex>& tmpSpectralFab = tmpSpectralField[mfi];
              const Box& spectralBox = tmpSpectralFab.box();
              std::cout << "spectralBox = " << spectralBox
                        << std::endl;
              // Set spiralFieldHatForward[mfi] component field_index to
              // Spiral's 3D R2C FFT on tmpRealField[mfi] component field_index.
              BaseFab<Complex>& spiralFieldHatForwardFab =
                spiralFieldHatForward[mfi];
              BaseFab<Complex> spectralFab(spiralFieldHatForwardFab,
                                           amrex::make_alias, field_index, 1);
              Array4<Complex> spectralArray = spectralFab.array();
              // Call forward 3D FFT on inputRealFab.
              init_warpxforward_only_80();
              std::cout << "SpectralFieldData::ForwardTransform calling warpxforward_only_80, field_index=" << field_index << ", i_comp=" << i_comp << std::endl;
              warpxforward_only_80((double*) spectralFab.dataPtr(),
                                   (double*) inputRealFab.dataPtr());
              destroy_warpxforward_only_80();
              // Spiral-generated code returns an array of real-imaginary pairs.
              // Now compare spectralArray with tmpSpectralFab.
              BaseFab<Complex> diffFab(spectralBox, 1);
              Array4<Complex> diffArray = diffFab.array();
              // components: source, dest, count
              diffFab.copy(tmpSpectralFab, 0, 0, 1);
              diffFab.minus(spectralFab, 0, 0, 1);
              /*
                std::cout << "At (10, 11, 12), "
                << "AMReX gives " << tmp_arr(10, 11, 12)
                << " and Spiral " << spectralArray(10, 11, 12)
                << " ratio " << (tmp_arr(10, 11, 12)/spectralArray(10, 11, 12))
                << std::endl;
              */
              Real amrex_max = 0.;
              Real spiral_max = 0.;
              Real diff_max = 0.;
              Real amrex2_sum = 0.;
              Real spiral2_sum = 0.;
              IntVect amrex_biggest = IntVect(0, 0, 0);
              IntVect spiral_biggest = IntVect(0, 0, 0);
              /*
              FILE *fp_amrex, *fp_spiral;
              if (field_index == SpectralFieldIndex::rho_new)
                {
                  fp_amrex = fopen("onlyfwdamrex_10.out", "w");
                  fp_spiral = fopen("onlyfwdspiral_10.out", "w");
                }
              */
              const Dim3 spectralLo = amrex::lbound(spectralBox);
              const Dim3 spectralHi = amrex::ubound(spectralBox);
              for (int k = spectralLo.z; k <= spectralHi.z; ++k)
                for (int j = spectralLo.y; j <= spectralHi.y; ++j)
                  for (int i = spectralLo.x; i <= spectralHi.x; ++i)
                    {
                      Complex amrex_val = tmp_arr(i, j, k);
                      Complex spiral_val = spectralArray(i, j, k);
                      /*
                      if (field_index == SpectralFieldIndex::rho_new)
                        {
                          fprintf(fp_amrex, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                                  amrex_val.real(), amrex_val.imag());
                          fprintf(fp_spiral, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                                  spiral_val.real(), spiral_val.imag());
                        }
                      */
                      Real amrex_abs = abs(amrex_val);
                      Real spiral_abs = abs(spiral_val);
                      amrex2_sum += amrex_abs * amrex_abs;
                      spiral2_sum += spiral_abs * spiral_abs;
                      if (amrex_abs > amrex_max)
                        {
                          amrex_max = amrex_abs;
                          amrex_biggest = IntVect(i, j, k);
                        }
                      if (spiral_abs > spiral_max)
                        {
                          spiral_max = spiral_abs;
                          spiral_biggest = IntVect(i, j, k);
                        }
                      Real diff_abs = abs(diffArray(i, j, k));
                      if (diff_abs > diff_max)
                        {
                          diff_max = diff_abs;
                        }
                    }
              /*
                std::cout << "Forward 3DFFT sum(|amrex|^2) = " << amrex2_sum
                << " sum(|spiral|^2) = " << spiral2_sum
                << " ratio " << (spiral2_sum/amrex2_sum)
                << std::endl;
              */
              std::cout << "Only Forward 3DFFT |diff| <= " << diff_max
                        << " |solution| <= " << amrex_max
                        << " relative " << (diff_max/amrex_max)
                        << std::endl;
              /*
                std::cout << "Location of biggest"
                << " amrex " << amrex_biggest
                << " spiral " << spiral_biggest
                << " within " << spectralBox
                << std::endl;
              */
              /*
              if (field_index == SpectralFieldIndex::rho_new)
                {
                  fclose(fp_amrex);
                  fclose(fp_spiral);
                }
              */
            }
#endif
            const Complex* xshift_arr = xshift_FFTfromCell[mfi].dataPtr();
#if (AMREX_SPACEDIM == 3)
            const Complex* yshift_arr = yshift_FFTfromCell[mfi].dataPtr();
#endif
            const Complex* zshift_arr = zshift_FFTfromCell[mfi].dataPtr();
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

#if WARPX_USE_SPIRAL
            /*
            {
              FILE* fp1 = fopen("x_from.out", "w");
              for (int i = spectralLo.x; i <= spectralHi.x; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          xshift_arr[i].real(), xshift_arr[i].imag());
                }
              fclose(fp1);
            }
            {
              FILE* fp1 = fopen("y_from.out", "w");
              for (int i = spectralLo.y; i <= spectralHi.y; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          yshift_arr[i].real(), yshift_arr[i].imag());
                }
              fclose(fp1);
            }
            {
              FILE* fp1 = fopen("z_from.out", "w");
              for (int i = spectralLo.z; i <= spectralHi.z; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          zshift_arr[i].real(), zshift_arr[i].imag());
                }
              fclose(fp1);
            }
            */
#endif
            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = tmp_arr(i,j,k);
                // Apply proper shift in each dimension
                if (is_nodal_x==false) spectral_field_value *= xshift_arr[i];
#if (AMREX_SPACEDIM == 3)
                if (is_nodal_y==false) spectral_field_value *= yshift_arr[j];
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[k];
#elif (AMREX_SPACEDIM == 2)
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[j];
#endif
                // Copy field into the right index
                fields_arr(i,j,k,field_index) = spectral_field_value;
            });
        }
    }
}


void
SpectralFieldData::setCopySpectralFieldBackward()
{
  // Loop over boxes
  for ( MFIter mfi(copySpectralFieldBackward); mfi.isValid(); ++mfi ){
    copySpectralFieldBackward[mfi].copy(SpectralFieldData::fields[mfi]);
  }
}


/* \brief Transform spectral field specified by `field_index` back to
 * real space, and store it in the component `i_comp` of `mf` */
void
SpectralFieldData::BackwardTransform( MultiFab& mf,
                                      const int field_index,
                                      const int i_comp )
{
    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_x = mf.is_nodal(0);
#if (AMREX_SPACEDIM == 3)
    const bool is_nodal_y = mf.is_nodal(1);
    const bool is_nodal_z = mf.is_nodal(2);
#else
    const bool is_nodal_z = mf.is_nodal(1);
#endif

    // Loop over boxes
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // field (specified by the input argument field_index)
        // and apply correcting shift factor if the field is to be transformed
        // to a cell-centered grid in real space instead of a nodal grid.
        {
            Array4<const Complex> field_arr = SpectralFieldData::fields[mfi].array();
            Array4<Complex> tmp_arr = tmpSpectralField[mfi].array();
            const Complex* xshift_arr = xshift_FFTtoCell[mfi].dataPtr();
#if (AMREX_SPACEDIM == 3)
            const Complex* yshift_arr = yshift_FFTtoCell[mfi].dataPtr();
#endif
            const Complex* zshift_arr = zshift_FFTtoCell[mfi].dataPtr();
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

#if WARPX_USE_SPIRAL
            /*
            const Dim3 spectralLo = amrex::lbound(spectralspace_bx);
            const Dim3 spectralHi = amrex::ubound(spectralspace_bx);
            {
              FILE* fp1 = fopen("x_to.out", "w");
              for (int i = spectralLo.x; i <= spectralHi.x; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          xshift_arr[i].real(), xshift_arr[i].imag());
                }
              fclose(fp1);
            }
            {
              FILE* fp1 = fopen("y_to.out", "w");
              for (int i = spectralLo.y; i <= spectralHi.y; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          yshift_arr[i].real(), yshift_arr[i].imag());
                }
              fclose(fp1);
            }
            {
              FILE* fp1 = fopen("z_to.out", "w");
              for (int i = spectralLo.z; i <= spectralHi.z; ++i)
                {
                  fprintf(fp1, " %3d%20.9e%20.9e\n", i,
                          zshift_arr[i].real(), zshift_arr[i].imag());
                }
              fclose(fp1);
            }
            */
#endif
            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = field_arr(i,j,k,field_index);
                // Apply proper shift in each dimension
                if (is_nodal_x==false) spectral_field_value *= xshift_arr[i];
#if (AMREX_SPACEDIM == 3)
                if (is_nodal_y==false) spectral_field_value *= yshift_arr[j];
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[k];
#elif (AMREX_SPACEDIM == 2)
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[j];
#endif
                // Copy field into temporary array
                tmp_arr(i,j,k) = spectral_field_value;
            });
#if WARPX_USE_SPIRAL
            // Array4<Complex> tmp_arr = tmpSpectralField[mfi].array();
            
            // copySpectralField is for single component of backward transform.
            // fieldsBackward is for all 6 components of backward transform.
            BaseFab<Complex>& copySpectralFieldFab = copySpectralField[mfi];
            copySpectralFieldFab.copy(tmpSpectralField[mfi]);
            BaseFab<Complex>& fieldsBackwardFab = fieldsBackward[mfi];
            // std::cout << "fieldsBackward[" << field_index
            // << "] = tmpSpectralField[" << 0 << "]" << std::endl;
            // components: source, dest, count
            fieldsBackwardFab.copy(tmpSpectralField[mfi], 0, field_index, 1);
            /*
            const Dim3 inputLo = amrex::lbound(spectralspace_bx);
            const Dim3 inputHi = amrex::ubound(spectralspace_bx);
            FILE* fp_input = fopen("amrex_in.out", "w");
            for (int k = inputLo.z; k <= inputHi.z; ++k)
              for (int j = inputLo.y; j <= inputHi.y; ++j)
                for (int i = inputLo.x; i <= inputHi.x; ++i)
                  {
                    // tmp_arr(i, j, k) = i * 1.; // FIXME
                    fprintf(fp_input, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                            tmp_arr(i, j, k).real(),
                            tmp_arr(i, j, k).imag());
                  }
            fclose(fp_input);
            */
#endif
        }

#if WARPX_USE_SPIRAL
        std::cout << "is_nodal_x=" << is_nodal_x 
                  << " is_nodal_y=" << is_nodal_y
                  << " is_nodal_z=" << is_nodal_z
                  << std::endl;
#endif
        // Perform Fourier transform from `tmpSpectralField` to `tmpRealField`
        AnyFFT::Execute(backward_plan[mfi]);

        // Copy the temporary field `tmpRealField` to the real-space field `mf`
        // (only in the valid cells ; not in the guard cells)
        // Normalize (divide by 1/N) since the FFT+IFFT results in a factor N
        {
            Array4<Real> mf_arr = mf[mfi].array();
            Array4<const Real> tmp_arr = tmpRealField[mfi].array();
            // Normalization: divide by the number of points in realspace
            // (includes the guard cells)
            const Box realspace_bx = tmpRealField[mfi].box();
            const Real inv_N = 1./realspace_bx.numPts();

            // added by petermc
            std::cout << "SpectralFieldData::BackwardTransform"
                      << " realspace_bx = " << realspace_bx
                      << " mf[mfi].box() = " << mf[mfi].box()
                      << std::endl;
#if WARPX_USE_SPIRAL
            const BaseFab<Complex>& inputSpectralFab = copySpectralField[mfi];
            const Box& spectralBox = inputSpectralFab.box();
            std::cout << "spectralBox = " << spectralBox
                      << std::endl;
            const BaseFab<Real>& outputRealFab = tmpRealField[mfi];
            const Box& outputBox = outputRealFab.box();
            std::cout << "outputBox = " << outputBox
                      << std::endl;
            BaseFab<Real> realFab(outputBox, 1);
            Array4<Real> realArray = realFab.array();
            /*
            const Dim3 inputLo = amrex::lbound(spectralBox);
            const Dim3 inputHi = amrex::ubound(spectralBox);
            Array4<const Complex> inputSpectralArray = inputSpectralFab.array();
            FILE* fp_input = fopen("spiral_in.out", "w");
            for (int k = inputLo.z; k <= inputHi.z; ++k)
              for (int j = inputLo.y; j <= inputHi.y; ++j)
                for (int i = inputLo.x; i <= inputHi.x; ++i)
                  {
                    // tmp_arr(i, j, k) = i * 1.; // FIXME
                    fprintf(fp_input, " %3d%3d%3d%20.9e%20.9e\n", i, j, k,
                            inputSpectralArray(i, j, k).real(),
                            inputSpectralArray(i, j, k).imag());
                  }
            fclose(fp_input);
            exit(0);
            */
            
            // Call backward 3D FFT on inputSpectralFab.
            init_warpxbackward_only_80();
            warpxbackward_only_80((double*) realFab.dataPtr(),
                                  (double*) inputSpectralFab.dataPtr());
            destroy_warpxbackward_only_80();
            // realFab.mult(inv_N, 0);
            BaseFab<Real> diffFab(outputBox, 1);
            Array4<Real> diffArray = diffFab.array();
            // components: source, dest, count
            diffFab.copy(outputRealFab, 0, 0, 1);
            diffFab.minus(realFab, 0, 0, 1);
            /*
            std::cout << "At (10, 11, 12), "
                      << "AMReX gives " << tmp_arr(10, 11, 12)
                      << " and Spiral " << realArray(10, 11, 12)
                      << " ratio " << (tmp_arr(10, 11, 12)/realArray(10, 11, 12))
                      << std::endl;
            */
            Real amrex_max = 0.;
            Real spiral_max = 0.;
            Real diff_max = 0.;
            Real amrex2_sum = 0.;
            Real spiral2_sum = 0.;
            IntVect amrex_biggest = IntVect(0, 0, 0);
            IntVect spiral_biggest = IntVect(0, 0, 0);
            const Dim3 outputLo = amrex::lbound(outputBox);
            const Dim3 outputHi = amrex::ubound(outputBox);
            // FILE* fp_amrex = fopen("amrex.out", "w");
            // FILE* fp_spiral = fopen("spiral.out", "w");
            for (int k = outputLo.z; k <= outputHi.z; ++k)
              for (int j = outputLo.y; j <= outputHi.y; ++j)
                for (int i = outputLo.x; i <= outputHi.x; ++i)
                  {
                    Real amrex_val = tmp_arr(i, j, k);
                    Real spiral_val = realArray(i, j, k);
                    // fprintf(fp_amrex, " %3d%3d%3d%20.9e\n", i, j, k,
                    // amrex_val);
                    // fprintf(fp_spiral, " %3d%3d%3d%20.9e\n", i, j, k,
                    // spiral_val);
                    Real amrex_abs = amrex::Math::abs(amrex_val);
                    Real spiral_abs = amrex::Math::abs(spiral_val);
                    amrex2_sum += amrex_abs * amrex_abs;
                    spiral2_sum += spiral_abs * spiral_abs;
                    if (amrex_abs > amrex_max)
                      {
                        amrex_max = amrex_abs;
                        amrex_biggest = IntVect(i, j, k);
                      }
                    if (spiral_abs > spiral_max)
                      {
                        spiral_max = spiral_abs;
                        spiral_biggest = IntVect(i, j, k);
                      }
                    Real diff_abs = amrex::Math::abs(diffArray(i, j, k));
                    if (diff_abs > diff_max)
                      {
                        diff_max = diff_abs;
                      }
                  }
            // fclose(fp_amrex);
            // fclose(fp_spiral);
            /*
            std::cout << "Backward 3DFFT sum(|amrex|^2) = " << amrex2_sum
                      << " sum(|spiral|^2) = " << spiral2_sum
                      << " ratio " << (spiral2_sum/amrex2_sum)
                      << std::endl;
            */
            std::cout << "Only Backward 3DFFT |diff| <= " << diff_max
                      << " |solution| <= " << amrex_max
                      << " relative " << (diff_max/amrex_max)
                      << std::endl;
            /*
            std::cout << "Location of biggest"
                      << " amrex " << amrex_biggest
                      << " spiral " << spiral_biggest
                      << " within " << outputBox
                      << std::endl;
            */
            // exit(0);
            std::cout << "SpectralField::BackwardTransform[" << i_comp << "]"
                      << " filling mfi.validbox()=" << mfi.validbox()
                      << " m_periodic_single_box=" << m_periodic_single_box
                      << std::endl;
#endif
            if (m_periodic_single_box) {
                // Enforce periodicity on the nodes, by using modulo in indices
                // This is because `tmp_arr` is cell-centered while `mf_arr` can be nodal
                int const nx = realspace_bx.length(0);
                int const ny = realspace_bx.length(1);
#if (AMREX_SPACEDIM == 3)
                int const nz = realspace_bx.length(2);
#else
                int constexpr nz = 1;
#endif
                ParallelFor(
                    mfi.validbox(),
                    /* GCC 8.1-8.2 work-around (ICE):
                     *   named capture in nonexcept lambda needed for modulo operands
                     *   https://godbolt.org/z/ppbAzd
                     */
                    [mf_arr, i_comp, inv_N, tmp_arr, nx, ny, nz]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        mf_arr(i,j,k,i_comp) = inv_N*tmp_arr(i%nx, j%ny, k%nz);
                    });
            } else {
                ParallelFor( mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    // Copy and normalize field
                    mf_arr(i,j,k,i_comp) = inv_N*tmp_arr(i,j,k);
                });
            }

        }
    }
}

#endif // WARPX_USE_PSATD
