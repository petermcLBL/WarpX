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
#include "warpx-forward_only_80.c"
#include "warpx-backward_only_80.c"
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
    // Need this because the backward FFT overwrites input tmpSpectralField.
    copySpectralField = SpectralField(spectralspace_ba, dm, 1, 0);
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
            std::cout << "SpectralFieldData::ForwardTransform"
                      << " mf[mfi].box() = " << mf[mfi].box()
                      << " realspace_bx = " << realspace_bx
                      << std::endl;
            AMREX_ALWAYS_ASSERT( realspace_bx.contains(tmpRealField[mfi].box()) );
            Array4<const Real> mf_arr = mf[mfi].array();
            Array4<Real> tmp_arr = tmpRealField[mfi].array();
            ParallelFor( tmpRealField[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_arr(i,j,k) = mf_arr(i,j,k,i_comp);
            });
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
            const BaseFab<Real>& inputRealFab = tmpRealField[mfi];
            const Box& inputBox = inputRealFab.box();
            std::cout << "inputBox = " << inputBox
                      << std::endl;
            BaseFab<Complex>& tmpSpectralFab = tmpSpectralField[mfi];
            const Box& spectralBox = tmpSpectralFab.box();
            std::cout << "spectralBox = " << spectralBox
                      << std::endl;
            BaseFab<Complex> spectralFab(spectralBox, 1);
            Array4<Complex> spectralArray = spectralFab.array();
            // Call forward 3D FFT on inputRealFab.
            init_warpxforward_only_80();
            warpxforward_only_80((double*) spectralFab.dataPtr(),
                                 (double*) inputRealFab.dataPtr());
            // Spiral-generated code returns an array of real-imaginary pairs.
            // Now compare spectralArray with tmpSpectralFab.
            BaseFab<Complex> diffFab(spectralBox, 1);
            Array4<Complex> diffArray = diffFab.array();
            // components: source, dest, number
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
            const Dim3 spectralLo = amrex::lbound(spectralBox);
            const Dim3 spectralHi = amrex::ubound(spectralBox);
            for (int k = spectralLo.z; k <= spectralHi.z; ++k)
              for (int j = spectralLo.y; j <= spectralHi.y; ++j)
                for (int i = spectralLo.x; i <= spectralHi.x; ++i)
                  {
                    Complex amrex_val = tmp_arr(i, j, k);
                    Complex spiral_val = spectralArray(i, j, k);
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
                      << " ratio " << (amrex2_sum/spiral2_sum)
                      << std::endl;
            */
            std::cout << "Forward 3DFFT |diff| <= " << diff_max
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
#endif
            const Complex* xshift_arr = xshift_FFTfromCell[mfi].dataPtr();
#if (AMREX_SPACEDIM == 3)
            const Complex* yshift_arr = yshift_FFTfromCell[mfi].dataPtr();
#endif
            const Complex* zshift_arr = zshift_FFTfromCell[mfi].dataPtr();
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

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
            BaseFab<Complex>& copySpectralFieldFab = copySpectralField[mfi];
            copySpectralFieldFab.copy(tmpSpectralField[mfi]);
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
            // realFab.mult(inv_N, 0);
            BaseFab<Real> diffFab(outputBox, 1);
            Array4<Real> diffArray = diffFab.array();
            // components: source, dest, number
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
                      << " ratio " << (amrex2_sum/spiral2_sum)
                      << std::endl;
            */
            std::cout << "Backward 3DFFT |diff| <= " << diff_max
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
