/* Copyright 2019-2020
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ANYFFT_H_
#define ANYFFT_H_

#ifdef AMREX_USE_GPU
#  include <cufft.h>
#else
#  include <fftw3.h>
#endif

#include <AMReX_LayoutData.H>

/**
 * Wrapper around FFT libraries. The header file defines the API and the base types
 * (Complex and VendorFFTPlan), and the implementation for different FFT libraries is
 * done in different cpp files. This wrapper only depends on the underlying FFT library
 * AND on AMReX (There is no dependence on WarpX).
 */
namespace AnyFFT
{
    // First, define library-dependent types (complex, FFT plan)

    /** Complex type for FFT, depends on FFT library */
#ifdef AMREX_USE_GPU
#  ifdef AMREX_USE_FLOAT
    using Complex = cuComplex;
#  else
    using Complex = cuDoubleComplex;
#  endif
#else
#  ifdef AMREX_USE_FLOAT
    using Complex = fftwf_complex;
#  else
    using Complex = fftw_complex;
#  endif
#endif

    /** Library-dependent FFT plans type, which holds one fft plan per box
     * (plans are only initialized for the boxes that are owned by the local MPI rank).
     */
#ifdef AMREX_USE_GPU
    using VendorFFTPlan = cufftHandle;
#else
#  ifdef AMREX_USE_FLOAT
    using VendorFFTPlan = fftwf_plan;
#  else
    using VendorFFTPlan = fftw_plan;
#  endif
#endif

    // Second, define library-independent API

    /** Direction in which the FFT is performed. */
    enum struct direction {R2C, C2R};

    /** This struct contains the vendor FFT plan and additional metadata
     */
    struct FFTplan
    {
        amrex::Real* m_real_array; /**< pointer to real array */
        Complex* m_complex_array; /**< pointer to complex array */
        VendorFFTPlan m_plan; /**< Vendor FFT plan */
        direction m_dir;  /**< direction (C2R or R2C) */
        int m_dim; /**< Dimensionality of the FFT plan */
    };

    /** Collection of FFT plans, one FFTplan per box */
    using FFTplans = amrex::LayoutData<FFTplan>;

    /** \brief create FFT plan for the backend FFT library.
     * \param[in] real_size Size of the real array, along each dimension.
     *                      Only the first dim elements are used.
     * \param[out] real_array Real array from/to where R2C/C2R FFT is performed
     * \param[out] complex_array Complex array to/from where R2C/C2R FFT is performed
     * \param[in] dir direction, either R2C or C2R
     * \param[in] dim direction, number of dimensions of the arrays. Must be <= AMREX_SPACEDIM.
     */
    FFTplan CreatePlan(const amrex::IntVect& real_size, amrex::Real * const real_array,
                       Complex * const complex_array, const direction dir, const int dim);

    /** \brief Destroy library FFT plan.
     * \param[out] fft_plan plan to destroy
     */
    void DestroyPlan(FFTplan& fft_plan);

    /** \brief Perform FFT with backend library.
     * \param[out] fft_plan plan for which the FFT is performed
     */
    void Execute(FFTplan& fft_plan);
}

#endif // ANYFFT_H_
