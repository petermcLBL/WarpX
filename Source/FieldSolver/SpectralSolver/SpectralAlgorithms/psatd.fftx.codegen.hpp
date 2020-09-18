

    #ifndef psatd_CODEGEN_H
    #define psatd_CODEGEN_H
 
    #include "fftx3.hpp"

    extern void init_psatd_spiral(); 
    extern void psatd_spiral(double** X, double** Y, double** symvar); 
    extern void destroy_psatd_spiral();

   namespace psatd
   {
    inline void init(){ init_psatd_spiral();}
    inline void trace();
    template<std::size_t IN_DIM, std::size_t OUT_DIM, std::size_t S_DIM>
    inline fftx::handle_t transform(std::array<fftx::array_t<3, double>,IN_DIM>& source,
                                    std::array<fftx::array_t<3, double>,OUT_DIM>& destination,
                                    std::array<fftx::array_t<3, double>,S_DIM>& symvar)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input[IN_DIM];
        double* output[OUT_DIM];
        double* sym[S_DIM];
        for(int i=0; i<IN_DIM; i++) input[i] = source[i].m_data.local();
        for(int i=0; i<OUT_DIM; i++) output[i] = destination[i].m_data.local();
        for(int i=0; i<S_DIM; i++) sym[i] = symvar[i].m_data.local();

        psatd_spiral(input, output, sym);
   
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    //inline void destroy(){ destroy_psatd_spiral();}
    inline void destroy(){ }
  };

 #endif  
