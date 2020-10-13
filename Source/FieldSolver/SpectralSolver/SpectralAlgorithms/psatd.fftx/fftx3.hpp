
/*
    ___________________  __
   / ____/ ____/_  __/ |/ /
  / /_  / /_    / /  |   / 
 / __/ / __/   / /  /   |  
/_/   /_/     /_/  /_/|_|  
                           

*/

#ifndef FFTX_HEADER
#define FFTX_HEADER


#include <complex>
#include <regex>
#include <memory>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <functional>
#include <list>
#include <iterator> // for std::next
#include <cassert>
#include <iomanip>


namespace fftx
{

  bool tracing = false;
  
  template<int DIM>
  struct plan_implem_t; // opaque handles

  struct handle_implem_t;
  
  struct handle_t
  {
  private:
    std::shared_ptr<handle_implem_t> m_implem;
  };
  struct context_t;
  
  typedef int intrank_t; // just useful for self-documenting code.

  //  This API has been shelved while I flesh out Phil's preferred design

  // Currently I have a bit of an ugly hack for dynamic extents...   what C++20 does
  // is set a dynamic range to std::numeric_limits<size_t> 
  //template <size_t Begin, size_t End, size_t Stride> struct span_t;


  ///non-owning global ptr object.  Can be used in place of upcxx::global_ptr
  template <typename T>
  class global_ptr
  {
    T* _ptr;
    intrank_t _domain;
    int _device;
  public:
    using element_type = T;
    global_ptr():_ptr{nullptr},_domain{0}, _device{0}{}
    /// strong constructor
    /** Franz would refer to this as the registration step */
    global_ptr(T* ptr, int domain=0, int device=0)
      :_ptr{ptr}, _domain{domain}, _device{device}{ }

    bool is_null() const;
    bool is_local() const;
    intrank_t where() const {return _domain;}
    int device() const {return _device;}
    T* local() {return _ptr;}
    const T* local() const {return _ptr;}
    operator global_ptr<void>(){ return global_ptr<void>(_ptr, _domain, _device);}
  };

  ///  multi-dimensional span.  a non-owning view of a data layout
  //  Can function as both the symbolic placeholder for a data structure
  //  for planning and for intermediate plan stages, and as input/output
  //  data placement when the gptr_t is not NULL.   It also allows
  //  a user to make plans that map between non-local data structures.

  // The evil part of this for our regular non-C++ users is that this is
  // a variadic template.  You can have an arbitrary extent of span_t members.
  // template <typename gptr_t, span_t... Span>
  //class mdspan
  // {
  //  gptr_t _data = NULL;
    
  // public:
  //  constexpr int dim();
  //  mdpsan(gptr_t data, int domain=0, int device=0)
  //    :_data(data),_domain(domain),_device(device){ };
  //  inline constexpr span_t span(int dim) const;
  //  inline gptr_t gptr(int dim);
  //};
    

  template<int DIM>
  struct point_t
  {
    int x[DIM];
    point_t<DIM-1> project() const;
    point_t<DIM-1> projectC() const;
    int operator[](unsigned char a_id) const {return x[a_id];}
    int& operator[](unsigned char a_id) {return x[a_id];}
    bool operator==(const point_t<DIM>& a_rhs) const;
    void operator=(int a_default);
    point_t<DIM> operator*(int scale) const;
    static point_t<DIM> Unit();
  };


                                 
  template<int DIM>
  struct box_t
  {
    box_t() = default;
    box_t(const point_t<DIM>&& a, const point_t<DIM>&& b)
      : lo(a), hi(b) { ; }
    point_t<DIM> lo, hi;
    std::size_t size() const;
    bool operator==(const box_t<DIM>& rhs) const {return lo==rhs.lo && hi == rhs.hi;}
    point_t<DIM> extents() const { point_t<DIM> rtn(hi); for(int i=0; i<DIM; i++) rtn[i]-=(lo[i]-1); return rtn;}
    box_t<DIM-1> projectC() const
    {
      return box_t<DIM-1>(lo.projectC(),hi.projectC());
    }
  };

  uint64_t ID;

  template<int DIM, typename T>
  struct array_t
  {
 
    array_t() = default;
    array_t(global_ptr<T>&& p, const box_t<DIM>& a_box)
      :m_data(p), m_domain(a_box) {;}
    array_t(const box_t<DIM>& m_box):m_domain(m_box)
    { if(tracing)
        {
          m_data = global_ptr<T>((T*)ID);
          std::cout<<"var_"<<ID<<":= var(\"var_"<<ID<<"\", BoxND("<<m_box.extents()<<", TReal));\n";
          ID++;
        }
      else m_data = global_ptr<T>(new T[m_box.size()]);
    }
    
    global_ptr<T> m_data;
    box_t<DIM>    m_domain;
    array_t<DIM, T> subArray(box_t<DIM>&& subbox);
    uint64_t id() const { assert(tracing); return (uint64_t)m_data.local();}
  };

  template<int DIM, typename T, typename Func>
  void forall(Func f, array_t<DIM, T>& array);

  

#ifdef MPI_VERSION
  void context_mpi(context_t& a_context, MPI_Comm a_comm);
#endif

  // Sort of a bridge function for making a plan that can assume MPI_COMM_WORLD is
  // active.  For many users the defualt MPI_COMM_WORLD plan is appropriate.  If special
  // communicators are needed then a valid MPI-enabled compilation is needed to access
  // context_mpi(context_t& a_context, MPI_Comm, a_comm)
  void context_mpi_comm_world(context_t& a_context);

  // a_threads can refer to the number of threads in the whole execution
  // or the number of threads within an omp team.
  //    FFTX is free to also use the omp simd directive if this is part of the
  // context, even if a_threads is set to zero.
  // FFTX will also use omp atomic primitives.
  void context_omp(context_t& a_context, unsigned int a_threads);

  // build FFTW3 plans internally and just invoke those plans in sequence
  void context_fftw3(context_t& a_context);


  // Tells the code generator how many transforms to perform simultaneously
  // The default batch size is set to 1.   This can help the code generator
  // explore another axis of parallelism and data reuse.
  void context_batch(context_t& a_context, unsigned int a_batchSize);
    
  // using C++ std::thread concurrency.  This option is incompatible with
  //context_omp.   This version tells FFTX to spin up std::threads on each
  // invocation of fftx_execute.  threads are 'join'ed when the fftx_handle_t
  // is waited on.
  void context_threads(context_t& a_context, unsigned int a_threads);

  // using C++ std::thread.  Same logic is used for planning as overloaded
  //sibing function, but user takes resposibility for 'join'ing threads after
  // waiting for fftx_handle_t completes.  
  //context_t context_threads(const context_t& input, const std::vector<std::thread>>& a_threadGroup);

#ifdef __CUDACC__
  // This is CUDA specific.  I will need to generalize this concept
  // better in the long term.   It is the responsibility of the user
  // to cudaSetDevice before the call to fftx_execute to match the
  // plan built on this context.
  // context_gpu can be combined with context_gpu.
  //  fftx does not assume the use of Unified Virtual Memory.
  // fftx_codegen plans built with this context enabled are non-blocking
  // by default. ie they can return before execution completes.
  void context_cuda(context_t& a_context, const cudaDeviceProp& a_device);

#endif

  enum KIND{I=0, DFT, IDFT, RDFT, IRDFT, M, DCT1, DCT2, DCT3, DCT4, DST1, DST2, DST3, DST4};

  // generalized FFT plan operation.  Two ways this can work.  plan_t
  //  can encode its input type, or it can type erase
  //  with strong typing
  // a_rank identifies the KIND of transform wanted in each dimenion. negative
  // means forward, 0 means untouched, positive means backwards
  template<int DIM, typename SOURCE, typename DEST>
  void transform(point_t<DIM> a_kind,
                 const array_t<DIM, SOURCE> & a_input,
                 array_t<DIM, DEST>& a_output);
  
  template<int DIM_S, int DIM_D, typename T>
  void scatter(const array_t<DIM_S, T>& a_src, array_t<DIM_D, T>& a_dst);

  template<int DIM, typename T>
  void resample(const point_t<DIM>& a_numerator, const point_t<DIM>& a_denominator,
                array_t<DIM, T>& a_data);
   
  // normalization after both a forward and an inverse transform.
  // FFTX does unnormalized transforms
  template<int DIM>
  std::size_t normalization(box_t<DIM> a_transformBox);
  

  template<int DIM, typename SOURCE, typename DEST>
  struct plan_t
  {
    box_t<DIM> m_input;
    box_t<DIM> m_output;
    box_t<DIM> m_transform;

    std::list<plan_implem_t<DIM>> m_implem;
  }; 
  
  template<int DIM, typename T>
  plan_t<DIM, T, T> copy(const box_t<DIM>& a_input,
                         const box_t<DIM>& a_output);

  
  template<int DIM, typename S>
  using fftx_callback_pw_box = std::function<void(array_t<DIM, std::complex<S>>& in_out)>;

  template<int DIM, typename T>
  using fftx_reshape_callback = std::function<void(array_t<DIM, T>& in_out)>;

  template<int DIM, typename S>
  using fftx_callback_pw = std::function<void(S& in_out, const point_t<DIM>& a_point, size_t a_normalize)>;

  template<int DIM, typename INPUT, typename OUTPUT>
  using fftx_callback_tc = std::function<void(OUTPUT& out, const INPUT& in, const point_t<DIM>&a_point, size_t a_normalize)>;
  
  template<int DIM, typename S>
  plan_t<DIM, S, S>
  kernel(fftx_callback_pw_box<DIM, S> a_op,
         box_t<DIM> & in_place, size_t normalize);

  template<int DIM, typename S>
  plan_t<DIM, S, S>
  kernel(fftx_callback_pw<DIM, S> a_op,
         box_t<DIM>& in_place, size_t normalize);

  template<int DIM, typename INPUT, typename OUTPUT>
  plan_t<DIM, INPUT, OUTPUT>
  kernel(fftx_callback_tc<DIM, INPUT, OUTPUT> a_op,
         box_t<DIM>& in_place, size_t a_normalize);
  
  // user captures return objects with std::tie
  //  plan_t<d, S, S> plan;
  //  array_t<d, S>   array;
  //  std::tie(plan, array) = scalarMult(box);
  // user expected to fill in array before a launch command is issued
  template<int DIM, typename S, typename D>
  std::pair<plan_t<DIM, D, D>,
            array_t<DIM, S>> scalarMult(const box_t<DIM>& a_region);

  template<int DIM, typename S>
  std::pair<plan_t<DIM, std::complex<S>, std::complex<S>>,
            array_t<DIM, std::complex<S>>> complexMult(const box_t<DIM>& a_region);

  template<typename T>
  inline void assignValue(T& a_dest,
                          const T& a_source)
  {
    a_dest = a_source;
  }

  template<typename T, std::size_t C>
  inline void assignValue(T(&a_dest)[C],
                          const T(&a_source)[C])
  {
    for (int comp = 0; comp < C; comp++)
      {
        a_dest[comp] = a_source[comp];
      }
  }


  template<typename DEST, typename SOURCE>
  inline void assignComplexValue(DEST& a_dest,
                                 const SOURCE& a_source,
                                 int a_comp);

  template<int C>
  inline void assignComplexValue(std::complex<double>& a_dest,
                                 const double(&a_source)[C],
                                 int a_comp)
  {
    a_dest = (a_source[a_comp], 0.);
  }

  template<int C>
  inline void assignComplexValue(std::complex<double>& a_dest,
                                 const std::complex<double>(&a_source)[C],
                                 int a_comp)
  {
    a_dest = a_source[a_comp];
  }

  template<int C>
  inline void assignComplexValue(std::complex<double>(&a_dest)[C],
                                 const std::complex<double>& a_source,
                                 int a_comp)
  {
    a_dest[a_comp] = a_source;
  }

  template<int C>
  inline void assignComplexValue(double(&a_dest)[C],
                                 const std::complex<double>& a_source,
                                 int a_comp)
  {
    a_dest[a_comp] = real(a_source);
  }

  template<>
  inline void assignComplexValue(std::complex<double>& a_dest,
                                 const std::complex<double>& a_source,
                                 int a_comp) // not used
  {
    a_dest = a_source;
  }

  template<>
  inline void assignComplexValue(double& a_dest,
                                 const std::complex<double>& a_source,
                                 int a_comp) // not used
  {
    a_dest = real(a_source);
  }

  template<>
  inline void assignComplexValue(std::complex<double>& a_dest,
                                 const double& a_source,
                                 int a_comp) // not used
  {
    a_dest = std::complex<double>(a_source, 0.);
  }

  template<int DIM, typename SOURCE, typename DEST>
  plan_t<DIM, SOURCE, DEST> arrayCopy(const box_t<DIM>& a_region);
            
  template<int DIM, typename T, std::size_t C_IN, std::size_t C_OUT>
  std::pair<plan_t<DIM, T[C_IN], T[C_OUT]>,
            array_t<DIM, T[C_IN][C_OUT]>> tensorContraction(const box_t<DIM>& a_region);

  template<int DIM, typename T>
  plan_t<DIM, T, T> reshape(const box_t<DIM>& a_input,
                            const box_t<DIM>& a_output,
                            fftx_reshape_callback<DIM, T> kernel);
                            
                            
  /* currently the design doesn't need this.....
  template<int DIM, typename SOURCE, typename DEST>
  handle_t launch(context_t a_context, const plan_t<DIM, SOURCE, DEST> p,
                  const array_t<DIM, SOURCE>& a_input,
                  array_t<DIM, DEST>& a_output);
  
  void wait(handle_t handle);
  */

  ///different user API now.  user calls trace after building a plan and prints to std::cout
  template<int DIM, typename SOURCE, typename DEST>
  void trace_g(context_t a_context,
               const plan_t<DIM, SOURCE, DEST>& a_plan, const char* name);
  
  // replace existing plan if name appears again, otherwise append plan to file
  template<int DIM, typename SOURCE, typename DEST>
  void export_spl(context_t a_context,
                  const plan_t<DIM, SOURCE, DEST>& a_plan,
                  std::ostream& a_splFile,
                  const std::string& a_name);

  // For code generation this will map a previously generated function that has been
  // compiled into the user's source code.  Returns a null generated plan if this
  // function does not exist.
  template<int DIM, typename SOURCE, typename DEST>
  using fftx_codegen = std::function<handle_t(const array_t<DIM, SOURCE>* input,
                                          array_t<DIM, DEST>* output,
                                          unsigned int a_count)>;

  /*  import is our init operation and also where we hand back
      a usable function that the user can execute */
  template<int DIM, typename SOURCE, typename DEST>
  fftx_codegen<DIM, SOURCE, DEST> import_spl(const std::string& plan_name);

 
  
  /* Seems we need to be able to destroy plans to clean up resources
     for GPU execution.  After this function is called, the fftx_codegen
     function returned by import_spl will have undefined behavior
  */
  template<int DIM, typename SOURCE, typename DEST>
  void destroy_plan(const std::string& plan_name);

  template<int DIM, typename T>
  array_t<DIM-1, T> nth(array_t<DIM, T>& array, int index)
  {
    box_t<DIM-1> b = array.m_domain.projectC();
    array_t<DIM-1, T> rtn(b);
    std::cout<<"var_"<<(uint64_t)rtn.m_data.local()<<":=nth(var_"<<(uint64_t)array.m_data.local()<<","<<index<<");\n";
    return rtn;
  }

  template<int DIM, typename T>
  void copy(array_t<DIM, T>& dest, const array_t<DIM, T>& src)
  {
    std::cout<<"    TDAGNode(TGath(fBox("<<src.m_domain.extents()<<")),var_"<<dest.id()<<", var_"<<src.id()<<"),\n";
  }

  void rawScript(const std::string& a_rawScript)
  {
    std::cout<<"\n"<<a_rawScript<<"\n";
  }

  template<typename T, std::size_t COUNT>
  inline std::ostream& operator<<(std::ostream& os, const std::array<T, COUNT>& arr)
  {
    os<<std::fixed<<std::setprecision(2);
    os<<"["<<arr[0];
    for(int i=1; i<COUNT; i++) os<<","<<arr[i];
    os<<"]";
    return os;
  }

  template<int DIM>
  void MDPRDFT(const point_t<DIM>& extent, int batch,
               array_t<DIM+1, double>& destination,
               array_t<DIM+1, double>& source)
  {
    std::cout<<"    TDAGNode(TTensorI(MDPRDFT("<<extent<<",-1),"<<batch<<",APar,APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  template<int DIM>
  void IMDPRDFT(const point_t<DIM>& extent, int batch,
               array_t<DIM+1, double>& destination,
               array_t<DIM+1, double>& source)
  {
    std::cout<<"    TDAGNode(TTensorI(IMDPRDFT("<<extent<<",1),"<<batch<<",APar,APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }


  template<int DIM, typename T, std::size_t COUNT>
  void setInputs(const std::array<array_t<DIM, T>, COUNT>& a_inputs)
  {
    for(int i=0; i<COUNT; i++)
      {
        std::cout<<"var_"<<a_inputs[i].id()<<":= nth(X,"<<i<<");\n";
      }
  }
  template<int DIM, typename T, std::size_t COUNT>
  void setOutputs(const std::array<array_t<DIM, T>, COUNT>& a_outputs)
  {
    for(int i=0; i<COUNT; i++)
      {
        std::cout<<"var_"<<a_outputs[i].id()<<":= nth(Y,"<<i<<");\n";
      }
  }
  
  template<int DIM, typename T>
  void resample(const std::array<double, DIM>& shift,
                array_t<DIM,T>& destination,
                const array_t<DIM,T>& source)
  {
    std::cout<<"    TDAGNode(TResample("
             <<destination.m_domain.extents()<<","
             <<source.m_domain.extents()<<","<<shift<<"),"
             <<"var_"<<destination.id()<<","
             <<"var_"<<source.id()<<"),\n";
  }
  void openDAG()
  {
  
    std::cout<<"transform:= TFCall(TDecl(TDAG([\n";
  }

  template<int DIM, unsigned long COUNT>
  void closeDAG(std::array<array_t<DIM,double>, COUNT>& localVars, const char* name)
  {
    static const char* header_template = R"(

    #ifndef PLAN_CODEGEN_H
    #define PLAN_CODEGEN_H

    #include "fftx3.hpp"

    extern void init_PLAN_spiral(); 
    extern void PLAN_spiral(double** X, double** Y, double** symvar); 
    extern void destroy_PLAN_spiral();

   namespace PLAN
   {
    inline void init(){ init_PLAN_spiral();}
    inline void trace();
    template<std::size_t IN_DIM, std::size_t OUT_DIM, std::size_t S_DIM>
    inline fftx::handle_t transform(std::array<fftx::array_t<DD, S_TYPE>,IN_DIM>& source,
                                    std::array<fftx::array_t<DD, D_TYPE>,OUT_DIM>& destination,
                                    std::array<fftx::array_t<DD, double>,S_DIM>& symvar)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input[IN_DIM];
        double* output[OUT_DIM];
        double* sym[S_DIM];
        for(int i=0; i<IN_DIM; i++) input[i] = source[i].m_data.local();
        for(int i=0; i<OUT_DIM; i++) output[i] = destination[i].m_data.local();
        for(int i=0; i<S_DIM; i++) sym[i] = symvar[i].m_data.local();

        PLAN_spiral(output, input, sym);
   
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    //inline void destroy(){ destroy_PLAN_spiral();}
    inline void destroy(){ }
  };

 #endif  )";

   std::string headerName = std::string(name)+std::string(".fftx.codegen.hpp");
   std::ofstream headerFile(headerName);
   //DataTypeT<SOURCE> s;
   //DataTypeT<DEST> d;
   std::string header_text = std::regex_replace(header_template,std::regex("PLAN"),name);
   header_text = std::regex_replace(header_text, std::regex("S_TYPE"), "double");
   header_text = std::regex_replace(header_text, std::regex("D_TYPE"), "double");
   header_text = std::regex_replace(header_text, std::regex("DD"), std::to_string(DIM-1));
   
   headerFile<<header_text<<"\n";
   headerFile.close();

    std::cout<<"\n]),\n   [";
    if(COUNT==0){}
    else
      {
        std::cout<<"var_"<<(uint64_t)localVars[0].m_data.local();
        for(int i=1; i<COUNT; i++) std::cout<<", var_"<<(uint64_t)localVars[i].m_data.local();
      }
     std::cout<<"]\n),\n";
     std::cout<<"rec(XType:= TPtr(TPtr(TReal)), YType:=TPtr(TPtr(TReal)), fname:=prefix::\"_spiral\", params:= [symvar])\n"
              <<").withTags(opts.tags);\n";
  }
      
      
  //============================================
  //============================================

  //Implementation details

  template<int DIM>
  struct ErasedFunction
  {
    ErasedFunction():function(nullptr) {}
    ErasedFunction(fftx_callback_pw<DIM, float>& f):function(&f) {}
    ErasedFunction(fftx_callback_pw<DIM, double>& f):function(&f) {}
    ErasedFunction(fftx_callback_pw<DIM, std::complex<float>>& f):function(&f) {}
    ErasedFunction(fftx_callback_pw<DIM, std::complex<double>>& f):function(&f) {}
    template<int C>
    ErasedFunction(fftx_callback_pw<DIM, double[C]>& f):function(&f) {}
    template<int C, int D>
    ErasedFunction(fftx_callback_tc<DIM, double[C], double[D]>& f): function(&f) {}
    template<int C, int D>
    ErasedFunction(fftx_callback_tc<DIM, std::complex<double>[C], std::complex<double>[D]>& f): function(&f) {}
    void* function;
  };


  enum DATA_TYPE { REAL_SINGLE=0, REAL_DOUBLE=1, COMPLEX_SINGLE=2, COMPLEX_DOUBLE=3 };

  struct DataType
  {
    int type = -1;
    int components = 1;
    const char* name() const
    {
      switch(type)
        {
        case REAL_SINGLE:
          return "float";
        case REAL_DOUBLE:
          return "double";
        case COMPLEX_SINGLE:
          return "std::complex<float>";
        case COMPLEX_DOUBLE:
          return "std::complex<double>";
        }
      return "unknown";
    }
  };
  template<typename T>
  class DataTypeT : public DataType
  {
  public:
    inline DataTypeT() {;}
  };
  template<typename T, std::size_t C>
  class DataTypeT<T[C]> : public DataTypeT<T>
  {
  public:
    inline DataTypeT(){ DataTypeT<T>::components=C;}
  };
  
 

  template<>
  inline DataTypeT<float>::DataTypeT(){ type=REAL_SINGLE;}
  template<>
  inline DataTypeT<double>::DataTypeT(){ type=REAL_DOUBLE;}
  template<>
  inline DataTypeT<std::complex<float>>::DataTypeT(){ type=COMPLEX_SINGLE;}
  template<>
  inline DataTypeT<std::complex<double>>::DataTypeT(){ type=COMPLEX_DOUBLE;}
  /*
  template<>
  inline DataTypeT<float>::DataTypeT(){ type=0; components=C}
  template<std::size_t C>
  inline DataTypeT<double[C]>::DataTypeT(){ type=1; components=C}
  template<>
  inline DataTypeT<std::complex<float>>::DataTypeT(){ type=2;}
  template<>
  inline DataTypeT<std::complex<double>>::DataTypeT(){ type=3;}
  */
  
  template<int DIM>
  inline point_t<DIM> lengthsBox(const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> hi = a_bx.hi;
    point_t<DIM> lengths;
    for (int d = 0; d < DIM; d++)
      {
        lengths.x[d] = hi[d] - lo[d] + 1;
      }
    return lengths;
  }

  template<int DIM>
  inline bool isInBox(point_t<DIM> a_pt, const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> hi = a_bx.hi;
    for (int d = 0; d < DIM; d++)
      {
        if (a_pt[d] < lo[d]) return false;
        if (a_pt[d] > hi[d]) return false;
      }
    return true;
  }

  template<int DIM>
  inline size_t positionInBox(point_t<DIM> a_pt, const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> hi = a_bx.hi;
    point_t<DIM> lengths = lengthsBox(a_bx);

    /*
    // Last dimension changes fastest.
    size_t disp = a_pt[0] - lo[0];
    for (int d = 1; d < DIM; d++)
      {
        disp *= lengths[d];
        disp += a_pt[d] - lo[d];
      }
    */

    // First dimension changes fastest.
    size_t disp = a_pt[DIM-1] - lo[DIM-1];
    for (int d = DIM-2; d >= 0; d--)
      {
        disp *= lengths[d];
        disp += a_pt[d] - lo[d];
      }

    return disp;
  }

 

  template<int DIM>
  struct plan_implem_t
  {
    plan_implem_t(const box_t<DIM>& a_input,
                  const box_t<DIM>& a_output,
                  const box_t<DIM>& a_transform,
                  point_t<DIM>& a_kind,
                  DataType a_inputType,
                  DataType a_outputType,
                  size_t a_normalize = 1,
                  std::string a_kernel = "")
      : inputBox(a_input),
        outputBox(a_output),
        transformBox(a_transform),
        kind(a_kind),
        inputType(a_inputType),
        outputType(a_outputType),
        normalize(a_normalize),
        reshape_kernel(a_kernel),
        erasedHolder(nullptr)
        
    {
      fftwflag = false;
#if USE_FFTW
      std::cout << "Long plan kind " << a_kind
                << " input type " << a_inputType.name()
                << " output type " << a_outputType.name() << std::endl;

      point_t<DIM> allI, allDFT, allIDFT, allDST1;
      allI = I;
      allDFT = DFT;
      allIDFT = IDFT;
      allDST1 = DST1;
      if ((a_kind == allDFT) || (a_kind == allIDFT) || (a_kind == allDST1))
        {
          fftwflag = true;
          // order of passes through dimensions
          box_t<DIM> boxStartHigh = (a_kind == allDFT) ? a_input : a_output;
          box_t<DIM> boxStartLow = (a_kind == allDFT) ? a_output : a_input;
          point_t<DIM> transformLo = a_transform.lo;
          point_t<DIM> transformHi = a_transform.hi;
          point_t<DIM> startHighLength = lengthsBox(boxStartHigh);
          point_t<DIM> startLowLength = lengthsBox(boxStartLow);
          point_t<DIM> transformLength = lengthsBox(a_transform);
          // distance between adjacent points in each dimension
          point_t<DIM> distance;

          point_t<DIM> zero;
          for (int d = 0; d < DIM; d++)
            {
              zero.x[d] = 0;
            }
          for (int d = 0; d < DIM; d++)
            {
              point_t<DIM> unit;
              for (int dd = 0; dd < DIM; dd++)
                {
                  unit.x[dd] = (dd == d) ? 1 : 0;
                }
              distance.x[d] = positionInBox(unit, transformBox) -
                positionInBox(zero, transformBox);
            }

          for (int tfmdir = 0; tfmdir < DIM; tfmdir++)
            {
              // low and high ends of box_t<DIM> startloop[tfmdir]
              point_t<DIM> loopLo, loopHi;
              // dimension of this transform: do all
              loopLo.x[tfmdir] = transformLo[tfmdir];
              loopHi.x[tfmdir] = transformLo[tfmdir];
              if (tfmdir < DIM-1)
                {
                  for (int otherdir = 0; otherdir < tfmdir; otherdir++)
                    { // lower dimensions, already done (DFT) or to do (IDFT)
                      loopLo.x[otherdir] = boxStartLow.lo[otherdir];
                      loopHi.x[otherdir] = boxStartLow.hi[otherdir];
                    }
                  for (int otherdir = tfmdir+1; otherdir < DIM-1; otherdir++)
                    { // higher dimensions, to do (DFT) or already done (IDFT)
                      loopLo.x[otherdir] = boxStartHigh.lo[otherdir];
                      loopHi.x[otherdir] = boxStartHigh.hi[otherdir];
                    }
                  // in highest dimension, loopLo == loopHi
                  loopLo.x[DIM-1] = boxStartHigh.lo[DIM-1];
                  loopHi.x[DIM-1] = boxStartHigh.lo[DIM-1];
                }
              else if ((DIM > 1) && (tfmdir == DIM-1))
                {
                  for (int otherdir = 0; otherdir < DIM-2; otherdir++)
                    { // lower dimensions, already done (DFT) or to do (IDFT)
                      loopLo.x[otherdir] = boxStartLow.lo[otherdir];
                      loopHi.x[otherdir] = boxStartLow.hi[otherdir];
                    }
                  // next-highest dimension: loopLo == loopHi
                  loopLo.x[DIM-2] = boxStartLow.lo[DIM-2];
                  loopHi.x[DIM-2] = boxStartLow.lo[DIM-2];
                }
              startloop[tfmdir] = box_t<DIM>(point_t<DIM>(loopLo),
                                             point_t<DIM>(loopHi));

              int fftlength = transformLength[tfmdir];
              int howmany, stride, dist;
              if (tfmdir < DIM-1)
                {
                  howmany = startHighLength[DIM-1];
                  dist = distance[DIM-1];
                  stride = distance[tfmdir];
                }
              else if (DIM > 1) // tfmdir == DIM-1
                {
                  howmany = startLowLength[DIM-2];
                  dist = distance[DIM-2];
                  stride = distance[DIM-1];
                }
              else // tfmdir == 0 and DIM == 1
                {
                  howmany = 1;
                  dist = distance[0];
                  stride = distance[0];
                }
              int fftlengtharray[1] = { transformLength[tfmdir] };
              // std::cout << "Maximum offset of any element in plan is "
              // << ((howmany-1) * dist + (fftlength-1)*stride)
              // << std::endl;
              int fftrank = 1;
              std::cout << "Defining fftwplandims[" << tfmdir << "]"
                        << " dimension=" << tfmdir
                        << " length=" << fftlengtharray[0]
                        << " howmany=" << howmany
                        << " stride=" << stride
                        << " dist=" << dist
                        << " startloop=" << startloop[tfmdir]
                        << " count " << startloop[tfmdir].size()
                        << std::endl;

              if ( (a_inputType.type == COMPLEX_DOUBLE) ||
                   (a_outputType.type == COMPLEX_DOUBLE) )
                {
                  int signfftw = FFTW_FORWARD;
                  if (a_kind == allIDFT)
                    { // If IDFT, do dimensions from high to low.
                      signfftw = FFTW_BACKWARD;
                    }

                  size_t bytes = a_transform.size() * sizeof(fftw_complex);
                  std::cout << "allocating " << bytes << " bytes of complex for "
                            << a_transform.size() << " points"
                            << std::endl;
                  fftwdataptr = (fftw_complex*) fftw_malloc(bytes);
                  
                  // howmany, stride, dist are all for tfmdir.
                  fftwplandims[tfmdir] =
                    fftw_plan_many_dft(fftrank, fftlengtharray, howmany,
                                       fftwdataptr,
                                       NULL, stride, dist,
                                       fftwdataptr,
                                       NULL, stride, dist,
                                       signfftw, FFTW_MEASURE);
                  // FFTW 3.3.8: check this alignment with those in execute_dft
                  fftwplanalignment[tfmdir] =
                    fftw_alignment_of((double *) fftwdataptr);
                  // std::cout << "defining fftw_plan_many_dft with alignment "
                  // << fftwplanalignment[tfmdir] << std::endl;
                }
              else 
                {
                  std::cout << "cannot define plan" << std::endl;
                }

              // std::cout << "done defining plan in dimension " << tfmdir << std::endl;
            }
        }
      else if (kind == allI)
        {
          // Nothing to do here.
        }
      else
        {
          std::cout << "Kind must be all I, or all DFT, or all IDFT" << std::endl;
        }
#endif
    }

    // input and output and transform boxes are all the same a_region
    plan_implem_t(const box_t<DIM>& a_region,
                  point_t<DIM>& a_kind,
                  DataType a_inputType,
                  DataType a_outputType, 
                  global_ptr<void> a_erasedHolder)
      : inputBox(a_region),
        outputBox(a_region),
        transformBox(a_region),
        kind(a_kind),
        inputType(a_inputType),
        outputType(a_outputType),
        erasedHolder(a_erasedHolder)
    {
#if USE_FFTW
      std::cout << "Short plan kind " << a_kind
                << " input type " << a_inputType.name()
                << " output type " << a_outputType.name() << std::endl;
#endif
      fftwflag = false;
    }

    ~plan_implem_t()
    {
#if USE_FFTW
      if (fftwflag)
        {
          // std::cout << "kind " << kind << " fftw_free(fftwdataptr);" << std::endl;
          // fftw_free(fftwdataptr);
          // std::cout << "fftw_destroy_plan(fftwplan);" << std::endl;
          // fftw_destroy_plan(fftwplan);
          // std::cout << "done ~plan_implem_t" << std::endl;
          fftwflag = false;
        }
#endif
    }
    
    box_t<DIM> inputBox, outputBox, transformBox;
    point_t<DIM>  kind;
    DataType inputType, outputType;
    size_t normalize = 1;
    std::string reshape_kernel;
    global_ptr<void> erasedHolder;
    bool fftwflag;
#if USE_FFTW
    fftw_complex* fftwdataptr;
    fftw_plan fftwplandims[DIM];
    int fftwplanalignment[DIM];
    box_t<DIM> startloop[DIM];
#endif
  };
  // helper meta functions===============
  template<int DIM>
  void projecti(int out[], const int in[] );

  template<>
  inline void projecti<0>(int out[], const int in[]) { return; }

  template<int DIM>
  inline void projecti(int out[], const int in[] )
  {
    out[DIM-1]=in[DIM-1]; projecti<DIM-1>(out, in);
  }

  template<int DIM>
  std::size_t bsize(int const lo[], int const hi[]);
  template<>
  inline std::size_t bsize<0>(int const lo[], int const hi[]){return 1;}
  template<int DIM>
  inline std::size_t bsize(int const lo[], int const hi[]){ return (hi[DIM-1]-lo[DIM-1]+1)*bsize<DIM-1>(lo, hi);}

  template<int DIM>
  inline point_t<DIM-1> point_t<DIM>::project() const
  {
    point_t<DIM-1> rtn;
    projecti<DIM-1>(rtn.x, x);
    return rtn;
  }

  template<int DIM>
  inline point_t<DIM-1> point_t<DIM>::projectC() const
  {
    point_t<DIM-1> rtn;
    for(int i=0; i<DIM-1; i++) rtn[i] = x[i+1];
    return rtn;
  }
  
  template<unsigned char DIM>
  inline bool equalInts(const int* a, const int* b) { return (a[DIM-1]==b[DIM-1])&&equalInts<DIM-1>(a, b);}
  template<>
  inline bool equalInts<0>(const int* a, const int* b) {return true;}
  
  template<int DIM>
  inline std::size_t box_t<DIM>::size() const { return bsize<DIM>(lo.x,hi.x);}

  template<int DIM>
  inline bool point_t<DIM>::operator==(const point_t<DIM>& a_rhs) const
  {
    return equalInts<DIM>(x, a_rhs.x);
  }
  
  template<int DIM>
  inline void point_t<DIM>::operator=(int a_value)
  {
    for(int i=0; i<DIM; i++) x[i]=a_value;
  }

  template<int DIM>
  inline point_t<DIM> point_t<DIM>::operator*(int a_scale) const
  {
    point_t<DIM> rtn(*this);
    for(int i=0; i<DIM; i++) rtn.x[i]*=a_scale;
    return rtn;
  }

  template<int DIM>
  inline point_t<DIM> point_t<DIM>::Unit()
  {
    point_t<DIM> rtn;
    for(int i=0; i<DIM; i++) rtn.x[i]=1;
    return rtn;
  } 
  
  template<int DIM, typename T, typename Func_P>
  struct forallHelper
  {
    static void f(T*& __restrict ptr, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[DIM-1]; i<=hi[DIM-1]; ++i)
        {
          pvect[DIM-1]=i;
          forallHelper<DIM-1, T, Func_P>::f(ptr, pvect, lo, hi, fp);
        }
    }
    template<typename T2>
    static void f2(T*& __restrict ptr1, const T2*& __restrict ptr2, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[DIM-1]; i<=hi[DIM-1]; ++i)
        {
          pvect[DIM-1]=i;
          forallHelper<DIM-1, T, Func_P>::f2(ptr1, ptr2, pvect, lo, hi, fp);
        }
    }
  };
  
  template<typename T, typename Func_P>
  struct forallHelper<1, T, Func_P>
  {
    static void f(T*& __restrict ptr, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[0]; i<=hi[0]; i++, ptr++)
        {
          pvect[0]=i;
          fp(*ptr);
        }
    }
    template<typename T2>
    static void f2(T*& __restrict ptr1,  const T2*& __restrict ptr2, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[0]; i<=hi[0]; i++, ptr1++, ptr2++)
        {
          pvect[0]=i;
          fp(*ptr1, *ptr2);
        }
    }
 
  };
  
  template<int DIM, typename T, typename Func>
  inline void forall(Func f, array_t<DIM, T>& array)
  {
    int* lo=array.m_domain.lo.x;
    int* hi=array.m_domain.hi.x;
    point_t<DIM> p = array.m_domain.lo;
    auto fp = [&](T& v){f(v, p);};
    T* ptr = array.m_data.local();
    forallHelper<DIM, T,decltype(fp) >::f(ptr, p.x, lo, hi,fp);
  }
  template<int DIM, typename T1, typename T2, typename Func>
  inline void forall(Func f, array_t<DIM, T1>& array, const array_t<DIM, T2>& array2)
  {
    int* lo=array.m_domain.lo.x;
    int* hi=array.m_domain.hi.x;
    point_t<DIM> p = array.m_domain.lo;
    auto fp = [&](T1& v, const T2& v2){f(v, v2, p);};
    T1* ptr = array.m_data.local();
    const T2* ptr2 = array2.m_data.local();
    forallHelper<DIM, T1,decltype(fp) >::f2(ptr, ptr2, p.x, lo, hi,fp);
  }
                  
  template<int DIM, typename SOURCE, typename DEST>
  inline plan_t<DIM, SOURCE, DEST> plan_dft(point_t<DIM> a_kind,
                                     const box_t<DIM> & a_input,
                                     const box_t<DIM>& a_output,
                                     const box_t<DIM>& a_transform)
  {
    plan_t<DIM, SOURCE, DEST> rtn;
    rtn.m_input  = a_input;
    rtn.m_output = a_output;
    rtn.m_transform = a_transform;
    rtn.m_implem.clear();
    DataTypeT<SOURCE> s;
    DataTypeT<DEST> d;
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_input, a_output, a_transform,a_kind,
                                       s, d));
    return rtn;
  }

  template<int DIM, typename S>
  inline plan_t<DIM, S, S>
  kernel(fftx_callback_pw<DIM, S> a_op,
         box_t<DIM>& in_place, size_t normalize)
  {
    plan_t<DIM, S, S> rtn;
    rtn.m_input = in_place;
    rtn.m_output = in_place;
    rtn.m_transform = in_place;
    DataTypeT<S> s;
    point_t<DIM> kind ; kind = I;
    rtn.m_implem.push_back(plan_implem_t<DIM>(in_place, in_place, in_place,
                                              kind, s, s,
                                              normalize, ErasedFunction<DIM>(a_op)));
    
    return rtn;
  }

  template<int DIM, typename INPUT, typename OUTPUT>
  plan_t<DIM, INPUT, OUTPUT>
  kernel(fftx_callback_tc<DIM, INPUT, OUTPUT> a_op,
         box_t<DIM>& in_place, size_t a_normalize)
  {
    plan_t<DIM, INPUT, OUTPUT> rtn;
    rtn.m_input = in_place;
    rtn.m_output = in_place;
    rtn.m_transform = in_place;
    DataTypeT<INPUT> s;
    DataTypeT<OUTPUT> d;
    point_t<DIM> kind ; kind = I;
    rtn.m_implem.push_back(plan_implem_t<DIM>(in_place, in_place, in_place,
                                              kind, s, d,
                                              a_normalize, ErasedFunction<DIM>(a_op)));
    
    return rtn;
  }
  template<unsigned char DIM>
  inline size_t dimHelper(int* lo, int* hi) {return (hi[DIM-1]-lo[DIM-1]+1)*dimHelper<DIM-1>(lo, hi);}
  template<>
  inline size_t dimHelper<0>(int* lo, int* hi){ return 1;}
  
  template<int DIM>
  inline std::size_t normalization(box_t<DIM> a_transformBox)
  {
    //return dimHelper<DIM>(a_transformBox.lo.x, a_transformBox.hi.x);
    return a_transformBox.size();
  }

  template<int DIM, typename S, typename SD, typename D>
  inline plan_t<DIM, S, D> chain(const plan_t<DIM,S, SD>& a_left,
                                 const plan_t<DIM,SD, D>& a_right)
  {
    if(!(a_left.m_output == a_right.m_input))
      {
        std::cerr<<"mismatched inputs and output "<<a_left.m_output<<" "<<a_right.m_input<<std::endl;
        abort();
      }
    // if(!(a_left.m_transform == a_right.m_transform))  not sure how to do this check for reshape plans
    //   {
    //     std::cerr<<" mismatched transform size box_t in chain operation "<<
    //       a_left.m_transform<<" "<<a_right.m_transform<<std::endl;
    //     abort();
    //   }
    //unsure what other checks should be imposed at this point (bvs)
    plan_t<DIM, S, D> rtn;
    rtn.m_input  = a_left.m_input;
    rtn.m_output = a_right.m_output;
    rtn.m_transform = a_left.m_transform;
    rtn.m_implem = a_left.m_implem;
    rtn.m_implem.insert(rtn.m_implem.end(),
                        a_right.m_implem.begin(),
                        a_right.m_implem.end());
    return rtn;
  }

  template<int DIM, typename S>
  inline std::pair<plan_t<DIM, std::complex<S>, std::complex<S>>,
                   array_t<DIM, std::complex<S>>>
                                   complexMult(const box_t<DIM>& a_region)
  {
    plan_t<DIM, std::complex<S>, std::complex<S>> rtn;
    rtn.m_input = a_region;
    rtn.m_output = a_region;
    rtn.m_transform = a_region;
    point_t<DIM> kind;
    kind = M;
    DataTypeT<std::complex<S>> s;
    array_t<DIM, std::complex<S>> holder(a_region);
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_region, a_region, a_region,
                                              kind, s, s,holder));
    return std::make_pair(rtn, holder);
  }

  template<int DIM, typename S, typename D>
  inline std::pair<plan_t<DIM, D, D>,
                   array_t<DIM, S>>
                                   scalarMult(const box_t<DIM>& a_region)
  {
    plan_t<DIM, D, D> rtn;
    rtn.m_input = a_region;
    rtn.m_output = a_region;
    rtn.m_transform = a_region;
    point_t<DIM> kind;
    kind = M;
    DataTypeT<S> s;
    DataTypeT<D> d;
    array_t<DIM, S> holder(a_region);
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_region,
                                              kind, s, d ,holder.m_data));
    return std::make_pair(rtn, holder);
  }

  template<int DIM, typename SOURCE, typename DEST>
  inline plan_t<DIM, SOURCE, DEST> arrayCopy(const box_t<DIM>& a_region)
  {
    plan_t<DIM, SOURCE, DEST> rtn;
    rtn.m_input = a_region;
    rtn.m_output = a_region;
    rtn.m_transform = a_region;
    point_t<DIM> kind;
    kind = I;
    DataTypeT<SOURCE> s;
    DataTypeT<DEST> d;
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_region, kind, s, d));
    return rtn;
  }

  template<int DIM, typename T, std::size_t C_IN, std::size_t C_OUT>
  inline std::pair<plan_t<DIM, T[C_IN], T[C_OUT]>,
                   array_t<DIM, T[C_IN][C_OUT]>>
                                   tensorContraction(const box_t<DIM>& a_region)
  {
    plan_t<DIM, T[C_IN], T[C_OUT]> rtn;
    rtn.m_input = a_region;
    rtn.m_output = a_region;
    rtn.m_transform = a_region;
    point_t<DIM> kind;
    kind = M;
    DataTypeT<T[C_IN]> s;
    DataTypeT<T[C_OUT]> d;
    array_t<DIM, T[C_IN][C_OUT]> holder(a_region);
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_region,
                                              kind, s, d ,holder.m_data));
    return std::make_pair(rtn, holder);
  }
  
  template<int DIM, typename T>
  plan_t<DIM, T, T> reshape(const box_t<DIM>& a_input,
                            const box_t<DIM>& a_output,
                            const std::string& a_reshapeFunction)
  {
    plan_t<DIM, T, T> rtn;
    rtn.m_input = a_input;
    rtn.m_output = a_output;
    rtn.m_transform = a_input;
    point_t<DIM> kind;
    kind = I;
    DataTypeT<T> s, d;
    rtn.m_implem.push_back(plan_implem_t<DIM>(a_input, a_output, a_input, kind, s, d, 1, a_reshapeFunction));
    return rtn;

  }
  template<int DIM>
  inline std::ostream& operator<<(std::ostream& output, const point_t<DIM> p)
  {
    output<<"[";
    for(int i=0; i<DIM-1; i++)
      {
        output<<p.x[i]<<",";
      }
    output<<p.x[DIM-1]<<"]";
    return output;
  }
  
  template<int DIM>
  inline std::ostream& operator<<(std::ostream& output, const box_t<DIM>& b)
  {
    output<<"["<<b.lo<<","<<b.hi<<"]";
    return output;
  }

  struct context_implem_t
  {
    unsigned int m_OMPThreads=0; // 0 indicates no openmp threading.
    unsigned int m_batch=1;
    bool m_mpiCommWorld = false;
    bool m_fftw3 = false;
  };
  
  struct context_t
  {
    context_t():m_implem(new context_implem_t){;}
    std::shared_ptr<context_implem_t> m_implem;
  };
  
  inline void context_mpi_comm_world(context_t& a_context)
  {
    a_context.m_implem->m_mpiCommWorld = true;
  }
  
  inline void context_omp(context_t& a_context, unsigned int a_threads)
  {
    a_context.m_implem->m_OMPThreads = a_threads;
  }

  inline void context_batch(context_t& a_context, unsigned int a_batchSize)
  {
    a_context.m_implem->m_batch = a_batchSize;
  }

  inline void context_fftw3(context_t& a_context)
  {
    a_context.m_implem->m_fftw3 = true;
  }

  // registration mappings ==================
  template<int DIM, typename SOURCE, typename DEST>
  std::map<std::string, std::pair<plan_t<DIM, SOURCE, DEST>,fftx_codegen<DIM, SOURCE, DEST>> > planMaps;

 
  template<int DIM>
  std::map<std::string, void (*)()> initMap;

  template<int DIM>
  std::map<std::string, void (*)()> destroyMap;
  //=========================================

  template<int DIM, typename SOURCE, typename DEST>
  void executeFFTW3(const plan_t<DIM, SOURCE, DEST>& a_plan,
                    const array_t<DIM, SOURCE>* input,
                    array_t<DIM, DEST>* output,
                    unsigned int a_count)
  {
    std::cout << "*** ENTER executeFFTW3" << std::endl;
    point_t<DIM> allI, allDFT, allIDFT, allM, allDST1;
    allI = I;
    allDFT = DFT;
    allIDFT = IDFT;
    allM = M;
    allDST1 = DST1;
    for(int c=0; c<a_count; c++)
      {
        const array_t<DIM, SOURCE>& in_array = input[c];
        array_t<DIM, DEST>& out_array = output[c];

        std::cout << "input array ";
        if (std::is_same< SOURCE, std::complex<double> >::value)
          {
            std::cout << "SOURCE == std::complex<double>";
          }
        else if (std::is_array<SOURCE>::value)
          {
            typedef typename std::remove_all_extents<SOURCE>::type sourcebase;
            if (std::is_same< sourcebase, std::complex<double> >::value)
              {
                std::cout << "SOURCE == std::complex<double>";
              }
            else if (std::is_same< sourcebase, double >::value)
              {
                std::cout << "SOURCE == double";
              }
            int sourceextent = std::extent<SOURCE, 0>::value;
            std::cout << "[" << sourceextent << "]";
          }
        std::cout << " on " << in_array.m_domain << std::endl;

        std::cout << "output array ";
        if (std::is_same< DEST, std::complex<double> >::value)
          {
            std::cout << "DEST == std::complex<double>";
          }
        else if (std::is_array<DEST>::value)
          {
            typedef typename std::remove_all_extents<DEST>::type sourcebase;
            if (std::is_same< sourcebase, std::complex<double> >::value)
              {
                std::cout << "DEST == std::complex<double>";
              }
            else if (std::is_same< sourcebase, double >::value)
              {
                std::cout << "DEST == double";
              }
            int destextent = std::extent<DEST, 0>::value;
            std::cout << "[" << destextent << "]";
          }
        std::cout << " on " << out_array.m_domain << std::endl;

        // Need to have a data structure to use between plans.
        const box_t<DIM>& transformBox = a_plan.m_transform;
        array_t< DIM, std::complex<double> > fftwarray(transformBox);

        // BEGIN DEBUG
        //        forall([](SOURCE(&v), const point_t<DIM>& p)
        //               {
        //                 if (real(v[5]) != 0.)
        //                   {
        //                     std::cout << "AT POS = " << p
        //                                   << " bz = " << v[5] << std::endl;
        //                   }
        //               }, in_array);
        if (false)
        {
          auto inputData = in_array.m_data.local();
          for (size_t pt = 0; pt < a_plan.m_input.size(); pt++)
            {
              std::complex<double> cval;
              assignComplexValue(cval, inputData[pt], 5);
              if (real(cval) != 0.)
                {
                  std::cout << "bz[" << pt << "] = " << cval << std::endl;
                }
            }
        }
        // END DEBUG

        // general code for executing FFTW3 plans.
        
        // const array_t<DIM, SOURCE>& in_array = input[c];
        // array_t<DIM, DEST>& out_array = output[c];
        void* inputArrayPtr = (void*) &in_array;
        void* outputArrayPtr = NULL;
        // void* inputArrayPtr[1] = {(void *) &in_array};
        // void* outputArrayPtr[1] = {NULL};

        // for (auto& plan_implem : a_plan.m_implem)
        auto implemList = a_plan.m_implem;
        for (typename std::list< plan_implem_t<DIM> >::iterator plan_implem = implemList.begin();
             plan_implem != implemList.end();
             ++plan_implem)
          {
            // loop over plans
            int nincomps = plan_implem->inputType.components;
            int noutcomps = plan_implem->outputType.components;
            std::cout << "plan_implem in executeFFTW3"
                      << " fftwflag=" << plan_implem->fftwflag
                      << " kind=" << plan_implem->kind
                      << " from " << plan_implem->inputType.name();
            if (std::is_array<SOURCE>::value)
              {
                std::cout << "[" << nincomps << "]";
              }
            std::cout << " to " << plan_implem->outputType.name();
            if (std::is_array<DEST>::value)
              {
                std::cout << "[" << nincomps << "]";
              }
            std::cout << std::endl;
            const box_t<DIM>& outputBox = plan_implem->outputBox;
            // int destcomps = DataType(DEST).components;
            bool lastInList = (std::next(plan_implem) == implemList.end());
            if (lastInList)
              { // This is the last subplan in the list.
                outputArrayPtr = (void*) &out_array;
              }
            else if ( (plan_implem->kind == allDFT) ||
                      (plan_implem->kind == allM) ||
                      (plan_implem->kind == allDST1) ||
                      (plan_implem->kind == allI) )
              {
                std::cout << "Allocating new array on " << outputBox << std::endl;
                outputArrayPtr = new array_t<DIM, SOURCE>(outputBox);
              }
            else
              {
                std::cout << "Not setting outputArrayPtr!" << std::endl;
              }

            if (plan_implem->kind == allDFT)
              {
                executeFFTW3transform((const array_t<DIM, SOURCE>*)inputArrayPtr,
                                      (array_t<DIM, SOURCE>*)outputArrayPtr,
                                      *plan_implem);
              }
            else if (plan_implem->kind == allIDFT)
              {
                executeFFTW3transform((const array_t<DIM, DEST>*)inputArrayPtr,
                                      (array_t<DIM, DEST>*)outputArrayPtr,
                                      *plan_implem);
              }
            else if (plan_implem->kind == allDST1)
              {
                executeFFTW3transform((const array_t<DIM, SOURCE>*)inputArrayPtr,
                                      (array_t<DIM, SOURCE>*)outputArrayPtr,
                                      *plan_implem);
              }
            else if (plan_implem->kind == allM)
              { // Tensor contraction.
                executeTensorContraction((const array_t<DIM, SOURCE>*) inputArrayPtr,
                                         (array_t<DIM, DEST>*) outputArrayPtr,
                                         *plan_implem);
              }
            else if (plan_implem->kind == allI)
              {
                executeKernel((const array_t<DIM, DEST>*) inputArrayPtr,
                              (array_t<DIM, DEST>*) outputArrayPtr,
                              *plan_implem);
              }
            else
              {
                std::cout << "executeFFTW3: unsupported kind "
                          << plan_implem->kind << std::endl;
              }

            if (plan_implem->kind == allM)
              {
                delete (array_t<DIM, SOURCE>*) inputArrayPtr;
              }
            else if (plan_implem->kind == allIDFT)
              {
                delete (array_t<DIM, DEST>*) inputArrayPtr;
              }
            else if (plan_implem->kind == allDST1)
              {
                if (lastInList)
                  {
                    delete (array_t<DIM, SOURCE>*) inputArrayPtr;
                  }
              }
            // input array for next subplan is output array for current one
            inputArrayPtr = outputArrayPtr;
          }
      }
    std::cout << "*** EXIT executeFFTW3" << std::endl;
  }

  template<int DIM, typename T>
  inline void executeFFTW3transform(const array_t<DIM, T>* a_inputArrayPtr,
                                    array_t<DIM, T>* a_outputArrayPtr,
                                    plan_implem_t<DIM>& a_plan_implem)
  {
#if USE_FFTW
    int ncomps = a_plan_implem.inputType.components;
    const box_t<DIM>& inputArrayBox = a_inputArrayPtr->m_domain;
    const box_t<DIM>& outputArrayBox = a_outputArrayPtr->m_domain;
    std::cout << "running executeFFTW3transform on "
              << ncomps << " component";
    if (ncomps > 1)
      {
        std::cout << "s";
      }
    std::cout << ", kind "
              << a_plan_implem.kind << std::endl;
    point_t<DIM> allDFT, allIDFT, allDST1;
    allDFT = DFT;
    allIDFT = IDFT;
    allDST1 = DST1;
    const box_t<DIM>& inputBox = a_plan_implem.inputBox;
    const box_t<DIM>& outputBox = a_plan_implem.outputBox;
    assert(inputBox == inputArrayBox);
    assert(outputBox == outputArrayBox);
    const box_t<DIM>& transformBox = a_plan_implem.transformBox;
    array_t< DIM, std::complex<double> > fftwarray(transformBox);
    // ncomps should be same as a_plan_implem.outputType.components
    auto inData = a_inputArrayPtr->m_data.local();
    auto outData = a_outputArrayPtr->m_data.local();
    for (int comp = 0; comp < ncomps; comp++)
      {
        // BEGIN DEBUG
        /*
        array_t<2, std::complex<double> >& inputArray =
          (array_t<2, std::complex<double> >&) *a_inputArrayPtr;
        forall([inputBox](std::complex<double>& v, const point_t<2>& p)
               {
                 std::cout << "FFT on input" << p
                           << " pos = " << positionInBox(p, inputBox)
                           << " v = " << v << std::endl;
               }, inputArray);

        size_t mypos = 0;
        for (int i = inputBox.lo[0]; i <= inputBox.hi[0]; i++)
          for (int j = inputBox.lo[1]; j <= inputBox.hi[1]; j++)
            {
              point_t<2> p = point_t<2>({{i, j}});
              size_t pos = positionInBox(p, inputBox);
              std::cout << "inData(" << i << ", " << j << ") at " << pos << " or " << mypos << " is " << inData[pos] << std::endl;
              mypos++;
            }
        */
        // END DEBUG
             
        forall([inputBox, inData, comp](std::complex<double>(&v), const point_t<DIM>& p)
               {
                 if (isInBox(p, inputBox))
                   {
                     size_t pos = positionInBox(p, inputBox);
                     // v = inData[pos][comp];
                     assignComplexValue(v, inData[pos], comp);
                     // BEGIN DEBUG
                     if (false) // (real(v) != 0.)
                       {
                         std::cout << "INPUT" << p << " AT POS = " << pos
                                   << " v = " << v << std::endl;
                       }
                     // END DEBUG
                   }
                 else
                   {
                     v = 0.;
                   }
               }, fftwarray);


        forall([inputBox, inData, comp](std::complex<double>(&v), const point_t<DIM>& p)
               {
                 if (isInBox(p, inputBox))
                   {
                     size_t pos = positionInBox(p, inputBox);
                     // v = inData[pos][comp];
                     assignComplexValue(v, inData[pos], comp);
                     // BEGIN DEBUG
                     if (false) // (real(v) != 0.)
                       {
                         std::cout << "INPUT" << p << " AT POS = " << pos
                                   << " v = " << v << std::endl;
                       }
                     // END DEBUG
                   }
                 else
                   {
                     v = 0.;
                   }
               }, fftwarray);
        auto fftwdata = fftwarray.m_data.local();
        // BEGIN DEBUG
        if (false) // (comp == 5)
          {
            if (a_plan_implem.kind == allDFT)
              {
                size_t pt = 0;
                for (int i = 0; i < 8; i++)
                  for (int j = 0; j < 8; j++)
                    for (int k = 0; k < 8; k++)
                      {
                        std::complex<double> cval;
                        assignComplexValue(cval, inData[pt], comp);
                        std::cout << "DFT input "
                                  << i << " " << j << " " << k << " "
                                  << cval << std::endl;
                        pt++;
                      }
              }
          }
        // END DEBUG

        // Now do the DFT or IDFT in each dimension.
        for (int dodir = 0; dodir < DIM; dodir++)
          {
            int tfmdir = dodir;
            if (a_plan_implem.kind == allIDFT) tfmdir = DIM-1 - dodir;

            // BEGIN DEBUG
            if (false)
              {
                std::cout << "Doing transformation " << dodir
                          << " in dimension " << tfmdir
                          << " offsets " << a_plan_implem.startloop[tfmdir]
                          << " transformBox=" << transformBox
                          << std::endl;
                forall([](std::complex<double>(&v), const point_t<DIM>& p)
                       {
                         std::cout << "1D DFT input " << p << " = "
                                   << v << std::endl;
                       }, fftwarray);
              }
            // END DEBUG
            // Note that for IDFT, tfmdir order is 2, 1, 0.
            fftw_plan fftwplanHere = a_plan_implem.fftwplandims[tfmdir];
            int alignmentPlan = a_plan_implem.fftwplanalignment[tfmdir];
            // kludge: forall must be over an array, not a box.
            array_t<DIM, double> looparr(a_plan_implem.startloop[tfmdir]);
            forall([fftwdata, fftwplanHere, transformBox, alignmentPlan]
                   (double(v), const point_t<DIM>& startp)
                   {
                     fftw_complex* planArray = (fftw_complex*) fftwdata;
                     planArray += positionInBox(startp, transformBox);
                     // FFTW 3.3.8: Check that alignment matches.
                     int alignmentArray =
                       fftw_alignment_of((double *) planArray);
                     // std::cout << "alignmentArray=" << alignmentArray
                     // << "  alignmentPlan=" << alignmentPlan
                     // << std::endl;
                     assert(alignmentArray == alignmentPlan);
                     fftw_execute_dft(fftwplanHere, planArray, planArray);
                   }, looparr);
            // BEGIN DEBUG
            if (false)
              {
                forall([](std::complex<double>(&v), const point_t<DIM>& p)
                       {
                         std::cout << "1D DFT output " << p << " = "
                                   << v << std::endl;
                       }, fftwarray);
              }
            // END DEBUG
          }

        forall([outputBox, outData, comp](std::complex<double>(&v), const point_t<DIM>& p)
               {
                 if (isInBox(p, outputBox))
                   {
                     size_t pos = positionInBox(p, outputBox);
                     // outData[pos][comp] = v;
                     assignComplexValue(outData[pos], v, comp);
                     // BEGIN DEBUG
                     if (false) // (real(v) != 0.)
                       {
                         std::cout << "OUTPUT " << p << " AT POS = " << pos
                                   << " v = " << v << std::endl;
                       }
                     // END DEBUG
                   }
               }, fftwarray);
        // BEGIN DEBUG
        if (false)
          {
            size_t pt = 0;
            for (int i = outputBox.lo[0]; i <= outputBox.hi[0]; i++)
              for (int j = outputBox.lo[1]; j <= outputBox.hi[1]; j++)
                for (int k = outputBox.lo[2]; k <= outputBox.hi[2]; k++)
                  {
                    std::complex<double> cval;
                    assignComplexValue(cval, outData[pt], comp);
                    if (a_plan_implem.kind == allIDFT)
                      {
                        std::cout << "I";
                      }
                    std::cout << "DFT "
                              << comp << " output[" << i
                              << "," << j 
                              << "," << k
                              << "] = " << cval << std::endl;
                    pt++;
                  }
          }
        // END DEBUG
      }
#endif
  }
  
  template<int DIM, typename SOURCE, typename DEST>
  inline void executeTensorContraction(const array_t< DIM, SOURCE >* a_inputArray,
                                       array_t< DIM, DEST >* a_outputArray,
                                       plan_implem_t<DIM>& a_plan_implem)
  {
    std::cout << "Unimplemented executeTensorContraction" << std::endl;
  };

  template<int DIM, std::size_t C_IN, std::size_t C_OUT>
  inline void executeTensorContraction(const array_t< DIM, std::complex<double>[C_IN] >* a_inputArray,
                                       array_t< DIM, std::complex<double>[C_OUT] >* a_outputArray,
                                       plan_implem_t<DIM>& a_plan_implem)
  {
    std::cout << "Doing tensor contraction from "
              << C_IN << " components to "
              << C_OUT << std::endl;
    const std::complex<double>(*inData)[C_IN] =
      (const std::complex<double>(*)[C_IN])(a_inputArray->m_data.local());
    std::complex<double>(*outData)[C_OUT] =
      (std::complex<double>(*)[C_OUT])(a_outputArray->m_data.local());
    global_ptr<void> gptr = a_plan_implem.erasedHolder;
    const std::complex<double>(*coeffs)[C_IN][C_OUT] =
      (const std::complex<double>(*)[C_IN][C_OUT])(gptr.local());
    // Assumes a_outputArray->m_domain == a_inputArray->m_domain.
    const box_t<DIM>& transformBox = a_inputArray->m_domain;
    // FIXME: this is not the kind of kludge we should be using.
    // And note that this requires all arrays be on transformBox.
    for (size_t pt = 0; pt < transformBox.size(); pt++)
      {
        const std::complex<double>(&matrix)[C_IN][C_OUT] = coeffs[pt];
        const std::complex<double>(&v_in)[C_IN] = inData[pt];
        std::complex<double>(&v_out)[C_OUT] = outData[pt];
        for (int compout = 0; compout < C_OUT; compout++)
          {
            // v_out[compout] = 0.0;
            // std::complex<double> zeroval = std::complex<double>(0., 0.);
            // outData[pt][compout] = zeroval;
            // FIXME: This does not compile, because
            // variable-sized array type "long int" is not a valid template argument.
            // assignComplexValue(outData[pt], zeroval, compout);
            for (int compin = 0; compin < C_IN; compin++)
              {
                v_out[compout] +=
                  matrix[compin][compout] * v_in[compin];
              }
            // BEGIN DEBUG
            if (false) // (real(v_out[compout]) != 0.)
              {
                std::cout << "tensor contraction output["
                          << compout << "]["
                          << pt << "] = "
                          << v_out[compout]
                          << std::endl;
              }
            // END DEBUG
          }
      }
  }

  template<int DIM>
  inline void executeTensorContraction(const array_t< DIM, std::complex<double> >* a_inputArray,
                                       array_t< DIM, std::complex<double> >* a_outputArray,
                                       plan_implem_t<DIM>& a_plan_implem)
  {
    std::cout << "Doing scalar multiplication" << std::endl;
    const std::complex<double>* inData =
      (const std::complex<double>*)(a_inputArray->m_data.local());
    std::complex<double>* outData =
      (std::complex<double>*)(a_outputArray->m_data.local());
    global_ptr<void> gptr = a_plan_implem.erasedHolder;
    const std::complex<double>* coeffs =
      (const std::complex<double>*)(gptr.local());
    // Assumes a_outputArray->m_domain == a_inputArray->m_domain.
    const box_t<DIM>& transformBox = a_inputArray->m_domain;
    // FIXME: shouldn't this be some kind of forall?
    // And note that this requires all arrays be on transformBox.
    for (size_t pt = 0; pt < transformBox.size(); pt++)
      {
        outData[pt] = coeffs[pt] * inData[pt];
        // BEGIN DEBUG
        if (false) // (real(v) != 0.)
          {
            std:: cout << "AT " << pt
                       << " inData = " << inData[pt]
                       << " outData = " << outData[pt]
                       << std::endl;
          }
        // END DEBUG
      }
  }

  template<int DIM, typename T>
  inline void copyArray(array_t<DIM, T>& a_outputArray,
                        const array_t<DIM, T>& a_inputArray)
  {
    T* outputDataPtr = a_outputArray.m_data.local();
    const T* inputDataPtr = a_inputArray.m_data.local();
    for (size_t pt = 0; pt < a_inputArray.m_domain.size(); pt++)
      {
        assignValue(outputDataPtr[pt], inputDataPtr[pt]);
      }
  }

  template<int DIM, typename T>
  inline void executeKernel(const array_t< DIM, T >* a_inputArrayPtr,
                            array_t< DIM, T >* a_outputArrayPtr,
                            plan_implem_t<DIM>& a_plan_implem)
  {
    std::cout << "running executeKernel" << std::endl;
    auto inputArray = *((array_t< DIM, T >*) a_inputArrayPtr);
    auto outputArray = *((array_t< DIM, T >*) a_outputArrayPtr);
    copyArray(outputArray, inputArray);

    typedef std::function<void(T& inout,
                               const point_t<DIM>& a_point,
                               size_t normalize)> kp;
    kp* kernel = (kp*)(a_plan_implem.kernel.function);
    size_t normalize = a_plan_implem.normalize;
    forall([kernel, normalize](T(&v), const point_t<DIM>& p)
           {
             kernel->operator()(v, p, normalize);
           }, outputArray);
  };

  /*
  template<int DIM>
  inline void executeKernel(const array_t< DIM, std::complex<double> >* a_inputArrayPtr,
                            array_t< DIM, std::complex<double> >* a_outputArrayPtr,
                            plan_implem_t<DIM>& a_plan_implem)
  {
    auto inputArray =
      *((array_t< DIM, std::complex<double> >*) a_inputArrayPtr);
    auto outputArray =
      *((array_t< DIM, std::complex<double> >*) a_outputArrayPtr);
    auto inputDataPtr = inputArray.m_data.local();
    auto outputDataPtr = outputArray.m_data.local();
    for (size_t pt = 0; pt < inputArray.m_domain.size(); pt++)
      {
        outputDataPtr[pt] = inputDataPtr[pt];
      }

    typedef std::function<void(std::complex<double>& inout,
                               const point_t<DIM>& a_point,
                               size_t normalize)> kp;
    kp* kernel = (kp*)(a_plan_implem.kernel.function);
    size_t normalize = a_plan_implem.normalize;
    forall([kernel, normalize](std::complex<double>(&v), const point_t<DIM>& p)
           {
             kernel->operator()(v, p, normalize);
           }, outputArray);
  }

  template<int DIM>
  inline void executeKernel(const array_t< DIM, double >* a_inputArrayPtr,
                            array_t< DIM, double >* a_outputArrayPtr,
                            plan_implem_t<DIM>& a_plan_implem)
  {
    auto inputArray = *((array_t< DIM, double >*) a_inputArrayPtr);
    auto outputArray = *((array_t< DIM, double >*) a_outputArrayPtr);
    auto inputDataPtr = inputArray.m_data.local();
    auto outputDataPtr = outputArray.m_data.local();
    for (size_t pt = 0; pt < inputArray.m_domain.size(); pt++)
      {
        outputDataPtr[pt] = inputDataPtr[pt];
      }

    typedef std::function<void(double& inout,
                               const point_t<DIM>& a_point,
                               size_t normalize)> kp;
    kp* kernel = (kp*)(a_plan_implem.kernel.function);
    size_t normalize = a_plan_implem.normalize;
    forall([kernel, normalize](double(&v), const point_t<DIM>& p)
           {
             kernel->operator()(v, p, normalize);
           }, outputArray);
  }
  */

  template<int DIM, typename SOURCE, typename DEST>
  inline void export_spl(context_t a_context,
                         const plan_t<DIM, SOURCE, DEST>& a_plan,
                         std::ostream& out,
                         const std::string& a_name)
  {
    std::cout << "In export_spl on " << a_name << std::endl;
        
    planMaps<DIM, SOURCE, DEST>[a_name].first = a_plan;
    DataTypeT<SOURCE> S;
    DataTypeT<DEST>   D;
    std::cout << "a_context.m_implem->m_fftw3 == " << a_context.m_implem->m_fftw3 << std::endl;
    if (a_context.m_implem->m_fftw3)
      {
        std::cout << "setting planMaps from " << S.name() << " to " << D.name() << std::endl;
        planMaps<DIM, SOURCE, DEST>[a_name].second =
          [a_name](const array_t<DIM, SOURCE>* input,
                   array_t<DIM, DEST>* output,
                   unsigned int a_count)
          {
            std::cout << "now calling executeFFTW3" << std::endl;
            executeFFTW3(planMaps<DIM, SOURCE, DEST>[a_name].first,
                         input, output, a_count);
            handle_t rtn;
            return rtn;
          };
        initMap<DIM>[a_name] = [](){ /* any initialization FFTW3 needs can go here*/};
      }

    if(initMap<DIM>.find(a_name) == initMap<DIM>.end())
      {
        // we haven't seen this plan before and there is no init function
        planMaps<DIM, SOURCE, DEST>[a_name].second =
          [](const array_t<DIM, SOURCE>* input,
             array_t<DIM, DEST>* output,
             unsigned int a_count)
          {
            std::cerr<<"This function has not had code generated for it yet\n"
            <<"You will need to run the fftx code generator on your output object\n"
            <<"and create the source to link into your executable and relink\n"
            <<"Your executable"<<std::endl;
            handle_t rtn;
            return rtn;
          };
      }
    

    out<<a_name<<":= rec(\n"
       <<" context:= rec("
       <<"    dim:="<<DIM<<",\n"
       <<"    ompThreads:="<<a_context.m_implem->m_OMPThreads<<",\n"
       <<"    batch:="<<a_context.m_implem->m_batch<<",\n"
       <<"    MPI_COMM_WORLD:="<<a_context.m_implem->m_mpiCommWorld<<",\n"
       <<"    FFTW3:="<<a_context.m_implem->m_fftw3
       <<"  ),\n"
       <<" header:=\"\\#include \\\"fftx2.h\\\""
       <<"namespace fftx { void "<<a_name<<"_init(){ std::cout<<\\\""<<a_name<<"_init \\n\\\""
       <<";}  handle_t "<<a_name<<"(const array_t<"<<DIM<<","<<S.name()<<">* input,"
       <<"array_t<"<<DIM<<","<<D.name()<<">* output, unsigned int count) { handle_t rtn;\",\n"
       <<" registration:=\"class "<<a_name<<"_register { static bool registerFunc() {"
       <<" fftx::planMaps<"<<DIM<<","<<S.name()<<","<<D.name()<<">[\\\""<<a_name<<"\\\"].second = fftx::"<<a_name<<";"
       <<" fftx::initMaps<"<<DIM<<">[\\\""<<a_name<<"\\\"] = fftx::"<<a_name<<"_init;"
       <<" return true;} static bool registered;};"
       <<" bool "<<a_name<<"_register::registered = "<<a_name<<"_register::registerFunc();\"\n";
    int planID=0;
    for(auto& p : a_plan.m_implem)
      {
        
        out<<", plan"<<planID<<":= rec(\n"
           <<" kind:= "<<p.kind<<",\n"
           <<" input:= rec(box:="<<p.inputBox<<",  type:=\""<<p.inputType.name()<<"\", components:="<<p.inputType.components<<"),\n"
           <<" output:=rec(box:="<<p.outputBox<<",  type:=\""<<p.outputType.name()<<"\", components:="<<p.outputType.components<<"),\n"
           <<" transform:= "<<p.transformBox<<",\n"
           <<" normalize:= "<<p.normalize<<"\n";
        if(p.kernel.function != nullptr)
          {
            out<<"  ,kernel:= \""
               <<"auto& pm = planMaps<"<<DIM<<","<<S.name()<<","<<D.name()<<">[\\\""<<a_name<<"\\\"];"
               <<"auto p = pm.first.m_implem.begin();"
               <<"for(int i=0; i<"<<planID<<";i++)p++;"
               <<"typedef std::function<void("<<p.inputType.name()<<"& inout, const point_t<"<<DIM<<"> & a_point, size_t normalize)> kp;"
               <<"kp* kernel_"<<a_name<<"_"<<planID<<" = (kp*)((p->kernel.function));\")\n";
          }
        else if(p.erasedHolder.local() != nullptr)
          {
            out<<"  ,erasedHolder:=\""
              <<"auto& pm = planMaps<"<<DIM<<","<<S.name()<<","<<D.name()<<">[\\\""<<a_name<<"\\\"];"
              <<"auto p = pm.first.m_implem.begin();"
              <<"for(int i=0; i<"<<planID<<";i++)p++;"
              <<"global_ptr<void> gptr = p->erasedHolder;"
              <<"const "<<p.inputType.name()<<"* dataHolder = (const "<<p.inputType.name()<<"*)(gptr.local());"
              <<"\")\n";
          }
        else
          {
            out<<")\n";
          }
              
        planID++;
      }
    out<<",\n";
    out<<"epilogue:=\"    std::cout<<\\\"fftx_codegen execution for "<<a_name<<"\\\"<<std::endl; return rtn; } }\""
       <<");\n";

  }

   ///different user API now.  user calls trace after building a plan and prints to std::cout
  template<int DIM, typename SOURCE, typename DEST>
  void trace_g(context_t a_context,
               const plan_t<DIM, SOURCE, DEST>& a_plan, const char* name)
  {
    static const char* header_template = R"(

    #ifndef PLAN_CODEGEN_H
    #define PLAN_CODEGEN_H

    extern void init_PLAN_spiral(); 
    extern void PLAN_spiral(double* X, double* Y, double** symvar); 
    extern void destroy_PLAN_spiral();

   namespace PLAN
   {
    inline void init(){ init_PLAN_spiral();}
    inline void trace();
    inline fftx::handle_t transform(fftx::array_t<DD, S_TYPE>* source, fftx::array_t<DD, D_TYPE>* destination, double** symvar, int count)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
      for(int i=0; i<count; i++)
      {
        PLAN_spiral((double*)(source[i].m_data.local()), (double*)(destination[i].m_data.local()), symvar);
      }
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    inline void destroy(){ destroy_PLAN_spiral();}
  };

 #endif  )";

   std::string headerName = std::string(name)+std::string(".fftx.codegen.hpp");
   std::ofstream headerFile(headerName);
   DataTypeT<SOURCE> s;
   DataTypeT<DEST> d;
   std::string header_text = std::regex_replace(header_template,std::regex("PLAN"),name);
   header_text = std::regex_replace(header_text, std::regex("S_TYPE"), s.name());
   header_text = std::regex_replace(header_text, std::regex("D_TYPE"), d.name());
   header_text = std::regex_replace(header_text, std::regex("DD"), std::to_string(DIM));
   
   headerFile<<header_text<<"\n";
   headerFile.close();
    std::cout<<"\n context:= rec("
             <<"    dim:="<<DIM<<",\n"
             <<"    ompThreads:="<<a_context.m_implem->m_OMPThreads<<",\n"
             <<"    batch:="<<a_context.m_implem->m_batch<<",\n"
             <<"    MPI_COMM_WORLD:="<<a_context.m_implem->m_mpiCommWorld<<",\n"
             <<"    FFTW3:="<<a_context.m_implem->m_fftw3
             <<"  );\n";

    std::cout<<"transformList:=[];\n";
      for(auto& p : a_plan.m_implem)
        {
          //std::cout<<p.kind<<"\n";
          if(p.kind[DIM-1] == I){
            // last DIM is the batch size
            const box_t<DIM>& inputBox = p.inputBox;
            const box_t<DIM>& transformBox = p.transformBox;
            int batch = inputBox.hi[DIM-1]-inputBox.lo[DIM-1]+1;
            point_t<DIM-1> kind = p.kind.project();
            //std::cout<<"batch:"<<batch<<" kind: "<<kind<<"\n";
            if(kind == point_t<DIM-1>::Unit()*RDFT)
              {
                point_t<DIM-1> tsize = lengthsBox(transformBox).project();
                std::cout<<"transformList:=tranformList::[TTensorI(MDPRDFT("<<tsize<<"),"<<batch<<",APar, APar)];\n";
             
              }
            if(kind == point_t<DIM-1>::Unit()*IRDFT)
              {
                point_t<DIM-1> tsize = lengthsBox(transformBox).project();
                std::cout<<"transformList:=tranformList::[TTensorI(IMDPRDFT("<<tsize<<"),"<<batch<<",APar, APar)];\n";
              }
            if(kind == point_t<DIM-1>::Unit()*I)
              {
                // kernel operation
            std::cout<<"transformList:=transformList::[FData(List(  ";
              }
          }
          
        }
      
  }
  // For code generation this will map a previously generated function that has been
  // compiled into the user's source code.  Returns a null generated plan if this
  // function does not exist.

  
  template<int DIM, typename SOURCE, typename DEST>
  fftx_codegen<DIM, SOURCE, DEST> import_spl(const std::string& a_name)
  {
 
    if(initMap<DIM>.find(a_name) == initMap<DIM>.end())
      {
        std::cerr<<"plan has not had code generation created yet"<<std::endl;
        return nullptr;
      }
    if(initMap<DIM>[a_name] == (void(*)())0x1)
      {
        // program has already called init for this plan
      }
    else
      {
        initMap<DIM>[a_name](); // initialize
        initMap<DIM>[a_name] = (void(*)())0x1;
      }
    if(planMaps<DIM, SOURCE, DEST>.find(a_name) == planMaps<DIM, SOURCE, DEST>.end())
      {
        std::cerr<<"named plan does not exist in database"<<std::endl;
        return nullptr;
      }
   
    return planMaps<DIM, SOURCE, DEST>[a_name].second;
  }


  
}


namespace fftx_helper
{
  inline size_t reverseBits(size_t x, int n) {
    size_t result = 0;
    for (int i = 0; i < n; i++, x >>= 1)
      result = (result << 1) | (x & 1U);
    return result;
  }



  inline void multiply(std::complex<double>& a, const std::complex<double>& b){ a*=b;}

  template<int C>
  inline void multiply(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]*=b[i]; }
  }

  inline void assign(std::complex<double>& a, const std::complex<double>& b){ a=b;}

  template<int C>
  inline void assign(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]=b[i]; }
  }
  inline void subtract(std::complex<double>& a, const std::complex<double>& b){ a-=b; }

  template<int C>
  inline void subtract(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]-=b[i]; }
  }
  inline void increment(std::complex<double>& a, const std::complex<double>& b){ a+=b; }

  template<int C>
  inline void increment(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]+=b[i]; }
  }


  template<int BATCH, typename T, int DIR = 1>
  static void batchtransformRadix2(int n, int stride, T* dvec[])

  {  
    static std::vector<std::complex<double>> expTable;
    int levels = 0;  // Compute levels = floor(log2(n))
    for (size_t temp = n; temp > 1U; temp >>= 1)
      {
        levels++;
      }
    if (static_cast<size_t>(1U) << levels != n)
      {
        throw std::domain_error("Length is not a power of 2");
      }
  
    // Trigonometric table
    if (expTable.size() != n/2)
      {
        expTable.resize(n/2);
        // This must be int, not size_t, because we will negate it.
        for (int i = 0; i < n / 2; i++)
          {
            // std::complex<double> k = std::complex<double>(0, -(2*DIR*i)*M_PI/n);
            // std::complex<double> tw = std::exp(k);
            double th = -(2*DIR*i)*M_PI/(n*1.);
            std::complex<double> tw = std::complex<double>(cos(th), sin(th));
            expTable[i] = tw;
          }
      }
    
    // Bit-reversed addressing permutation
    for (size_t i = 0; i < n; i++)
      {
        size_t j = reverseBits(i, levels);
        // If j == i, then no change.
        // If j != i, then swap, but only if j > i, so as not to duplicate.
        if (j > i)
          {
            for(int b=0; b<BATCH; b++)
              {
                std::swap(dvec[b][i*stride], dvec[b][j*stride]);
              }
          }
      }
  
    // Cooley-Tukey decimation-in-time radix-2 FFT.
    // From algorithm iterative-fft in Wikipedia "Cooley-Tukey FFT algorithm"
    for (size_t size = 2; size <= n; size *= 2)
      {
        size_t halfsize = size / 2;
        size_t tablestep = n / size;
        for (size_t k = 0; k < n; k += size)
          {
            for (size_t j = 0; j < halfsize; j++)
              {
                size_t indkj = (k + j)*stride;
                size_t indkjhalfsize = (k + j + halfsize)*stride;
                for (int b=0; b<BATCH; b++)
                  {
                    T* vec = dvec[b];
                    T temp1;
                    assign(temp1, vec[indkjhalfsize]);
                    // std::cout << "size=" << size << " tablestep=" << tablestep << " : k=" << k << " j=" << j << " j*tablestep = " << (j*tablestep) << "\n";
                    multiply(temp1, expTable[j*tablestep]);
                    T temp2;
                    assign(temp2, vec[indkj]);
                    increment(vec[indkj], temp1);
                    assign(vec[indkjhalfsize], temp2);
                    subtract(vec[indkjhalfsize], temp1);
                  }
                // std::cout << "k=" << k << " j=" << j << "\n";
              }
          }
        if (size == n)  // Prevent overflow in 'size *= 2'
          {
            break;
          }
      }
  }
}

#endif /*  end include guard FFTX_H */
