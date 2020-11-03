#include <iostream>
#include <fstream>
#include <cmath>

#include <cufft.h>
#include <cufftXt.h>

#include <helper_cuda.h>
#include "WarpXConst.H"

// #include "psatd.fftx.source.cpp"
#include "warpx80.cu"

#define DIM 3

size_t product3(size_t* vec)
{
  return (vec[0] * vec[1] * vec[2]);
}

void readComponent(std::ifstream& is,
                   double*& data,
                   size_t len)
{
  data = new double[len];
  size_t bytes = len * sizeof(double);
  is.read((char*) data, bytes);
}

void readAll(const std::string& str,
             double**& dataPtr,
             int nfields,
             const size_t* lengths)
{
  dataPtr = new double*[nfields];
  std::ifstream is(str, std::ios::binary);
  if (!is.is_open())
    {
      std::cout << "Error: missing file " << str << std::endl;
    }
  for (int comp = 0; comp < nfields; comp++)
    {
      readComponent(is, dataPtr[comp], lengths[comp]);
    }
  is.close();
}

void deleteAll(double**& dataPtr, int nfields)
{
  for (int comp = 0; comp < nfields; comp++)
    {
      delete[] dataPtr[comp];
    }
  delete[] dataPtr;
}
               
int main(int argc, char* argv[])
{
  int nbase = 64;
  int ng = 8;
  int n = nbase + 2*ng;
  int np = n + 1;
  int nf = (n + 2)/2;

  size_t nodesAll[DIM] = {np, np, np};
  size_t nodesR2C[DIM] = {nf, n, n};
  size_t edges[DIM][DIM] = {{n, np, np},
                            {np, n, np},
                            {np, np, n}};
  size_t faces[DIM][DIM] = {{np, n, n},
                            {n, np, n},
                            {n, n, np}};
  
  const int inFields = 3*DIM + 2;
  size_t inputLength[inFields] = {
    product3(edges[0]), product3(edges[1]), product3(edges[2]), // E
    product3(faces[0]), product3(faces[1]), product3(faces[2]), // B
    product3(edges[0]), product3(edges[1]), product3(edges[2]), // J
    product3(nodesAll), product3(nodesAll) // rho_old and rho_new
  };
  double** inputPtr; //  = new double*[inFields];
  readAll("input.bin", inputPtr, inFields, inputLength);

  // spiral generated cuda code assumes in/out/symbol will be in device memory
  
  checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  cufftDoubleReal **cudain, **hostin;
  cudaMalloc     ( &cudain, sizeof(cufftDoubleReal) * inFields );
  cudaMallocHost ( &hostin, sizeof(cufftDoubleReal) * inFields );
  for (int comp = 0; comp < inFields; comp++) {
	  cudaMalloc ( &hostin[comp], sizeof(cufftDoubleReal) * inputLength[comp] );
	  cudaMemcpy ( hostin[comp], inputPtr[comp], sizeof(cufftDoubleReal) * inputLength[comp],
				   cudaMemcpyHostToDevice );
  }
  cudaMemcpy ( cudain, hostin, sizeof(cufftDoubleReal) * inFields, cudaMemcpyHostToDevice );

  const int symFields = DIM + 5;
  size_t n3DR2C = product3(nodesR2C);
  size_t symLength[symFields] =
    {nf, n, n, n3DR2C, n3DR2C, n3DR2C, n3DR2C, n3DR2C};
  double** symPtr; // = new double*[symFields];
  readAll("sym.bin", symPtr, symFields, symLength);
  
  cufftDoubleReal **cudasym, **hostsym;
  cudaMalloc     ( &cudasym, sizeof(cufftDoubleReal) * symFields );
  cudaMallocHost ( &hostsym, sizeof(cufftDoubleReal) * symFields );
  for (int comp = 0; comp < symFields; comp++) {
	  cudaMalloc ( &hostsym[comp], sizeof(cufftDoubleReal) * symLength[comp] );
	  cudaMemcpy ( hostsym[comp], symPtr[comp], sizeof(cufftDoubleReal) * symLength[comp],
				   cudaMemcpyHostToDevice );
  }
  cudaMemcpy ( cudasym, hostsym, sizeof(cufftDoubleReal) * symFields, cudaMemcpyHostToDevice );

  const int outFields = 2*DIM;
  size_t** outputDims = new size_t*[outFields];
  size_t* outputLength = new size_t[outFields];
  for (int idir = 0; idir < DIM; idir++)
    {
      outputDims[idir] = edges[idir]; // E
      outputDims[idir + DIM] = faces[idir]; // B
      outputLength[idir] = product3(outputDims[idir]);
      outputLength[idir + DIM] = product3(outputDims[idir + DIM]);
    }
  double** outputWarpXPtr; // = new double*[outFields];
  readAll("output.bin", outputWarpXPtr, outFields, outputLength);

  double** outputSpiralPtr = new double*[outFields];
  for (int comp = 0; comp < outFields; comp++)
    {
      outputSpiralPtr[comp] = new double[outputLength[comp]];
    }

  cufftDoubleReal **cudaout, **hostout;
  cudaMalloc     ( &cudaout, sizeof(cufftDoubleReal) * outFields );
  cudaMallocHost ( &hostout, sizeof(cufftDoubleReal) * outFields );
  for (int comp = 0; comp < outFields; comp++) {
	  cudaMalloc ( &hostout[comp], sizeof(cufftDoubleReal) * outputLength[comp] );
  }
  cudaMemcpy ( cudaout, hostout, sizeof(cufftDoubleReal) * outFields, cudaMemcpyHostToDevice );

//  init_psatd_spiral();
//  psatd_spiral(outputSpiralPtr, inputPtr, symPtr);
//  destroy_psatd_spiral();

  init_warpx();
  warpx(cudaout, cudain, cudasym);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
  destroy_warpx();

  // copy the output data from device memory to host
  cudaMemcpy ( hostout, cudaout, sizeof(cufftDoubleReal) * outFields, cudaMemcpyDeviceToHost );
  for (int comp = 0; comp < outFields; comp++) {
	  cudaMemcpy ( outputSpiralPtr[comp], hostout[comp],
				   sizeof(cufftDoubleReal) * outputLength[comp], cudaMemcpyDeviceToHost );
  }



  std::string names[6] = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
  double E2max = 0.;
  double diffE2max = 0.;
  double B2max = 0.;
  double diffB2max = 0.;
  for (int comp = 0; comp < outFields; comp++)
    {
      std::string name = names[comp];
      // int idir = comp % DIM;
      size_t* dims = outputDims[comp];
      double* WarpX = outputWarpXPtr[comp];
      double* Spiral = outputSpiralPtr[comp];
      double maxWarpX = 0.;
      double maxdiff = 0.;
      size_t pt = 0;
      for (int k = 0; k < dims[2]; k++)
        for (int j = 0; j < dims[1]; j++)
          for (int i = 0; i < dims[0]; i++)
            {
              // We need to compare the valid parts of the output only.
              if ((k >= ng) && (k < dims[2]-ng) &&
                  (j >= ng) && (j < dims[1]-ng) &&
                  (i >= ng) && (i < dims[0]-ng))
                {
                  double absWarpX = std::abs(WarpX[pt]);
                  if (absWarpX > maxWarpX)
                    {
                      maxWarpX = absWarpX;
                    }
                  double diff = std::abs(WarpX[pt] - Spiral[pt]);
                  if (diff > maxdiff)
                    {
                      maxdiff = diff;
                    }
                }
              pt++;
            }
      std::cout << "|diff(" << name << ")| <= " << maxdiff
                << " of |" << name << "| <= " << maxWarpX
                << " relative " << (maxdiff/maxWarpX)
                << std::endl;
      if (comp < DIM)
        {
          E2max += maxWarpX * maxWarpX;
          diffE2max += maxdiff * maxdiff;
        }
      else
        {
          B2max += maxWarpX * maxWarpX;
          diffB2max += maxdiff * maxdiff;
        }
    }

  double diffEmax = std::sqrt(diffE2max);
  double diffBmax = std::sqrt(diffB2max);
  double Emax = std::sqrt(E2max);
  double Bmax = std::sqrt(B2max);
  double const c2 = PhysConst::c * PhysConst::c;
  double compositeEmax = std::sqrt(E2max + c2*B2max);
  double compositeBmax = std::sqrt(E2max/c2 + B2max);

  std::cout << "||diff(E)|| <= " << diffEmax
            << " of ||E|| <= " << Emax
            << " relative " << (diffEmax / Emax)
            << std::endl;
  std::cout << "||diff(E)|| <= " << diffEmax
            << " of sqrt(||E||^2 + c^2*||B||^2) <= " << compositeEmax
            << " relative " << (diffEmax / compositeEmax)
            << std::endl;

  std::cout << "||diff(B)|| <= " << diffBmax
            << " of ||B|| <= " << Bmax
            << " relative " << (diffBmax / Bmax)
            << std::endl;
  std::cout << "||diff(B)|| <= " << diffBmax
            << " of sqrt(||E||^2/c^2 + ||B||^2) <= " << compositeBmax
            << " relative " << (diffBmax / compositeBmax)
            << std::endl;
  
  deleteAll(inputPtr, inFields);
  deleteAll(symPtr, symFields);
  deleteAll(outputWarpXPtr, outFields);
  deleteAll(outputSpiralPtr, outFields);
}
