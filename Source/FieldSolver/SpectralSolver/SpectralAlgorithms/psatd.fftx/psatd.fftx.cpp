// make

#include "fftx3.hpp"
#include <cstdio>
#include <cassert>

using namespace fftx;

#include "defineArrays.hpp"

static const char* CONTRACTION = R"(
    ix := Ind(xdim);
    iy := Ind(ydim);
    iz := Ind(zdim);
    c:=1.0;
    ep0 := 1.0;
    invep0 := 1.0 / ep0;
    ii := lin_idx(iz, iy, ix);
    fmkx := nth(nth(symvar, 0), ix);
    fmky := nth(nth(symvar, 1), iy);
    fmkz := nth(nth(symvar, 2), iz);
    fcv := nth(nth(symvar, 3), ii);
    fsckv := nth(nth(symvar, 4), ii);
    fx1v := nth(nth(symvar, 5), ii);
    fx2v := nth(nth(symvar, 6), ii);
    fx3v := nth(nth(symvar, 7), ii);

    rmat := TSparseMat([6,11], [
        [0, [0, fcv],
            [4, cxpack(0, -fmkz * c^2 * fsckv)],
            [5, cxpack(0, fmky * c^2 * fsckv)],
            [6, -invep0 * fsckv],
            [9, cxpack(0, -fmkx * fx2v)],
            [10, cxpack(0, fmkx * fx3v)]],
        [1, [1, fcv],
            [3, cxpack(0, fmkz * c^2 * fsckv)], 
            [5, cxpack(0, -fmkx * c^2 * fsckv)],
            [7, -invep0 * fsckv],
            [9, cxpack(0, -fmky * fx2v)],
            [10, cxpack(0, fmky * fx3v)]],
        [2, [2, fcv],
            [3, cxpack(0, -fmky * c^2 * fsckv)],
            [4,  cxpack(0, fmkx * c^2 * fsckv)],
            [8, -invep0 * fsckv],
            [9, cxpack(0, -fmkz * fx2v)],
            [10, cxpack(0, fmkz * fx3v)]],
    
        [3, [1, cxpack(0, fmkz * fsckv)],
            [2, cxpack(0, -fmky * fsckv)],
            [3, fcv],
            [7, cxpack(0, -fmkz * fx1v)],
            [8, cxpack(0, fmky * fx1v)]],
        [4, [0, cxpack(0, -fmkz * fsckv)],
            [2, cxpack(0, fmkx * fsckv)],
            [4, fcv],
            [6, cxpack(0, fmkz * fx1v)],
            [8, cxpack(0, -fmkx * fx1v)]],
        [5, [0, cxpack(0, fmky * fsckv)],      
            [1, cxpack(0, -fmkx * fsckv)],
            [5, fcv],
            [6, cxpack(0, -fmky * fx1v)],
            [7, cxpack(0, fmkx * fx1v)]]
        ]);)";

void trace(const char* name)
{

  int xdim = (nx+2)/2;
  int ydim = ny;
  int zdim = nz;


  std::array<array_t<3,double>,11> inputs;
  std::array<array_t<4, double>,4> bigBoxes;
  std::array<array_t<3,double>,6>  outputs;
  std::array<array_t<3,double>,8>  symvars;
  
  defineArrays(inputs, outputs, symvars);
  defineBigBoxes(bigBoxes);
  std::array<array_t<3,double>,11> bigIn;
  for(int i=0; i<11; i++)
    {bigIn[i] = nth(bigBoxes[0],i);}
  std::array<array_t<3,double>,6> bigOut;
  for(int i=0; i<6; i++)
    {bigOut[i] = nth(bigBoxes[3],i);}

  setInputs(inputs);
  setOutputs(outputs);
  
  std::string contraction = std::regex_replace(CONTRACTION,std::regex("xdim"),std::to_string(xdim));
  contraction = std::regex_replace(contraction,std::regex("ydim"),std::to_string(ydim));
  contraction = std::regex_replace(contraction,std::regex("zdim"),std::to_string(zdim));
  contraction = std::regex_replace(contraction,std::regex("fmkx"),"var_"+std::to_string(symvars[0].id()));
  contraction = std::regex_replace(contraction,std::regex("fmky"),"var_"+std::to_string(symvars[1].id()));
  contraction = std::regex_replace(contraction,std::regex("fmkz"),"var_"+std::to_string(symvars[2].id()));
  contraction = std::regex_replace(contraction,std::regex("fcv"),"var_"+std::to_string(symvars[3].id()));
  contraction = std::regex_replace(contraction,std::regex("fsckv"),"var_"+std::to_string(symvars[4].id()));
  contraction = std::regex_replace(contraction,std::regex("fx1v"),"var_"+std::to_string(symvars[5].id()));
  contraction = std::regex_replace(contraction,std::regex("fx2v"),"var_"+std::to_string(symvars[6].id()));
  contraction = std::regex_replace(contraction,std::regex("fx3v"),"var_"+std::to_string(symvars[7].id()));


  rawScript(contraction);

  openDAG();
  resample({{0,0,-0.5}},bigIn[0],inputs[0]);
  resample({{0,-0.5,0}},bigIn[1],inputs[1]);
  resample({{-0.5,0,0}},bigIn[2],inputs[2]);

  resample({{-0.5,-0.5,0}},bigIn[3],inputs[3]);
  resample({{-0.5,0,-0.5}},bigIn[4],inputs[4]);
  resample({{0,-0.5,-0.5}},bigIn[5],inputs[5]);

  resample({{0,0,-0.5}},bigIn[6],inputs[6]);
  resample({{0,-0.5,0}},bigIn[7],inputs[7]);
  resample({{-0.5,0,0}},bigIn[8],inputs[8]);

  copy(bigIn[9],inputs[9]);
  copy(bigIn[10],inputs[10]);


  MDPRDFT(bigBoxes[0].m_domain.extents().projectC(), 11, bigBoxes[1], bigBoxes[0]);
  rawScript("    TDAGNode(TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)),var_"+std::to_string(bigBoxes[2].id())+", var_"+std::to_string(bigBoxes[1].id())+"),\n");

  IMDPRDFT(bigBoxes[0].m_domain.extents().projectC(), 6, bigBoxes[3], bigBoxes[2]);

  resample({{0,0,0.5}}, outputs[0], bigOut[0]);
  resample({{0,0.5,0}}, outputs[1], bigOut[1]);
  resample({{0.5,0,0}}, outputs[2], bigOut[2]);

  resample({{0.5,0.5,0}}, outputs[3], bigOut[3]);
  resample({{0.5,0,0.5}}, outputs[4], bigOut[4]);
  resample({{0,0.5,0.5}}, outputs[5], bigOut[5]);

  closeDAG(bigBoxes, name);
  
}
