DESCRIPTION:

The runspiral program in this directory runs the Spiral-generated C
code for a PSATD step and compares its results with those from WarpX.

Specifically, runspiral reads in binary files produced by WarpX for
one step of PSATD -- input arrays, symbol arrays, and output arrays --
and uses the input arrays and symbol arrays to compute new output
arrays with the Spiral-generated C code built from FFTX.  The
runspiral program then finds norms of differences between the two sets
of output arrays.

BUILDING:

Building runspiral requires the source file psatd.fftx.source.cpp .
This can be either copied or linked from the psatd.fftx build
directory:
cp ../Source/FieldSolver/SpectralSolver/SpectralAlgorithms/psatd.fftx/build/psatd.fftx.source.cpp .
or
ln -s ../Source/FieldSolver/SpectralSolver/SpectralAlgorithms/psatd.fftx/build/psatd.fftx.source.cpp .

To build:
g++ -o runspiral runspiral.cpp

RUNNING:

Running runspiral requires binary files containing the input, symbol,
and output arrays from WarpX.  To get these, run the WarpX
application in the myexamples directory, and it will write out these
arrays to the files input.bin, sym.bin, and output.bin .  So then in
the present directory, copy or link those files here:
cp ../myexamples/sym.bin .
cp ../myexamples/input.bin .
cp ../myexamples/output.bin .
or
ln -s ../myexamples/sym.bin .
ln -s ../myexamples/input.bin .
ln -s ../myexamples/output.bin .

Then to run:
./runspiral
