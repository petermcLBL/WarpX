
comment("right now I don't know what goes in a preamble");

Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);

symvar := var("sym", TPtr(TPtr(TReal)));

