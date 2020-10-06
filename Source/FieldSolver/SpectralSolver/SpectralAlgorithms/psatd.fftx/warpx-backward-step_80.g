Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "BS";

# inFields := 11;
outFields := 6;

n := 80;
np := n + 1;

nf := n + 2;
# xdim := nf/2;
# ydim := n;
# zdim := n;

# BaseFab in Fortran array order, as in AMReX.
# BaseFab := (dims, comps, stype) -> BoxND([comps]::Reversed(dims), stype);
BaseFab := (dims, comps, stype) -> BoxND([comps]::dims, stype);

FAB := (dims, comps) -> BaseFab(dims, comps, TReal);

t := let(name := "warpxbackward_step_80",
    outputFab := var("outputFab", FAB([n, n, n], outFields)),

    TFCall(
        TDecl(
            TDAG([ # list of TDAGNode(kernel, output, input).
                  # Do 3D FFT
                  TDAGNode(TTensorI(IMDPRDFT([n, n, n], 1),
                                    outFields, APar, APar),
                           outputFab, X),
                  # Should multiply by 1/n^3 here.
                  # Copy E
                  TDAGNode(TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]),
                           nth(Y, 0), nth(outputFab, 0)),
                  TDAGNode(TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]),
                           nth(Y, 1), nth(outputFab, 1)),
                  TDAGNode(TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]),
                           nth(Y, 2), nth(outputFab, 2)),
                  # Copy B
                  TDAGNode(TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]),
                           nth(Y, 3), nth(outputFab, 3)),
                  TDAGNode(TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]),
                           nth(Y, 4), nth(outputFab, 4)),
                  TDAGNode(TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]),
                           nth(Y, 5), nth(outputFab, 5)),
                  ]),
                  [ outputFab ]
        ), 
        rec(XType := TPtr(TReal), YType := TPtr(TPtr(TReal)),
            fname := name, params := [ ])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-backward-step_80.c", opts.prettyPrint(c));
