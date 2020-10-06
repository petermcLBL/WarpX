Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "FS";

inFields := 11;
# outFields := 6;

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

t := let(name := "warpxforward_step_80",
    inputFab := var("inputFab", FAB([n, n, n], inFields)),
    # outputFab := var("outputFab", FAB([nf, n, n], inFields)),

    TFCall(
        TDecl(
            TDAG([ # list of TDAGNode(kernel, output, input).
                  # Copy E
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]),
                           nth(inputFab, 0), nth(X, 0)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]),
                           nth(inputFab, 1), nth(X, 1)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]),
                           nth(inputFab, 2), nth(X, 2)),
                  # Copy B
                  TDAGNode(TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]),
                           nth(inputFab, 3), nth(X, 3)),
                  TDAGNode(TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]),
                           nth(inputFab, 4), nth(X, 4)),
                  TDAGNode(TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]),
                           nth(inputFab, 5), nth(X, 5)),
                  # Copy J
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]),
                           nth(inputFab, 6), nth(X, 6)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]),
                           nth(inputFab, 7), nth(X, 7)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]),
                           nth(inputFab, 8), nth(X, 8)),
                  # Copy rho
                  TDAGNode(TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]),
                           nth(inputFab, 9), nth(X, 9)),
                  TDAGNode(TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]),
                           nth(inputFab, 10), nth(X, 10)),

                  # Do 3D FFT
                  TDAGNode(TTensorI(MDPRDFT([n, n, n], -1),
                                    inFields, APar, APar),
                           Y, inputFab), # was outputFab instead of Y

                  # # Copy E
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 0), nth(outputFab, 0)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 1), nth(outputFab, 1)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 2), nth(outputFab, 2)),
                  # # Copy B
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 3), nth(outputFab, 3)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 4), nth(outputFab, 4)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 5), nth(outputFab, 5)),
                  # # Copy J
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 6), nth(outputFab, 6)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 7), nth(outputFab, 7)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 8), nth(outputFab, 8)),
                  # # Copy rho
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 9), nth(outputFab, 9)),
                  # TDAGNode(TScat(fBox([nf, n, n])),
                  #          nth(Y, 10), nth(outputFab, 10)),
                  ]),
                  [ inputFab ]
        ), 
        rec(XType := TPtr(TPtr(TReal)), YType := TPtr(TReal),
            fname := name, params := [ ])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-forward-step_80.c", opts.prettyPrint(c));
