Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "FO";

n := 80;
np := n + 1;

nf := n + 2;

t := let(name := "warpxforward_only_80",
   
    TFCall(
        TDecl(
            TDAG([ # list of TDAGNode(kernel, output, input).
                  TDAGNode(MDPRDFT([n, n, n], -1), Y, X),
                  ]),
                  [ ]
        ), 
        rec(XType := TPtr(TReal), YType := TPtr(TReal), fname := name, params := [ ])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-forward-only_80.c", opts.prettyPrint(c));
