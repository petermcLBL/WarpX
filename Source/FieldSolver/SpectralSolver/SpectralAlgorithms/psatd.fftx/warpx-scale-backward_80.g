Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "SB";

# inFields := 11;
outFields := 6;

n := 80;
# np := n + 1;

nf := n + 2;
xdim := nf/2;
ydim := n;
zdim := n;

t := let(name := "warpxscale_backward_80",
    symvar := var("sym", TPtr(TPtr(TReal))),
    ix := Ind(xdim),
    iy := Ind(ydim),
    iz := Ind(zdim),

    # ii := lin_idx(iz, iy, ix),
    # xshift := nth(nth(symvar, 0), ix),
    # yshift := nth(nth(symvar, 1), iy),
    # zshift := nth(nth(symvar, 2), iz),
    xreal := nth(nth(symvar, 0), 2*ix),
    ximag := nth(nth(symvar, 0), 2*ix+1),
    yreal := nth(nth(symvar, 1), 2*iy),
    yimag := nth(nth(symvar, 1), 2*iy+1),
    zreal := nth(nth(symvar, 2), 2*iz),
    zimag := nth(nth(symvar, 2), 2*iz+1),

    xc := cxpack(xreal, ximag),
    yc := cxpack(yreal, yimag),
    zc := cxpack(zreal, zimag),

    # SpectralFieldIndex order, and nodal (1) or central (0):
    # 0: Ex  0 1 1   xshift
    # 1: Ey  1 0 1   yshift
    # 2: Ez  1 1 0   zshift
    # 3: Bx  1 0 0   yshift * zshift
    # 4: By  0 1 0   xshift * zshift
    # 5: Bz  0 0 1   xshift * yshift
    rmat := TSparseMat([outFields, outFields], [
        [0, [0, xc]],
        [1, [1, yc]],
        [2, [2, zc]],
        [3, [3,      yc * zc]],
        [4, [4, xc      * zc]],
        [5, [5, xc * yc     ]]
        ]),

    TFCall(TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), 
        rec(fname := name, params := [symvar])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-scale-backward_80.c", opts.prettyPrint(c));
