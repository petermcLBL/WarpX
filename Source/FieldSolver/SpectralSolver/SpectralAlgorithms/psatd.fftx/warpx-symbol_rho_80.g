Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "SR";

inFields := 11;
outFields := 6;

n := 80;
# np := n + 1;

nf := n + 2;
xdim := nf/2;
ydim := n;
zdim := n;
# c := 1.0;
# ep0 := 1.0;
# invep0 := 1.0 / ep0;

t := let(name := "warpxsym_rho_80",
    symvar := var("sym", TPtr(TPtr(TReal))),
    cvar := var("PhysConst::c", TReal),
    ep0var := var("PhysConst::ep0", TReal),
    c2 := cvar^2,
    invep0 := 1.0 / ep0var,
    ix := Ind(xdim),
    iy := Ind(ydim),
    iz := Ind(zdim),

    ii := lin_idx(iz, iy, ix),
    fmkx := nth(nth(symvar, 0), ix),
    fmky := nth(nth(symvar, 1), iy),
    fmkz := nth(nth(symvar, 2), iz),
    fcv  := nth(nth(symvar, 3), ii),
    fsckv:= nth(nth(symvar, 4), ii),
    fx1v := nth(nth(symvar, 5), ii),
    fx2v := nth(nth(symvar, 6), ii),
    fx3v := nth(nth(symvar, 7), ii),
    
    # SpectralFieldIndex order:
    # 0: Ex
    # 1: Ey
    # 2: Ez
    # 3: Bx
    # 4: By
    # 5: Bz
    # 6: Jx
    # 7: Jy
    # 8: Jz
    # 9: rho_old
    #10: rho_new
    rmat := TSparseMat([outFields,inFields], [
        [0, [0, fcv],
            [4, cxpack(0, -fmkz * c2 * fsckv)],
            [5, cxpack(0, fmky * c2 * fsckv)],
            [6, -invep0 * fsckv],
            [9, cxpack(0,   fmkx * fx3v)],
            [10, cxpack(0, -fmkx * fx2v)]],
        [1, [1, fcv],
            [3, cxpack(0, fmkz * c2 * fsckv)], 
            [5, cxpack(0, -fmkx * c2 * fsckv)],
            [7, -invep0 * fsckv],
            [9, cxpack(0,   fmky * fx3v)],
            [10, cxpack(0, -fmky * fx2v)]],
        [2, [2, fcv],
            [3, cxpack(0, -fmky * c2 * fsckv)],
            [4,  cxpack(0, fmkx * c2 * fsckv)],
            [8, -invep0 * fsckv],
            [9, cxpack(0,   fmkz * fx3v)],
            [10, cxpack(0, -fmkz * fx2v)]],
    
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
        ]),

    TFCall(TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), 
        rec(fname := name, params := [symvar])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-symbol_rho_80.c", opts.prettyPrint(c));