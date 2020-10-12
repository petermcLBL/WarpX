Load(packages.fftx);

Import(packages.fftx, packages.fftx.baseline, fftx.baseline.nonterms,
    fftx.baseline.breakdown, fftx.baseline.rewrite, fftx.baseline.sigma,
    filtering, realdft);

conf := FFTXGlobals.defaultWarpXConf();
opts := FFTXGlobals.getOpts(conf);
opts.includes := [ ];
opts.buffPrefixExtra := "STEPNR";

inFields := 9;
outFields := 6;

n := 80;
np := n + 1;

nf := n + 2;
xdim := nf/2;
ydim := n;
zdim := n;

t := let(name := "warpxfullstep_norho_80",
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

    rmat := TSparseMat([outFields,inFields], [
        [0, [0, fcv / n^3 + fmkx*fx2v*fmkx / n^3],
            [1,       fmkx*fx2v*fmky / n^3],
            [2,       fmkx*fx2v*fmkz / n^3],
            [4, cxpack(0, -fmkz * c2 * fsckv / n^3)],
            [5, cxpack(0, fmky * c2 * fsckv / n^3)],
            [6, invep0 * (fmkx*fx3v*fmkx - fsckv) / n^3],
            [7, invep0 * (fmkx*fx3v*fmky) / n^3],
            [8, invep0 * (fmkx*fx3v*fmkz) / n^3]],
        [1, [0,       fmky*fx2v*fmkx / n^3],
            [1, fcv / n^3 + fmky*fx2v*fmky / n^3],
            [2,       fmky*fx2v*fmkz / n^3],
            [3, cxpack(0, fmkz * c2 * fsckv / n^3)], 
            [5, cxpack(0, -fmkx * c2 * fsckv / n^3)],
            [6, invep0 * (fmky*fx3v*fmkx) / n^3],
            [7, invep0 * (fmky*fx3v*fmky - fsckv) / n^3],
            [8, invep0 * (fmky*fx3v*fmkz) / n^3]],
        [2, [0,       fmkz*fx2v*fmkx / n^3],
            [1,       fmkz*fx2v*fmky / n^3],
            [2, fcv / n^3 + fmkz*fx2v*fmkz / n^3],
            [3, cxpack(0, -fmky * c2 * fsckv / n^3)],
            [4,  cxpack(0, fmkx * c2 * fsckv / n^3)],
            [6, invep0 * (fmkz*fx3v*fmkx) / n^3],
            [7, invep0 * (fmkz*fx3v*fmky) / n^3],
            [8, invep0 * (fmkz*fx3v*fmkz - fsckv) / n^3]],

        [3, [1, cxpack(0, fmkz * fsckv / n^3)],
            [2, cxpack(0, -fmky * fsckv / n^3)],
            [3, fcv / n^3],
            [7, cxpack(0, -fmkz * fx1v / n^3)],
            [8, cxpack(0, fmky * fx1v / n^3)]],
        [4, [0, cxpack(0, -fmkz * fsckv / n^3)],
            [2, cxpack(0, fmkx * fsckv / n^3)],
            [4, fcv / n^3],
            [6, cxpack(0, fmkz * fx1v/ n^3)],
            [8, cxpack(0, -fmkx * fx1v / n^3)]],
        [5, [0, cxpack(0, fmky * fsckv / n^3)],      
            [1, cxpack(0, -fmkx * fsckv / n^3)],
            [5, fcv / n^3],
            [6, cxpack(0, -fmky * fx1v / n^3)],
            [7, cxpack(0, fmkx * fx1v / n^3)]]
        ]),
        
    inputFab := var("inputFab", BoxND([inFields, n, n, n], TReal)),
    inputHatFab := var("inputHatFab", BoxND([inFields, n, n, nf], TReal)),
    outputHatFab := var("outputHatFab", BoxND([outFields, n, n, nf], TReal)),
    outputFab := var("outputFab", BoxND([outFields, n, n, n], TReal)),
   
    TFCall(
        TDecl(
            TDAG([
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]),
                           nth(inputFab, 0), nth(X, 0)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]),
                           nth(inputFab, 1), nth(X, 1)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]),
                           nth(inputFab, 2), nth(X, 2)),

                  TDAGNode(TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]),
                           nth(inputFab, 3), nth(X, 3)),
                  TDAGNode(TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]),
                           nth(inputFab, 4), nth(X, 4)),
                  TDAGNode(TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]),
                           nth(inputFab, 5), nth(X, 5)),
                  
                  TDAGNode(TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]),
                           nth(inputFab, 6), nth(X, 6)),
                  TDAGNode(TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]),
                           nth(inputFab, 7), nth(X, 7)),
                  TDAGNode(TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]),
                           nth(inputFab, 8), nth(X, 8)),

                  TDAGNode(TTensorI(MDPRDFT([n, n, n], -1),
                                    inFields, APar, APar),
                           inputHatFab, inputFab),
                  TDAGNode(TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)),
                           outputHatFab, inputHatFab),
                  TDAGNode(TTensorI(IMDPRDFT([n, n, n], 1),
                                    outFields, APar, APar),
                           outputFab, outputHatFab),

                  TDAGNode(TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]),
                           nth(Y, 0), nth(outputFab, 0)),
                  TDAGNode(TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]),
                           nth(Y, 1), nth(outputFab, 1)),
                  TDAGNode(TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]),
                           nth(Y, 2), nth(outputFab, 2)),
                  
                  TDAGNode(TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]),
                           nth(Y, 3), nth(outputFab, 3)),
                  TDAGNode(TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]),
                           nth(Y, 4), nth(outputFab, 4)),
                  TDAGNode(TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]),
                           nth(Y, 5), nth(outputFab, 5)),
            ]),          
            [inputFab, inputHatFab, outputHatFab, outputFab]        
        ), 
        rec(XType := TPtr(TPtr(TReal)), YType := TPtr(TPtr(TReal)),
            fname := name, params := [symvar ])
    ).withTags(opts.tags)
);

c := opts.fftxGen(t);
opts.prettyPrint(c);
PrintTo("warpx-fullstep-norho_80.c", opts.prettyPrint(c));

