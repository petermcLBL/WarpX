
/*
 * This code was generated by Spiral 8.2.0, www.spiral.net
 */


void init_warpxsym_norho_80() {
}

void warpxsym_norho_80(double  *Y, double  *X, double  * *sym) {
    for(int i3 = 0; i3 <= 79; i3++) {
        for(int i2 = 0; i2 <= 79; i2++) {
            for(int i1 = 0; i1 <= 40; i1++) {
                double a1316, a1317, a1318, a1319, a1320, a1321, a1322, a1323, 
                        a1324, a1325, a1326, a1327, a1328, a1329, a1330, a1331, 
                        a1332, a1333, a1334, a1335, a1336, a1337, a1338, a1339, 
                        a1340, a1341, a1342, a1343, a1344, a1345, a1346, a1347, 
                        a1348, a1349, a1350, a1351, a1352, a1353, a1354, a1355, 
                        a1356, a1357, a1358, s100, s83, s84, s85, s86, 
                        s87, s88, s89, s90, s91, s92, s93, s94, 
                        s95, s96, s97, s98, s99;
                int a1303, a1304, a1305, a1306, a1307, a1308, a1309, a1310, 
                        a1311, a1312, a1313, a1314, a1315;
                a1303 = ((6560*i3) + (82*i2) + (2*i1));
                s83 = X[a1303];
                a1304 = (a1303 + 1);
                s84 = X[a1304];
                a1305 = (a1303 + 524800);
                s85 = X[a1305];
                a1306 = (a1303 + 524801);
                s86 = X[a1306];
                a1307 = (a1303 + 1049600);
                s87 = X[a1307];
                a1308 = (a1303 + 1049601);
                s88 = X[a1308];
                a1309 = (a1303 + 1574400);
                s89 = X[a1309];
                a1310 = (a1303 + 1574401);
                s90 = X[a1310];
                a1311 = (a1303 + 2099200);
                s91 = X[a1311];
                a1312 = (a1303 + 2099201);
                s92 = X[a1312];
                a1313 = (a1303 + 2624000);
                s93 = X[a1313];
                a1314 = (a1303 + 2624001);
                s94 = X[a1314];
                s95 = X[(a1303 + 3148800)];
                s96 = X[(a1303 + 3148801)];
                s97 = X[(a1303 + 3673600)];
                s98 = X[(a1303 + 3673601)];
                s99 = X[(a1303 + 4198400)];
                s100 = X[(a1303 + 4198401)];
                a1315 = ((3280*i3) + (41*i2) + i1);
                a1316 = sym[3][a1315];
                a1317 = sym[0][i1];
                a1318 = sym[6][a1315];
                a1319 = (a1317*a1318);
                a1320 = (a1316 + (a1319*a1317));
                a1321 = sym[1][i2];
                a1322 = (a1319*a1321);
                a1323 = sym[2][i3];
                a1324 = (a1319*a1323);
                a1325 = (PhysConst::c*PhysConst::c);
                a1326 = sym[4][a1315];
                a1327 = ((a1323*a1325)*a1326);
                a1328 = ((a1321*a1325)*a1326);
                a1329 = (1.0 / PhysConst::ep0);
                a1330 = sym[7][a1315];
                a1331 = (a1317*a1330);
                a1332 = (a1329*((a1331*a1317) - a1326));
                a1333 = (a1329*(a1331*a1321));
                a1334 = (a1329*(a1331*a1323));
                a1335 = (a1321*a1318);
                a1336 = (a1335*a1317);
                a1337 = (a1316 + (a1335*a1321));
                a1338 = (a1335*a1323);
                a1339 = ((a1317*a1325)*a1326);
                a1340 = (a1321*a1330);
                a1341 = (a1329*(a1340*a1317));
                a1342 = (a1329*((a1340*a1321) - a1326));
                a1343 = (a1329*(a1340*a1323));
                a1344 = (a1323*a1318);
                a1345 = (a1344*a1317);
                a1346 = (a1344*a1321);
                a1347 = (a1316 + (a1344*a1323));
                a1348 = (a1323*a1330);
                a1349 = (a1329*(a1348*a1317));
                a1350 = (a1329*(a1348*a1321));
                a1351 = (a1329*((a1348*a1323) - a1326));
                a1352 = (a1323*a1326);
                a1353 = (a1321*a1326);
                a1354 = sym[5][a1315];
                a1355 = (a1323*a1354);
                a1356 = (a1321*a1354);
                a1357 = (a1317*a1326);
                a1358 = (a1317*a1354);
                Y[a1303] = ((((a1320*s83) + (a1322*s85) + (a1324*s87) + (a1327*s92)) - (a1328*s94)) + (a1332*s95) + (a1333*s97) + (a1334*s99));
                Y[a1304] = ((((a1320*s84) + (a1322*s86) + (a1324*s88)) - (a1327*s91)) + (a1328*s93) + (a1332*s96) + (a1333*s98) + (a1334*s100));
                Y[a1305] = ((((a1336*s83) + (a1337*s85) + (a1338*s87)) - (a1327*s90)) + (a1339*s94) + (a1341*s95) + (a1342*s97) + (a1343*s99));
                Y[a1306] = ((((a1336*s84) + (a1337*s86) + (a1338*s88) + (a1327*s89)) - (a1339*s93)) + (a1341*s96) + (a1342*s98) + (a1343*s100));
                Y[a1307] = ((((a1345*s83) + (a1346*s85) + (a1347*s87) + (a1328*s90)) - (a1339*s92)) + (a1349*s95) + (a1350*s97) + (a1351*s99));
                Y[a1308] = ((((a1345*s84) + (a1346*s86) + (a1347*s88)) - (a1328*s89)) + (a1339*s91) + (a1349*s96) + (a1350*s98) + (a1351*s100));
                Y[a1309] = ((((a1353*s88) - (a1352*s86)) + (a1316*s89) + (a1355*s98)) - (a1356*s100));
                Y[a1310] = (((((a1352*s85) - (a1353*s87)) + (a1316*s90)) - (a1355*s97)) + (a1356*s99));
                Y[a1311] = (((((a1352*s84) - (a1357*s88)) + (a1316*s91)) - (a1355*s96)) + (a1358*s100));
                Y[a1312] = ((((a1357*s87) - (a1352*s83)) + (a1316*s92) + (a1355*s95)) - (a1358*s99));
                Y[a1313] = ((((a1357*s86) - (a1353*s84)) + (a1316*s93) + (a1356*s96)) - (a1358*s98));
                Y[a1314] = (((((a1353*s83) - (a1357*s85)) + (a1316*s94)) - (a1356*s95)) + (a1358*s97));
            }
        }
    }
}
