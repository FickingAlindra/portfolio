from __future__ import annotations

import io
import json
import math
import subprocess
from dataclasses import dataclass, asdict

import numpy as np
import logging
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import ndimage, stats


@dataclass
class Setup:
    anchor: tuple = (112.55, -2.35)
    anchors: tuple = ((113.0, -2.0), (110.6, -1.8), (140.4, -6.5))
    search_km: float = 120
    center: tuple | None = None
    window_km: tuple = (15, 10)
    years: tuple = (2010, 2026)
    res: int = 30
    bare_nbr: float = 0.30
    cloud_blue: float = 0.12
    min_obs_bare: int = 2
    min_regrowth: float = 0.15
    regrowth_level: float = 0.40
    regrowth_years: int = 5
    min_obs_years: int = 10
    mmu_ha: float = 2.0
    snic_size: int = 12
    s2_cs: float = 0.6
    palm_share: float = 0.5
    author: str = 'Ficking Alindra Wijayadarma'
    dpi: int = 200


SENSORS = [('LANDSAT/LT05/C02/T1_L2', False), ('LANDSAT/LE07/C02/T1_L2', False),
           ('LANDSAT/LC08/C02/T1_L2', True), ('LANDSAT/LC09/C02/T1_L2', True)]
TM_BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
OLI_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
ROY_SLOPE = [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]
ROY_ITCP = [-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]
HANSEN = 'UMD/hansen/global_forest_change_2025_v1_13'
AGE_CLASSES = [(0, 3, '0–3 yr', 'Immature, not yet harvested'), (4, 8, '4–8 yr', 'Young, yield rising'),
               (9, 18, '9–18 yr', 'Prime yield'), (19, 25, '19–25 yr', 'Old, yield falling'),
               (26, 999, '26+ yr', 'Past economic life')]
OLD, PLANTED, RECENT, UNCLEAR = 1, 2, 3, 4


class UTM:
    a, f, k0 = 6378137.0, 1/298.257223563, 0.9996

    def __init__(self, zone: int, south: bool):
        self.zone, self.south = zone, south
        self.e2 = self.f*(2-self.f); self.ep2 = self.e2/(1-self.e2)
        self.epsg = (32700 if south else 32600)+zone

    @classmethod
    def at(cls, lon, lat): return cls(int((lon+180)//6)+1, lat < 0)

    def fwd(self, lon, lat):
        e2, ep2, a, k0 = self.e2, self.ep2, self.a, self.k0
        phi = np.radians(lat); lam = np.radians(lon); lam0 = math.radians(self.zone*6-183)
        N = a/np.sqrt(1-e2*np.sin(phi)**2); T = np.tan(phi)**2; C = ep2*np.cos(phi)**2; A = np.cos(phi)*(lam-lam0)
        M = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi-(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*phi)
               +(15*e2**2/256+45*e2**3/1024)*np.sin(4*phi)-(35*e2**3/3072)*np.sin(6*phi))
        x = k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T**2+72*C-58*ep2)*A**5/120)+500000
        y = k0*(M+N*np.tan(phi)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24+(61-58*T+T**2+600*C-330*ep2)*A**6/720))
        return x, (y+1e7 if self.south else y)

    def inv(self, x, y):
        lon, lat = self.zone*6-183.0, (-5.0 if self.south else 5.0)
        for _ in range(40):
            x0, y0 = self.fwd(lon, lat); d = 1e-7
            x1, y1 = self.fwd(lon+d, lat); x2, y2 = self.fwd(lon, lat+d)
            J = np.array([[(x1-x0)/d, (x2-x0)/d], [(y1-y0)/d, (y2-y0)/d]])
            step = np.linalg.solve(J, [x-x0, y-y0]); lon += step[0]; lat += step[1]
            if abs(step).max() < 1e-11: break
        return float(lon), float(lat)


@dataclass
class Grid:
    utm: UTM
    x0: float
    y1: float
    res: float
    nx: int
    ny: int

    @property
    def extent(self): return [self.x0, self.x0+self.nx*self.res, self.y1-self.ny*self.res, self.y1]

    @classmethod
    def box(cls, utm, lon, lat, w_km, h_km, res):
        cx, cy = utm.fwd(lon, lat); nx, ny = int(w_km*1000/res), int(h_km*1000/res)
        x0 = math.floor((cx-nx*res/2)/res)*res; y1 = math.ceil((cy+ny*res/2)/res)*res
        return cls(utm, x0, y1, res, nx, ny)

    def finer(self, k): return Grid(self.utm, self.x0, self.y1, self.res/k, self.nx*k, self.ny*k)

    def rect(self, ee):
        e = self.extent
        return ee.Geometry.Rectangle([e[0], e[2], e[1], e[3]], 'EPSG:%d' % self.utm.epsg, False)

    def pixel_lonlat(self, r, c): return self.utm.inv(self.x0+(c+.5)*self.res, self.y1-(r+.5)*self.res)


def pull(ee, image, grid: Grid, bands, dtype=np.float32, tile=256):
    out = {b: np.zeros((grid.ny, grid.nx), dtype) for b in bands}
    for r0 in range(0, grid.ny, tile):
        for c0 in range(0, grid.nx, tile):
            h, w = min(tile, grid.ny-r0), min(tile, grid.nx-c0)
            req = {'expression': image, 'fileFormat': 'NUMPY_NDARRAY',
                   'grid': {'dimensions': {'width': w, 'height': h}, 'crsCode': 'EPSG:%d' % grid.utm.epsg,
                            'affineTransform': {'scaleX': grid.res, 'shearX': 0, 'translateX': grid.x0+c0*grid.res,
                                                'shearY': 0, 'scaleY': -grid.res, 'translateY': grid.y1-r0*grid.res}}}
            for attempt in range(3):
                try:
                    arr = ee.data.computePixels(req); break
                except Exception as err:
                    msg = str(err).lower()
                    if tile > 32 and any(k in msg for k in ('memory', 'too large', 'timed out', 'limit', 'too many', 'deadline')):
                        return pull(ee, image, grid, bands, dtype, tile//2)
                    if attempt == 2: raise
            for b in bands: out[b][r0:r0+h, c0:c0+w] = np.asarray(arr[b])
    return out




class PalmAge:
    def __init__(self, ee, setup: Setup):
        self.ee, self.s = ee, setup
        self.years = list(range(setup.years[0], setup.years[1]+1))

    def say(self, msg): print(msg, flush=True)

    def palm_layers(self):
        col = self.ee.ImageCollection('BIOPAMA/GlobalOilPalm/v1_2')
        return col.select('classification').mosaic(), col.select('planting_year').mosaic()

    def landsat(self, region, d0, d1):
        ee = self.ee; out = None
        slope, itcp = ee.Image.constant(ROY_SLOPE), ee.Image.constant(ROY_ITCP)
        for cid, oli in SENSORS:
            def prep(im, oli=oli):
                clear = im.select('QA_PIXEL').bitwiseAnd(31).eq(0).And(im.select('QA_RADSAT').eq(0))
                sr = im.select(OLI_BANDS if oli else TM_BANDS, NAMES).multiply(0.0000275).add(-0.2)
                if oli: sr = sr.subtract(itcp).divide(slope).rename(NAMES)
                ok = clear.And(sr.reduce(ee.Reducer.min()).gt(0)).And(sr.reduce(ee.Reducer.max()).lt(1))
                return ee.Image(sr.updateMask(ok).copyProperties(im, ['system:time_start']))
            col = (ee.ImageCollection(cid).filterBounds(region).filterDate(d0, d1)
                   .filter(ee.Filter.lt('CLOUD_COVER', 80)).map(prep))
            out = col if out is None else out.merge(col)
        return out

    def sentinel2(self, region, d0, d1):
        ee = self.ee
        cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(d0, d1)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).linkCollection(cs, ['cs_cdf']))
        return col.map(lambda im: im.select(['B2', 'B3', 'B4', 'B8', 'B11']).updateMask(im.select('cs_cdf').gte(self.s.s2_cs)).divide(10000)).median()

    def haze_free(self, im):
        b = im.select('blue')
        return b.lte(self.s.cloud_blue).Or(im.select('swir1').gt(b.multiply(1.3)))

    def year_cut(self, y): return '%d-01-01' % y, '%d-01-01' % (y+1)

    def annual_stack(self, col):
        ee, s = self.ee, self.s
        nil = ee.Image.constant(0).toFloat().rename('nbr').updateMask(ee.Image.constant(0))
        red = ee.Reducer.percentile([25]).combine(ee.Reducer.median(), sharedInputs=True).combine(ee.Reducer.count(), sharedInputs=True)
        out = []
        for y in self.years:
            c = col.filterDate(*self.year_cut(y)).map(
                lambda im: im.normalizedDifference(['nir', 'swir2']).rename('nbr').updateMask(self.haze_free(im)))
            r = ee.ImageCollection([nil]).merge(c).reduce(red)
            out += [r.select('nbr_p25').multiply(1000).unmask(-9999).rename('q%d' % y),
                    r.select('nbr_median').multiply(1000).unmask(-9999).rename('n%d' % y),
                    r.select('nbr_count').unmask(0).rename('c%d' % y)]
        return ee.Image.cat(out).toShort()

    def find_site(self):
        ee, s = self.ee, self.s; y0 = s.years[0]
        cls, py = self.palm_layers()
        p = cls.gt(0).unmask(0); yy = py.unmask(0); tc = ee.Image(HANSEN).select('treecover2000').unmask(0)
        edges = np.linspace(y0+1, 2022, 4).round().astype(int)
        bins = [(int(edges[i]), int(edges[i+1])-1 if i < len(edges)-2 else 2022) for i in range(len(edges)-1)]
        img = ee.Image.cat([p.rename('palm'), cls.eq(1).unmask(0).rename('ind'), tc.updateMask(p).divide(100).rename('tc')]
                           + [p.And(yy.gte(a)).And(yy.lte(b)).rename('b%d' % i) for i, (a, b) in enumerate(bins)])
        best = []
        for anc in (s.anchors or (s.anchor,)):
            utm = UTM.at(*anc); cx, cy = utm.fwd(*anc)
            w, h, half, step = s.window_km[0]*1000, s.window_km[1]*1000, s.search_km*500, 5000
            feats = []
            for x in np.arange(cx-half+w/2, cx+half-w/2+1, step):
                for y in np.arange(cy-half+h/2, cy+half-h/2+1, step):
                    r = ee.Geometry.Rectangle([x-w/2, y-h/2, x+w/2, y+h/2], 'EPSG:%d' % utm.epsg, False)
                    feats.append(ee.Feature(r, {'x': float(x), 'y': float(y)}))
            rows = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.mean(), scale=300, crs='EPSG:%d' % utm.epsg).getInfo()['features']
            for f in rows:
                q = f['properties']; palm, ind, tc0 = q.get('palm') or 0, q.get('ind') or 0, q.get('tc') or 0
                sh = np.array([q.get('b%d' % i) or 0 for i in range(len(bins))]); post = sh.sum()/max(palm, 1e-9); sh = sh/max(sh.sum(), 1e-9)
                ent = -sum(v*math.log(v) for v in sh if v > 0)/math.log(len(bins))
                score = palm*ind*min(post, 1)*ent*(.5+.5*min(tc0/.6, 1))
                if not (palm >= .6 and ind >= .6 and post >= .6): score *= .01
                lon, lat = utm.inv(q['x'], q['y'])
                best.append((score, palm, post, sh, lon, lat))
        best.sort(key=lambda t: -t[0])
        self.say('      best windows (palm · planted after %d · %s):' % (y0, ' / '.join('%d–%02d' % (a, b % 100) for a, b in bins)))
        for sc, palm, post, sh, lon, lat in best[:3]:
            self.say('      %.4f, %.4f   %.0f%% · %.0f%% · %s%%' % (lon, lat, palm*100, post*100, ' / '.join('%.0f' % v for v in sh*100)))
        if not best or best[0][0] == 0:
            raise RuntimeError('No oil palm near the anchors; change Setup.anchors.')
        if best[0][1] < .6 or best[0][2] < .6:
            self.say('      PERINGATAN: tidak ada jendela dengan ≥ 60%% sawit yang ditanam setelah %d; yang terbaik dipakai.' % y0)
        return best[0][4], best[0][5]

    def run(self):
        ee, s = self.ee, self.s
        y0, y1 = s.years
        self.say('[1/6] Choosing the window ...')
        lon, lat = s.center or self.find_site()
        utm = UTM.at(lon, lat); g = Grid.box(utm, lon, lat, *s.window_km, s.res); self.g = g
        region = g.rect(ee); self.region = region
        try:
            adm = ee.FeatureCollection('FAO/GAUL/2015/level2').filterBounds(ee.Geometry.Point([lon, lat])).first().getInfo()['properties']
            place = {'adm2': adm.get('ADM2_NAME'), 'adm1': adm.get('ADM1_NAME')}
        except Exception:
            place = {}
        self.say('      %.4f, %.4f  %s' % (lon, lat, ', '.join(v for v in place.values() if v)))

        self.say('[2/6] Annual Landsat NBR, %d–%d ...' % (y0, y1))
        col = self.landsat(region, '%d-01-01' % y0, '%d-01-01' % (y1+1)); self.col = col
        names = [k % y for y in self.years for k in ('q%d', 'n%d', 'c%d')]
        Y = pull(ee, self.annual_stack(col), g, names, np.int16)
        N = np.stack([Y['n%d' % y] for y in self.years]).astype(np.float32)/1000
        B = np.stack([Y['q%d' % y] for y in self.years]).astype(np.float32)/1000
        C = np.stack([Y['c%d' % y] for y in self.years]).astype(np.int16)
        miss = (C == 0) | (N < -9)
        N[miss] = np.nan; B[miss] = np.nan; del Y
        self.N, self.B, self.C = N, B, C

        self.say('[3/6] Reference layers (BIOPAMA, Hansen, GEDI) ...')
        cls, ref = self.palm_layers()
        han = ee.Image(HANSEN)

        def gq(im):
            ok = im.select('quality_flag').eq(1).And(im.select('degrade_flag').eq(0)).And(im.select('sensitivity').gt(0.95))
            yr = ee.Image(ee.Number(ee.Date(im.get('system:time_start')).get('year'))).toFloat()
            return im.select('rh98').toFloat().addBands(yr.rename('gy')).updateMask(ok)
        gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(region).map(gq).sort('system:time_start').mosaic()
        recent = col.filterDate('%d-01-01' % (y1-1), '%d-01-01' % (y1+1)).select(['red', 'green', 'blue']).median()
        stack = ee.Image.cat([
            ref.unmask(0).rename('ref'), han.select('lossyear').unmask(0).rename('loss'),
            gedi.select('rh98').unmask(-1).rename('gedi'), gedi.select('gy').unmask(0).rename('gy'),
            recent.unmask(0)]).toFloat()
        A = pull(ee, stack, g, ['ref', 'loss', 'gedi', 'gy', 'red', 'green', 'blue'], tile=512)
        f = g.finer(3)
        K = pull(ee, cls.unmask(0).rename('cls').toByte(), f, ['cls'], np.uint8, tile=512)['cls']
        self.K10 = K
        self.say('      Sentinel-2 10 m composite and block segmentation ...')
        try:
            s2 = self.sentinel2(region, '%d-01-01' % (y1-1), '%d-12-31' % y1)
            seg = ee.Algorithms.Image.Segmentation.SNIC(image=s2.select(['B4', 'B8', 'B11']), size=s.snic_size, compactness=1,
                                                        connectivity=8, neighborhoodSize=4*s.snic_size)
            img = ee.Image.cat([seg.select('clusters').unmask(-1).toFloat(), s2.select(['B4', 'B3', 'B2']).unmask(0).toFloat()])
            S10 = pull(ee, img, f, ['clusters', 'B4', 'B3', 'B2'], tile=256)
            self.S10 = S10
        except Exception as err:
            self.say('      Sentinel-2 segmentation skipped: %s' % str(err)[:120]); self.S10 = None
        blk = K.reshape(g.ny, 3, g.nx, 3)
        A['palm'] = (blk > 0).mean((1, 3)); A['ind'] = (blk == 1).mean((1, 3))
        self.A = A

        self.say('[4/6] Landsat availability per year ...')
        nsc = ee.List([col.filterDate(*self.year_cut(y)).size() for y in self.years]).getInfo()
        last = ee.Date(col.aggregate_max('system:time_start')).format('YYYY-MM-dd').getInfo()
        self.avail = {y: {'coverage': float((C[i] > 0).mean()), 'scenes': int(n), 'median_obs': float(np.median(C[i]))}
                      for i, (y, n) in enumerate(zip(self.years, nsc))}

        self.say('[5/6] Reading the planting year of every pixel ...')
        R = self.classify()
        R.update(center=[lon, lat], place=place, window_km=list(s.window_km), res=s.res, latest_image=last,
                 availability={str(k): v for k, v in self.avail.items()}, setup=asdict(s), utm_epsg=utm.epsg)
        R['examples'] = self.examples()
        self.say('[6/6] Locator map ...')
        box = ee.Geometry.Rectangle([94, -11.5, 142, 7.5])
        try:
            self.inset = (ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filterBounds(box)
                          .map(lambda ft: ft.setGeometry(ft.geometry().intersection(box, 5000).simplify(8000)))).getInfo()
        except Exception:
            self.inset = None
        pt = ee.Geometry.Point([lon, lat])
        try:
            pr = ee.Feature(ee.FeatureCollection('FAO/GAUL/2015/level1').filterBounds(pt).first())
            self.prov = pr.geometry().simplify(1500).getInfo()
            self.regencies = (ee.FeatureCollection('FAO/GAUL/2015/level2').filterBounds(pr.geometry())
                              .map(lambda ft: ft.setGeometry(ft.geometry().simplify(1500)))).getInfo()
        except Exception:
            self.prov, self.regencies = None, None
        for q in R['validation'].get('qc', []): self.say('  PERINGATAN: ' + q)
        if not R['validation'].get('qc'): self.say('  Kontrol kualitas: semua pemeriksaan lulus.')
        self.R = R
        return R

    def read_years(self):
        s, N, B, C = self.s, self.N, self.B, self.C
        T, ny, nx = N.shape
        valid = ~np.isnan(N)
        n9, b9 = np.nan_to_num(N, nan=9), np.nan_to_num(B, nan=9)
        bare = valid & (((C >= 3) & (b9 <= s.bare_nbr) & (n9 <= s.bare_nbr+.15)) | ((C >= s.min_obs_bare) & (n9 <= s.bare_nbr)))
        has = bare.any(0)
        last = np.where(has, T-1-np.argmax(bare[::-1], 0), -1)
        active = has.copy(); best_t = last.copy(); best_v = np.full((ny, nx), 9.0, np.float32)
        runs = np.zeros((ny, nx), np.int16); gens = np.zeros((ny, nx), np.int16); prev_bare = np.zeros((ny, nx), bool); prev_v = np.full((ny, nx), np.nan, np.float32)
        for t in range(T):
            start = bare[t] & ~prev_bare & (np.nan_to_num(prev_v, nan=0) >= .5)
            gens += start.astype(np.int16); runs += (bare[t] & ~prev_bare).astype(np.int16)
            prev_bare = np.where(valid[t], bare[t], prev_bare); prev_v = np.where(valid[t], N[t], prev_v)
        for t in range(T-1, -1, -1):
            here = has & (t <= last)
            active &= ~(here & valid[t] & ~bare[t])
            take = here & active & bare[t] & (N[t] < best_v)
            best_t = np.where(take, t, best_t); best_v = np.where(take, N[t], best_v)
        post = np.full((ny, nx), -9.0, np.float32)
        for k in range(1, s.regrowth_years+1):
            idx = np.clip(last+k, 0, T-1)
            v = np.take_along_axis(N, idx[None], 0)[0]
            ok = has & (last+k <= T-1) & ~np.isnan(v)
            post = np.where(ok, np.maximum(post, v), post)
        at_last = np.take_along_axis(N, np.clip(last, 0, T-1)[None], 0)[0]
        regrew = (post >= s.regrowth_level) & (post-np.nan_to_num(at_last, nan=9) >= s.min_regrowth)
        return dict(bare=bare, has=has, last=last, plant_t=best_t, regrew=regrew, gens=gens, runs=runs, nvalid=valid.sum(0))

    def classify(self):
        A, s = self.A, self.s; y0, y1 = s.years
        Z = self.read_years(); self.Z = Z
        yrs = np.array(self.years)
        palm = A['palm'] >= s.palm_share
        py_raw = np.where(Z['has'], yrs[np.clip(Z['plant_t'], 0, len(yrs)-1)], 0)
        last_y = np.where(Z['has'], yrs[np.clip(Z['last'], 0, len(yrs)-1)], 0)
        recent = palm & Z['has'] & (last_y >= y1-1)
        planted = palm & Z['has'] & ~recent & Z['regrew']
        old = palm & ~Z['has'] & (Z['nvalid'] >= s.min_obs_years)
        cat = np.zeros(palm.shape, np.uint8)
        cat[old], cat[planted], cat[recent] = OLD, PLANTED, RECENT
        cat[palm & (cat == 0)] = UNCLEAR
        py = np.where((cat == PLANTED) | (cat == RECENT), py_raw, np.where(cat == OLD, y0, 0))
        A.update(cat=cat, py=py, ndist=Z['gens'])
        lab = np.where(cat == PLANTED, py, np.where(cat == RECENT, 5000+py, np.where(cat == OLD, 1, np.where(cat == UNCLEAR, 2, 0))))
        A['py_raw'] = py
        gl, gm, palm_m, ind_m = self.block_map(lab, palm)
        px_ha = gm.res**2/1e4
        cm = np.zeros(gl.shape, np.uint8); cm[(gl >= 1900) & (gl < 5000)] = PLANTED; cm[gl >= 5000] = RECENT; cm[gl == 1] = OLD; cm[gl == 2] = UNCLEAR
        pm = np.where(gl >= 5000, gl-5000, np.where(gl >= 1900, gl, np.where(gl == 1, y0, 0)))
        self.gm = gm
        k = 3 if gm.res < s.res else 1
        if k > 1:
            A['cat_map'] = cm[1::3, 1::3]
            A['py_map'] = pm[1::3, 1::3]
            S10 = self.S10
            self.M = {'cat_map': cm, 'py_map': pm, 'red': S10['B4'], 'green': S10['B3'], 'blue': S10['B2']}
        else:
            A['cat_map'], A['py_map'] = cm, pm
            self.M = A
        cat, py, palm, ind = cm, pm, palm_m, ind_m
        ha = lambda m: float(m.sum()*px_ha)
        known = (cat == OLD) | (cat == PLANTED) | (cat == RECENT)
        age = np.where(known, y1-py, -1)
        by_year = {int(y): ha((py == y) & ((cat == PLANTED) | (cat == RECENT))) for y in range(y0, y1+1)}
        aidx, abins = age_index(cat, py, y0, y1)
        classes = [{'label': lab_, 'meaning': mean, 'lo': lo, 'ha': ha(aidx == k)} for k, (lo, hi, lab_, mean, _) in enumerate(abins)]
        for c in classes: c['pct'] = 100*c['ha']/max(ha(known), 1e-9)
        dated = (cat == PLANTED) | (cat == RECENT)
        med = lambda m: int(np.median(py[m])) if m.any() else '–'
        gens = np.repeat(np.repeat(Z['gens'], 3, 0), 3, 1) if k > 1 else Z['gens']
        R = {'years': [y0, y1], 'palm_ha': ha(palm), 'industrial_ha': ha(ind), 'smallholder_ha': ha(palm & ~ind),
             'dated_ha': ha(dated), 'old_ha': ha(cat == OLD), 'recent_ha': ha(cat == RECENT), 'unclear_ha': ha(cat == UNCLEAR),
             'by_year': by_year, 'age_classes': classes, 'median_year': med(known),
             'mean_age': float(np.mean(age[known])) if known.any() else None,
             'replanted_in_record_ha': ha(dated & (gens >= 2)), 'map_res': gm.res,
             'segmented': k > 1,
             'median_year_industrial': med(known & ind), 'median_year_smallholder': med(known & ~ind)}
        R['blocks'] = int(sum(ndimage.label(gl == v)[1] for v in np.unique(gl[gl > 0])))
        R['validation'] = self.validate(A['cat'], A['py_raw'])
        return R

    def block_map(self, lab, palm):
        s, g = self.s, self.g
        S10 = getattr(self, 'S10', None)
        if S10 is None:
            px_ha = g.res**2/1e4
            return generalize(lab, palm, max(1, int(round(s.mmu_ha/px_ha)))), g, palm, palm & (self.A['ind'] >= .5)
        f = g.finer(3)
        palm10 = self.K10 > 0; ind10 = self.K10 == 1
        lab10 = np.repeat(np.repeat(lab, 3, 0), 3, 1)
        valid = palm10 & (lab10 > 0)
        lab10 = np.where(valid, lab10, 0)
        hole = palm10 & ~valid
        if hole.any() and valid.any():
            _, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
            lab10[hole] = lab10[ri[hole], ci[hole]]
        cl = S10['clusters'].astype(np.int64)
        rr, cc = np.indices(cl.shape)
        key = np.where(cl >= 0, cl*10000+(rr//256)*100+(cc//256), -1)
        m = palm10 & (key >= 0)
        vals, inv = np.unique(lab10[m], return_inverse=True)
        keys, kinv = np.unique(key[m], return_inverse=True)
        cnt = np.zeros((len(keys), len(vals)), np.int32)
        np.add.at(cnt, (kinv, inv), 1)
        out = lab10.copy(); out[m] = vals[cnt.argmax(1)][kinv]
        out = np.where(palm10, out, 0)
        out = generalize(out, palm10, max(1, int(round(s.mmu_ha/(f.res**2/1e4)))), passes=0)
        out = drop_strips(out, out >= 5000, 90/f.res)
        palm10 = palm10 & (out > 0)
        return out, f, palm10, ind10

    def validate(self, cat, py):
        A, s = self.A, self.s; px_ha = s.res**2/1e4
        ref = A['ref'].astype(int); V = {}
        lo = max(s.years[0]+1, 1991)
        both = (cat == PLANTED) & (py >= lo) & (py <= 2021) & (ref >= lo) & (ref <= 2021)
        d = (py-ref)[both]
        if d.size > 2:
            V['reference'] = {'n_px': int(d.size), 'ha': float(d.size*px_ha), 'bias': float(d.mean()), 'mae': float(np.abs(d).mean()),
                              'rmse': float(np.sqrt((d**2).mean())), 'exact': float((d == 0).mean()), 'within1': float((np.abs(d) <= 1).mean()),
                              'within3': float((np.abs(d) <= 3).mean()), 'r': float(np.corrcoef(py[both], ref[both])[0, 1])}
        palm = cat > 0; dated = (cat == PLANTED) | (cat == RECENT)
        V['coverage'] = {'ref_dated_ha': float((palm & (ref >= lo)).sum()*px_ha),
                         'both_dated_ha': float((dated & (ref >= lo)).sum()*px_ha),
                         'ours_old_ref_dated_ha': float(((cat == OLD) & (ref >= lo)).sum()*px_ha),
                         'ours_old_far_ha': float(((cat == OLD) & (ref >= s.years[0]+3)).sum()*px_ha)}
        rc = palm & (ref >= 1989) & (ref <= s.years[0]-1)
        V['old_agree'] = {'ref_old_ha': float(rc.sum()*px_ha), 'also_old': float((cat[rc] == OLD).mean()) if rc.any() else None}
        post = dated & (py >= 2022) & (ref >= 1989) & (ref <= 2021)
        if post.any():
            V['after_reference'] = {'ha': float(post.sum()*px_ha), 'ref_median': int(np.median(ref[post])),
                                    'age_at_replanting': float(np.median(py[post]-np.maximum(ref[post], 1990)))}
        loss = A['loss'].astype(int)
        hl = (cat == PLANTED) & (py >= 2002) & (loss > 0)
        if hl.any():
            lag = (py-(loss+2000))[hl]
            V['hansen'] = {'n_px': int(hl.sum()), 'median_lag': float(np.median(lag)), 'within0_2': float(((lag >= -1) & (lag <= 2)).mean()),
                           'lag_hist': {int(k): int((lag == k).sum()) for k in range(-4, 7)},
                           'share_dated_with_loss': float(hl.sum()/max(((cat == PLANTED) & (py >= 2002)).sum(), 1))}
        gd = (cat == PLANTED) & (A['gedi'] > 0) & (A['gy'] >= py) & (A['gedi'] < 40)
        V['qc'] = []
        if gd.sum() > 20:
            ag = (A['gy']-py)[gd]; ht = A['gedi'][gd]
            rho = stats.spearmanr(ag, ht)
            V['gedi'] = {'n': int(gd.sum()), 'rho': float(rho.correlation), 'p': float(rho.pvalue), 'age_span': [int(ag.min()), int(ag.max())],
                         'median_by_age': {int(k): float(np.median(ht[ag == k])) for k in range(0, 40) if (ag == k).sum() >= 5}}
            self.gedi_pts = (ag, ht)
        else:
            self.gedi_pts = (np.array([]), np.array([]))
        qc, r, cv = V['qc'], V.get('reference'), V['coverage']
        if r is None: qc.append('Tidak ada piksel yang bisa dibandingkan dengan peta Descals.')
        elif r['within3'] < .7: qc.append('Kecocokan ±3 th dengan Descals hanya %.0f%% (harapan ≥ 70%%).' % (r['within3']*100))
        if cv['ref_dated_ha'] > 0 and cv['ours_old_far_ha']/cv['ref_dated_ha'] > .1:
            qc.append('%.0f%% sawit yang oleh Descals ditanam %d atau sesudahnya terbaca "sebelum %d" di sini.' % (100*cv['ours_old_far_ha']/cv['ref_dated_ha'], s.years[0]+3, s.years[0]))
        gq = V.get('gedi')
        if gq and gq['age_span'][1]-gq['age_span'][0] >= 10 and gq['rho'] < .4:
            qc.append('Tinggi GEDI tidak naik seiring umur (ρ = %.2f).' % gq['rho'])
        un = float((cat == UNCLEAR).sum())/max(float((cat > 0).sum()), 1)
        if un > .10: qc.append('%.0f%% sawit tidak bisa diberi tahun tanam.' % (un*100))
        return V

    def examples(self):
        A, s, g = self.A, self.s, self.g; y0, y1 = s.years
        cat, py, gens = A['cat'], A['py'], A['ndist']
        same = ndimage.uniform_filter((py == A['py_map']).astype(float), 5) > .999
        groups = [('Planted in the 1990s', (cat == PLANTED) & (py <= 1999)),
                  ('Planted in the 2000s', (cat == PLANTED) & (py >= 2000) & (py <= 2009)),
                  ('Replanted, second cycle', (cat == PLANTED) & (gens >= 2)),
                  ('Cleared %d–%d, new cycle' % (y1-1, y1), cat == RECENT),
                  ('Planted 2010 or later', (cat == PLANTED) & (py >= 2010))]
        pts, used = [], set()
        for name, m in groups:
            if len(pts) == 4: break
            cand = m & same
            if not cand.any(): cand = m
            if not cand.any(): continue
            depth = np.where(cand, np.where(self.Z['bare'], np.nan_to_num(self.N, nan=9), 9).min(0), 9)
            r, c = np.unravel_index(np.argmin(depth), depth.shape)
            if (r, c) in used: continue
            used.add((r, c)); lon, lat = g.pixel_lonlat(r, c)
            pts.append({'name': name, 'lon': lon, 'lat': lat, 'py': int(py[r, c]), 'ref': int(A['ref'][r, c]),
                        'year': self.years, 'nbr': [None if np.isnan(v) else float(v) for v in self.N[:, r, c]],
                        'bare': [bool(b) for b in self.Z['bare'][:, r, c]]})
        return pts

    def frames_data(self):
        ee, g = self.ee, self.g
        nil = ee.Image.constant([0, 0, 0]).toFloat().rename(['red', 'green', 'blue']).updateMask(ee.Image.constant(0))
        clean = self.col.map(lambda im: im.select(['red', 'green', 'blue']).updateMask(self.haze_free(im)))

        def comp(a, b):
            return ee.ImageCollection([nil]).merge(clean.filterDate('%d-01-01' % a, '%d-01-01' % (b+1))).median()
        F = {}
        for y in self.years:
            im = comp(y, y).unmask(comp(y-1, y+1)).unmask(comp(y-2, y+2))
            d = pull(ee, im.unmask(-1).toFloat(), g, ['red', 'green', 'blue'], tile=512)
            F[y] = np.dstack([d['red'], d['green'], d['blue']])
            self.say('      frame %d' % y)
        return F

def generalize(lab, mask, mmu_px, passes=2, size=5):
    out = np.where(mask, lab, 0).astype(np.int32)
    for _ in range(passes):
        vals = [v for v in np.unique(out[mask]) if v > 0]
        best = np.zeros(out.shape, np.float32); arg = out.copy()
        for v in vals:
            c = ndimage.uniform_filter((out == v).astype(np.float32), size)
            upd = c > best; best = np.where(upd, c, best); arg = np.where(upd, v, arg)
        out = np.where(mask, arg, 0)
    for v in [v for v in np.unique(out[mask]) if v > 0]:
        L, n = ndimage.label(out == v)
        if n == 0: continue
        sizes = ndimage.sum(np.ones_like(L), L, range(1, n+1))
        small = np.isin(L, np.where(sizes < mmu_px)[0]+1)
        out[small] = 0
    hole = mask & (out == 0)
    if hole.any() and (mask & (out > 0)).any():
        _, (ri, ci) = ndimage.distance_transform_edt(~(mask & (out > 0)), return_indices=True)
        out[hole] = out[ri[hole], ci[hole]]
    return out


def drop_strips(lab, mask, max_width_px):
    L, n = ndimage.label(mask)
    if n == 0: return lab
    dt = ndimage.distance_transform_edt(mask)
    w = np.asarray(ndimage.maximum(dt, L, range(1, n+1)))*2
    area = np.asarray(ndimage.sum(mask, L, range(1, n+1)))
    thin = np.isin(L, np.where((w <= max_width_px) & (area >= 4*w**2))[0]+1)
    out = lab.copy(); out[thin] = 0
    return out


def mode3(py, mask):
    p = np.where(mask, py, 0).astype(np.int32)
    def f(w):
        w = w[w > 0].astype(int)
        return np.bincount(w).argmax() if w.size else 0
    m = ndimage.generic_filter(p, f, size=3, mode='constant')
    return np.where(mask, np.where(m > 0, m, p), 0)


PAPER, INK, MUTED, LINE, ACCENT, CLEAR, OLDC = '#f4efe6', '#1f2a24', '#6f6a5f', '#d8cfbf', '#c2562b', '#e07a3f', '#8f8577'
YEAR_CMAP = LinearSegmentedColormap.from_list('palm', ['#3d2f5b', '#2d5f86', '#21908c', '#5ec26a', '#c4d94e', '#f5e663'])
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'FreeSans', 'Helvetica', 'DejaVu Sans'], 'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.edgecolor': LINE, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
                     'savefig.facecolor': PAPER})


def fmt(v, d=0): return ('{:,.%df}' % d).format(v)
def idn(v, d=0): return fmt(v, d).replace(',', 'X').replace('.', ',').replace('X', '.')


def place_label(R):
    p = R.get('place') or {}
    if p.get('adm2'): return '%s, %s' % (p['adm2'], p.get('adm1', ''))
    return '%.3f, %.3f' % tuple(R['center'])


def ynorm(y, y0, y1): return np.clip((np.asarray(y)-y0)/max(y1-2-y0, 1), 0, 1)


def year_rgba(py, cat, y0, y1, upto=None, faded=None):
    out = np.zeros(py.shape+(4,))
    col = YEAR_CMAP(ynorm(py, y0, y1))
    show = (cat == PLANTED) if upto is None else (cat == PLANTED) & (py <= upto)
    out[show] = col[show]
    o = cat == OLD; out[o] = (*matplotlib.colors.to_rgb(OLDC), 1)
    rc = (cat == RECENT) if upto is None else (cat == RECENT) & (py <= upto)
    out[rc] = (*matplotlib.colors.to_rgb(CLEAR), 1)
    if faded is not None:
        wait = (cat > 0) & ~show & ~o & ~rc; out[wait] = (*matplotlib.colors.to_rgb(faded), 1)
    else:
        u = cat == UNCLEAR; out[u] = (*matplotlib.colors.to_rgb('#bdb6a8'), 1)
    return out


def grey_base(A, k=.55):
    rgb = np.dstack([A['red'], A['green'], A['blue']]).astype(float)
    lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
    gr = np.clip((rgb.mean(2)-lo)/(hi-lo+1e-9), 0, 1)
    paper = np.array(matplotlib.colors.to_rgb(PAPER))
    return paper*(1-k)+(gr[..., None]*np.array([.92, .92, .9]))*k


def _scalebar(ax, g, km=2, color=INK, fs=8):
    e = g.extent; x0 = e[0]+(e[1]-e[0])*.04; y0 = e[2]+(e[3]-e[2])*.05; L = km*1000
    ax.add_patch(Rectangle((x0, y0), L/2, (e[3]-e[2])*.008, fc=color, ec=color))
    ax.add_patch(Rectangle((x0+L/2, y0), L/2, (e[3]-e[2])*.008, fc='white', ec=color))
    ax.text(x0, y0+(e[3]-e[2])*.02, '0', fontsize=fs, color=color, ha='center')
    ax.text(x0+L, y0+(e[3]-e[2])*.02, '%d km' % km, fontsize=fs, color=color, ha='center')
    xn, yn = e[1]-(e[1]-e[0])*.04, e[3]-(e[3]-e[2])*.06
    ax.annotate('N', xy=(xn, yn), xytext=(xn, yn-(e[3]-e[2])*.08), ha='center', va='top', color=color, fontsize=fs+3,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))


def _inset(ax, study, R):
    ax.set_facecolor('#dfe7ea')
    if study.inset:
        from matplotlib.patches import Polygon
        for ft in study.inset['features']:
            gm = ft['geometry']; polys = []
            if gm['type'] == 'Polygon': polys = [gm['coordinates']]
            elif gm['type'] == 'MultiPolygon': polys = gm['coordinates']
            elif gm['type'] == 'GeometryCollection':
                for gg in gm['geometries']:
                    if gg['type'] == 'Polygon': polys.append(gg['coordinates'])
                    elif gg['type'] == 'MultiPolygon': polys += gg['coordinates']
            for p in polys: ax.add_patch(Polygon(np.array(p[0]), fc='#f7f3ea', ec='#9a9283', lw=.4))
    lon, lat = R['center']
    ax.plot(lon, lat, 's', ms=6, mfc=ACCENT, mec=INK, mew=.6)
    ax.set_xlim(lon-6, lon+6); ax.set_ylim(lat-5, lat+5); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(LINE)


def poster(R, study, path):
    A, g = study.A, study.g; y0, y1 = R['years']
    fig = plt.figure(figsize=(16.54, 11.69), dpi=study.s.dpi)
    fig.text(.04, .945, 'Oil Palm Planting Year, %s' % place_label(R), fontsize=24, fontweight='bold', va='top')
    fig.text(.04, .9, 'Landsat %d-%d, %g x %g km area' % (y0, y1, *R['window_km']), fontsize=11.5, color=MUTED, va='top')
    aw = .56; ah = aw*16.54/11.69*g.ny/g.nx
    if ah > .76: ah = .76; aw = ah*11.69/16.54*g.nx/g.ny
    ax = fig.add_axes([.04, .845-ah, aw, ah])
    ax.imshow(grey_base(A), extent=g.extent, interpolation='bilinear')
    disp = A['py_map']
    ax.imshow(year_rgba(disp, A['cat_map'], y0, y1), extent=g.extent, interpolation='nearest')
    ax.set_xlim(g.extent[:2]); ax.set_ylim(g.extent[2:]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(LINE)
    _scalebar(ax, g)
    cy = .845-ah-.035
    cax = fig.add_axes([.04, cy, aw*.55, .012])
    cax.imshow(np.linspace(0, 1, 256)[None], cmap=YEAR_CMAP, aspect='auto'); cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values(): sp.set_visible(False)
    fig.text(.04, cy-.006, str(y0), fontsize=9, color=MUTED, va='top')
    fig.text(.04+aw*.55, cy-.006, str(y1-2), fontsize=9, color=MUTED, va='top', ha='right')
    fig.text(.04, cy+.02, 'Year planted', fontsize=9.5, fontweight='bold', va='bottom')
    lx = .04+aw*.6
    for i, (c, t) in enumerate([(OLDC, 'Planted before %d' % y0), (CLEAR, 'Cleared %d-%d (replanting)' % (y1-1, y1)), ('#bdb6a8', 'Oil palm, date unclear')]):
        fig.patches.append(Rectangle((lx, cy+.012-i*.02), .012, .011, transform=fig.transFigure, fc=c, ec='none'))
        fig.text(lx+.017, cy+.0175-i*.02, t, fontsize=9, color=MUTED, va='center')
    rx = .04+aw+.04; rw = .96-rx
    big = [(fmt(R['palm_ha']), 'ha of oil palm'), (str(R['median_year'] or '–'), 'median planting year'),
           ('%.0f%%' % (100*R['dated_ha']/max(R['palm_ha'], 1e-9)), 'planted since %d' % R['years'][0])]
    for i, (n, l) in enumerate(big):
        x = rx+i*rw/3
        fig.text(x, .845, n, fontsize=24, fontweight='bold', color=ACCENT if i == 2 else INK, va='top')
        fig.text(x, .808, l, fontsize=9, color=MUTED, va='top')
    b = fig.add_axes([rx+.02, .585, rw-.02, .165])
    ys = sorted(R['by_year']); vals = [R['by_year'][y] for y in ys]
    cols = [YEAR_CMAP(ynorm(y, y0, y1)) if y < y1-1 else CLEAR for y in ys]
    b.bar(ys, vals, color=cols, width=.82)
    b.set_title('Hectares planted per year (current planting)', fontsize=10.5, loc='left', color=INK, fontweight='bold')
    for sp in ('top', 'right'): b.spines[sp].set_visible(False)
    b.tick_params(labelsize=8); b.set_xlim(y0-.7, y1+.7)
    a2 = fig.add_axes([rx, .31, rw, .215])
    cl = R['age_classes']; yy = np.arange(len(cl))[::-1]
    a2.barh(yy, [c['ha'] for c in cl], color=age_cols(age_bins(y0, y1), AGE_COLORS), height=.62)
    for yv, c in zip(yy, cl):
        a2.text(c['ha'], yv, '  %s ha (%.0f%%)' % (fmt(c['ha']), c['pct']), va='center', fontsize=8.5, color=INK)
        a2.text(0, yv+.42, '%s, %s' % (c['label'], c['meaning'].lower()), va='bottom', fontsize=8.5, color=MUTED)
    a2.set_yticks([]); a2.set_ylim(-.5, len(cl)-.05)
    a2.set_xticks([]); a2.set_xlim(0, max(c['ha'] for c in cl)*1.45+1)
    for sp in ('top', 'right', 'bottom', 'left'): a2.spines[sp].set_visible(False)
    a2.set_title('Age in %d' % y1, fontsize=10.5, loc='left', color=INK, fontweight='bold')
    V = R['validation']; t = []
    if 'reference' in V:
        r = V['reference']; t.append('Descals et al. (2024) planting-year map: %.0f%% within ±3 yr, RMSE %.1f yr (n = %s px)' % (r['within3']*100, r['rmse'], fmt(r['n_px'])))
    if 'gedi' in V:
        t.append('GEDI lidar height rises with estimated age: Spearman ρ = %.2f (n = %s)' % (V['gedi']['rho'], fmt(V['gedi']['n'])))
    if 'hansen' in V:
        t.append('Hansen tree-cover loss year within −1 to +2 yr of planting for %.0f%% of blocks' % (V['hansen']['within0_2']*100))
    fig.text(rx, .285, 'Checked against', fontsize=10.5, fontweight='bold', va='top')
    fig.text(rx, .262, '\n'.join(t), fontsize=8.6, color=MUTED, va='top', linespacing=1.6)
    ia = fig.add_axes([.96-.13, .065, .13, .13]); _inset(ia, study, R)
    fig.text(.04, .03, 'Data: Landsat 5, 7, 8, 9 Collection 2 surface reflectance (USGS), annual median NBR; BIOPAMA Global Oil Palm v1.2 '
             '(extent; planting year by Descals et al. 2024 for comparison) · Hansen GFC v1.13 · GEDI L2A · FAO GAUL.\n'
             'Planting year = barest year (lowest annual median NBR) of the last bare-soil spell, confirmed by regrowth. Areas are 30 m pixel counts; map smoothed with a 3 × 3 mode filter.   Map: %s'
             % R['setup']['author'], fontsize=7.6, color=MUTED, va='bottom', linespacing=1.5)
    fig.savefig(path); plt.close(fig)



def validation_chart(R, study, path, dpi=200):
    A = study.A; V = R['validation']; y0, y1 = R['years']
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=dpi)
    ax = axs[0, 0]; cat, py, ref = A['cat'], A['py'], A['ref'].astype(int)
    m = (cat == PLANTED) & (py >= 1991) & (py <= 2021) & (ref >= 1991) & (ref <= 2021)
    if m.any():
        h = ax.hist2d(ref[m], py[m], bins=[np.arange(1990.5, 2022.5), np.arange(1990.5, 2022.5)], cmap='Greens', cmin=1,
                      norm=matplotlib.colors.LogNorm())
        ax.plot([1991, 2021], [1991, 2021], color=INK, lw=1)
        for k in (-3, 3): ax.plot([1991, 2021], [1991+k, 2021+k], color=INK, lw=.7, ls=':')
        r = V['reference']
        ax.text(.97, .04, 'n = %s px (%s ha)\nexact %.0f%% · ±1 yr %.0f%% · ±3 yr %.0f%%\nbias %+.1f yr · MAE %.1f · RMSE %.1f yr'
                % (fmt(r['n_px']), fmt(r['ha']), r['exact']*100, r['within1']*100, r['within3']*100, r['bias'], r['mae'], r['rmse']),
                transform=ax.transAxes, va='bottom', ha='right', fontsize=8.5)
    ax.set_xlabel('Descals et al. 2024 (BIOPAMA v1.2) planting year'); ax.set_ylabel('This study')
    ax.set_title('a. Comparison with Descals et al. (2024)', loc='left', fontsize=10.5, fontweight='bold')
    ax = axs[0, 1]; ag, ht = study.gedi_pts
    if ag.size:
        bins = list(range(0, 32, 2)); data = [ht[(ag >= b) & (ag < b+2)] for b in bins]
        keep = [i for i, d in enumerate(data) if d.size >= 5]
        if keep:
            ax.boxplot([data[i] for i in keep], positions=[bins[i]+1 for i in keep], widths=1.4, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor='#cfe3c4', edgecolor=INK), medianprops=dict(color=ACCENT))
        g = V['gedi']; ax.text(.03, .97, 'Spearman ρ = %.2f · n = %s footprints' % (g['rho'], fmt(g['n'])), transform=ax.transAxes, va='top', fontsize=8.5)
        ax.set_xticks(range(0, 32, 4)); ax.set_xticklabels(range(0, 32, 4))
    else:
        ax.text(.5, .5, 'Too few GEDI footprints', ha='center', transform=ax.transAxes)
    ax.set_xlabel('Age at the time of the GEDI shot (yr)'); ax.set_ylabel('GEDI RH98 height (m)')
    ax.set_title('b. GEDI canopy height by planting age', loc='left', fontsize=10.5, fontweight='bold')
    ax = axs[1, 0]
    if 'hansen' in V:
        hh = V['hansen']['lag_hist']; ks = sorted(hh)
        ax.bar(ks, [hh[k] for k in ks], color=['#21908c' if -1 <= k <= 2 else '#b9b2a3' for k in ks])
        ax.text(.03, .97, '%.0f%% of blocks planted since 2002 also show Hansen loss\nof these, %.0f%% within −1 to +2 yr (median %.0f)'
                % (V['hansen']['share_dated_with_loss']*100, V['hansen']['within0_2']*100, V['hansen']['median_lag']), transform=ax.transAxes, va='top', fontsize=8.5)
    ax.set_xlabel('Planting year minus Hansen loss year'); ax.set_ylabel('pixels')
    ax.set_title('c. Planting year minus Hansen forest loss year', loc='left', fontsize=10.5, fontweight='bold')
    ax = axs[1, 1]; av = R['availability']; ys = sorted(int(k) for k in av)
    ax.bar(ys, [av[str(y)]['coverage']*100 for y in ys], color=['#2d5f86' if av[str(y)]['coverage'] >= .5 else '#b9b2a3' for y in ys])
    ax2 = ax.twinx(); ax2.plot(ys, [av[str(y)]['median_obs'] for y in ys], color=ACCENT, lw=1.4); ax2.set_ylabel('clear views per pixel (median)', color=ACCENT)
    ax2.tick_params(labelsize=8, colors=ACCENT); ax2.spines['top'].set_visible(False)
    ax.set_xticks([y for y in ys if y % 2 == 0]); ax.set_ylim(0, 105); ax.set_ylabel('% of window with a clear view')
    ax.set_title('d. Landsat data availability', loc='left', fontsize=10.5, fontweight='bold')
    for a in axs.flat:
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)



def low_line(v, b, R): return ~b & (np.nan_to_num(v, nan=9) <= R['setup']['bare_nbr'])


def pixel_stories(R, path, dpi=200):
    ex = R['examples']; y0, y1 = R['years']
    n = max(len(ex), 1); fig, axs = plt.subplots(n, 1, figsize=(10, 2.1*n+.8), dpi=dpi, sharex=True)
    axs = np.atleast_1d(axs)
    fig.suptitle('Annual Landsat NBR at four sample blocks', x=.06, ha='left', fontsize=13, fontweight='bold', y=.995)
    for ax, p in zip(axs, ex):
        yr = np.array(p['year']); v = np.array([np.nan if x is None else x for x in p['nbr']]); b = np.array(p['bare'])
        ok = ~np.isnan(v)
        ax.axhspan(-.2, R['setup']['bare_nbr'], color='#efe3d0', zorder=0)
        ax.plot(yr[ok & ~low_line(v, b, R)], v[ok & ~low_line(v, b, R)], '-', color=INK, lw=1.2, zorder=2)
        low = ok & ~b & (np.nan_to_num(v, nan=9) <= R['setup']['bare_nbr'])
        ax.plot(yr[ok & ~b & ~low], v[ok & ~b & ~low], 'o', ms=4, color='#3f7d4e', zorder=3, label='vegetated')
        ax.plot(yr[low], v[low], 'o', ms=4, mfc='none', mec='#9a9283', zorder=3, label='cloud or haze, ignored')
        ax.plot(yr[b], v[b], 'o', ms=5, color=CLEAR, zorder=3, label='bare soil')
        ax.axvline(p['py'], color=ACCENT, lw=1.4)
        ax.text(p['py']+.3, .86, 'planted %d' % p['py'], color=ACCENT, fontsize=8.5, transform=ax.get_xaxis_transform())
        if 1990 <= p['ref'] <= 2022:
            ax.axvline(p['ref'], color='#2d5f86', lw=1.1, ls='--')
            ax.text(p['ref']+.3, .70, 'Descals et al. 2024: %d' % p['ref'], color='#2d5f86', fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_ylim(-.2, .9); ax.set_ylabel('NBR', fontsize=8.5)
        ax.set_title('%s (%.4f, %.4f)' % (p['name'], p['lat'], p['lon']), loc='left', fontsize=9.5, color=INK)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc='upper right', ncol=3, bbox_to_anchor=(.98, .985))
    axs[-1].set_xlim(y0-.5, y1+.5)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(path); plt.close(fig)


def _stretch_frames(F):
    raw = {}
    for y, f in F.items():
        ok = f[..., 0] > 0
        if ok.mean() < .05: raw[y] = None; continue
        raw[y] = (np.percentile(f[ok], 1, axis=0), np.percentile(f[ok], 99, axis=0))
    ys = sorted(F); out = {}
    for i, y in enumerate(ys):
        near = [raw[ys[j]] for j in range(max(0, i-1), min(len(ys), i+2)) if raw[ys[j]] is not None]
        f = F[y]; miss = f[..., 0] <= 0
        if not near:
            out[y] = (np.full(f.shape, .82), 0.); continue
        lo = np.median([n[0] for n in near], 0); hi = np.median([n[1] for n in near], 0)
        rgb = np.clip((f-lo)/(hi-lo+1e-9), 0, 1)**.85
        rgb[miss] = .82
        out[y] = (rgb, 1-miss.mean())
    return out


def timelapse(R, study, F, path, fps=24, hold=18, fade=0, web_path=None):
    A, g = study.A, study.g; y0, y1 = R['years']
    CS = chip_stretch(F, sorted(F))
    S = {y: (CS[y], float((F[y][..., 0] > 0).mean())) for y in F}
    ys = sorted(R['by_year']); vals = np.array([R['by_year'][y] for y in ys])
    W, H = 1920, 1080
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)
    base = grey_base(A, .25)
    cum = np.cumsum(vals)
    cols = [YEAR_CMAP(ynorm(yy, y0, y1)) if yy < y1-1 else CLEAR for yy in ys]
    pmap, cat = A['py_map'], A['cat_map']
    last_good = None

    def render(y, final=False):
        nonlocal last_good
        rgb, cov = S[y]
        if cov >= .6 or last_good is None: last_good = (y, rgb)
        vy, vrgb = last_good
        fig = plt.figure(figsize=(W/100, H/100), dpi=100); fig.patch.set_facecolor(PAPER)
        fig.text(.035, .9, 'Oil Palm Planting Year, %s' % place_label(R), color=INK, fontsize=26, fontweight='bold')
        fig.text(.965, .872, str(y), color=INK, fontsize=66, fontweight='bold', ha='right')
        pw = .45; ph = pw*W/H*g.ny/g.nx
        a1 = fig.add_axes([.035, .80-ph, pw, ph]); a1.imshow(vrgb, extent=g.extent, interpolation='bilinear')
        new = ((cat == PLANTED) | (cat == RECENT)) & (pmap == y)
        if new.any():
            a1.contour(new.astype(float), levels=[.5], colors=['#ffd23f'], linewidths=1.1, extent=g.extent, origin='upper')
        a1.set_title('Landsat %d, natural colour%s' % (vy, '' if vy == y else '  (no clear view in %d)' % y), color=MUTED, fontsize=12.5, loc='left')
        a2 = fig.add_axes([.515, .80-ph, pw, ph]); a2.imshow(base, extent=g.extent)
        a2.imshow(year_rgba(pmap, cat, y0, y1, upto=y, faded='#e2dccf'), extent=g.extent, interpolation='nearest')
        a2.set_title('Oil palm by planting year', color=MUTED, fontsize=12.5, loc='left')
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color(LINE)
        fig.text(.035, .80-ph-.03, 'Yellow outline: blocks planted in this year', color=MUTED, fontsize=12)
        b = fig.add_axes([.035, .07, .93, .80-ph-.16]); b.set_facecolor(PAPER)
        on = np.array(ys) <= y
        b.bar(np.array(ys)[on], vals[on], color=[c for c, o in zip(cols, on) if o], width=.8)
        b.bar(np.array(ys)[~on], vals[~on], color='none', edgecolor=LINE, width=.8)
        b.set_xlim(y0-.7, y1+.7); b.tick_params(colors=MUTED, labelsize=11)
        for sp in ('top', 'right', 'left'): b.spines[sp].set_visible(False)
        b.spines['bottom'].set_color(LINE); b.set_yticks([])
        k = int(on.sum())-1
        done = (cum[k] if k >= 0 else 0)+R['old_ha']
        fig.text(.965, .80-ph-.03, 'Planted area by %d: %s of %s ha' % (y, fmt(done), fmt(R['palm_ha'])), color=INK, fontsize=15, ha='right')
        if final:
            r = R['validation'].get('reference')
            msg = 'Annual Landsat NBR, %d–%d  ·  ' % (y0, y1)
            if r: msg += '%.0f%% within ±3 yr of the published map (Descals et al. 2024)  ·  ' % (r['within3']*100)
            msg += 'map: %s' % R['setup']['author']
            fig.text(.035, .015, msg, color=MUTED, fontsize=12)
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
        return im

    prev = None
    for y in study.years:
        cur = render(y, final=(y == y1))
        if prev is not None:
            for i in range(1, fade+1):
                t = i/(fade+1); proc.stdin.write((prev*(1-t)+cur*t).astype(np.uint8).tobytes())
        for _ in range(hold if y < y1 else fps*4): proc.stdin.write(cur.tobytes())
        prev = cur.astype(float)
    proc.stdin.close(); proc.wait()
    if web_path:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-vf', 'scale=1280:-2', '-c:v', 'libx264', '-preset', 'veryfast',
                        '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', web_path], check=True)


def report_id(R):
    y0, y1 = R['years']; V = R['validation']; S = R['setup']; L = []
    L += ['UMUR KEBUN SAWIT DARI ARSIP LANDSAT %d–%d' % (y0, y1), place_label(R),
          'Pusat %.4f, %.4f · jendela %g × %g km · grid %d m (EPSG:%d) · citra terakhir %s' % (*R['center'], *R['window_km'], R['res'], R['utm_epsg'], R['latest_image']), '',
          '1. METODE',
          '- Landsat 5, 7, 8, 9 Collection 2 Level-2 (sudah terkoreksi atmosfer); awan, bayangan, dan saturasi dimasker (QA_PIXEL, QA_RADSAT). OLI diselaraskan ke ETM+ (Roy dkk. 2016).',
          '- Pengamatan yang biru > %s dan SWIR1 < 1,3 × biru dibuang (sisa awan tipis/kabut; tanah pasir putih tetap dipakai karena SWIR1-nya tinggi). Per tahun dihitung median dan persentil 25 NBR = (NIR − SWIR2)/(NIR + SWIR2).' % idn(S['cloud_blue'], 2),
          '- Tahun tanah terbuka: persentil 25 NBR ≤ %s dan median ≤ %s (≥ 3 pengamatan), atau median NBR ≤ %s (≥ %d pengamatan).' % (idn(S['bare_nbr'], 2), idn(S['bare_nbr']+.15, 2), idn(S['bare_nbr'], 2), S['min_obs_bare']),
          '- Tahun tanam = tahun dengan NBR terendah pada periode tanah terbuka terakhir, lalu wajib disusul pertumbuhan kembali dalam %d tahun (NBR ≥ %s dan naik ≥ %s).'
          % (S['regrowth_years'], idn(S['regrowth_level'], 2), idn(S['min_regrowth'], 2)),
          '- Tidak pernah terbuka sejak %d = ditanam sebelum %d. Terbuka %d–%d = siklus baru, pertumbuhan belum terlihat.' % (y0, y0, y1-1, y1),
          '- Batas sawit: BIOPAMA Global Oil Palm v1.2 (10 m, ≥ 50% piksel 30 m).', '',
          '2. HASIL',
          '- Sawit di jendela: %s ha (industri %s ha, rakyat %s ha).' % (idn(R['palm_ha']), idn(R['industrial_ha']), idn(R['smallholder_ha'])),
          ('- Peta disusun menjadi %s blok: citra Sentinel-2 10 m disegmentasi (SNIC), tiap segmen mengambil tahun tanam mayoritas, unit pemetaan minimum %s ha. Luas dan kelas umur dihitung dari peta blok 10 m; validasi memakai piksel Landsat asli.'
           if R.get('segmented') else '- Peta digeneralisasi menjadi %s blok (modus 5 × 5 piksel, unit pemetaan minimum %s ha). Luas dan kelas umur dihitung dari peta blok; validasi memakai piksel asli.') % (idn(R['blocks']), idn(S['mmu_ha'], 0)),
          '- Bertanggal: %s ha · ditanam sebelum %d: %s ha · siklus baru %d–%d: %s ha · tidak jelas: %s ha.' % (idn(R['dated_ha']), y0, idn(R['old_ha']), y1-1, y1, idn(R['recent_ha']), idn(R['unclear_ha'])),
          '- Median tahun tanam %s (industri %s, rakyat %s); rata-rata umur %s tahun.' % (R['median_year'], R['median_year_industrial'], R['median_year_smallholder'], idn(R['mean_age'] or 0, 1)),
          '- Terlihat dua siklus tanam dalam rekaman (pembukaan hutan lalu peremajaan): %s ha.' % idn(R['replanted_in_record_ha']),
          '- Kelas umur tahun %d:' % y1]
    for c in R['age_classes']: L.append('    %-8s %-30s %10s ha  (%s%%)' % (c['label'], c['meaning'], idn(c['ha']), idn(c['pct'], 0)))
    L += ['', '3. VALIDASI', '- Kontrol kualitas: ' + ('LULUS, semua pemeriksaan otomatis terpenuhi.' if not V.get('qc') else 'ADA PERINGATAN, hasil belum layak dipublikasikan:')]
    for q in V.get('qc', []): L.append('    ! ' + q)
    if 'reference' in V:
        r = V['reference']
        L.append('- vs peta tahun tanam Descals dkk. 2024 (metode lain, arsip Landsat yang sama; n = %s piksel, %s ha): tepat %s%%, ±1 th %s%%, ±3 th %s%%, bias %s th, MAE %s, RMSE %s th, r = %s.'
                 % (idn(r['n_px']), idn(r['ha']), idn(r['exact']*100), idn(r['within1']*100), idn(r['within3']*100), idn(r['bias'], 1), idn(r['mae'], 1), idn(r['rmse'], 1), idn(r['r'], 2)))
        L.append('  Sebagai patokan, peta Descals sendiri punya RMSE 2,0 th (industri) dan 4,9 th (rakyat) terhadap data lapangan.')
    cv = V.get('coverage', {})
    if cv:
        L.append('- Sawit yang diberi tahun tanam oleh Descals: %s ha; di sini %s ha di antaranya juga bertanggal, %s ha terbaca "sebelum %d" (%s ha di antaranya oleh Descals ditanam %d atau sesudahnya).'
                 % (idn(cv['ref_dated_ha']), idn(cv['both_dated_ha']), idn(cv['ours_old_ref_dated_ha']), y0, idn(cv['ours_old_far_ha']), y0+3))
    oa = V.get('old_agree', {})
    if oa.get('also_old') is not None:
        L.append('- Blok yang oleh Descals ditanam sebelum %d (%s ha): %s%% juga terbaca sebelum %d di sini.' % (y0, idn(oa['ref_old_ha']), idn(oa['also_old']*100), y0))
    if 'after_reference' in V:
        a = V['after_reference']
        L.append('- Ditanam ulang 2022 ke atas (di luar jangkauan peta Descals): %s ha; median tahun tanam generasi sebelumnya %d, umur saat diremajakan ± %s th.' % (idn(a['ha']), a['ref_median'], idn(a['age_at_replanting'])))
    if 'hansen' in V:
        h = V['hansen']
        L.append('- Hansen GFC: %s%% blok yang ditanam sejak 2002 juga tercatat kehilangan tutupan pohon; dari itu %s%% selisihnya −1 s.d. +2 th (median %s th).' % (idn(h['share_dated_with_loss']*100), idn(h['within0_2']*100), idn(h['median_lag'])))
    if 'gedi' in V:
        gd = V['gedi']; mb = gd['median_by_age']
        pick = ', '.join('%d th %s m' % (k, idn(mb[k], 1)) for k in sorted(mb) if k in (2, 3, 5, 8, 10, 15, 20, 25, 30))
        L.append('- GEDI RH98 (n = %s, umur %d–%d th): Spearman ρ umur vs tinggi = %s. Median: %s.' % (idn(gd['n']), *gd['age_span'], idn(gd['rho'], 2), pick))
    L += ['', '4. KETERBATASAN',
          '- Tahun dengan sedikit citra bersih (lihat grafik d) lebih rawan meleset; pembukaan di tahun tanpa citra ditarik ke tahun terdekat yang ada datanya.',
          '- Tanam sisip (under-planting) tanpa pembukaan total tidak terdeteksi. Blok kecil di tepi bisa tercampur (piksel 30 m).',
          '- Batas sawit BIOPAMA dibuat 2016–2021, jadi kebun baru di lahan yang belum ada di peta itu tidak ikut dihitung.',
          '- Peta Descals juga dari Landsat, jadi perbandingan ini menguji konsistensi metode, bukan akurasi lapangan. Luas = jumlah piksel 30 m.']
    return '\n'.join(L)


AGE_COLORS = ['#c7e9a0', '#74c476', '#1f7a3d', '#f4a259', '#c0392b']
UNRESOLVED = '#8f8577'


def age_bins(y0, y1):
    mx = y1-y0; out = []
    for i, (lo, hi, lab, mean) in enumerate(AGE_CLASSES):
        if lo > mx: continue
        h = min(hi, mx)
        out.append((lo, h, lab if h == hi else '%d–%d yr' % (lo, h), mean, i))
    if mx < 26: out.append((mx+1, 999, '%d+ yr' % (mx+1), 'Planted before %d' % y0, -1))
    return out


def age_index(cat, py, y0, y1):
    bins = age_bins(y0, y1); idx = np.full(cat.shape, -1, np.int16)
    dated = (cat == PLANTED) | (cat == RECENT); age = y1-py
    for k, (lo, hi, _, _, j) in enumerate(bins):
        if j >= 0: idx[dated & (age >= lo) & (age <= hi)] = k
    if y1-y0 < 26: idx[cat == OLD] = len(bins)-1
    else: idx[cat == OLD] = [k for k, b in enumerate(bins) if b[1] == 999][0]
    return idx, bins


def age_cols(bins, palette): return [palette[b[4]] if b[4] >= 0 else UNRESOLVED for b in bins]
NONPALM_NOTE = 'Not oil palm (Landsat %d–%d, greyscale)'


def year_classes(y0, y1, by_year=None):
    cl = [(-1, -1, 'Before %d' % y0)]
    if by_year and y1-y0 <= 20:
        area = {int(k): float(v) for k, v in by_year.items()}
        tot = sum(area.get(y, 0) for y in range(y0, y1-1)) or 1
        run = []
        def flush():
            if run and sum(area.get(y, 0) for y in run) > 0:
                cl.append((run[0], run[-1], str(run[0]) if len(run) == 1 else '%d-%d' % (run[0], run[-1])))
            run.clear()
        for y in range(y0, y1-1):
            if area.get(y, 0) >= .05*tot:
                flush(); run.append(y); flush()
            else:
                run.append(y)
        flush()
    else:
        stp = 5 if y1-y0 > 20 else 3
        for a in range(y0, y1-1, stp): cl.append((a, min(a+stp-1, y1-2), '%d-%d' % (a, min(a+stp-1, y1-2))))
    cl.append((y1-1, y1, '%d-%d (replanting)' % (y1-1, y1)))
    n = len(cl)-2
    cols = ['#8f8577'] + [matplotlib.colors.to_hex(YEAR_CMAP(i/max(n-1, 1))) for i in range(n)] + [CLEAR]
    return cl, cols


def class_raster(A, R, kind):
    y0, y1 = R['years']; cat, py = A['cat_map'], A['py_map']
    out = np.full(cat.shape, -1, np.int16)
    if kind == 'year':
        cl, cols = year_classes(y0, y1, R.get('by_year'))
        out[cat == OLD] = 0
        for i, (a, b, _) in enumerate(cl[1:-1], 1):
            out[(cat == PLANTED) & (py >= a) & (py <= b)] = i
        out[cat == RECENT] = len(cl)-1
        labels = [c[2] for c in cl]
    else:
        idx, bins = age_index(cat, py, y0, y1)
        cols = age_cols(bins, AGE_COLORS); labels = ['%s, %s' % (b[2], b[3][0].lower()+b[3][1:]) for b in bins]
        out = idx.copy()
    out[cat == UNCLEAR] = 98
    return out, labels, cols


def _dms(v, pos, neg):
    h = pos if v >= 0 else neg; v = abs(v); d = int(v); m = round((v-d)*60)
    if m == 60: d, m = d+1, 0
    return "%d°%02d'%s" % (d, m, h)


def graticule(ax, g, step_min=2, fs=8):
    e = g.extent; u = g.utm
    corners = [u.inv(x, y) for x in e[:2] for y in e[2:]]
    lons = [c[0] for c in corners]; lats = [c[1] for c in corners]
    st = step_min/60
    lon_t = np.arange(math.ceil(min(lons)/st)*st, max(lons), st); lat_t = np.arange(math.ceil(min(lats)/st)*st, max(lats), st)
    lat_mid = u.inv((e[0]+e[1])/2, (e[2]+e[3])/2)[1]; lon_mid = u.inv((e[0]+e[1])/2, (e[2]+e[3])/2)[0]
    span = np.linspace(min(lats)-.05, max(lats)+.05, 50)
    for lo in lon_t:
        xs, ys = u.fwd(np.full(50, lo), span); ax.plot(xs, ys, color='#ffffff', lw=.6, alpha=.75, zorder=5)
    span = np.linspace(min(lons)-.05, max(lons)+.05, 50)
    for la in lat_t:
        xs, ys = u.fwd(span, np.full(50, la)); ax.plot(xs, ys, color='#ffffff', lw=.6, alpha=.75, zorder=5)
    xt = [u.fwd(lo, lat_mid)[0] for lo in lon_t]; yt = [u.fwd(lon_mid, la)[1] for la in lat_t]
    ax.set_xticks(xt); ax.set_xticklabels([_dms(v, 'E', 'W') for v in lon_t], fontsize=fs)
    ax.set_yticks(yt); ax.set_yticklabels([_dms(v, 'N', 'S') for v in lat_t], fontsize=fs, rotation=90, va='center')
    ax.tick_params(top=True, labeltop=True, right=True, labelright=True, direction='in', length=5, color=INK, labelcolor=INK)
    ax.set_xlim(e[:2]); ax.set_ylim(e[2:])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(1.1)


def _rf(g, aw, fig_w_in):
    ground = g.nx*g.res; paper = aw*fig_w_in*0.0254
    r = ground/paper; step = 5000 if r > 20000 else 1000
    return int(round(r/step)*step)


def _scale_fig(fig, x, y, g, aw, km, fs=8.5):
    frac = km*1000/(g.nx*g.res)*aw; seg = frac/4
    for i in range(4):
        fig.patches.append(Rectangle((x+i*seg, y), seg, .007, transform=fig.transFigure, fc=INK if i % 2 == 0 else 'white', ec=INK, lw=.8))
    for i, t in enumerate([0, km/4, km/2, 3*km/4, km]):
        lab = ('%g' % t) + (' km' if i == 4 else '')
        fig.text(x+i*seg, y+.012, lab, fontsize=fs, ha='center', va='bottom', color=INK)


def _north_fig(fig, x, y, h=.05):
    ax = fig.add_axes([x-.015, y, .03, h]); ax.set_xlim(-1, 1); ax.set_ylim(0, 2.6); ax.axis('off')
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon([[0, 2], [-.6, 0], [0, .55]], fc=INK, ec=INK, lw=.8))
    ax.add_patch(Polygon([[0, 2], [.6, 0], [0, .55]], fc='white', ec=INK, lw=.8))
    ax.text(0, 2.12, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold', color=INK)


def _poly(ax, gm, **kw):
    from matplotlib.patches import Polygon
    if not gm: return
    polys = []
    if gm['type'] == 'Polygon': polys = [gm['coordinates']]
    elif gm['type'] == 'MultiPolygon': polys = gm['coordinates']
    elif gm['type'] == 'GeometryCollection':
        for gg in gm['geometries']:
            if gg['type'] == 'Polygon': polys.append(gg['coordinates'])
            elif gg['type'] == 'MultiPolygon': polys += gg['coordinates']
    for p in polys: ax.add_patch(Polygon(np.array(p[0]), **kw))


def _coords(gm):
    out = []
    if not gm: return out
    t = gm['type']
    if t == 'Polygon': out = [gm['coordinates']]
    elif t == 'MultiPolygon': out = gm['coordinates']
    elif t == 'GeometryCollection':
        for g2 in gm['geometries']: out += _coords(g2)
    return out


def _bounds(gm):
    pts = np.concatenate([np.array(p[0]) for p in _coords(gm)]) if _coords(gm) else np.array([[0, 0], [1, 1]])
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(.6)


def inset_country(ax, study, R):
    ax.set_facecolor('#dce8ee')
    if study.inset:
        for ft in study.inset['features']: _poly(ax, ft['geometry'], fc='#f6f2e9', ec='#8a8274', lw=.25)
    _poly(ax, getattr(study, 'prov', None), fc='#e2c98f', ec=INK, lw=.4)
    lon, lat = R['center']; ax.plot(lon, lat, 's', ms=4, mfc=ACCENT, mec=INK, mew=.5)
    ax.set_xlim(lon-6.5, lon+6.5); ax.set_ylim(lat-6.5, lat+6.5); ax.set_aspect('equal'); _frame(ax)
    ax.text(.04, .04, 'Regional context', transform=ax.transAxes, fontsize=6.5, color=MUTED)


def inset_province(ax, study, R):
    ax.set_facecolor('#dce8ee')
    p = R.get('place') or {}
    rg = getattr(study, 'regencies', None)
    if rg:
        for ft in rg['features']:
            hit = ft['properties'].get('ADM2_NAME') == p.get('adm2')
            _poly(ax, ft['geometry'], fc='#e2c98f' if hit else '#f6f2e9', ec='#8a8274', lw=.3)
    pr = getattr(study, 'prov', None)
    for c in _coords(pr): ax.plot(np.array(c[0])[:, 0], np.array(c[0])[:, 1], color=INK, lw=.6)
    g = study.g; e = g.extent
    cs = np.array([g.utm.inv(x, y) for x, y in ((e[0], e[2]), (e[1], e[2]), (e[1], e[3]), (e[0], e[3]), (e[0], e[2]))])
    ax.fill(cs[:, 0], cs[:, 1], fc=ACCENT, ec=INK, lw=.5)
    if pr:
        x0, x1, y0, y1 = _bounds(pr); pad = .04*max(x1-x0, y1-y0)
        ax.set_xlim(x0-pad, x1+pad); ax.set_ylim(y0-pad, y1+pad)
    ax.set_aspect('equal'); _frame(ax)
    ax.text(.03, .05, '%s Province' % p.get('adm1', ''), transform=ax.transAxes, fontsize=6.5, color=MUTED)


def natural_base(A, sat=.45, light=.3):
    rgb = np.dstack([A['red'], A['green'], A['blue']]).astype(float)
    ok = rgb.sum(2) > 0
    if ok.any():
        lo = np.percentile(rgb[ok], 2, axis=0); hi = np.percentile(rgb[ok], 98, axis=0)
        rgb = np.clip((rgb-lo)/(hi-lo+1e-9), 0, 1)
    grey = rgb.mean(2, keepdims=True)
    rgb = grey+(rgb-grey)*sat
    return rgb*(1-light)+light


def block_edges(lab):
    e = np.zeros(lab.shape, bool)
    e[:-1] |= lab[:-1] != lab[1:]; e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return e & ((lab > 0) | np.roll(lab > 0, 1, 0) | np.roll(lab > 0, 1, 1))


def _box(fig, x, y, w, h, head=None):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, fill=False, ec=INK, lw=.8))
    if head:
        fig.text(x+w/2, y+h-.011, head, ha='center', va='top', fontsize=9, fontweight='bold', color=INK)
        fig.add_artist(plt.Line2D([x+.008, x+w-.008], [y+h-.032, y+h-.032], color=LINE, lw=.6, transform=fig.transFigure))


def map_sheet(R, study, path, kind='year'):
    import textwrap
    A, g = getattr(study, 'M', study.A), getattr(study, 'gm', study.g); y0, y1 = R['years']; S = R['setup']; p = R.get('place') or {}
    W, H = 16.54, 11.69
    fig = plt.figure(figsize=(W, H), dpi=study.s.dpi); fig.patch.set_facecolor('white')
    fig.patches.append(Rectangle((.012, .017), .976, .966, transform=fig.transFigure, fill=False, ec=INK, lw=1.6))
    fig.patches.append(Rectangle((.018, .025), .964, .950, transform=fig.transFigure, fill=False, ec=INK, lw=.5))
    cx0, cx1 = .715, .968; cw = cx1-cx0
    mw = .64; mh = mw*W/H*g.ny/g.nx
    if mh > .82: mh = .82; mw = mh*H/W*g.nx/g.ny
    mx = .045+(.655-mw)/2; my = .5-mh/2+.01
    ax = fig.add_axes([mx, my, mw, mh])
    ax.imshow(natural_base(A), extent=g.extent, interpolation='bilinear')
    cr, labels, cols = class_raster(A, R, kind)
    rgba = np.zeros(cr.shape+(4,))
    for i, c in enumerate(cols): rgba[cr == i] = (*matplotlib.colors.to_rgb(c), 1)
    rgba[cr == 98] = (*matplotlib.colors.to_rgb('#d6d0c4'), 1)
    ed = block_edges(np.where(cr >= 0, cr+1, 0)); rgba[ed & (cr >= 0)] = (.13, .13, .13, .55)
    ax.imshow(rgba, extent=g.extent, interpolation='nearest', zorder=2)
    graticule(ax, g)
    fig.text(mx+mw/2, my-.055, 'Map area %g x %g km, pixel size %d m, minimum mapping unit %g ha' % (*R['window_km'], g.res, S['mmu_ha']),
             ha='center', fontsize=8, color=MUTED)
    top = .968; gap = .006
    h1 = .135; y = top-h1; _box(fig, cx0, y, cw, h1)
    title = 'OIL PALM PLANTING YEAR' if kind == 'year' else 'OIL PALM AGE CLASS %d' % y1
    sub = 'Year each block was last planted, %d–%d' % (y0, y1) if kind == 'year' else 'Plantation age and stage in the crop cycle'
    fig.text(cx0+cw/2, y+h1-.022, title, ha='center', va='top', fontsize=16, fontweight='bold', color=INK)
    fig.text(cx0+cw/2, y+h1-.058, sub, ha='center', va='top', fontsize=9.5, color=MUTED)
    fig.text(cx0+cw/2, y+h1-.082, ('%s Regency' % p.get('adm2', '')).upper() if p.get('adm2') else '%.4f, %.4f' % tuple(R['center']),
             ha='center', va='top', fontsize=9.5, fontweight='bold', color=INK)
    fig.text(cx0+cw/2, y+h1-.103, ('%s Province, Indonesia' % p.get('adm1', '')) if p.get('adm1') else 'Indonesia', ha='center', va='top', fontsize=8.5, color=MUTED)
    h2 = .125; y -= gap+h2; _box(fig, cx0, y, cw, h2, 'SCALE AND ORIENTATION')
    _north_fig(fig, cx0+.035, y+.03, .055)
    rf = _rf(g, mw, W)
    fig.text(cx0+.075, y+h2-.045, 'Scale 1 : %s' % fmt(rf), fontsize=9.5, va='top', color=INK)
    km = 4 if g.nx*g.res >= 12000 else 2
    _scale_fig(fig, cx0+.08, y+.045, g, mw, km, fs=7.5)
    fig.text(cx0+.075, y+.012, 'WGS 1984 UTM Zone %d%s (EPSG:%d)' % (g.utm.zone, 'S' if g.utm.south else 'N', g.utm.epsg), fontsize=7.6, color=MUTED)
    nrow = len(labels)+3
    h3 = .075+nrow*.0215; y -= gap+h3; _box(fig, cx0, y, cw, h3, 'LEGEND')
    fig.text(cx0+.01, y+h3-.042, 'Planting year (current cycle)' if kind == 'year' else 'Age in %d' % y1, fontsize=8.3, va='top', color=MUTED)
    px_ha = g.res**2/1e4
    areas = [float((cr == i).sum()*px_ha) for i in range(len(labels))]; amax = max(areas+[1])
    yy = y+h3-.07
    for lab, c, a_ in zip(labels, cols, areas):
        fig.patches.append(Rectangle((cx0+.01, yy-.006), .02, .013, transform=fig.transFigure, fc=c, ec=INK, lw=.4))
        fig.text(cx0+.036, yy, lab, fontsize=8, va='center', color=INK)
        fig.patches.append(Rectangle((cx0+.165, yy-.004), .05*a_/amax, .008, transform=fig.transFigure, fc=c, ec='none'))
        fig.text(cx0+cw-.008, yy, fmt(a_), fontsize=7.6, va='center', ha='right', color=MUTED)
        yy -= .0215
    for c, lab in (('#d6d0c4', 'Oil palm, year not determined'), ('edge', 'Block boundary'), ('base', 'Other land (Landsat %d–%d)' % (y1-1, y1))):
        if c == 'edge':
            fig.add_artist(plt.Line2D([cx0+.01, cx0+.03], [yy, yy], color='#222222', lw=.9, alpha=.7, transform=fig.transFigure))
        elif c == 'base':
            gi = fig.add_axes([cx0+.01, yy-.006, .02, .013]); gi.imshow(natural_base(A)[g.ny//2:g.ny//2+20, g.nx//2:g.nx//2+30], aspect='auto'); _frame(gi)
        else:
            fig.patches.append(Rectangle((cx0+.01, yy-.006), .02, .013, transform=fig.transFigure, fc=c, ec=INK, lw=.4))
        fig.text(cx0+.036, yy, lab, fontsize=8, va='center', color=INK); yy -= .0215
    fig.text(cx0+cw-.008, y+h3-.042, 'ha', fontsize=7.6, va='top', ha='right', color=MUTED)
    iw = (cw-.03)/2; ih = iw*W/H; h4 = ih+.05; y -= gap+h4; _box(fig, cx0, y, cw, h4, 'LOCATION')
    inset_country(fig.add_axes([cx0+.01, y+.01, iw, ih]), study, R)
    inset_province(fig.add_axes([cx0+.02+iw, y+.01, iw, ih]), study, R)
    h6 = .045; h5 = y-gap-.032-h6-gap; y5 = y-gap-h5; _box(fig, cx0, y5, cw, h5, 'SOURCES')
    src = ['Landsat 5, 7, 8, 9 Collection 2 Level-2, USGS (%d–%d)' % (y0, y1), 'Oil palm extent: BIOPAMA Global Oil Palm v1.2 (Descals et al. 2024)',
           'Boundaries: FAO GAUL 2015; USDOS LSIB 2017',
           'Method: barest year of the last bare-soil spell in the yearly Landsat NBR record, confirmed by regrowth within %d years' % S['regrowth_years']]
    txt = '\n'.join('\n'.join(textwrap.wrap(t, 66, subsequent_indent='  ')) for t in src)
    fig.text(cx0+.01, y5+h5-.042, txt, fontsize=6.9, va='top', color=MUTED, linespacing=1.35)
    _box(fig, cx0, .032, cw, h6)
    fig.text(cx0+cw/2, .032+h6*.62, 'Cartographer: %s' % S['author'], ha='center', va='center', fontsize=9, color=INK)
    fig.text(cx0+cw/2, .032+h6*.28, 'fickingalindra.my.id', ha='center', va='center', fontsize=7.4, color=MUTED)
    fig.savefig(path, facecolor='white'); plt.close(fig)


def method_sheets(R, study, F, folder):
    import os, textwrap
    os.makedirs(folder, exist_ok=True)
    A, g, Z = study.A, study.g, study.Z; y0, y1 = R['years']; S = R['setup']; ext = g.extent
    CS = chip_stretch(F, sorted(F))
    SF = {y: (CS[y], float((F[y][..., 0] > 0).mean())) for y in F}
    good = [y for y in sorted(SF) if SF[y][1] >= .9]
    ya, yb = (good[0], good[-1]) if good else (y0, y1)
    palm = A['palm'] >= S['palm_share']
    base = grey_base(A, .6)

    def paint(mask_cols):
        out = base.copy()
        for m, c in mask_cols: out[m] = matplotlib.colors.to_rgb(c)
        return out
    runs = Z['runs']
    run_cols = [(palm & (runs == 0), '#d6d0c4'), (palm & (runs == 1), '#f4a259'), (palm & (runs == 2), '#c0392b'), (palm & (runs >= 3), '#5b1a12')]
    ext_cols = [(palm & (A['ind'] >= .5), '#2f7d4a'), (palm & (A['ind'] < .5), '#9ccf7a')]
    cr, labels, cols = class_raster(A, R, 'year')
    res_img = base.copy()
    for i, c in enumerate(cols): res_img[cr == i] = matplotlib.colors.to_rgb(c)
    res_img[cr == 98] = matplotlib.colors.to_rgb('#d6d0c4')
    steps = [
        ('Landsat %d, start of the record' % ya, 'Natural colour, median of the clear views that year. Clouds, shadows and haze are masked; '
         'Landsat 8–9 are matched to Landsat 5–7 so 37 years read as one record.', SF[ya][0], []),
        ('Landsat %d, today' % yb, 'The same window now. Every year in between gets its own composite, and for each pixel the '
         'yearly NBR = (NIR − SWIR2) / (NIR + SWIR2) is kept.', SF[yb][0], []),
        ('Bare-soil spells', 'Years where NBR fell to bare soil (≤ %.2f). One spell is a first clearing; two or more mean the block '
         'was cleared again for replanting.' % S['bare_nbr'], paint(run_cols),
         [('#d6d0c4', 'None since %d' % y0), ('#f4a259', 'One spell'), ('#c0392b', 'Two spells'), ('#5b1a12', 'Three or more')]),
        ('Oil palm extent', 'Only pixels mapped as closed-canopy oil palm by BIOPAMA v1.2 (10 m, Descals et al. 2024) are dated, '
         'so other land is left out.', paint(ext_cols), [('#2f7d4a', 'Industrial estate'), ('#9ccf7a', 'Smallholder')]),
        ('Result: planting year', 'The barest year of the last bare-soil spell, kept only when the palms grew back within %d years. '
         'Blocks never bare since %d were planted earlier.' % (S['regrowth_years'], y0), res_img,
         [(c, l) for c, l in zip(cols, labels)]),
    ]
    out = []
    for i, (title, sub, img, leg) in enumerate(steps, 1):
        fig = plt.figure(figsize=(14, 7.6), dpi=170)
        aw = .62; ah = aw*14/7.6*g.ny/g.nx
        if ah > .9: ah = .9; aw = ah*7.6/14*g.nx/g.ny
        ax = fig.add_axes([.04, (1-ah)/2, aw, ah]); ax.imshow(img, extent=ext, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(.9)
        L = 2000; x0 = ext[0]+(ext[1]-ext[0])*.04; yb0 = ext[2]+(ext[3]-ext[2])*.05
        ax.add_patch(Rectangle((x0, yb0), L/2, (ext[3]-ext[2])*.012, fc=INK, ec=INK))
        ax.add_patch(Rectangle((x0+L/2, yb0), L/2, (ext[3]-ext[2])*.012, fc='white', ec=INK))
        ax.text(x0+L, yb0+(ext[3]-ext[2])*.03, '2 km', fontsize=10, ha='center', color=INK,
                bbox=dict(fc='white', ec='none', alpha=.7, pad=1))
        xn, yn = ext[1]-(ext[1]-ext[0])*.04, ext[3]-(ext[3]-ext[2])*.06
        ax.annotate('N', xy=(xn, yn), xytext=(xn, yn-(ext[3]-ext[2])*.12), ha='center', va='top', color=INK, fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6), bbox=dict(fc='white', ec='none', alpha=.7, pad=1))
        R0 = .04+aw+.035; W0 = .965-R0
        fig.text(R0, .93, 'Step %d of %d' % (i, len(steps)), fontsize=10.5, color=ACCENT, fontweight='bold', va='top')
        fig.text(R0, .885, '\n'.join(textwrap.wrap(title, 24)), fontsize=19, fontweight='bold', va='top', linespacing=1.15)
        fig.text(R0, .75, '\n'.join(textwrap.wrap(sub, 38)), fontsize=10.5, color=MUTED, va='top', linespacing=1.5)
        yy = .53 if len(leg) > 6 else .44
        fs = 9.5 if len(leg) > 6 else 10.5
        for c, l in leg:
            fig.patches.append(Rectangle((R0, yy), .018, .022, transform=fig.transFigure, fc=c, ec=INK, lw=.4))
            fig.text(R0+.028, yy+.011, l, fontsize=fs, color=INK, va='center'); yy -= .034 if len(leg) > 6 else .04
        fig.text(R0, .1, '%g x %g km area, %s\ncentre %.4f, %.4f' % (*R['window_km'], place_label(R), R['center'][1], R['center'][0]),
                 fontsize=9, color=MUTED, va='top')
        fig.text(R0, .045, 'Map: %s' % S['author'], fontsize=9.5, color=INK)
        pth = os.path.join(folder, '%02d.png' % i); fig.savefig(pth); plt.close(fig); out.append(pth)
    return out


def chip_stretch(F, years):
    vals = np.concatenate([F[y][F[y][..., 0] > 0].reshape(-1, 3)[::20] for y in years if (F[y][..., 0] > 0).any()])
    lo = np.percentile(vals, 1, axis=0); hi = np.percentile(vals, 99.2, axis=0)
    out = {}
    for y in years:
        f = F[y]; miss = f[..., 0] <= 0
        rgb = np.clip((f-lo)/(hi-lo+1e-9), 0, 1)**.8
        gr = rgb.mean(2, keepdims=True); rgb = np.clip(gr+(rgb-gr)*1.35, 0, 1)
        rgb[miss] = .85
        out[y] = rgb
    return out


def notable_blocks(cr, labels, px_ha, n=3):
    picks = []
    order = sorted(range(1, len(labels)), key=lambda i: -(cr == i).sum())
    for i in order:
        L, k = ndimage.label(cr == i)
        if k == 0: continue
        sizes = ndimage.sum(np.ones_like(L), L, range(1, k+1)); j = int(np.argmax(sizes))+1
        m = L == j
        dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]; r, c = np.unravel_index(np.argmax(dist), dist.shape)
        picks.append({'cls': i, 'label': labels[i], 'ha': float(sizes[j-1]*px_ha), 'r': int(r), 'c': int(c)})
        if len(picks) == n: break
    return picks


def feature_map(R, study, chips, path, kind='year', inset_imgs=None):
    import textwrap
    A, g = getattr(study, 'M', study.A), getattr(study, 'gm', study.g); y0, y1 = R['years']; S = R['setup']; p = R.get('place') or {}
    W, H = 16.54, 11.69; BG = '#f6f3ec'
    fig = plt.figure(figsize=(W, H), dpi=study.s.dpi); fig.patch.set_facecolor(BG)
    title = 'OIL PALM PLANTING YEAR MAP' if kind == 'year' else 'OIL PALM AGE CLASS MAP %d' % y1
    fig.text(.035, .945, title, fontsize=24, fontweight='bold', color=INK, va='top')
    fig.text(.035, .897, ('%s Regency, %s Province, Indonesia' % (p.get('adm2', ''), p.get('adm1', ''))) if p.get('adm2') else place_label(R),
             fontsize=13, color=INK, va='top')
    new_pct = 100*R['dated_ha']/max(R['palm_ha'], 1e-9)
    rows = [('Oil palm area', '%s ha' % fmt(R['palm_ha'])), ('Planting blocks', fmt(R.get('blocks', 0))),
            ('Median planting year', str(R['median_year'])), ('Planted since %d' % y0, '%.0f%% of area' % new_pct)]
    for i, (k, v) in enumerate(rows):
        x = .62+(i % 2)*.18; yv = .94-(i//2)*.032
        fig.text(x, yv, k, fontsize=9.5, color=MUTED, va='top'); fig.text(x+.165, yv, v, fontsize=9.5, color=INK, va='top', ha='right', fontweight='bold')
    fig.add_artist(plt.Line2D([.035, .965], [.855, .855], color=INK, lw=1.0, transform=fig.transFigure))

    mh = .56; mw = mh*H/W*g.nx/g.ny; mx, my = .045, .255
    ax = fig.add_axes([mx, my, mw, mh])
    ax.imshow(natural_base(A, .55, .22), extent=g.extent, interpolation='bilinear')
    cr, labels, cols = class_raster(A, R, kind)
    rgba = np.zeros(cr.shape+(4,))
    for i, c in enumerate(cols): rgba[cr == i] = (*matplotlib.colors.to_rgb(c), 1)
    rgba[cr == 98] = (*matplotlib.colors.to_rgb('#d6d0c4'), 1)
    ed = block_edges(np.where(cr >= 0, cr+1, 0)); rgba[ed & (cr >= 0)] = (1, 1, 1, .85)
    ax.imshow(rgba, extent=g.extent, interpolation='nearest', zorder=2)
    graticule(ax, g, fs=7.5)
    e = g.extent; px_ha = g.res**2/1e4
    picks = notable_blocks(cr, labels, px_ha)
    for k, pk in enumerate(picks):
        x = e[0]+(pk['c']+.5)*g.res; y = e[3]-(pk['r']+.5)*g.res
        ax.plot(x, y, 'o', ms=17, mfc='white', mec=INK, mew=1.2, zorder=6)
        ax.text(x, y, 'ABC'[k], ha='center', va='center', fontsize=10, fontweight='bold', color=INK, zorder=7)
    bx = ax.inset_axes([.015, .02, .3, .13]); bx.set_facecolor((1, 1, 1, .85)); bx.set_xticks([]); bx.set_yticks([])
    for sp in bx.spines.values(): sp.set_visible(False)
    bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    km = 4 if g.nx*g.res >= 12000 else 2; frac = km*1000/(g.nx*g.res)/.3*.62
    for i in range(4):
        bx.add_patch(Rectangle((.06+i*frac/4, .3), frac/4, .14, fc=INK if i % 2 == 0 else 'white', ec=INK, lw=.7))
    for i, t in enumerate([0, km/2, km]):
        bx.text(.06+i*frac/2, .52, ('%g' % t)+(' km' if i == 2 else ''), ha='center', fontsize=7.5, color=INK)
    bx.text(.06, .82, 'Scale 1:%s' % fmt(_rf(g, mw, W)), fontsize=7.5, color=INK, va='center')
    from matplotlib.patches import Polygon
    bx.add_patch(Polygon([[.86, .82], [.81, .22], [.86, .38]], fc=INK, ec=INK, lw=.6))
    bx.add_patch(Polygon([[.86, .82], [.91, .22], [.86, .38]], fc='white', ec=INK, lw=.6))
    bx.text(.86, .9, 'N', ha='center', fontsize=9, fontweight='bold', color=INK)
    fig.text(mx, my-.026, 'Projection: WGS 1984 UTM Zone %d%s (EPSG:%d). Pixel size %d m, minimum mapping unit %g ha.'
             % (g.utm.zone, 'S' if g.utm.south else 'N', g.utm.epsg, g.res, S['mmu_ha']), fontsize=7.5, color=MUTED, va='top')

    rx = mx+mw+.04; rw = .965-rx
    fig.text(rx, .815, 'Legend', fontsize=11, fontweight='bold', color=INK, va='top')
    fig.text(rx, .795, 'Planting year, area in hectares' if kind == 'year' else 'Age class in %d' % y1, fontsize=8.5, color=MUTED, va='top')
    if kind == 'year':
        lg = fig.add_axes([rx+.045, .63, rw-.045, .15]); lg.set_facecolor(BG)
        ys = sorted(int(k) for k in R['by_year']); vals = [R['by_year'][str(y)] if str(y) in R['by_year'] else R['by_year'][y] for y in ys]
        cl, ccols = year_classes(y0, y1, R.get('by_year'))
        def cidx(y):
            for i, (a, b, _) in enumerate(cl[1:], 1):
                if a <= y <= b: return i
            return 0
        lg.bar(ys, vals, color=[ccols[cidx(y)] for y in ys], width=.85)
        for sp in ('top', 'right', 'left'): lg.spines[sp].set_visible(False)
        lg.set_yticks([]); lg.tick_params(labelsize=7.5, colors=MUTED); lg.set_xlim(y0-.7, y1+.7)
        lg.set_xticks([a for a, _, _ in cl[1:-1]]+[y1]); lg.set_xticklabels([str(a) for a, _, _ in cl[1:-1]]+[str(y1)], fontsize=7)
        mx_ = max(vals); im_ = int(np.argmax(vals))
        lg.text(ys[im_], mx_, ' %s ha' % fmt(mx_), fontsize=7.5, color=INK, va='bottom', ha='center')
        ob = fig.add_axes([rx, .63, .03, .15]); ob.set_facecolor(BG)
        ob.bar([0], [R['old_ha']], color=ccols[0], width=.8); ob.set_ylim(0, max(R['old_ha'], mx_)*1.1)
        lg.set_ylim(0, max(R['old_ha'], mx_)*1.1)
        ob.set_xticks([0]); ob.set_xticklabels(['< %d' % y0], fontsize=7); ob.set_yticks([]); ob.tick_params(colors=MUTED)
        for sp in ('top', 'right', 'left'): ob.spines[sp].set_visible(False)
        ob.text(0, R['old_ha'], ' %s' % fmt(R['old_ha']), fontsize=7, ha='center', va='bottom', color=INK)
        yy = .595
    else:
        yy = .78
        for c, cc in zip(cols, R['age_classes']):
            fig.patches.append(Rectangle((rx, yy-.012), .02, .016, transform=fig.transFigure, fc=c, ec=INK, lw=.4))
            fig.text(rx+.028, yy-.004, '%s, %s' % (cc['label'], cc['meaning'][0].lower()+cc['meaning'][1:]), fontsize=8.5, va='center', color=INK)
            fig.text(rx+rw, yy-.004, '%s ha (%.0f%%)' % (fmt(cc['ha']), cc['pct']), fontsize=8.2, va='center', ha='right', color=MUTED)
            yy -= .026
        yy -= .01
    fig.text(rx, yy, 'Age structure in %d (%% of area)' % y1, fontsize=10, fontweight='bold', color=INK, va='top')
    sb = fig.add_axes([rx, yy-.06, rw, .03]); sb.set_xlim(0, 100); sb.set_ylim(0, 1); sb.axis('off')
    acc = 0
    acols = age_cols(age_bins(y0, y1), AGE_COLORS); nb = len(acols)
    for c, cc in zip(acols, R['age_classes']):
        sb.add_patch(Rectangle((acc, 0), cc['pct'], 1, fc=c, ec=BG, lw=1))
        if cc['pct'] >= 9: sb.text(acc+cc['pct']/2, .5, '%.0f%%' % cc['pct'], ha='center', va='center', fontsize=8, color='white' if c in ('#1f7a3d', '#c0392b', UNRESOLVED) else INK, fontweight='bold')
        acc += cc['pct']
    for i, (c, cc) in enumerate(zip(acols, R['age_classes'])):
        fig.patches.append(Rectangle((rx+i*rw/nb, yy-.083), .01, .01, transform=fig.transFigure, fc=c, ec='none'))
        fig.text(rx+i*rw/nb+.013, yy-.078, cc['label'], fontsize=7.4, color=MUTED, va='center')
    yy -= .115
    fig.text(rx, yy, 'Reference blocks', fontsize=10, fontweight='bold', color=INK, va='top'); yy -= .025
    for k, pk in enumerate(picks):
        fig.text(rx, yy, 'ABC'[k], fontsize=10, fontweight='bold', color=INK, va='top',
                 bbox=dict(boxstyle='circle,pad=.25', fc='white', ec=INK, lw=1))
        fig.text(rx+.025, yy, '%s, %s ha' % (pk['label'].replace(' (replanting)', ''), fmt(pk['ha'])), fontsize=8.6, color=INK, va='top')
        yy -= .027
    yy -= .008
    fig.text(rx, yy, 'Location map', fontsize=10, fontweight='bold', color=INK, va='top')
    ih = yy-.03-.255; iw = (rw-.01)/2
    if inset_imgs:
        for k, im in enumerate(inset_imgs):
            ia = fig.add_axes([rx+k*(iw+.01), .255, iw, ih]); ia.imshow(im); ia.axis('off')
    else:
        inset_country(fig.add_axes([rx, .255, iw, ih]), study, R)
        inset_province(fig.add_axes([rx+iw+.01, .255, iw, ih]), study, R)

    fig.text(.035, .2, 'Landsat imagery %d-%d (natural colour)' % (min(chips), max(chips)), fontsize=10, fontweight='bold', color=INK, va='top')
    yrs = sorted(chips); n = len(yrs); cw_ = (.93-(n-1)*.008)/n; ch_ = cw_*W/H*g.ny/g.nx
    for k, y in enumerate(yrs):
        ca = fig.add_axes([.035+k*(cw_+.008), .178-ch_, cw_, ch_]); ca.imshow(chips[y], interpolation='bilinear', aspect='auto'); ca.set_xticks([]); ca.set_yticks([])
        for sp in ca.spines.values(): sp.set_color(INK); sp.set_linewidth(.6)
        ca.text(.04, .93, str(y), transform=ca.transAxes, fontsize=10, fontweight='bold', color='white', va='top',
                bbox=dict(fc=INK, ec='none', pad=2, alpha=.85))
    fig.text(.035, .025, 'Data sources: Landsat 5, 7, 8 and 9 Collection 2 Level-2 (USGS); oil palm extent from BIOPAMA Global Oil Palm v1.2 (Descals et al., 2024); '
             'administrative boundaries from FAO GAUL 2015 and USDOS LSIB.', fontsize=7.5, color=MUTED, va='bottom')
    fig.text(.965, .025, 'Cartographer: %s' % S['author'], fontsize=8.5, color=INK, va='bottom', ha='right')
    fig.savefig(path, facecolor=BG); plt.close(fig)


FONT_URLS = {'display': 'https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf',
             'body': 'https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Medium.ttf',
             'bold': 'https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-SemiBold.ttf'}
SHOW_COLS = ['#5a4e7a', '#9c179e', '#d8576b', '#f48849', '#fdca26', '#f0f921', '#2ee6d6']
AGE_SHOW = ['#2ee6d6', '#7ee081', '#f0f921', '#f48849', '#e8336b']


def load_fonts(folder='fonts'):
    import os, urllib.request
    from matplotlib import font_manager
    os.makedirs(folder, exist_ok=True); names = {}
    for k, u in FONT_URLS.items():
        f = os.path.join(folder, u.rsplit('/', 1)[1])
        try:
            if not os.path.exists(f) or os.path.getsize(f) < 10000: urllib.request.urlretrieve(u, f)
            font_manager.fontManager.addfont(f); names[k] = font_manager.FontProperties(fname=f).get_name()
        except Exception:
            names[k] = None
    fb = 'DejaVu Sans'
    return {'display': names.get('display') or fb, 'body': names.get('body') or fb, 'bold': names.get('bold') or fb}


SHOW_WIN = (.335, .285, .945, .795)


def showcase_extent(g, W=16.54, H=11.69):
    gw, gh = g.nx*g.res, g.ny*g.res
    xr = max(gw/(SHOW_WIN[2]-SHOW_WIN[0]), gh/(SHOW_WIN[3]-SHOW_WIN[1])*W/H); yr = xr*H/W
    e = g.extent
    cx = (SHOW_WIN[0]+SHOW_WIN[2])/2; cy = (SHOW_WIN[1]+SHOW_WIN[3])/2
    x0 = (e[0]+e[1])/2-cx*xr; y0 = (e[2]+e[3])/2-cy*yr
    return [x0, x0+xr, y0, y0+yr]


def extent_grid(g, ext):
    nx = int((ext[1]-ext[0])/g.res)+1; ny = int((ext[3]-ext[2])/g.res)+1
    return Grid(g.utm, math.floor(ext[0]/g.res)*g.res, math.ceil(ext[3]/g.res)*g.res, g.res, nx, ny)


def dark_base(rgb, k=.6, gamma=1.1):
    ok = rgb.sum(2) > 0
    gr = rgb.mean(2)
    if ok.any():
        lo, hi = np.percentile(gr[ok], [1, 99]); gr = np.clip((gr-lo)/(hi-lo+1e-9), 0, 1)**gamma
    gr[~ok] = 0
    return np.dstack([gr*k*.92, gr*k*.96, gr*k])


def glow(rgba, sigma=3, strength=.55):
    a = ndimage.gaussian_filter(rgba[..., 3], sigma)
    out = np.zeros_like(rgba)
    for i in range(3): out[..., i] = ndimage.gaussian_filter(rgba[..., i]*rgba[..., 3], sigma)/np.maximum(a, 1e-6)
    out[..., 3] = np.clip(a*strength, 0, 1)
    return out


def zoom_window(cr, size=(80, 55), img=None):
    best, pos = -1, (0, 0); st = max(10, size[0]//5)
    ed = block_edges(np.where(cr >= 0, cr+1, 0)) & (cr >= 0) & (cr < 98)
    if img is not None:
        gr = np.hypot(ndimage.sobel(img, 0), ndimage.sobel(img, 1)); gr = ndimage.uniform_filter(gr, 3)
        gm_ = float(np.mean(gr))+1e-9
    for r in range(0, cr.shape[0]-size[1], st):
        for c in range(0, cr.shape[1]-size[0], st):
            w = cr[r:r+size[1], c:c+size[0]]
            k = len(set(np.unique(w[(w >= 0) & (w < 98)])))
            v = min(k, 4) + 3*min(ed[r:r+size[1], c:c+size[0]].mean()*20, 1) + 2*((w >= 0) & (w < 98)).mean()
            if img is not None:
                ew = ed[r:r+size[1], c:c+size[0]]
                if ew.sum() > 20: v += 3*min(gr[r:r+size[1], c:c+size[0]][ew].mean()/gm_/2, 1)
                v += .6*(c/cr.shape[1]) + .6*(r/cr.shape[0])
            if v > best: best, pos = v, (r, c)
    return pos


def evidence_window(study, frames, size=(50, 33)):
    A = study.A; cat, py = A['cat_map'], A['py_map']; w, h = size
    best, out = -1e9, None
    for Y in sorted(frames):
        m = (cat == PLANTED) & (py == Y)
        if m.sum() < 150: continue
        br = frames[Y].mean(2)
        for r in range(0, cat.shape[0]-h, 6):
            for c in range(0, cat.shape[1]-w, 6):
                mm = m[r:r+h, c:c+w]; f = mm.mean()
                if f < .15 or f > .65: continue
                b = br[r:r+h, c:c+w]
                sc = (b[mm].mean()-b[~mm].mean())*min(f/.35, 1)
                if sc > best: best, out = sc, (r, c, Y)
    return out


def showcase_map(R, study, chips, path, kind='year', ctx=None, fonts=None, inset_imgs=None, dpi=None, frames=None):
    from matplotlib.patches import ConnectionPatch, Polygon
    import textwrap
    A, g = getattr(study, 'M', study.A), getattr(study, 'gm', study.g); y0, y1 = R['years']; S = R['setup']; p = R.get('place') or {}
    fonts = fonts or {'display': 'DejaVu Sans', 'body': 'DejaVu Sans', 'bold': 'DejaVu Sans'}
    FD, FB, FS = fonts['display'], fonts['body'], fonts['bold']
    W, H = 16.54, 11.69; BG = '#0d1012'; FG = '#eef0ee'; MU = '#9aa3a6'; LN = '#5c666a'
    fig = plt.figure(figsize=(W, H), dpi=dpi or study.s.dpi); fig.patch.set_facecolor(BG)
    cr, labels, _ = class_raster(A, R, kind)
    if kind == 'year':
        n_ = len(labels)-2; pl = matplotlib.colormaps['plasma']
        cols = ['#5a4e7a'] + [matplotlib.colors.to_hex(pl(.18+.77*i/max(n_-1, 1))) for i in range(n_)] + ['#2ee6d6']
    else:
        cols = age_cols(age_bins(y0, y1), AGE_SHOW)
    rgba = np.zeros(cr.shape+(4,))
    for i, c in enumerate(cols[:len(labels)]): rgba[cr == i] = (*matplotlib.colors.to_rgb(c), .78)
    rgba[cr == 98] = (.55, .55, .55, .7)
    ed = block_edges(np.where(cr >= 0, cr+1, 0)); rgba[ed & (cr >= 0)] = (0.05, 0.06, 0.07, .9)
    e = g.extent
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis('off')
    if ctx is not None:
        cg, crgb = ctx
        ax.imshow(dark_base(crgb, .5), extent=cg.extent, interpolation='bilinear')
    win = dark_base(np.dstack([A['red'], A['green'], A['blue']]), .62)
    ax.imshow(win, extent=e, interpolation='bilinear')
    ax.imshow(glow(rgba, 4, .7), extent=e, interpolation='bilinear')
    ax.imshow(rgba, extent=e, interpolation='nearest')
    ax.plot([e[0], e[1], e[1], e[0], e[0]], [e[2], e[2], e[3], e[3], e[2]], color=FG, lw=.8, ls=(0, (4, 3)), alpha=.8)
    ce = showcase_extent(g, W, H); xr = ce[1]-ce[0]; yr = ce[3]-ce[2]
    ax.set_xlim(ce[0], ce[1]); ax.set_ylim(ce[2], ce[3])
    u = g.utm; st = 2/60
    lons = [u.inv(x, y)[0] for x in ce[:2] for y in ce[2:]]; lats = [u.inv(x, y)[1] for x in ce[:2] for y in ce[2:]]
    lon_t = np.arange(math.ceil(min(lons)/st)*st, max(lons), st); lat_t = np.arange(math.ceil(min(lats)/st)*st, max(lats), st)
    for lo in lon_t:
        for la in lat_t:
            x, y = u.fwd(lo, la)
            if e[0] < x < e[1] and e[2] < y < e[3]: ax.plot(x, y, marker='+', ms=11, mew=.9, color=FG, alpha=.6, zorder=5)
    lat_mid = u.inv((ce[0]+ce[1])/2, (ce[2]+ce[3])/2)[1]; lon_mid = u.inv((ce[0]+ce[1])/2, (ce[2]+ce[3])/2)[0]
    for lo in lon_t:
        x = u.fwd(lo, lat_mid)[0]
        if ce[0]+.3*xr < x < ce[1]-.02*xr:
            for yy_, va in ((ce[3], 'top'), (ce[2], 'bottom')):
                ax.text(x, yy_, ' %s ' % _dms(lo, 'E', 'W'), color=FG, fontsize=8.5, fontfamily=FB, ha='center', va=va,
                        bbox=dict(fc=BG, ec='none', pad=1.5))
    for la in lat_t:
        y = u.fwd(lon_mid, la)[1]
        if ce[2]+.04*(ce[3]-ce[2]) < y < ce[3]-.04*(ce[3]-ce[2]):
            ax.text(ce[1], y, ' %s ' % _dms(la, 'N', 'S'), color=FG, fontsize=8.5, fontfamily=FB, ha='right', va='center', rotation=90,
                    bbox=dict(fc=BG, ec='none', pad=1.5))
    fig.patches.append(Rectangle((0, 0), .285, 1, transform=fig.transFigure, fc=BG, ec='none', alpha=.86, zorder=1))
    fig.add_artist(plt.Line2D([.285, .285], [0, 1], color=LN, lw=.8, transform=fig.transFigure))
    lx = .025; tw = .25
    t1 = 'OIL PALM' if kind == 'year' else 'OIL PALM'
    t2 = 'PLANTING YEAR' if kind == 'year' else 'AGE CLASS %d' % y1
    fig.text(lx, .965, t1, fontsize=40, fontfamily=FD, color=FG, va='top')
    fig.text(lx, .905, t2, fontsize=40, fontfamily=FD, color=cols[3] if kind == 'year' else cols[2], va='top')
    fig.text(lx, .842, ('%s Regency, %s, Indonesia' % (p.get('adm2', ''), p.get('adm1', ''))) if p.get('adm2') else place_label(R),
             fontsize=12.5, fontfamily=FS, color=FG, va='top')
    desc = ('Planting year of oil palm blocks in a %g x %g km area, derived from annual Landsat imagery %d-%d. '
            'Each block is dated to the year it was last cleared and planted: the lowest NBR value in the last bare-soil period, '
            'followed by canopy regrowth within %d years. Blocks with no bare soil since %d were planted earlier.'
            % (*R['window_km'], y0, y1, S['regrowth_years'], y0)) if kind == 'year' else \
           ('Age of each oil palm block in %d, grouped by stage in the crop cycle: %s.' % (y1, '; '.join(
               '%s %s' % (b[3].lower() if b[4] >= 0 else 'planted before %d' % y0, '(%s)' % b[2]) for b in age_bins(y0, y1))))
    dl = textwrap.wrap(desc, 66)
    fig.text(lx, .805, '\n'.join(dl), fontsize=9.5, fontfamily=FB, color=MU, va='top', linespacing=1.2)
    yy = .805-len(dl)*.0175-.03
    nl = len(labels)
    need = .064+(nl+2)*.024+.075+nl*.021+.05+.06
    k_ = min(1, (yy-.135)/need); rs = .024*k_; brs = .021*k_
    fig.text(lx, yy, 'LEGEND', fontsize=17, fontfamily=FD, color=FG, va='top'); yy -= .038
    fig.text(lx, yy, 'Planting year (current cycle)' if kind == 'year' else 'Age in %d' % y1, fontsize=10.5, fontfamily=FS, color=FG, va='top'); yy -= .026
    px_ha = g.res**2/1e4
    for i, lab in enumerate(labels):
        c = cols[i]
        fig.patches.append(Rectangle((lx, yy-.013), .022, .015, transform=fig.transFigure, fc=c, ec='none', zorder=3))
        fig.patches.append(Rectangle((lx-.002, yy-.015), .026, .019, transform=fig.transFigure, fc=c, ec='none', alpha=.25, zorder=2))
        fig.text(lx+.032, yy-.005, lab.replace(' yr', ' years'), fontsize=10, fontfamily=FB, color=FG, va='center')
        fig.text(lx+tw, yy-.005, '%s ha' % fmt(float((cr == i).sum()*px_ha)), fontsize=9.5, fontfamily=FB, color=MU, va='center', ha='right')
        yy -= rs
    fig.add_artist(plt.Line2D([lx, lx+.022], [yy-.005, yy-.005], color=FG, lw=.8, ls=(0, (4, 3)), transform=fig.transFigure))
    fig.text(lx+.032, yy-.005, 'Study area', fontsize=10, fontfamily=FB, color=FG, va='center'); yy -= rs
    fig.text(lx+.006, yy-.006, '+', fontsize=13, color=FG, va='center', ha='center')
    fig.text(lx+.032, yy-.005, "Graticule, 2' interval", fontsize=10, fontfamily=FB, color=FG, va='center'); yy -= .04
    fig.text(lx, yy, 'AREA BY PERIOD (HA)' if kind == 'year' else 'AREA BY AGE CLASS (HA)', fontsize=17, fontfamily=FD, color=FG, va='top'); yy -= .035
    areas = [float((cr == i).sum()*px_ha) for i in range(len(labels))]
    bh = brs*.76; amax = max(areas+[1])
    for i, (lab, a_) in enumerate(zip(labels, areas)):
        fig.text(lx+.07, yy-bh/2, lab.split(' (')[0].split(',')[0], fontsize=8.5, fontfamily=FB, color=MU, va='center', ha='right')
        fig.patches.append(Rectangle((lx+.075, yy-bh), (tw-.12)*a_/amax, bh*.8, transform=fig.transFigure, fc=cols[i], ec='none'))
        fig.text(lx+.078+(tw-.12)*a_/amax, yy-bh/2, ' %s' % fmt(a_), fontsize=8.5, fontfamily=FB, color=FG, va='center')
        yy -= brs
    yy -= .05
    km = 4 if g.nx*g.res >= 12000 else 2; frac = km*1000/xr
    for i in range(4):
        fig.patches.append(Rectangle((lx+i*frac/4, yy-.008), frac/4, .007, transform=fig.transFigure, fc=FG if i % 2 == 0 else BG, ec=FG, lw=.7))
    for i, t in enumerate([0, km/2, km]):
        fig.text(lx+i*frac/2, yy+.004, ('%g' % t)+(' km' if i == 2 else ''), fontsize=8.5, fontfamily=FB, color=FG, ha='center', va='bottom')
    fig.text(lx, yy-.025, 'Scale 1:%s (A3)' % fmt(int(round(xr/(W*.0254)/5000)*5000)), fontsize=8.5, fontfamily=FB, color=MU, va='top')
    nx_, ny_ = lx+tw-.03, yy-.03
    fig.patches.append(Polygon([[nx_, ny_+.045], [nx_-.011, ny_-.02], [nx_, ny_-.006]], closed=True, transform=fig.transFigure, fc=FG, ec=FG, lw=.8))
    fig.patches.append(Polygon([[nx_, ny_+.045], [nx_+.011, ny_-.02], [nx_, ny_-.006]], closed=True, transform=fig.transFigure, fc=BG, ec=FG, lw=.8))
    fig.text(nx_, ny_+.05, 'N', fontsize=13, fontfamily=FS, color=FG, ha='center', va='bottom')
    fig.text(lx, .092, 'Datum WGS 1984, UTM Zone %d%s (EPSG:%d)\nData: Landsat 5, 7, 8, 9 C2 L2 (USGS); Sentinel-2 L2A (ESA); BIOPAMA Global\nOil Palm v1.2 (Descals et al., 2024); FAO GAUL 2015; USDOS LSIB. MMU %g ha.'
             % (g.utm.zone, 'S' if g.utm.south else 'N', g.utm.epsg, S['mmu_ha']), fontsize=7.8, fontfamily=FB, color=MU, va='top', linespacing=1.2)
    fig.text(lx, .03, 'Map by %s' % S['author'], fontsize=10, fontfamily=FS, color=FG, va='bottom')
    fig.text(lx+tw, .03, 'fickingalindra.my.id', fontsize=9, fontfamily=FB, color=MU, va='bottom', ha='right')

    fx0 = (e[0]-ce[0])/xr; fx1 = (e[1]-ce[0])/xr; fy0 = (e[2]-ce[2])/yr; fy1 = (e[3]-ce[2])/yr
    gap = .012; top = .962; cap_w = .1
    yrs = sorted(chips); pick = [yrs[0], yrs[len(yrs)//2], yrs[-1]]
    chh = top-(fy1+.022); cw = chh*H/W*g.nx/g.ny
    if 3*cw+2*gap > fx1-fx0-cap_w: cw = (fx1-fx0-cap_w-2*gap)/3; chh = cw*W/H*g.ny/g.nx
    for k, y in enumerate(pick):
        ca = fig.add_axes([fx0+k*(cw+gap), top-chh, cw, chh]); ca.imshow(chips[y], aspect='auto', interpolation='bilinear')
        ca.set_xticks([]); ca.set_yticks([])
        for sp in ca.spines.values(): sp.set_color(FG); sp.set_linewidth(.7)
        ca.text(.035, .93, str(y), transform=ca.transAxes, fontsize=14, fontfamily=FD, color=FG, va='top',
                bbox=dict(fc=BG, ec='none', alpha=.8, pad=2))
    tx = fx0+3*cw+2*gap+.014
    fig.text(tx, top, 'LANDSAT\nTIME SERIES', fontsize=15, fontfamily=FD, color=FG, va='top', linespacing=1.25)
    fig.text(tx, top-.055, 'Natural colour,\nsame extent as\nthe study area', fontsize=8.5, fontfamily=FB, color=MU, va='top', linespacing=1.3)

    bot = .035; bh = fy0-.022-bot
    g30 = study.g; kf = int(round(g30.res/g.res))
    ev = evidence_window(study, frames) if frames else None
    if ev:
        r30, c30, YE = ev; zw30, zh30 = 50, 33
        r0, c0, zw, zh = r30*kf, c30*kf, zw30*kf, zh30*kf
    else:
        zw, zh = int(round(1500/g.res)), int(round(1000/g.res))
        bright = np.dstack([A['red'], A['green'], A['blue']]).astype(np.float32).mean(2)
        r0, c0 = zoom_window(cr, (zw, zh), bright); YE = None
    zx0 = e[0]+c0*g.res; zy1 = e[3]-r0*g.res; zx1 = zx0+zw*g.res; zy0 = zy1-zh*g.res
    ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0, fill=False, ec=FG, lw=1.3, zorder=6))
    iw = bh*H/W
    pw = bh*H/W*zw/zh; gap2 = .012
    avail = fx1-(fx0+iw+.015)
    if YE is not None and 2*pw+gap2 > avail:
        pw = (avail-gap2)/2; ph = pw*W/H*zh/zw
    else:
        ph = bh
    panels = []
    sub = cr[r0:r0+zh, c0:c0+zw]; ext_z = [zx0, zx1, zy0, zy1]

    def local(rgb):
        okp = rgb.sum(2) > 0
        if not okp.any(): return rgb
        lo_ = np.percentile(rgb[okp], 1, axis=0); hi_ = np.percentile(rgb[okp], 99, axis=0)
        return np.clip((rgb-lo_)/(hi_-lo_+1e-9), 0, 1)**.9
    s2 = local(np.dstack([A['red'], A['green'], A['blue']]).astype(np.float32)[r0:r0+zh, c0:c0+zw])
    if YE is not None:
        ls = frames[YE][r30:r30+zh30, c30:c30+zw30]
        panels = [(fx1-2*pw-gap2, ls, 'LANDSAT %d' % YE, 'Year of planting: blocks are bare soil'),
                  (fx1-pw, s2, 'SENTINEL-2 %d' % max(chips), 'Same blocks today, %d m pixels' % g.res)]
    else:
        panels = [(fx1-pw, s2, 'SENTINEL-2 DETAIL, %d' % max(chips), '%.1f x %.1f km, %d m pixels' % (zw*g.res/1000, zh*g.res/1000, g.res))]
    yid = None
    if YE is not None:
        for i_, (a_, b_, _) in enumerate(year_classes(y0, y1, R.get('by_year'))[0]):
            if a_ <= YE <= b_ and i_ > 0: yid = i_
    axes_z = []
    for x_, im_, t1_, t2_ in panels:
        za = fig.add_axes([x_, bot+(bh-ph), pw, ph]); axes_z.append(za)
        za.imshow(im_, extent=ext_z, interpolation='bilinear')
        for i, c in enumerate(cols[:len(labels)]):
            m = (sub == i).astype(float)
            if m.any() and not m.all():
                hl = (i == yid) if kind == 'year' else False
                za.contour(m, levels=[.5], colors=['black'], linewidths=3.4 if hl else 2.4, extent=ext_z, origin='upper', alpha=.55)
                za.contour(m, levels=[.5], colors=[c], linewidths=2.0 if hl else 1.2, extent=ext_z, origin='upper')
        za.set_xlim(zx0, zx1); za.set_ylim(zy0, zy1); za.set_xticks([]); za.set_yticks([])
        for sp in za.spines.values(): sp.set_color(FG); sp.set_linewidth(1.1)
        za.text(.03, .95, t1_, transform=za.transAxes, fontsize=13, fontfamily=FD, color=FG, va='top', bbox=dict(fc=BG, ec='none', alpha=.8, pad=2))
        za.text(.03, .04, t2_, transform=za.transAxes, fontsize=8, fontfamily=FB, color=FG, va='bottom', bbox=dict(fc=BG, ec='none', alpha=.8, pad=2))
    fig.add_artist(ConnectionPatch(xyA=(zx0, zy0), coordsA=ax.transData, xyB=(0, 1), coordsB=axes_z[0].transAxes, color=FG, lw=.8, alpha=.85))
    fig.add_artist(ConnectionPatch(xyA=(zx1, zy0), coordsA=ax.transData, xyB=(1, 1), coordsB=axes_z[-1].transAxes, color=FG, lw=.8, alpha=.85))
    if YE is not None:
        fig.text(fx1-2*pw-gap2, bot+(bh-ph)-.008, 'Detail %.1f x %.1f km. Outlines: block boundaries coloured by planting year; the %d blocks are drawn thicker.'
                 % (zw*g.res/1000, zh*g.res/1000, YE), fontsize=8, fontfamily=FB, color=MU, va='top')

    ia = fig.add_axes([fx0, bot, iw, bh]); ia.set_facecolor('#1b2226')
    if inset_imgs:
        ia.imshow(inset_imgs[0], aspect='auto'); ia.set_xticks([]); ia.set_yticks([])
    else:
        if study.inset:
            for ft in study.inset['features']: _poly(ia, ft['geometry'], fc='#3a4246', ec='#6e787c', lw=.3)
        rg = getattr(study, 'regencies', None)
        if rg:
            for ft in rg['features']:
                hit = ft['properties'].get('ADM2_NAME') == p.get('adm2')
                _poly(ia, ft['geometry'], fc='#f48849' if hit else '#4a5458', ec='#2a3236', lw=.25, alpha=.9 if hit else 1)
        lon, lat = R['center']; ia.plot(lon, lat, 's', ms=5, mfc=FG, mec=BG, mew=.6)
        ia.set_xlim(lon-4, lon+4); ia.set_ylim(lat-4, lat+4); ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values(): sp.set_color(FG); sp.set_linewidth(.8)
    ia.text(.04, .955, 'LOCATION', transform=ia.transAxes, fontsize=13, fontfamily=FD, color=FG, va='top', bbox=dict(fc=BG, ec='none', alpha=.8, pad=2))
    ia.text(.04, .04, '%s Regency\n%s Province' % (p.get('adm2', ''), p.get('adm1', '')), transform=ia.transAxes, fontsize=8.5, fontfamily=FB,
            color=FG, va='bottom', linespacing=1.2, bbox=dict(fc=BG, ec='none', alpha=.8, pad=2))
    fig.savefig(path, facecolor=BG); plt.close(fig)


def context_image(study, ext, res=20):
    ee = study.ee; y1 = study.s.years[1]; g = study.g
    grid = Grid(g.utm, math.floor(ext[0]/res)*res, math.ceil(ext[3]/res)*res, res, int((ext[1]-ext[0])/res)+1, int((ext[3]-ext[2])/res)+1)
    im = study.sentinel2(grid.rect(ee), '%d-01-01' % (y1-1), '%d-12-31' % y1).select(['B4', 'B3', 'B2']).unmask(0)
    d = pull(ee, im.toFloat(), grid, ['B4', 'B3', 'B2'], tile=512)
    return grid, np.dstack([d['B4'], d['B3'], d['B2']])


def overlay_sheets(R, study, frames, folder, fonts=None):
    import os, textwrap
    from matplotlib.patches import Polygon
    os.makedirs(folder, exist_ok=True)
    fonts = fonts or {'display': 'DejaVu Sans', 'body': 'DejaVu Sans', 'bold': 'DejaVu Sans'}
    FD, FB = fonts['display'], fonts['body']
    A, g30 = study.A, study.g; M, g = getattr(study, 'M', study.A), getattr(study, 'gm', study.g)
    y0, y1 = R['years']; S = R['setup']; p = R.get('place') or {}
    BG, FG, MU = '#0d1012', '#eef0ee', '#9aa3a6'
    s2 = np.dstack([M['red'], M['green'], M['blue']]).astype(np.float32)
    ok = s2.sum(2) > 0
    lo = np.percentile(s2[ok], 1, axis=0); hi = np.percentile(s2[ok], 99, axis=0)
    s2 = np.clip((s2-lo)/(hi-lo+1e-9), 0, 1)**.9
    dark = dark_base(np.dstack([M['red'], M['green'], M['blue']]).astype(np.float32), .55)
    cr, labels, _ = class_raster(M, R, 'year')
    n_ = len(labels)-2; pl = matplotlib.colormaps['plasma']
    cols = ['#5a4e7a'] + [matplotlib.colors.to_hex(pl(.18+.77*i/max(n_-1, 1))) for i in range(n_)] + ['#2ee6d6']

    def paint(base, masks):
        out = base.copy()
        for m, c in masks: out[m] = matplotlib.colors.to_rgb(c)
        return out
    pk = M['cat_map'] > 0
    ind = np.repeat(np.repeat(A['ind'] >= .5, 3, 0), 3, 1)[:pk.shape[0], :pk.shape[1]] if g.res < g30.res else A['ind'] >= .5
    raw = A['py_raw'].astype(int); rc = A['cat']
    yrs = list(range(y0, y1+1)); ycols = [matplotlib.colors.to_hex(pl(.12+.83*i/(len(yrs)-1))) for i in range(len(yrs))]
    rawimg = dark_base(np.dstack([A['red'], A['green'], A['blue']]).astype(np.float32), .55)
    for i, y in enumerate(yrs):
        rawimg[((rc == PLANTED) | (rc == RECENT)) & (raw == y)] = matplotlib.colors.to_rgb(ycols[i])
    rawimg[rc == OLD] = matplotlib.colors.to_rgb('#5a4e7a')
    res_img = dark.copy()
    for i, c in enumerate(cols[:len(labels)]): res_img[cr == i] = matplotlib.colors.to_rgb(c)
    ed = block_edges(np.where(cr >= 0, cr+1, 0)) & (cr >= 0); res_img[ed] = (.05, .06, .07)
    fy = sorted(frames)
    steps = [
        ('Landsat %d' % fy[0], 'Start of the record. Annual natural-colour composite from Landsat 7 and 8, clouds, shadows and haze masked.', frames[fy[0]], g30, []),
        ('Sentinel-2 %d' % y1, 'The same area today at 10 m. Field roads and block edges are visible, and they are used to draw the block boundaries.', s2, g, []),
        ('Oil palm extent', 'Only land mapped as oil palm by BIOPAMA Global Oil Palm v1.2 (Descals et al., 2024) is dated.',
         paint(dark, [(pk & ind, '#3fae6a'), (pk & ~ind, '#b6e07a')]), g, [('#3fae6a', 'Industrial estate'), ('#b6e07a', 'Smallholder')]),
        ('Year of bare soil, per pixel', 'For every 30 m Landsat pixel: the year with the lowest NBR in its last bare-soil period, kept only if the canopy grew back.',
         rawimg, g30, [(ycols[0], str(y0)), (ycols[len(yrs)//2], str(yrs[len(yrs)//2])), (ycols[-1], str(y1)), ('#5a4e7a', 'No bare soil since %d' % y0)]),
        ('Result: planting year by block', 'Sentinel-2 segments take the majority year of the pixels inside them, giving clean blocks along roads and canals.',
         res_img, g, [(c, l) for c, l in zip(cols, labels)]),
    ]
    out = []
    for i, (title, sub, img, gg, leg) in enumerate(steps, 1):
        W, H = 14, 7.6
        fig = plt.figure(figsize=(W, H), dpi=170); fig.patch.set_facecolor(BG)
        ext = gg.extent; aw = .64; ah = aw*W/H*gg.ny/gg.nx
        if ah > .88: ah = .88; aw = ah*H/W*gg.nx/gg.ny
        ax = fig.add_axes([.04, (1-ah)/2, aw, ah]); ax.imshow(img, extent=ext, interpolation='bilinear' if i <= 2 else 'nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(FG); sp.set_linewidth(.8)
        L = 2000; xs = ext[0]+(ext[1]-ext[0])*.04; ys = ext[2]+(ext[3]-ext[2])*.05; hh = (ext[3]-ext[2])*.012
        ax.add_patch(Rectangle((xs, ys), L/2, hh, fc=FG, ec=FG)); ax.add_patch(Rectangle((xs+L/2, ys), L/2, hh, fc=BG, ec=FG))
        for t, xx in ((0, xs), (1, xs+L/2), (2, xs+L)):
            ax.text(xx, ys+hh*2.2, ('%d' % t)+(' km' if t == 2 else ''), color=FG, fontsize=9, fontfamily=FB, ha='center', va='bottom',
                    bbox=dict(fc=BG, ec='none', alpha=.6, pad=1))
        xn = ext[1]-(ext[1]-ext[0])*.05; yn = ext[3]-(ext[3]-ext[2])*.08; dx = (ext[1]-ext[0])*.012; dy = (ext[3]-ext[2])*.09
        ax.add_patch(Polygon([[xn, yn], [xn-dx, yn-dy], [xn, yn-dy*.75]], fc=FG, ec=FG, lw=.6, zorder=8))
        ax.add_patch(Polygon([[xn, yn], [xn+dx, yn-dy], [xn, yn-dy*.75]], fc=BG, ec=FG, lw=.6, zorder=8))
        ax.text(xn, yn+(ext[3]-ext[2])*.01, 'N', color=FG, fontsize=11, fontfamily=FB, ha='center', va='bottom', fontweight='bold')
        x0 = .04+aw+.035
        fig.text(x0, .93, 'PALM AGE  |  STEP %d OF %d' % (i, len(steps)), fontsize=11, fontfamily=FB, color='#f48849', va='top')
        tl = textwrap.wrap(title.upper(), 22)
        fig.text(x0, .88, '\n'.join(tl), fontsize=26, fontfamily=FD, color=FG, va='top', linespacing=1.05)
        fig.text(x0, .88-.075*len(tl)-.02, '\n'.join(textwrap.wrap(sub, 40)), fontsize=11, fontfamily=FB, color=MU, va='top', linespacing=1.35)
        yy = .5 if len(leg) > 5 else .42; dy_ = .036 if len(leg) > 5 else .045
        for c, l in leg:
            fig.patches.append(Rectangle((x0, yy), .02, .024, transform=fig.transFigure, fc=c, ec='none'))
            fig.text(x0+.03, yy+.012, l, fontsize=11, fontfamily=FB, color=FG, va='center'); yy -= dy_
        fig.text(x0, .135, '%s Regency, %s\n%g x %g km, centre %.4f, %.4f' % (p.get('adm2', ''), p.get('adm1', ''), *R['window_km'], R['center'][1], R['center'][0]),
                 fontsize=9.5, fontfamily=FB, color=MU, va='top', linespacing=1.3)
        fig.text(x0, .04, 'Map: %s' % S['author'], fontsize=10.5, fontfamily=FB, color=FG)
        pth = os.path.join(folder, '%02d.png' % i); fig.savefig(pth, facecolor=BG); plt.close(fig); out.append(pth)
    return out


PROJECT = 'YOUR_EE_PROJECT'

import ee
ee.Authenticate()
ee.Initialize(project=PROJECT)

setup = Setup(center=(110.4875, -2.2975), window_km=(15, 10), years=(2010, 2026))

import os, csv, shutil
study = PalmAge(ee, setup)
R = study.run()
F = study.frames_data()
CS = chip_stretch(F, sorted(F))
chips = {y: CS[y] for y in sorted({int(round(v)) for v in np.linspace(setup.years[0], setup.years[1], 7)})}
fonts = load_fonts()
ctx = context_image(study, showcase_extent(study.gm))

out = 'palm_age'
os.makedirs(out, exist_ok=True)
showcase_map(R, study, chips, f'{out}/01_map_planting_year.png', 'year', ctx=ctx, fonts=fonts, frames=CS)
showcase_map(R, study, chips, f'{out}/02_map_age_class.png', 'age', ctx=ctx, fonts=fonts, frames=CS)
validation_chart(R, study, f'{out}/03_validation.png', setup.dpi)
pixel_stories(R, f'{out}/04_pixel_stories.png', setup.dpi)
overlay_sheets(R, study, CS, f'{out}/overlay', fonts)
timelapse(R, study, F, f'{out}/05_timelapse_1080p.mp4', web_path=f'{out}/06_timelapse_web.mp4')
with open(f'{out}/07_report.txt', 'w', encoding='utf-8') as fh: fh.write(report_id(R))
with open(f'{out}/08_results.json', 'w') as fh: json.dump(R, fh, indent=1, default=float)
with open(f'{out}/09_area_by_planting_year.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['planting_year', 'ha'])
    w.writerow(['before %d' % setup.years[0], round(R['old_ha'], 2)])
    for y, v in sorted(R['by_year'].items()): w.writerow([y, round(v, 2)])
shutil.make_archive(out, 'zip', out)
