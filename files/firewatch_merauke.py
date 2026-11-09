from __future__ import annotations
PROJECT = 'YOUR_EE_PROJECT'

import ee
ee.Authenticate(force=True)
ee.Initialize(project=PROJECT)


import base64
import io
import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy import ndimage


@dataclass
class Setup:
    region_name: str = 'Merauke'
    region_bbox: tuple = (137.55, -9.15, 141.05, -6.35)
    season: tuple = ('2026-07-01', None)
    pre_window: tuple = ('2026-04-15', '2026-06-30')
    post_window: tuple = ('2026-07-15', None)
    cloud_score: float = 0.60
    s2_cloudy_max: int = 60
    modis_conf_min: int = 30
    viirs_conf_min: int = 1
    dnbr_burn: float = 0.27
    hotspot_radius_m: int = 375
    char_nir_max: float = 0.20
    bare_swir2_min: float = 0.30
    min_patch_px: int = 13
    focus_km: tuple = (24, 16)
    focus_center: tuple | None = None
    stats_scale: int = 30
    overview_max_px: int = 1300
    author: str = 'Ficking Alindra Wijayadarma'
    dpi: int = 200


def _today():
    import datetime as dt
    return (dt.date.today()+dt.timedelta(days=1)).isoformat()


SEVERITY = [
    ('Low', 0.10, '#f6d55c'),
    ('Moderate-low', 0.27, '#f29e4c'),
    ('Moderate-high', 0.44, '#e4572e'),
    ('High', 0.66, '#8f1d21'),
]
WORLDCOVER = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 50: 'Built-up', 60: 'Bare / sparse',
              70: 'Snow / ice', 80: 'Open water', 90: 'Herbaceous wetland', 95: 'Mangroves', 100: 'Moss / lichen'}


class UTM:

    a, f, k0 = 6378137.0, 1/298.257223563, 0.9996

    def __init__(self, zone: int, south: bool):
        self.zone, self.south = zone, south
        self.e2 = self.f*(2-self.f); self.ep2 = self.e2/(1-self.e2)
        self.epsg = (32700 if south else 32600)+zone

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
    def around(cls, utm, lon0, lat0, lon1, lat1, max_px):
        xs, ys = utm.fwd(np.array([lon0, lon1, lon0, lon1]), np.array([lat0, lat0, lat1, lat1]))
        w, h = xs.max()-xs.min(), ys.max()-ys.min()
        res = max(20, math.ceil(max(w, h)/max_px/10)*10)
        x0 = math.floor(xs.min()/res)*res; y1 = math.ceil(ys.max()/res)*res
        return cls(utm, x0, y1, res, int(math.ceil((xs.max()-x0)/res)), int(math.ceil((y1-ys.min())/res)))

    @classmethod
    def box(cls, utm, lon, lat, w_km, h_km, res):
        cx, cy = utm.fwd(lon, lat); nx, ny = int(w_km*1000/res), int(h_km*1000/res)
        x0 = math.floor((cx-nx*res/2)/res)*res; y1 = math.ceil((cy+ny*res/2)/res)*res
        return cls(utm, x0, y1, res, nx, ny)

    def pixel_lonlat(self, r, c): return self.utm.inv(self.x0+(c+.5)*self.res, self.y1-(r+.5)*self.res)


def pull(ee, image, grid: Grid, bands, dtype=np.float32, tile=400):

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
                    if tile > 40 and any(k in msg for k in ('memory', 'too large', 'timed out', 'limit', 'too many')):
                        return pull(ee, image, grid, bands, dtype, tile//2)
                    if attempt == 2: raise
            for b in bands: out[b][r0:r0+h, c0:c0+w] = np.asarray(arr[b])
    return out


class FireStudy:
    def __init__(self, ee, setup: Setup):
        self.ee, self.s = ee, setup
        self.log = []
        if setup.season[1] is None: setup.season = (setup.season[0], _today())
        if setup.post_window[1] is None: setup.post_window = (setup.post_window[0], _today())

    def say(self, msg):
        print('  · ' + msg); self.log.append(msg)


    def region(self):
        ee = self.ee
        fc = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM2_NAME', self.s.region_name))
        if fc.size().getInfo() > 0:
            self.say('Region: %s regency (FAO GAUL 2015 level 2)' % self.s.region_name)
            self.geom = fc.geometry()
        else:
            self.say('GAUL lookup failed, using the fallback bounding box')
            self.geom = ee.Geometry.Rectangle(list(self.s.region_bbox))
        b = self.geom.bounds().coordinates().get(0).getInfo()
        lons, lats = [p[0] for p in b], [p[1] for p in b]
        self.bbox = (min(lons), min(lats), max(lons), max(lats))
        clon = (self.bbox[0]+self.bbox[2])/2
        self.utm = UTM(int((clon+180)//6)+1, True)
        self.area_km2 = self.geom.area(1000).divide(1e6).getInfo()
        return self


    def _clip_dates(self, col, label):
        ee = self.ee
        last = ee.Date(col.aggregate_max('system:time_start'))
        info = ee.Dictionary({'n': col.size(), 'last': last.format('YYYY-MM-dd')}).getInfo()
        self.say('%s: %d daily images, latest %s' % (label, info['n'], info['last']))
        return info

    def hotspot_collections(self):
        ee, s = self.ee, self.s
        start, end = s.season
        modis = ee.ImageCollection('FIRMS').filterDate(start, end).filterBounds(self.geom)
        snpp = ee.ImageCollection('NASA/LANCE/SNPP_VIIRS/C2').filterDate(start, end).filterBounds(self.geom)
        noaa = ee.ImageCollection('NASA/LANCE/NOAA20_VIIRS/C2').filterDate(start, end).filterBounds(self.geom)
        self.avail = {'MODIS (FIRMS)': self._clip_dates(modis, 'MODIS FIRMS 1 km'),
                      'VIIRS S-NPP': self._clip_dates(snpp, 'VIIRS S-NPP 375 m'),
                      'VIIRS NOAA-20': self._clip_dates(noaa, 'VIIRS NOAA-20 375 m')}
        self.modis = modis.map(lambda im: im.select('confidence').gte(s.modis_conf_min).selfMask().rename('fire')
                               .copyProperties(im, ['system:time_start']))
        viirs_m = lambda im: im.select('confidence').gte(s.viirs_conf_min).selfMask().rename('fire').copyProperties(im, ['system:time_start'])
        self.snpp, self.noaa = snpp.map(viirs_m), noaa.map(viirs_m)
        return self

    def daily_counts(self):
        ee = self.ee; geom = self.geom
        def per_day(col, scale):
            def f(im):
                n = im.reduceRegion(ee.Reducer.count(), geom, scale, maxPixels=1e10).get('fire')
                return ee.Feature(None, {'d': im.date().format('YYYY-MM-dd'), 'n': ee.Algorithms.If(n, n, 0)})
            fc = col.map(f)
            return fc.aggregate_array('d'), fc.aggregate_array('n')
        out = {}
        for key, col, scale in (('modis', self.modis, 1000), ('snpp', self.snpp, 375), ('noaa', self.noaa, 375)):
            d, n = per_day(col, scale)
            dd = ee.Dictionary({'d': d, 'n': n}).getInfo()
            acc = {}
            for k, v in zip(dd['d'], dd['n']): acc[k] = acc.get(k, 0)+(v or 0)
            out[key] = acc
        self.daily = out
        tot = {k: int(sum(v.values())) for k, v in out.items()}
        self.say('Hotspot pixels in season: MODIS %d · VIIRS S-NPP %d · VIIRS NOAA-20 %d' % (tot['modis'], tot['snpp'], tot['noaa']))
        return self


    def _s2(self, window, darkest=False):


        ee, s = self.ee, self.s
        cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(self.geom).filterDate(*window)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s.s2_cloudy_max)).linkCollection(cs, ['cs_cdf']))
        clean = col.map(lambda im: im.updateMask(im.select('cs_cdf').gte(s.cloud_score))
                        .select(['B3', 'B4', 'B8', 'B11', 'B12']).divide(10000))

        comp = clean.reduce(ee.Reducer.percentile([25])).rename(['B3', 'B4', 'B8', 'B11', 'B12']) if darkest else clean.median()
        return comp.setDefaultProjection('EPSG:%d' % self.utm.epsg, None, 20), col.size()

    def burn_scars(self):
        ee, s = self.ee, self.s
        pre, npre = self._s2(s.pre_window); post, npost = self._s2(s.post_window, darkest=True)
        post_med, _ = self._s2(s.post_window)
        s2last = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(self.geom).filterDate(*s.post_window)
                  .aggregate_max('system:time_start'))
        n = ee.Dictionary({'pre': npre, 'post': npost, 'last': ee.Date(s2last).format('YYYY-MM-dd')}).getInfo()
        self.s2_last = n.get('last')
        self.say('Sentinel-2 scenes: before %d, after %d (latest %s)' % (n['pre'], n['post'], self.s2_last))
        nbr = lambda im: im.normalizedDifference(['B8', 'B12'])
        mndwi = lambda im: im.normalizedDifference(['B3', 'B11'])
        wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        water = mndwi(pre).gt(0).Or(mndwi(post_med).gt(0)).Or(wc.eq(80))
        dnbr = nbr(pre).subtract(nbr(post)).rename('dnbr')
        valid = pre.select('B8').mask().And(post.select('B8').mask()).And(water.Not())
        sev = (dnbr.gte(SEVERITY[0][1]).add(dnbr.gte(SEVERITY[1][1])).add(dnbr.gte(SEVERITY[2][1]))
               .add(dnbr.gte(SEVERITY[3][1])).rename('sev'))
        change = dnbr.gte(s.dnbr_burn).And(pre.select('B8').subtract(post.select('B8')).gte(0.03)).And(valid)
        char = post.select('B8').lt(s.char_nir_max)
        bare = post.select('B12').gte(s.bare_swir2_min)
        patch = change.updateMask(change).connectedPixelCount(64, True).reproject(crs='EPSG:%d' % self.utm.epsg, scale=20)
        change = patch.gte(s.min_patch_px).unmask(0).And(change)
        burned = change.And(char)
        clearing = change.And(char.Not()).And(bare)

        viirs_proj = ee.Image(self.snpp.first()).projection() if self.avail['VIIRS S-NPP']['n'] else None
        hot = ee.Image(0)
        for col in (self.snpp, self.noaa):
            if col.size().getInfo():
                k = max(1, round(s.hotspot_radius_m/375))
                hot = hot.Or(col.max().unmask(0).focalMax(k, 'square', 'pixels').reproject(viirs_proj or col.first().projection()))
        if self.modis.size().getInfo():
            hot = hot.Or(self.modis.max().unmask(0).reproject(ee.Image(self.modis.first()).projection()))
        self.hot = hot.rename('hot')
        self.pre, self.post, self.dnbr, self.sev, self.burned, self.valid, self.wc = pre, post, dnbr, sev, burned, valid, wc
        self.clearing = clearing
        self.jrc = ee.Image('JRC/GFC2020/V4').select('Map').eq(1).unmask(0)
        return self


    def overview(self):


        ee, s = self.ee, self.s
        g = Grid.around(self.utm, *self.bbox, s.overview_max_px); self.g_over = g
        self.say('Overview grid %d × %d at %d m' % (g.nx, g.ny, g.res))
        crs = 'EPSG:%d' % g.utm.epsg
        inside = ee.Image.constant(1).clip(self.geom).unmask(0)
        days = ee.Image.cat([self.modis.count().unmask(0).rename('dm'), self.snpp.merge(self.noaa).count().unmask(0).rename('dv'),
                             inside.rename('aoi')]).toInt16()
        A = pull(ee, days, g, ['dm', 'dv', 'aoi'], np.int16)
        mcd = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(*s.season).filterBounds(self.geom)
        self.mcd_months = mcd.aggregate_array('system:index').getInfo() if mcd.size().getInfo() else []
        if self.mcd_months:
            A.update(pull(ee, mcd.select('BurnDate').max().gt(0).unmask(0).toInt16().rename('mcd'), g, ['mcd'], np.int16))
        conf = self.burned.And(self.hot); spec = self.burned.And(self.hot.Not())
        bands = []
        for k in (2, 3, 4):
            bands += [conf.And(self.sev.eq(k)).rename('c%d' % k), spec.And(self.sev.eq(k)).rename('s%d' % k)]
        bands += [self.clearing.rename('clr'), conf.And(self.jrc).rename('fst'), self.valid.rename('val')]
        bands += [conf.And(self.wc.eq(code)).rename('w%d' % code) for code in WORLDCOVER]
        names = ['c2', 's2', 'c3', 's3', 'c4', 's4', 'clr', 'fst', 'val'] + ['w%d' % c for c in WORLDCOVER]
        stack = (ee.Image.cat(bands).unmask(0).multiply(inside).toFloat().reproject(crs=crs, scale=20)
                 .reduceResolution(ee.Reducer.mean(), maxPixels=65536))
        fc = self.post.select(['B12', 'B8', 'B4']).multiply(10000).toInt16().rename(['r', 'g', 'b'])
        A.update(pull(ee, fc.unmask(0), g, ['r', 'g', 'b'], np.int16))
        self.say('Burn statistics from 20 m pixels, %d bands (progress below)' % len(names))
        F = {n: np.zeros((g.ny, g.nx), np.float32) for n in names}
        T = 96; jobs = [(r0, c0) for r0 in range(0, g.ny, T) for c0 in range(0, g.nx, T) if A['aoi'][r0:r0+T, c0:c0+T].any()]
        for i, (r0, c0) in enumerate(jobs, 1):
            sub = Grid(g.utm, g.x0+c0*g.res, g.y1-r0*g.res, g.res, min(T, g.nx-c0), min(T, g.ny-r0))
            part = pull(ee, stack, sub, names, tile=T)
            for n in names: F[n][r0:r0+sub.ny, c0:c0+sub.nx] = part[n]
            print('\r      %d / %d blocks' % (i, len(jobs)), end='', flush=True)
        print()
        A['bf'] = F['c2']+F['c3']+F['c4']+F['s2']+F['s3']+F['s4']
        A['cf'] = F['c2']+F['c3']+F['c4']
        self.A_over, self.frac = A, F
        return self

    def statistics(self):

        A, F, g = self.A_over, self.frac, self.g_over
        cell = g.res*g.res/1e4
        ha = lambda a: float(a.sum()*cell)
        self.area_sev = {12: ha(F['c2']), 13: ha(F['c3']), 14: ha(F['c4']), 2: ha(F['s2']), 3: ha(F['s3']), 4: ha(F['s4']), 9: ha(F['clr'])}
        self.area_lc = {code: ha(F['w%d' % code]) for code in WORLDCOVER}
        self.area_lc[1000] = 0.0
        self.forest_ha = ha(F['fst'])
        self.stats_scale_used = 20
        inside = A['aoi'] > 0; burnt = A['bf'] > 0
        val = {}
        for key, m in (('viirs', A['dv'] > 0), ('modis', A['dm'] > 0)):
            n = int((m & inside).sum())
            val[key] = dict(share_on_scar=(float((m & inside & burnt).sum())/n if n else None), n_cells=n)
        if self.mcd_months and 'mcd' in A:
            mc = (A['mcd'] > 0) & inside
            val['mcd64'] = dict(months=self.mcd_months, mcd_ha=ha(mc.astype(float)), ours_ha=ha(A['bf']*inside),
                                both_ha=ha(A['bf']*mc))
        self.validation = val
        self.say('Burned (confirmed) %s ha · burned without hotspot %s ha · clearing-like %s ha'
                 % (fmt(self.area_sev[12]+self.area_sev[13]+self.area_sev[14]), fmt(self.area_sev[2]+self.area_sev[3]+self.area_sev[4]), fmt(self.area_sev[9])))
        return self

    def focus(self):
        ee, s = self.ee, self.s
        if s.focus_center:
            lon, lat = s.focus_center
        else:
            A, g = self.A_over, self.g_over
            heat = (A['dv'] + 3*A['dm']).astype(float)*(A['aoi'] > 0)
            k = (max(1, int(s.focus_km[1]*1000/g.res)), max(1, int(s.focus_km[0]*1000/g.res)))
            sm = ndimage.uniform_filter(heat, k, mode='constant')
            r, c = np.unravel_index(np.argmax(sm), sm.shape)
            lon, lat = g.pixel_lonlat(r, c)
        self.focus_ll = (round(lon, 4), round(lat, 4))
        self.say('Focus window centre %.4f, %.4f' % (lat, lon))
        g = Grid.box(self.utm, lon, lat, *s.focus_km, 20); self.g_focus = g
        loose_s = Setup(**{**asdict(s), 's2_cloudy_max': 90, 'cloud_score': 0.45}); keep = self.s; self.s = loose_s
        pre_loose, _ = self._s2(s.pre_window); self.s = keep
        pre_show = self.pre.unmask(pre_loose)
        img = ee.Image.cat([pre_show.select(['B12', 'B8', 'B4']).multiply(10000).rename(['pr', 'pg', 'pb']),
                            self.post.select(['B12', 'B8', 'B4']).multiply(10000).rename(['qr', 'qg', 'qb']),
                            self.dnbr.multiply(1000).rename('dn')]).unmask(-9999).toInt16()
        F = pull(ee, img, g, ['pr', 'pg', 'pb', 'qr', 'qg', 'qb', 'dn'], np.int16, tile=300)
        cls = (self.burned.multiply(self.sev).add(self.burned.And(self.hot).multiply(10)).add(self.clearing.multiply(9))
               .where(self.valid.Not(), 99).unmask(99).rename('c'))
        F.update(pull(ee, ee.Image.cat([cls, self.hot.unmask(0).rename('h'), self.wc.rename('wc')]).toInt16(), g, ['c', 'h', 'wc'], np.int16, tile=300))
        self.F = F
        return self

    def run(self):
        for step in (self.region, self.hotspot_collections, self.daily_counts, self.burn_scars, self.overview, self.statistics, self.focus):
            print('▶ ' + step.__name__.replace('_', ' '))
            step()
        return self.results()

    def results(self):
        s = self.s
        data_end = max(a['last'] for a in self.avail.values() if a.get('last'))
        sev_conf = {SEVERITY[i-1][0]: self.area_sev.get(10+i, 0) for i in range(2, 5)}
        sev_spec = {SEVERITY[i-1][0]: self.area_sev.get(i, 0) for i in range(2, 5)}
        lc = {}
        for code, ha in self.area_lc.items():
            if code >= 1000: continue
            name = WORLDCOVER.get(code, 'Other')
            lc[name] = lc.get(name, 0)+ha
        forest_ha = self.forest_ha
        return dict(
            region=s.region_name, region_km2=self.area_km2, season=[s.season[0], data_end], pre=list(s.pre_window),
            post=[s.post_window[0], getattr(self, 's2_last', None) or s.post_window[1]], data_end=data_end,
            availability=self.avail, daily=self.daily, focus_center=self.focus_ll,
            burned_confirmed_ha=sum(sev_conf.values()), burned_spectral_only_ha=sum(sev_spec.values()),
            clearing_like_ha=self.area_sev.get(9, 0),
            severity_confirmed_ha=sev_conf, severity_spectral_only_ha=sev_spec,
            landcover_confirmed_ha=dict(sorted(lc.items(), key=lambda kv: -kv[1])), forest2020_confirmed_ha=forest_ha,
            validation=self.validation, stats_scale_m=self.stats_scale_used, overview_res_m=self.g_over.res, log=self.log, setup=asdict(s))


INK, PAPER, MUTED, LINE, EMBER = '#f3ede3', '#14110f', '#a39b8f', '#3a332d', '#ff7a1a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.edgecolor': LINE, 'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
                     'savefig.facecolor': PAPER})


def stretch(r, g, b, lo=None, hi=None):
    a = np.dstack([r, g, b]).astype(float); a[a < 0] = np.nan
    lo = np.nanpercentile(a, 2) if lo is None else lo; hi = np.nanpercentile(a, 98) if hi is None else hi
    return np.nan_to_num(np.clip((a-lo)/(hi-lo+1e-9), 0, 1), nan=0)**0.9, lo, hi


def fmt(v, d=0): return ('{:,.%df}' % d).format(v)
def idn(v, d=0): return fmt(v, d).replace(',', 'X').replace('.', ',').replace('X', '.')


def _daily_frame(R):
    import datetime as dt
    start = dt.date.fromisoformat(R['season'][0]); end = dt.date.fromisoformat(R['season'][1])
    days = [start+dt.timedelta(i) for i in range((end-start).days+1)]
    key = lambda d: d.isoformat()
    v = np.array([R['daily']['snpp'].get(key(d), 0)+R['daily']['noaa'].get(key(d), 0) for d in days], float)
    m = np.array([R['daily']['modis'].get(key(d), 0) for d in days], float)
    return days, v, m


def timeline(ax, R, small=False):
    days, v, m = _daily_frame(R)
    ax.bar(days, v, width=1, color=EMBER, alpha=.9, label='VIIRS 375 m (S-NPP + NOAA-20)')
    ax.plot(days, np.convolve(m, np.ones(7)/7, 'same'), color=INK, lw=1.2, label='MODIS 1 km, 7-day mean')
    ax.set_ylabel('fire pixels detected / day', fontsize=8 if small else 9)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5 if small else 8.5)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(frameon=False, fontsize=7.5 if small else 8.5, loc='upper left')
    if days:
        p = int(np.argmax(v+m)); ax.annotate('peak %s' % days[p].strftime('%d %b'), (days[p], (v+m)[p]), xytext=(8, -2),
                                             textcoords='offset points', fontsize=8, color=INK)


def sev_rgb(F):
    c = F['c']; out = np.zeros(c.shape+(4,))
    for i, (_, _, col) in enumerate(SEVERITY, start=1):
        rgb = matplotlib.colors.to_rgb(col)
        out[c == 10+i] = (*rgb, 1)
        out[c == i] = (*rgb, .45)
    out[c == 9] = (*matplotlib.colors.to_rgb('#6fa8c9'), .55)
    return out


def poster(R, study, path):

    A, g, F, gf = study.A_over, study.g_over, study.F, study.g_focus
    fig = plt.figure(figsize=(14, 19.8), dpi=study.s.dpi)
    fig.text(.05, .965, 'FIREWATCH · %s' % R['region'].upper(), fontsize=13, color=EMBER, fontweight='bold', va='top')
    fig.text(.05, .948, 'Where the 2026 fire season burned', fontsize=34, fontweight='bold', va='top')
    fig.text(.05, .917, '%s regency, South Papua · %s to %s · satellite hotspots and Sentinel-2 burn scars'
             % (R['region'], R['season'][0], R['season'][1]), fontsize=12, color=MUTED, va='top')
    tot_v = sum(R['daily']['snpp'].values())+sum(R['daily']['noaa'].values()); tot_m = sum(R['daily']['modis'].values())
    big = [(fmt(R['burned_confirmed_ha']), 'ha burned, confirmed by hotspots'), (fmt(R['burned_spectral_only_ha']), 'ha burned, no hotspot nearby'),
           (fmt(tot_v), 'VIIRS fire detections (pixel-days)'), (fmt(R['forest2020_confirmed_ha']), 'ha of 2020 forest burned')]
    for i, (n, l) in enumerate(big):
        x = .05+i*.23
        fig.text(x, .885, n, fontsize=27, fontweight='bold', color=EMBER if i == 0 else INK, va='top')
        fig.text(x, .858, l, fontsize=9.5, color=MUTED, va='top')
    fig.add_artist(plt.Line2D([.05, .95], [.842, .842], color=LINE, lw=1))

    ah = .42; aw = ah*19.8/14*g.nx/g.ny
    if aw > .68: aw = .68; ah = aw*14/19.8*g.ny/g.nx
    ax = fig.add_axes([.05, .83-ah, aw, ah])
    base, _, _ = stretch(A['r'], A['g'], A['b']); grey = base.mean(2, keepdims=True)*.55
    img = np.where(A['aoi'][..., None] > 0, grey.repeat(3, 2), np.array(matplotlib.colors.to_rgb(PAPER)))
    ax.imshow(img, extent=g.extent, interpolation='nearest')
    heat = np.ma.masked_where(A['bf'] < .02, A['bf'])
    ax.imshow(heat, extent=g.extent, cmap='inferno', vmin=0, vmax=.8, alpha=.95, interpolation='nearest')
    yy, xx = np.nonzero((A['dv'] >= 3) & (A['aoi'] > 0))
    ax.scatter(g.x0+(xx+.5)*g.res, g.y1-(yy+.5)*g.res, s=.35, c='#ffe08a', lw=0, alpha=.55)
    ax.contour(np.flipud(A['aoi'].astype(float)), levels=[.5], colors=[INK], linewidths=.7, extent=g.extent, origin='lower')
    e = gf.extent; ax.add_patch(Rectangle((e[0], e[2]), e[1]-e[0], e[3]-e[2], fill=False, ec=INK, lw=1.4))
    ax.text(e[1]+3000, e[3], 'focus', fontsize=9, color=INK, va='top')
    mx, my = g.utm.fwd(140.40, -8.49); ax.plot(mx, my, 'o', ms=4, mfc=INK, mec=PAPER); ax.text(mx+3000, my, 'Merauke', fontsize=8.5, color=INK, va='center')
    ax.set_xlim(g.extent[:2]); ax.set_ylim(g.extent[2:]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    L = 50000; x0 = g.extent[0]+(g.extent[1]-g.extent[0])*.05; y0 = g.extent[2]+(g.extent[3]-g.extent[2])*.05
    ax.plot([x0, x0+L], [y0, y0], color=INK, lw=2); ax.text(x0+L/2, y0+2500, '50 km', ha='center', fontsize=8.5, color=INK)
    ax.annotate('N', xy=(g.extent[1]-15000, g.extent[3]-8000), xytext=(g.extent[1]-15000, g.extent[3]-30000), ha='center', color=INK,
                fontsize=11, fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4))
    lx = .05+aw+.03; lw = .95-lx
    fig.text(lx, .825, 'Reading the map', fontsize=12, fontweight='bold', va='top')
    fig.text(lx, .800, 'Share of each cell mapped\nas a burn scar', fontsize=9, color=MUTED, va='top')
    cax = fig.add_axes([lx, .772, lw, .009]); cax.imshow(np.linspace(0, 1, 100)[None], cmap='inferno', aspect='auto'); cax.axis('off')
    fig.text(lx, .767, '0%', fontsize=8, color=MUTED, va='top'); fig.text(lx+lw, .767, '80%+', fontsize=8, color=MUTED, va='top', ha='right')
    yy0 = .735
    for sym, c, t in (('·', '#ffe08a', 'Fire seen by VIIRS on\n3 or more days'), ('━', INK, 'Regency boundary\n(FAO GAUL)'), ('▢', INK, 'Focus window,\nenlarged below')):
        fig.text(lx, yy0, sym, color=c, fontsize=11, va='top'); fig.text(lx+.025, yy0, t, fontsize=9, color=MUTED, va='top'); yy0 -= .032
    fig.text(lx, yy0-.01, 'Base: Sentinel-2 after the\nseason, shown in grey', fontsize=8.5, color=MUTED, va='top')

    y0 = .83-ah
    fig.text(.05, y0-.03, 'When it burned', fontsize=14, fontweight='bold', va='bottom')
    tl = fig.add_axes([.07, y0-.125, .86, .085]); timeline(tl, R, small=True)

    fig.text(.05, y0-.165, 'Up close: %s × %s km around the densest hotspot cluster' % R['setup']['focus_km'], fontsize=14, fontweight='bold', va='bottom')
    pre, lo, hi = stretch(F['pr'], F['pg'], F['pb']); post, _, _ = stretch(F['qr'], F['qg'], F['qb'], lo, hi)
    hole = (F['pr'] <= 0)
    if hole.any(): pre[hole] = matplotlib.colors.to_rgb(LINE)
    w = .285; h = w*14/19.8*gf.ny/gf.nx
    for i, (im, t) in enumerate([(pre, 'Before · %s to %s%s' % (*R['pre'], '  (grey = no cloud-free view)' if hole.any() else '')), (post, 'After · darkest-quarter composite, %s to %s' % tuple(R['post'])), (None, 'Burn severity (dNBR)')]):
        a = fig.add_axes([.05+i*(w+.0175), y0-.185-h, w, h])
        if im is None:
            a.imshow(post.mean(2), cmap='gray', vmin=0, vmax=1.4, extent=gf.extent); a.imshow(sev_rgb(F), extent=gf.extent, interpolation='nearest')
        else:
            a.imshow(im, extent=gf.extent)
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=9.5, color=MUTED, loc='left')
        for sp in a.spines.values(): sp.set_color(LINE)
    ly = y0-.185-h-.02
    for i, (lab, lo_, col) in enumerate(SEVERITY[1:]):
        x = .05+i*.21
        fig.patches.append(Rectangle((x, ly-.006), .014, .01, transform=fig.transFigure, fc=col, ec='none'))
        fig.text(x+.02, ly, '%s severity · dNBR ≥ %.2f' % (lab, lo_), fontsize=8.5, color=MUTED, va='center')
    fig.patches.append(Rectangle((.68, ly-.006), .014, .01, transform=fig.transFigure, fc='#6fa8c9', ec='none'))
    fig.text(.70, ly, 'Bare or cleared ground, not counted', fontsize=8.5, color=MUTED, va='center')
    fig.text(.05, ly-.02, 'Solid = burn scar on or next to a VIIRS fire pixel (≤ %d m) · faded = burn scar without a hotspot'
             % R['setup']['hotspot_radius_m'], fontsize=8, color=MUTED, va='center')
    fig.text(.05, .022, 'Data: NASA FIRMS (MODIS, VIIRS NRT) · Copernicus Sentinel-2 L2A with Cloud Score+ · ESA WorldCover 2021 · EC JRC Forest 2020 · FAO GAUL.\n'
             'False colour = SWIR2 / NIR / red. Screening estimate, not an official figure.   Map: %s' % R['setup']['author'], fontsize=7.8, color=MUTED, va='bottom')
    fig.savefig(path); plt.close(fig)


def landcover_chart(R, path, dpi=200):
    lc = [(k, v) for k, v in R['landcover_confirmed_ha'].items() if v > 0][:7]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=dpi)
    names = [k for k, _ in lc][::-1]; vals = [v for _, v in lc][::-1]
    ax.barh(names, vals, color=EMBER, height=.6)
    for i, v in enumerate(vals): ax.text(v, i, '  '+fmt(v)+' ha', va='center', fontsize=9, color=INK)
    ax.set_title('What burned: confirmed burn scars by land cover (ESA WorldCover 2021)', loc='left', fontsize=11, color=INK)
    for sp in ('top', 'right', 'bottom'): ax.spines[sp].set_visible(False)
    ax.set_xticks([]); ax.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def validation_chart(R, path, dpi=200):
    V = R['validation']; rows = []
    for k, lab in (('viirs', 'Cells with a VIIRS fire that hold a mapped scar'), ('modis', 'Cells with a MODIS fire that hold a mapped scar')):
        if k in V and V[k]['share_on_scar'] is not None: rows.append((lab, V[k]['share_on_scar']*100, '%s cells' % fmt(V[k]['n_cells'])))
    if 'mcd64' in V and V['mcd64']['mcd_ha']:
        m = V['mcd64']; rows.append(('MODIS MCD64A1 burned area also mapped here', 100*m['both_ha']/m['mcd_ha'], '%s months' % len(m['months'])))
    fig, ax = plt.subplots(figsize=(9, 1.2+.55*max(1, len(rows))), dpi=dpi)
    for i, (lab, pct, note) in enumerate(rows[::-1]):
        ax.barh(i, 100, color=LINE, height=.5); ax.barh(i, pct, color=EMBER, height=.5)
        ax.text(0, i+.42, lab, fontsize=9, color=INK); ax.text(100, i+.42, '%.0f%%  (%s)' % (pct, note), fontsize=9, color=INK, ha='right')
    ax.set_xlim(0, 100); ax.set_ylim(-.5, len(rows)-.2); ax.axis('off')
    ax.set_title('Agreement with independent fire records', loc='left', fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _png(arr):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


def story_html(R, study, path):
    F = study.F
    pre, lo, hi = stretch(F['pr'], F['pg'], F['pb']); post, _, _ = stretch(F['qr'], F['qg'], F['qb'], lo, hi)
    sev = sev_rgb(F); comp = post*(1-sev[..., 3:])+sev[..., :3]*sev[..., 3:]
    days, v, m = _daily_frame(R)
    data = dict(days=[d.isoformat() for d in days], viirs=v.tolist(), modis=m.tolist(), pre=_png(pre), post=_png(post), sev=_png(comp),
                R={k: R[k] for k in ('region', 'season', 'pre', 'post', 'burned_confirmed_ha', 'burned_spectral_only_ha', 'forest2020_confirmed_ha',
                                    'severity_confirmed_ha', 'landcover_confirmed_ha', 'validation', 'focus_center', 'stats_scale_m')},
                author=R['setup']['author'], km=R['setup']['focus_km'])
    open(path, 'w', encoding='utf-8').write(STORY.replace('/*DATA*/null', json.dumps(data, default=float)))


STORY = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FireWatch Merauke 2026</title>
<style>
:root{--bg:#14110f;--ink:#f3ede3;--mut:#a39b8f;--line:#3a332d;--ember:#ff7a1a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 Georgia,"Times New Roman",serif}
.wrap{max-width:980px;margin:0 auto;padding:56px 20px 80px}
.kick{font:600 12px/1 system-ui,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--ember)}
h1{font-size:clamp(34px,6vw,58px);line-height:1.05;margin:14px 0 12px;font-weight:700}
.lede{color:var(--mut);font-size:19px;max-width:720px}
.nums{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--line);margin:40px 0;border:1px solid var(--line)}
.nums div{background:var(--bg);padding:18px}.nums b{display:block;font:700 30px/1.1 system-ui,sans-serif}.nums b.e{color:var(--ember)}
.nums span{font:13px/1.4 system-ui,sans-serif;color:var(--mut)}
h2{font-size:26px;margin:56px 0 8px}p{color:#d9d2c7}
.sans{font-family:system-ui,sans-serif}
svg text{font:11px system-ui,sans-serif;fill:var(--mut)}
.cmp{position:relative;width:100%;overflow:hidden;border:1px solid var(--line);user-select:none;touch-action:none;cursor:ew-resize}
.cmp img{display:block;width:100%;height:auto}.cmp .top{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}
.cmp .bar{position:absolute;top:0;bottom:0;left:50%;width:2px;background:var(--ink);transform:translateX(-1px)}
.cmp .bar:after{content:"⟷";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--ink);color:var(--bg);border-radius:50%;width:34px;height:34px;display:grid;place-items:center;font:16px system-ui}
.tag{position:absolute;top:10px;font:600 12px system-ui;background:rgba(0,0,0,.6);padding:4px 8px}
.tabs{display:flex;gap:6px;margin:14px 0}.tabs button{font:14px system-ui;background:none;color:var(--mut);border:1px solid var(--line);padding:7px 12px;cursor:pointer}
.tabs button[aria-pressed=true]{color:var(--bg);background:var(--ember);border-color:var(--ember)}
.bars div{display:grid;grid-template-columns:170px 1fr 110px;gap:10px;align-items:center;font:14px system-ui;margin:6px 0}
.bars i{display:block;height:12px;background:var(--ember)}.bars em{font-style:normal;color:var(--mut);text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:14px;font:13px system-ui;color:var(--mut);margin-top:10px}.legend s{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-1px;text-decoration:none}
.note{border-left:3px solid var(--ember);padding:4px 0 4px 16px;color:var(--mut);font-size:15px}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);font:13px/1.6 system-ui;color:var(--mut)}
@media(max-width:600px){.bars div{grid-template-columns:110px 1fr 80px}}
</style></head><body><div class="wrap">
<div class="kick">FireWatch · South Papua</div>
<h1 id="h1"></h1><p class="lede" id="lede"></p>
<div class="nums" id="nums"></div>
<h2>When it burned</h2><p>Daily satellite hotspot detections over the regency. A hotspot is a pixel the sensor flagged as hotter than its surroundings; one fire can light up many pixels over several days.</p>
<svg id="tl" viewBox="0 0 960 260" width="100%"></svg>
<h2>Before and after</h2><p id="cmpText"></p>
<div class="tabs"><button id="tPost" aria-pressed="true">After</button><button id="tSev" aria-pressed="false">Burn severity</button></div>
<div class="cmp" id="cmp"><img id="imB" alt="before"><img class="top" id="imA" alt="after"><div class="bar" id="bar"></div><span class="tag" style="left:10px">Before</span><span class="tag" id="rtag" style="right:10px">After</span></div>
<div class="legend" id="lg"></div>
<h2>What burned</h2><p>Burn scars confirmed by a hotspot, split by the land cover mapped before the fires (ESA WorldCover 2021).</p><div class="bars" id="lc"></div>
<h2>Can the numbers be trusted?</h2><div class="bars" id="val"></div>
<p class="note" id="caveat"></p>
<h2>Method</h2><p class="sans" style="font-size:15px">Hotspots: NASA FIRMS MODIS (confidence ≥ 30) and VIIRS NRT S-NPP and NOAA-20 (nominal or high). Burn scars: Sentinel-2 surface reflectance with Cloud Score+ masking; the normalized burn ratio NBR = (NIR − SWIR2)/(NIR + SWIR2) is compared before and after the season (dNBR). A pixel counts as burned when dNBR ≥ 0.27, near-infrared fell by at least 0.03, the surface is dark after the season (NIR &lt; 0.20, the signature of char), it is not water in either period, and the patch is at least about 0.5 ha. Change that ends on bright bare soil (SWIR2 ≥ 0.30) is set aside as clearing-like and not counted. Scars on or next to a VIIRS fire pixel are reported as confirmed. Areas are counted from <span id="sc"></span> m pixels.</p>
<footer id="ft"></footer></div>
<script>
const D=/*DATA*/null,R=D.R,$=id=>document.getElementById(id),f=n=>Math.round(n).toLocaleString('en-US');
$('h1').textContent='Where the '+R.season[0].slice(0,4)+' fire season burned in '+R.region;
$('lede').textContent='Satellite hotspots and Sentinel-2 burn scars for '+R.region+' regency, South Papua, from '+R.season[0]+' to '+R.season[1]+'.';
const tv=D.viirs.reduce((a,b)=>a+b,0);
$('nums').innerHTML=[[f(R.burned_confirmed_ha),'ha burned, confirmed by hotspots',1],[f(R.burned_spectral_only_ha),'ha burned, no hotspot nearby'],[f(tv),'VIIRS fire detections (pixel-days)'],[f(R.forest2020_confirmed_ha),'ha of 2020 forest burned']]
 .map(([n,l,e])=>`<div><b class="${e?'e':''}">${n}</b><span>${l}</span></div>`).join('');
(function(){const W=960,H=260,P={l:44,r:10,t:14,b:28},n=D.days.length,mx=Math.max(1,...D.viirs.map((v,i)=>v+D.modis[i]));
 const x=i=>P.l+i*(W-P.l-P.r)/n,y=v=>H-P.b-v*(H-P.t-P.b)/mx,bw=Math.max(1,(W-P.l-P.r)/n-.6);let s='';
 for(let k=0;k<=4;k++){const v=mx*k/4;s+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y(v)}" y2="${y(v)}" stroke="#3a332d"/><text x="${P.l-6}" y="${y(v)+4}" text-anchor="end">${Math.round(v)}</text>`}
 D.viirs.forEach((v,i)=>{if(v>0)s+=`<rect x="${x(i)}" y="${y(v)}" width="${bw}" height="${H-P.b-y(v)}" fill="#ff7a1a"><title>${D.days[i]}: ${v} VIIRS, ${D.modis[i]} MODIS</title></rect>`});
 let pts=D.modis.map((v,i)=>`${x(i)+bw/2},${y(v)}`).join(' ');s+=`<polyline points="${pts}" fill="none" stroke="#f3ede3" stroke-width="1.2"/>`;
 let last='';D.days.forEach((d,i)=>{const m=d.slice(0,7);if(m!==last){last=m;s+=`<text x="${x(i)}" y="${H-8}">${new Date(d).toLocaleString('en',{month:'short'})}</text>`}});
 s+=`<rect x="${W-250}" y="6" width="10" height="10" fill="#ff7a1a"/><text x="${W-235}" y="15">VIIRS 375 m</text><line x1="${W-150}" x2="${W-136}" y1="11" y2="11" stroke="#f3ede3"/><text x="${W-130}" y="15">MODIS 1 km</text>`;
 $('tl').innerHTML=s})();
$('cmpText').textContent='A '+D.km[0]+' × '+D.km[1]+' km window around the densest hotspot cluster (centre '+R.focus_center[1]+', '+R.focus_center[0]+'). False colour: SWIR2, NIR, red, so fresh burn scars look dark brown to black. The after view keeps, per pixel, the darker quarter of clear views since '+R.post[0]+', so scars from July stay visible. Drag the handle.';
$('imB').src=D.pre;$('imA').src=D.post;
let pos=.5;const cmp=$('cmp'),set=p=>{pos=Math.min(1,Math.max(0,p));$('imA').style.clipPath=`inset(0 0 0 ${pos*100}%)`;$('bar').style.left=pos*100+'%'};
const mv=e=>{const r=cmp.getBoundingClientRect();set((e.clientX-r.left)/r.width)};
cmp.addEventListener('pointerdown',e=>{cmp.setPointerCapture(e.pointerId);mv(e);cmp.onpointermove=mv});cmp.addEventListener('pointerup',()=>cmp.onpointermove=null);set(.5);
const sevCols=[['Low','#f6d55c'],['Moderate-low','#f29e4c'],['Moderate-high','#e4572e'],['High','#8f1d21']];
const lgSev=sevCols.slice(1).map(([l,c])=>`<span><s style="background:${c}"></s>${l}</span>`).join('')+'<span><s style="background:#6fa8c9"></s>bare or cleared, not counted</span><span>solid = confirmed by hotspot, faded = no hotspot</span>';
$('tPost').onclick=()=>{$('imA').src=D.post;$('rtag').textContent='After';$('tPost').setAttribute('aria-pressed',true);$('tSev').setAttribute('aria-pressed',false);$('lg').innerHTML=''};
$('tSev').onclick=()=>{$('imA').src=D.sev;$('rtag').textContent='Severity';$('tSev').setAttribute('aria-pressed',true);$('tPost').setAttribute('aria-pressed',false);$('lg').innerHTML=lgSev};
const lc=Object.entries(R.landcover_confirmed_ha).filter(e=>e[1]>0).slice(0,7),lm=Math.max(1,...lc.map(e=>e[1]));
$('lc').innerHTML=lc.map(([k,v])=>`<div><span>${k}</span><i style="width:${100*v/lm}%"></i><em>${f(v)} ha</em></div>`).join('');
const V=R.validation,rows=[];
if(V.viirs&&V.viirs.share_on_scar!=null)rows.push(['VIIRS fire cells holding a scar',V.viirs.share_on_scar*100]);
if(V.modis&&V.modis.share_on_scar!=null)rows.push(['MODIS fire cells holding a scar',V.modis.share_on_scar*100]);
if(V.mcd64&&V.mcd64.mcd_ha)rows.push(['MODIS MCD64A1 area also mapped',100*V.mcd64.both_ha/V.mcd64.mcd_ha]);
$('val').innerHTML=rows.map(([k,p])=>`<div><span>${k}</span><i style="width:${p}%"></i><em>${Math.round(p)}%</em></div>`).join('')||'<p>No independent record available for this period.</p>';
$('caveat').textContent='These are satellite screening estimates. Official figures from BNPB or KLHK use field reports and different methods, so totals will differ. Hotspots under thick smoke or cloud can be missed, and burn-like change without a hotspot can also be land clearing.';
$('sc').textContent=R.stats_scale_m;
$('ft').innerHTML='Data: NASA FIRMS (MODIS, VIIRS NRT), Copernicus Sentinel-2 L2A, Google Cloud Score+, ESA WorldCover 2021, EC JRC Global Forest Cover 2020, FAO GAUL 2015.<br>Analysis and design: '+D.author;
</script></body></html>'''


def report_id(R):
    V = R['validation']; tv = sum(R['daily']['snpp'].values())+sum(R['daily']['noaa'].values()); tm = sum(R['daily']['modis'].values())
    days, v, m = _daily_frame(R)
    BLN = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    pk = days[int(np.argmax(v+m))] if days else None
    peak = '%d %s %d' % (pk.day, BLN[pk.month-1], pk.year) if pk else '-'
    L = ['LAPORAN KEBAKARAN HUTAN DAN LAHAN — KABUPATEN %s, PAPUA SELATAN' % R['region'].upper(),
         'Data terbaru yang dipakai: titik panas s.d. %s, Sentinel-2 s.d. %s.' % (R['data_end'], R['post'][1]),
         'Periode %s s.d. %s · luas wilayah %s km²' % (R['season'][0], R['season'][1], idn(R['region_km2'])), '',
         'KETERSEDIAAN DATA']
    for k, a in R['availability'].items(): L.append('- %s: %d citra harian, data terakhir %s' % (k, a['n'], a['last']))
    L += ['', 'TEMUAN',
          '- Deteksi api (piksel-hari): VIIRS %s, MODIS %s. Puncak harian pada %s.' % (idn(tv), idn(tm), peak),
          '- Bekas terbakar terkonfirmasi titik panas (≤ %d m): %s ha, atau %s%% luas kabupaten.' % (R['setup']['hotspot_radius_m'], idn(R['burned_confirmed_ha']), idn(R['burned_confirmed_ha']/(R['region_km2']*100)*100, 1)),
          '- Bekas terbakar tanpa titik panas di dekatnya: %s ha (kebakaran yang tak terdeteksi satelit, mis. tertutup asap/awan).' % idn(R['burned_spectral_only_ha']),
          '- Perubahan yang berakhir sebagai tanah terbuka terang (mirip pembukaan lahan), tidak dihitung sebagai terbakar: %s ha.' % idn(R.get('clearing_like_ha', 0)),
          '- Hutan 2020 (JRC) yang terbakar dan terkonfirmasi: %s ha.' % idn(R['forest2020_confirmed_ha']),
          '- Tingkat keparahan (terkonfirmasi): ' + ', '.join('%s %s ha' % (k, idn(v)) for k, v in R['severity_confirmed_ha'].items()) + '.',
          '- Tutupan lahan yang terbakar (terkonfirmasi): ' + ', '.join('%s %s ha' % (k, idn(v)) for k, v in list(R['landcover_confirmed_ha'].items())[:5]) + '.', '',
          'VALIDASI']
    for k, lab in (('viirs', 'VIIRS'), ('modis', 'MODIS')):
        if k in V and V[k]['share_on_scar'] is not None:
            L.append('- %s%% sel %d m yang berisi titik panas %s juga berisi bekas terbakar terpetakan (n = %s sel).' % (idn(V[k]['share_on_scar']*100), R['overview_res_m'], lab, idn(V[k]['n_cells'])))
    if 'mcd64' in V and V['mcd64']['mcd_ha']:
        mm = V['mcd64']; L.append('- MODIS MCD64A1 (%s bulan tersedia): %s ha; %s%% di antaranya juga terpetakan di sini.' % (len(mm['months']), idn(mm['mcd_ha']), idn(100*mm['both_ha']/mm['mcd_ha'])))
    else:
        L.append('- MODIS MCD64A1 belum tersedia untuk periode ini (rilis tertunda ± 2 bulan).')
    L += ['', 'KETERBATASAN',
          '- Estimasi satelit (screening), bukan angka resmi. Angka BNPB/KLHK memakai laporan lapangan dan metode berbeda.',
          '- Titik panas bisa terlewat karena asap tebal atau awan; satu kebakaran bisa menghasilkan banyak piksel.',
          '- Di savana Trans-Fly, rumput mengering di musim kemarau sehingga NBR ikut turun; karena itu bekas terbakar juga disyaratkan gelap (NIR < 0,20). Tanah terbuka terang dipisah sebagai mirip pembukaan lahan.',
          '- Air dimasker dengan MNDWI dan WorldCover. Bekas terbakar lama yang sudah pulih hijau sebelum Oktober bisa tidak terhitung.',
          '- Data VIIRS NRT di Earth Engine berstatus near-real-time, bukan kualitas sains.',
          '- "Hutan 2020" mengikuti peta JRC (definisi FAO, tutupan pohon ≥ 10%%); di Merauke ini termasuk hutan savana terbuka (kayu putih, eukaliptus) yang lantainya biasa terbakar.',
          '- Batas kabupaten dari FAO GAUL 2015 (%s km²), sedikit berbeda dari luas resmi.' % idn(R['region_km2']),
          '- Komposit "sesudah" memakai persentil 25 per piksel sejak %s, sehingga bekas terbakar sejak Juli tetap terekam.' % R['post'][0],
          '- Luas dihitung dari piksel %d m (dijumlahkan per sel %d m).' % (R['stats_scale_m'], R['overview_res_m'])]
    return '\n'.join(L)

setup = Setup(
    region_name='Merauke',
    season=('2026-07-01', None),
    pre_window=('2026-04-15', '2026-06-30'),
    post_window=('2026-07-15', None),
    s2_cloudy_max=30,
)

import os, shutil
study = FireStudy(ee, setup)
R = study.run()

out = 'firewatch_merauke'
os.makedirs(out, exist_ok=True)
poster(R, study, f'{out}/01_poster.png')
landcover_chart(R, f'{out}/02_landcover.png', setup.dpi)
validation_chart(R, f'{out}/03_validation.png', setup.dpi)
story_html(R, study, f'{out}/04_story.html')
with open(f'{out}/05_laporan.txt', 'w', encoding='utf-8') as fh: fh.write(report_id(R))
with open(f'{out}/06_results.json', 'w') as fh: json.dump(R, fh, indent=1, default=float)
print(report_id(R))

shutil.make_archive(out, 'zip', out)
try:
    from google.colab import files; files.download(out + '.zip')
except ImportError:
    pass
