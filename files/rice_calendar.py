from __future__ import annotations
PROJECT = 'YOUR_EE_PROJECT'

import ee
ee.Authenticate()
ee.Initialize(project=PROJECT)



import os
import json
import math
import logging
import warnings
import subprocess
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Polygon
from scipy import ndimage

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass
class Setup:
    regency: str = 'Indramayu'
    province: str = 'Jawa Barat'
    res: int = 30
    start: str = '2024-08-01'
    end: str = '2026-01-20'
    window: tuple = ('2024-09-23', '2025-09-22')
    bin_days: int = 12
    flood_db: float = -18.0
    rise_db: float = 4.0
    peak_db: float = -17.0
    rise_from: int = 3
    rise_to: int = 8
    min_gap_bins: int = 5
    field_days: int = 100
    repeat_days: int = 320
    lbs_ha: float = 126088
    bps_harvest_ha: float | None = None
    s2_samples: int = 1500
    sea_name: str | None = 'Java Sea'
    map_aspect: float = 1.79
    pad: float = .07
    block_km: int = 6
    dpi: int = 200


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






def fmt(v, d=0): return ('{:,.%df}' % d).format(v)
def idn(v, d=0): return fmt(v, d).replace(',', 'X').replace('.', ',').replace('X', '.')




def _dms(v, pos, neg):
    h = pos if v >= 0 else neg; v = abs(v); d = int(v); m = round((v-d)*60)
    if m == 60: d, m = d+1, 0
    return "%d°%02d'%s" % (d, m, h)




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


def coarse_frac(m, f, ny, nx):
    return m[:ny*f, :nx*f].reshape(ny, f, nx, f).mean((1, 3))


def _main_pts(gm, keep=.05):
    polys = [np.array(p[0]) for p in _coords(gm)]
    area = [abs(np.dot(p[:, 0], np.roll(p[:, 1], 1))-np.dot(p[:, 1], np.roll(p[:, 0], 1)))/2 for p in polys]
    big = max(area)
    return np.concatenate([p for p, a_ in zip(polys, area) if a_ >= keep*big])


def _bounds(gm):
    pts = np.concatenate([np.array(p[0]) for p in _coords(gm)]) if _coords(gm) else np.array([[0, 0], [1, 1]])
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(.6)




FONT_FILES = {'title': 'ofl/spectral/Spectral-SemiBold.ttf', 'serif': 'ofl/spectral/Spectral-Regular.ttf',
              'italic': 'ofl/spectral/Spectral-Italic.ttf', 'sans': 'ofl/firasans/FiraSans-Regular.ttf',
              'medium': 'ofl/firasans/FiraSans-Medium.ttf', 'light': 'ofl/firasans/FiraSans-Light.ttf'}


def load_fonts(folder='fonts', source='https://github.com/google/fonts/raw/main/'):
    import shutil, urllib.request
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
    os.makedirs(folder, exist_ok=True); out = {}
    for k, rel in FONT_FILES.items():
        f = os.path.join(folder, rel.rsplit('/', 1)[1])
        try:
            if not os.path.exists(f) or os.path.getsize(f) < 10000:
                if source.startswith('http'): urllib.request.urlretrieve(source+rel, f)
                else: shutil.copy(os.path.join(source, rel), f)
            font_manager.fontManager.addfont(f); out[k] = FontProperties(fname=f)
        except Exception:
            out[k] = FontProperties(family='DejaVu Serif' if k in ('title', 'serif', 'italic') else 'DejaVu Sans',
                                    style='italic' if k == 'italic' else 'normal', weight='bold' if k == 'title' else 'normal')
    return out




class RiceCalendar:
    def __init__(self, ee, setup: Setup):
        self.ee, self.s = ee, setup
        s = setup
        d0, d1 = np.datetime64(s.start), np.datetime64(s.end)
        n = int((d1-d0).astype(int)//s.bin_days)
        self.bins = [d0+np.timedelta64(i*s.bin_days, 'D') for i in range(n)]
        self.ref = np.datetime64('2025-01-01')
        self.doff = np.array([int((b+np.timedelta64(s.bin_days//2, 'D')-self.ref).astype(int)) for b in self.bins])
        w0, w1 = [int((np.datetime64(x)-self.ref).astype(int)) for x in s.window]
        self.target = [i for i, d in enumerate(self.doff) if w0 <= d <= w1 and i >= 3 and i+s.rise_to < n]

    def say(self, m): print(m, flush=True)

    def region(self):
        ee, s = self.ee, self.s
        fc = (ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM2_NAME', s.regency))
              .filter(ee.Filter.eq('ADM1_NAME', s.province)))
        geom = fc.geometry()
        info = geom.simplify(50).getInfo()
        return geom, info

    def s1_collection(self, geom):
        ee, s = self.ee, self.s
        col = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(geom).filterDate(s.start, s.end)
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
        rows = col.reduceColumns(ee.Reducer.toList(3), ['orbitProperties_pass', 'relativeOrbitNumber_start', 'system:time_start']).getInfo()['list']
        d0 = np.datetime64(s.start); tracks = {}
        for ps, orb, t in rows:
            k = (ps, int(orb)); b = int(((np.datetime64(int(t), 'ms')-d0).astype('timedelta64[D]').astype(int))//s.bin_days)
            tracks.setdefault(k, set()).add(b)
        area = geom.area(100)
        info = []
        for (ps, orb), bins in tracks.items():
            sub = col.filter(ee.Filter.eq('orbitProperties_pass', ps)).filter(ee.Filter.eq('relativeOrbitNumber_start', orb))
            cov = sub.geometry(1000).intersection(geom, 100).area(100).divide(area).getInfo()
            info.append({'pass': ps, 'orbit': orb, 'cover': round(cov, 3), 'bins': len([x for x in bins if 0 <= x < len(self.bins)]), 'set': bins})
        info.sort(key=lambda d: (d['cover'] >= .95, d['bins']), reverse=True)
        for d in info:
            self.say('      track %s %d: covers %d%% of the regency, data in %d of %d periods' % (d['pass'].lower(), d['orbit'], round(d['cover']*100), d['bins'], len(self.bins)))
        full = [d for d in info if d['cover'] >= .95]
        to_vh = lambda c: c.map(lambda im: ee.Image(ee.Image(10).pow(im.select('VH').updateMask(im.select('VH').gt(-30)).divide(10)).rename('vh')
                                .copyProperties(im, ['system:time_start'])))
        pick = lambda d: to_vh(col.filter(ee.Filter.eq('orbitProperties_pass', d['pass'])).filter(ee.Filter.eq('relativeOrbitNumber_start', d['orbit'])))
        if full:
            main = full[0]; self.fill = [(pick(d), d) for d in full[1:]]
            covered = set(main['set'])
            for d in full[1:]: covered |= d['set']
            self.pass_ = '%s orbit %d' % (main['pass'].lower(), main['orbit'])
            self.s1_info = {'primary': self.pass_, 'primary_periods': main['bins'],
                            'fill_tracks': ['%s orbit %d' % (d['pass'].lower(), d['orbit']) for d in full[1:]],
                            'periods_with_data': len([x for x in covered if 0 <= x < len(self.bins)]), 'periods': len(self.bins)}
            self.say('      using %s; %d other full-coverage track(s) fill missing periods' % (self.pass_, len(self.fill)))
            return pick(main)
        hist = col.aggregate_histogram('orbitProperties_pass').getInfo(); best = max(hist, key=hist.get)
        self.fill = []; self.pass_ = best.lower()
        self.s1_info = {'primary': best.lower()+' (all tracks)', 'primary_periods': None, 'fill_tracks': [], 'periods_with_data': None, 'periods': len(self.bins)}
        self.say('      WARNING: no single track covers the regency; mixing all %s tracks' % best.lower())
        return to_vh(col.filter(ee.Filter.eq('orbitProperties_pass', best)))

    def vh_bins(self, col):
        ee, s = self.ee, self.s
        nil = ee.Image.constant(0).toFloat().rename('vh').updateMask(ee.Image.constant(0))
        db = lambda im: im.log10().multiply(10).rename('vh')
        fills = []
        for fc, _ in getattr(self, 'fill', []):
            off = db(col.mean()).subtract(db(fc.mean())).focalMean(100, 'square', 'meters')
            fills.append((fc, off))
        out = []
        for b in self.bins:
            a = str(b); z = str(b+np.timedelta64(s.bin_days, 'D'))
            m = db(ee.ImageCollection([nil]).merge(col.filterDate(a, z)).median().focalMean(20, 'square', 'meters'))
            for fc, off in fills:
                f = db(ee.ImageCollection([nil]).merge(fc.filterDate(a, z)).median().focalMean(20, 'square', 'meters')).add(off)
                m = m.unmask(f)
            out.append(m)
        n = len(out); filled = []
        for i in range(n):
            prv = None; nxt = None
            for j in range(i-1, max(-1, i-4), -1): prv = out[j] if prv is None else prv.unmask(out[j])
            for j in range(i+1, min(n, i+4)): nxt = out[j] if nxt is None else nxt.unmask(out[j])
            f = out[i]
            if prv is not None and nxt is not None: f = f.unmask(prv.add(nxt).divide(2))
            if prv is not None: f = f.unmask(prv)
            if nxt is not None: f = f.unmask(nxt)
            filled.append(f.rename('vh'))
        sm = []
        for i in range(n):
            sm.append(filled[i-1].add(filled[i]).add(filled[i+1]).divide(3).rename('vh') if 0 < i < n-1 else filled[i])
        return sm

    def seasons(self, sm):
        ee, s = self.ee, self.s
        mn = lambda L: ee.Image.cat(L).reduce(ee.Reducer.min())
        mx = lambda L: ee.Image.cat(L).reduce(ee.Reducer.max())
        kept = {}
        for i in self.target:
            si = sm[i]
            left, right = mn(sm[i-3:i]), mn(sm[i+1:i+4])
            fut = mx(sm[i+s.rise_from:i+s.rise_to+1])
            c = (si.lte(s.flood_db).And(si.lt(left)).And(si.lte(right))
                 .And(fut.subtract(si).gte(s.rise_db)).And(fut.gte(s.peak_db))).unmask(0)
            prev = [kept[j] for j in range(i-s.min_gap_bins, i) if j in kept]
            if prev: c = c.And(ee.Image.cat(prev).reduce(ee.Reducer.max()).Not())
            kept[i] = c
        n = ee.Image.cat([kept[i] for i in self.target]).reduce(ee.Reducer.sum()).rename('n')
        cum = ee.Image.constant(0); dates = {1: [], 2: [], 3: []}
        for i in self.target:
            cum = cum.add(kept[i])
            for k in (1, 2, 3):
                dates[k].append(ee.Image.constant(int(self.doff[i])).updateMask(kept[i].And(cum.eq(k))).toInt16())
        ds = [ee.ImageCollection(dates[k]).min().unmask(-999).rename('d%d' % k) for k in (1, 2, 3)]
        return n.addBands(ds)

    def sentinel2(self, geom, d0, d1, bands):
        ee = self.ee
        cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geom).filterDate(d0, d1)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)).linkCollection(cs, ['cs_cdf']))
        return col.map(lambda im: im.select(bands).updateMask(im.select('cs_cdf').gte(.6)).divide(10000)
                       .copyProperties(im, ['system:time_start']))

    def run(self):
        ee, s = self.ee, self.s
        self.say('[1/6] %s Regency boundary ...' % s.regency)
        geom, info = self.region(); self.geom, self.outline = geom, info
        pts = _main_pts(info)
        lon_c, lat_c = pts[:, 0].mean(), pts[:, 1].mean()
        utm = UTM.at(lon_c, lat_c)
        xs, ys = utm.fwd(pts[:, 0], pts[:, 1])
        w, h = xs.max()-xs.min(), ys.max()-ys.min()
        w, h = w*(1+2*s.pad), h*(1+2*s.pad)
        if w/h < s.map_aspect: w = h*s.map_aspect
        else: h = w/s.map_aspect
        cx, cy = (xs.max()+xs.min())/2, (ys.max()+ys.min())/2
        nx, ny = int(round(w/s.res)), int(round(h/s.res))
        x0 = math.floor((cx-w/2)/s.res)*s.res; y1 = math.ceil((cy+h/2)/s.res)*s.res
        g = Grid(utm, x0, y1, s.res, nx, ny); self.g = g
        self.center = (float(lon_c), float(lat_c))
        self.say('      grid %d x %d at %d m (EPSG:%d)' % (g.nx, g.ny, g.res, utm.epsg))

        self.say('[2/6] Sentinel-1 VH time series, %d bins of %d days ...' % (len(self.bins), s.bin_days))
        col = self.s1_collection(geom); self.col = col
        self.sm = self.vh_bins(col)
        inreg = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(geom)]), 1).rename('inreg')
        res = self.seasons(self.sm).updateMask(inreg).unmask(ee.Image.constant([0, -999, -999, -999]).rename(['n', 'd1', 'd2', 'd3']))
        wc = ee.ImageCollection('ESA/WorldCover/v200').first().unmask(0).rename('wc')
        stack = res.addBands(wc).addBands(inreg).toInt16()
        self.say('      detecting planting dates (slow step) ...')
        A = pull(ee, stack, g, ['n', 'd1', 'd2', 'd3', 'wc', 'inreg'], np.int16, tile=384)

        self.say('[3/6] Sentinel-2 background ...')
        s2 = self.sentinel2(geom, '2025-06-01', '2025-10-31', ['B4', 'B3', 'B2']).median().multiply(10000).unmask(0).toInt16()
        B = pull(ee, s2, g, ['B4', 'B3', 'B2'], np.int16, tile=512)
        A.update(B); self.A = A

        self.say('[4/6] Area statistics ...')
        R = self.stats()
        self.block = self.pick_site(s.block_km)
        self.say('[5/6] Sentinel-2 check of %d sample pixels ...' % s.s2_samples)
        R['validation'] = self.validate()
        R['examples'] = self.examples()
        R.update(setup=asdict(s), center=list(self.center), utm_epsg=utm.epsg, orbit_pass=self.pass_, s1=self.s1_info)
        r0, c0, k = self.block
        R['field_block'] = {'row': r0, 'col': c0, 'size_px': k, 'center': list(g.pixel_lonlat(r0+k//2, c0+k//2))}
        self.say('[6/6] Boundaries for the map ...')
        R['qc'] = self.qc(R)
        for q in R['qc']: self.say('  WARNING: ' + q)
        if not R['qc']: self.say('  Quality control: all checks passed.')
        try:
            prov = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', s.province))
            self.neigh = prov.map(lambda f: f.setGeometry(f.geometry().simplify(500))).getInfo()
            box = ee.Geometry.Rectangle([104, -9, 116, -4.5])
            self.java = (ee.FeatureCollection('FAO/GAUL/2015/level1').filterBounds(box)
                         .map(lambda f: f.setGeometry(f.geometry().simplify(3000)))).getInfo()
        except Exception:
            self.neigh = self.java = None
        self.R = R
        return R

    def stats(self):
        A, s = self.A, self.s
        inreg = A['inreg'] == 1
        crop = np.isin(A['wc'], [30, 40, 90])
        d1, d2, d3 = A['d1'], A['d2'], A['d3']
        rep3 = (d1 > -999) & (d3 > -999) & (d3-d1 > s.repeat_days)
        rep2 = (d1 > -999) & (d2 > -999) & (d2-d1 > s.repeat_days)
        A['d3'] = np.where(rep3 | rep2, -999, d3).astype(np.int16)
        A['d2'] = np.where(rep2, -999, d2).astype(np.int16)
        A['n'] = ((A['d1'] > -999).astype(np.int16)+(A['d2'] > -999)+(A['d3'] > -999)).astype(np.int16)
        n = np.where(inreg & crop, A['n'], 0)
        A['nn'] = n
        px = s.res**2/1e4
        rice = n >= 1
        dates = [A['d%d' % k] for k in (1, 2, 3)]
        harvest_month = np.zeros(13)
        plant_bin = {}
        for d in dates:
            m = rice & (d > -999)
            hv = (np.datetime64('2025-01-01')+(d[m]+s.field_days).astype('timedelta64[D]')).astype('datetime64[M]')
            for k, c in zip(*np.unique(hv, return_counts=True)):
                yr, mo = int(str(k)[:4]), int(str(k)[5:7])
                if yr == 2025: harvest_month[mo] += c*px
            for k, c in zip(*np.unique(d[m], return_counts=True)): plant_bin[int(k)] = plant_bin.get(int(k), 0)+c*px
        R = {'regency': s.regency, 'province': s.province, 'pixel_ha': px,
             'rice_ha': float(rice.sum()*px), 'cropland_ha': float((inreg & crop).sum()*px),
             'seasons_ha': {k: float((n == k).sum()*px) for k in (1, 2, 3)},
             'harvest_ha': float(n[rice].sum()*px),
             'harvest_2025_ha': float(harvest_month.sum()),
             'harvest_by_month': {m: float(harvest_month[m]) for m in range(1, 13)},
             'planting_by_bin': {int(k): float(v) for k, v in sorted(plant_bin.items())}}
        R['cropping_index'] = R['harvest_ha']/max(R['rice_ha'], 1e-9)
        return R

    def sample_points(self):
        A, s, g = self.A, self.s, self.g
        rng = np.random.default_rng(7)
        r, c = np.where(A['nn'] >= 1)
        k = rng.choice(len(r), size=min(s.s2_samples, len(r)), replace=False)
        return [(int(r[i]), int(c[i])) for i in k]

    def ndvi_peaks(self, v):
        w0, w1 = [int((np.datetime64(x)-self.ref).astype(int)) for x in self.s.window]
        off = self.ndvi_off; out = []
        for j in np.where(np.nan_to_num(v, nan=-1) >= .65)[0]:
            win = v[max(0, j-3):j+4]
            if v[j] < np.nanmax(win): continue
            p_ = off[j]
            if not (w0+35 <= p_ <= w1+60): continue
            pre = v[(off >= p_-100) & (off <= p_-25)]
            if np.all(np.isnan(pre)) or np.nanmin(pre) > .35: continue
            if out and p_-out[-1] < 80: continue
            out.append(int(p_))
        return out

    def ndvi_stack(self):
        ee, s = self.ee, self.s
        col = self.sentinel2(self.geom, s.start, s.end, ['B4', 'B8'])
        col = col.map(lambda im: im.normalizedDifference(['B8', 'B4']).rename('ndvi').copyProperties(im, ['system:time_start']))
        nil = ee.Image.constant(0).toFloat().rename('ndvi').updateMask(ee.Image.constant(0))
        d0, d1 = np.datetime64(s.start), np.datetime64(s.end)
        steps = [d0+np.timedelta64(i*10, 'D') for i in range(int((d1-d0).astype(int)//10))]
        self.ndvi_off = np.array([int((t+np.timedelta64(5, 'D')-self.ref).astype(int)) for t in steps])
        ims = [ee.ImageCollection([nil]).merge(col.filterDate(str(t), str(t+np.timedelta64(10, 'D')))).max().rename('t%d' % i)
               for i, t in enumerate(steps)]
        return ee.Image.cat(ims).unmask(-9)

    def sample(self, image, pts, scale):
        ee, g = self.ee, self.g; out = {}
        for k in range(0, len(pts), 250):
            ch = pts[k:k+250]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(list(g.pixel_lonlat(r, c))), {'i': k+j}) for j, (r, c) in enumerate(ch)])
            for f in image.sampleRegions(collection=fc, scale=scale, geometries=False).getInfo()['features']:
                out[int(f['properties']['i'])] = f['properties']
        return out

    def validate(self):
        A, s = self.A, self.s
        pts = self.sample_points(); self.pts = pts
        nd = self.ndvi_stack()
        S = self.sample(nd, pts, 10)
        T = len(self.ndvi_off)
        conf = rej = noview = 0; lags = []; seen = caught = 0
        self.ndvi_samples = {}; status = []
        for i, (r, c) in enumerate(pts):
            p = S.get(i)
            status.append('none')
            if not p: continue
            pc = pr = 0
            v = np.array([p.get('t%d' % j, -9) for j in range(T)], float); v[v < -1] = np.nan
            self.ndvi_samples[i] = v
            for k in (1, 2, 3):
                d = int(A['d%d' % k][r, c])
                if d == -999: continue
                aft = (self.ndvi_off >= d+35) & (self.ndvi_off <= d+100)
                bef = (self.ndvi_off >= d-20) & (self.ndvi_off <= d+15)
                va, vb = v[aft], v[bef]
                if np.all(np.isnan(va)): noview += 1; continue
                top = np.nanmax(va); low = np.nanmin(vb) if not np.all(np.isnan(vb)) else np.nan
                ok = top >= .55 and (np.isnan(low) or low <= .40 or top-low >= .25)
                if ok:
                    conf += 1; pc += 1; lags.append(int(self.ndvi_off[aft][np.nanargmax(va)]-d))
                else:
                    rej += 1; pr += 1
            status[-1] = 'none' if not (pc or pr) else ('confirmed' if not pr else ('rejected' if not pc else 'partly'))
            dates = [int(A['d%d' % k][r, c]) for k in (1, 2, 3) if A['d%d' % k][r, c] != -999]
            for p_ in self.ndvi_peaks(v):
                seen += 1
                if any(35 <= p_-d <= 100 for d in dates): caught += 1
        V = {'s2': {'seasons_checked': conf+rej, 'confirmed': conf, 'rejected': rej, 'no_clear_view': noview,
                    'rate': conf/max(conf+rej, 1), 'median_peak_lag_days': float(np.median(lags)) if lags else None,
                    'points': len(pts), 'ndvi_crops': seen, 'ndvi_crops_caught': caught, 'recall': caught/max(seen, 1)}}
        V['points'] = [[round(float(x), 5) for x in self.g.pixel_lonlat(r, c)]+[st_] for (r, c), st_ in zip(pts, status)]
        R0 = self.stats()
        V['lbs'] = {'reference_ha': s.lbs_ha, 'mapped_ha': R0['rice_ha'], 'diff_pct': 100*(R0['rice_ha']-s.lbs_ha)/s.lbs_ha}
        if s.bps_harvest_ha:
            V['bps'] = {'reference_ha': s.bps_harvest_ha, 'mapped_ha': R0['harvest_2025_ha'],
                        'diff_pct': 100*(R0['harvest_2025_ha']-s.bps_harvest_ha)/s.bps_harvest_ha}
        return V

    def examples(self):
        ee, A = self.ee, self.A
        pts = getattr(self, 'pts', [])
        picks = []
        for k in (3, 2, 1):
            for i, (r, c) in enumerate(pts):
                if A['nn'][r, c] == k and i in self.ndvi_samples and np.isfinite(self.ndvi_samples[i]).sum() >= 12:
                    picks.append((i, r, c)); break
        if not picks: return []
        img = ee.Image.cat([b.rename('b%d' % j) for j, b in enumerate(self.sm)]).unmask(-99)
        S = self.sample(img, [(r, c) for _, r, c in picks], 30)
        out = []
        for j, (i, r, c) in enumerate(picks):
            p = S.get(j, {}); vh = [p.get('b%d' % t, -99) for t in range(len(self.sm))]
            lon, lat = self.g.pixel_lonlat(r, c)
            out.append({'lon': lon, 'lat': lat, 'n': int(A['nn'][r, c]), 'vh': [None if v < -90 else float(v) for v in vh],
                        'vh_off': [int(x) for x in self.doff], 'ndvi': [None if np.isnan(x) else float(x) for x in self.ndvi_samples[i]],
                        'ndvi_off': [int(x) for x in self.ndvi_off],
                        'dates': [int(A['d%d' % k][r, c]) for k in (1, 2, 3) if A['d%d' % k][r, c] != -999]})
        return out

    def save(self, R, out):
        os.makedirs(out, exist_ok=True); g = self.g
        with open(f'{out}/08_results.json', 'w') as fh: json.dump(R, fh, indent=1, default=float)
        np.savez_compressed(f'{out}/09_layers.npz', **{k: self.A[k] for k in ('n', 'nn', 'd1', 'd2', 'd3', 'wc', 'inreg', 'B4', 'B3', 'B2') if k in self.A},
                            grid=np.array([g.utm.zone, int(g.utm.south), g.x0, g.y1, g.res, g.nx, g.ny]))
        with open(f'{out}/10_geometry.json', 'w') as fh: json.dump({'outline': self.outline, 'neigh': self.neigh, 'java': self.java}, fh)

    def restore(self, out):
        with open(f'{out}/08_results.json') as fh: R = json.load(fh)
        Z = np.load(f'{out}/09_layers.npz'); gr = Z['grid']
        self.A = {k: Z[k] for k in Z.files if k != 'grid'}
        self.g = Grid(UTM(int(gr[0]), bool(gr[1])), float(gr[2]), float(gr[3]), float(gr[4]), int(gr[5]), int(gr[6]))
        with open(f'{out}/10_geometry.json') as fh: G = json.load(fh)
        self.outline, self.neigh, self.java = G['outline'], G['neigh'], G['java']
        fb = R['field_block']; self.block = (fb['row'], fb['col'], fb['size_px'])
        self.center = tuple(R['center'])
        return R

    def radar(self):
        self.geom, _ = self.region()
        self.col = self.s1_collection(self.geom)

    def pick_site(self, km):
        A, g = self.A, self.g
        k = int(round(km*1000/g.res)); h = k//2
        rice = (A['nn'] >= 1).astype(float)
        d = np.where(A['d1'] > -999, A['d1'], 0).astype(float)*rice
        f = ndimage.uniform_filter(rice, k, mode='constant')
        m1 = ndimage.uniform_filter(d, k, mode='constant')/np.maximum(f, 1e-6)
        m2 = ndimage.uniform_filter(d*d, k, mode='constant')/np.maximum(f, 1e-6)
        score = f*np.clip(np.sqrt(np.clip(m2-m1**2, 0, None))/30, 0, 1)
        if score.max() <= 0: score = f
        score[:h] = 0; score[-h:] = 0; score[:, :h] = 0; score[:, -h:] = 0
        r, c = np.unravel_index(np.argmax(score), score.shape)
        return int(r-h), int(c-h), k

    def site_series(self, res=10):
        g, A, s = self.g, self.A, self.s
        r0, c0, k = self.block; f = int(round(g.res/res))
        sg = Grid(g.utm, g.x0+c0*g.res, g.y1-r0*g.res, res, k*f, k*f)
        up = lambda a: np.repeat(np.repeat(a[r0:r0+k, c0:c0+k], f, 0), f, 1)
        self.site = self._series(sg, up(A['nn']) >= 1, [up(A['d%d' % j]) for j in (1, 2, 3)], 30, 'field block')
        self.site['km'] = s.block_km
        return self.site

    def regency_series(self, res=60):
        g, A = self.g, self.A; f = max(1, int(round(res/g.res)))
        ny, nx = g.ny//f, g.nx//f
        sg = Grid(g.utm, g.x0, g.y1, g.res*f, nx, ny)
        dn = lambda a: a[:ny*f:f, :nx*f:f]
        rice = coarse_frac(A['nn'] >= 1, f, ny, nx) >= .5
        inreg = coarse_frac(A['inreg'] == 1, f, ny, nx) >= .5
        self.regts = self._series(sg, rice & inreg, [dn(A['d%d' % j]) for j in (1, 2, 3)], 0, 'regency', inreg=inreg, keep_clear=False)
        self.regts['inreg'] = inreg
        return self.regts

    def _series(self, sg, rice, dd, focal, label, inreg=None, keep_clear=True):
        ee, s = self.ee, self.s
        rect = sg.rect(ee)
        cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(rect).filterDate(s.start, s.end)
              .linkCollection(cs, ['cs_cdf']).map(lambda im: im.select(['B4', 'B3', 'B2', 'B8', 'cs_cdf']).toFloat()))
        nil1 = ee.Image.constant(0).toFloat().rename('vh').updateMask(ee.Image.constant(0))
        nil2 = ee.Image.constant([0, 0, 0, 0, 0]).toFloat().rename(['B4', 'B3', 'B2', 'B8', 'cs_cdf']).updateMask(ee.Image.constant(0))
        T = len(self.bins); H, W = sg.ny, sg.nx
        vh = np.zeros((T, H, W), np.int16); rgb = np.zeros((T, H, W, 3), np.uint8)
        clear = np.zeros((T, H, W), bool) if keep_clear else None
        st = {k_: np.full(T, np.nan) for k_ in ('water', 'green', 'cloud', 'planted')}
        st['s1'] = np.zeros(T, bool); st['s2'] = np.zeros(T, bool)
        n_rice = max(rice.sum(), 1); area = inreg if inreg is not None else np.ones((H, W), bool)
        for i, b in enumerate(self.bins):
            a = str(b); z = str(b+np.timedelta64(s.bin_days, 'D'))
            v = ee.ImageCollection([nil1]).merge(self.col.filterDate(a, z)).median()
            if focal: v = v.focalMean(focal, 'circle', 'meters')
            v = v.log10().multiply(1000).unmask(-9999).toInt16().rename('vh')
            o = ee.ImageCollection([nil2]).merge(s2.filterDate(a, z)).qualityMosaic('cs_cdf')
            o = o.select(['B4', 'B3', 'B2', 'B8']).addBands(o.select('cs_cdf').multiply(100)).unmask(0).toInt16()
            P = pull(ee, v.addBands(o), sg, ['vh', 'B4', 'B3', 'B2', 'B8', 'cs_cdf'], np.int16, tile=min(1024, max(W, H)))
            vh[i] = P['vh']
            refl = np.dstack([P['B4'], P['B3'], P['B2']]).astype(np.float32)
            rgb[i] = (np.clip(refl/2500, 0, 1)**.8*255).astype(np.uint8)
            has1 = (P['vh'][area] > -9000).mean() > .5; has2 = (refl[area].sum(-1) > 0).mean() > .5
            st['s1'][i], st['s2'][i] = has1, has2
            if has1:
                st['water'][i] = ((P['vh']/100 <= s.flood_db) & rice).sum()/n_rice*100
            if has2:
                ok = P['cs_cdf'] >= 60
                if keep_clear: clear[i] = ok
                st['cloud'][i] = 100-ok[area].mean()*100
                rc = ok & rice
                if rc.sum() > .3*n_rice:
                    nd = (P['B8']-P['B4'])/np.maximum(P['B8']+P['B4'], 1)
                    st['green'][i] = ((nd >= .5) & rc).sum()/rc.sum()*100
            if i in self.target:
                st['planted'][i] = sum(((x == self.doff[i]) & rice).sum() for x in dd)/n_rice*100
            if (i+1) % 11 == 0: self.say('      %s: %d/%d periods' % (label, i+1, T))
        lon, lat = sg.pixel_lonlat(H//2, W//2)
        return {'grid': sg, 'vh': vh, 'rgb': rgb, 'clear': clear, 'rice': rice, 'dates': dd, 'stats': st, 'center': (float(lon), float(lat))}

    def site_arrays(self):
        S = self.site; g = S['grid']
        out = {'vh': S['vh'], 'rgb': S['rgb'], 'clear': S['clear'], 'rice': S['rice'],
               'd1': S['dates'][0], 'd2': S['dates'][1], 'd3': S['dates'][2],
               'grid': np.array([g.utm.zone, int(g.utm.south), g.x0, g.y1, g.res, g.nx, g.ny]),
               'bins': np.array([str(b) for b in self.bins]), 'doff': self.doff}
        out.update({'st_'+k: np.asarray(v, float) for k, v in S['stats'].items()})
        return out

    def qc(self, R):
        V = R['validation']; q = []
        if abs(V['lbs']['diff_pct']) > 20:
            q.append('Mapped rice area differs %.0f%% from the official paddy-field baseline (limit 20%%).' % V['lbs']['diff_pct'])
        if V['s2']['seasons_checked'] < 200:
            q.append('Only %d detected seasons could be checked with Sentinel-2.' % V['s2']['seasons_checked'])
        elif V['s2']['rate'] < .75:
            q.append('Only %.0f%% of detected seasons were confirmed by Sentinel-2 (limit 75%%).' % (V['s2']['rate']*100))
        if V['s2'].get('ndvi_crops', 0) >= 100 and V['s2']['recall'] < .75:
            q.append('Radar caught only %.0f%% of the crops visible in Sentinel-2 NDVI (limit 75%%): seasons are being missed.' % (V['s2']['recall']*100))
        if 'bps' in V and abs(V['bps']['diff_pct']) > 20:
            q.append('Harvested area differs %.0f%% from BPS (limit 20%%).' % V['bps']['diff_pct'])
        if not 1.0 <= R['cropping_index'] <= 3.0:
            q.append('Cropping index %.2f is outside the plausible range.' % R['cropping_index'])
        S1 = R.get('s1', {})
        if S1.get('periods_with_data') is not None and S1['periods']-S1['periods_with_data'] > 3:
            q.append('Sentinel-1 has no data in %d of %d periods.' % (S1['periods']-S1['periods_with_data'], S1['periods']))
        return q



import textwrap
import datetime
import matplotlib.patheffects as pe
from matplotlib.path import Path

BG, INK, MU, LN, ACC = '#f6f4ef', '#24282b', '#6e706c', '#cfcac0', '#b4462b'
SEA, SEA_INK = '#d3e1e8', '#5d7b88'
SEASON_COLS = {1: '#e2c15a', 2: '#7ea94a', 3: '#2d6a4f'}
DATE_CMAP = LinearSegmentedColormap.from_list('plant', ['#24467a', '#2f7f9e', '#5db39a', '#cfcb6c', '#e39e48', '#bf532c'])
AGE_CMAP = LinearSegmentedColormap.from_list('age', [(0, '#e4702a'), (.22, '#c4c35a'), (.5, '#5a9a3e'), (.8, '#2c6634'), (1, '#d6b04c')])
RADAR_CMAP = LinearSegmentedColormap.from_list('radar', [(0, '#15304f'), (.3, '#2f6ea3'), (.52, '#cdb78f'), (.74, '#9cbc5a'), (1, '#2b6a2e')])
RADAR_LO, RADAR_HI = -25.0, -11.0
WATER, GREEN, ORANGE = '#2f6ea3', '#4f8a2c', '#e07b2e'
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
REF = np.datetime64('2025-01-01')


def F(fp, size):
    p = fp.copy(); p.set_size(size); return p


def halo(w=2.4, c=BG): return [pe.withStroke(linewidth=w, foreground=c)]


def to_date(d): return REF+np.timedelta64(int(d), 'D')


def day_label(d, year=True):
    t = str(to_date(d)); return '%d %s%s' % (int(t[8:10]), MONTHS[int(t[5:7])-1], ' '+t[:4] if year else '')


def bin_label(b, year=True):
    t = str(b); return '%d %s%s' % (int(t[8:10]), MONTHS[int(t[5:7])-1], ' '+t[:4] if year else '')


def part_month(d):
    t = str(to_date(d)); day = int(t[8:10])
    return '%s %s %s' % ('early' if day <= 10 else ('mid' if day <= 20 else 'late'), MONTHS_LONG[int(t[5:7])-1], t[:4])


def season_hist(study, k):
    A = study.A; d = A['d%d' % k]; m = (A['nn'] >= 1) & (d > -999)
    u, c = np.unique(d[m], return_counts=True)
    return u.astype(int), c*study.s.res**2/1e4


def date_range(study, k):
    A = study.A; d = A['d%d' % k][(A['nn'] >= 1) & (A['d%d' % k] > -999)]
    if d.size == 0: return -90, 60
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi-lo < 36: hi = lo+36
    return float(lo), float(hi)


def base_rgb(study):
    A = study.A; wc = A['wc']; inreg = A['inreg'] == 1
    c = lambda h: np.array(matplotlib.colors.to_rgb(h))
    water = np.isin(wc, [80, 0])
    out = np.empty(wc.shape+(3,), np.float32); out[:] = c('#ece9e2')
    out[inreg] = c('#dcd7cc')
    out[inreg & np.isin(wc, [10, 20, 95])] = c('#cdd1bd')
    out[~inreg & (wc == 50)] = c('#dad5cb')
    out[inreg & (wc == 50)] = c('#a39c90')
    out[water] = c(SEA)
    out[water & inreg & (wc == 80)] = c('#bcd2dc')
    return out, water


def coarse(mask, q):
    ny, nx = mask.shape; ny2, nx2 = ny//q, nx//q
    return mask[:ny2*q, :nx2*q].reshape(ny2, q, nx2, q).mean((1, 3))


class Placer:
    def __init__(self, busy, aspect):
        self.B = busy; self.boxes = []; self.aspect = aspect

    def busy(self, b):
        x0, y0, x1, y1 = b; ny, nx = self.B.shape
        c0, c1 = int(max(0, x0)*nx), int(np.ceil(min(1, x1)*nx)); r0, r1 = int((1-min(1, y1))*ny), int(np.ceil((1-max(0, y0))*ny))
        if c1 <= c0 or r1 <= r0: return 1.0
        return float(self.B[r0:r1, c0:c1].mean())

    def overlap(self, b):
        t = 0.0
        for o in self.boxes:
            w = min(b[2], o[2])-max(b[0], o[0]); h = min(b[3], o[3])-max(b[1], o[1])
            if w > 0 and h > 0: t += w*h/((b[2]-b[0])*(b[3]-b[1]))
        return t

    def corner(self, bw, bh, pad=.018):
        c = {'tl': (pad, 1-pad-bh), 'tr': (1-pad-bw, 1-pad-bh), 'bl': (pad, pad), 'br': (1-pad-bw, pad)}
        best = min(c, key=lambda k: self.busy((c[k][0], c[k][1], c[k][0]+bw, c[k][1]+bh))+5*self.overlap((c[k][0], c[k][1], c[k][0]+bw, c[k][1]+bh)))
        x, y = c[best]; b = (x, y, x+bw, y+bh); self.boxes.append(b)
        return best, b


def text_box(lines, size, inch_w, inch_h, char=.5, lead=1.3):
    n = max(len(l) for l in lines)
    return n*size*char/72/inch_w, len(lines)*size*lead/72/inch_h


def callout(ax, P, pt, text, fp, size, iw, ih, dists=(.04, .06, .09, .13)):
    lines = text.split('\n'); w, h = text_box(lines, size, iw, ih)
    best = None
    for d in dists:
        for ang in range(0, 360, 30):
            a = np.radians(ang); cx = pt[0]+d*np.cos(a)*ih/iw; cy = pt[1]+d*np.sin(a)
            ha = 'left' if np.cos(a) > .3 else ('right' if np.cos(a) < -.3 else 'center')
            x0 = cx if ha == 'left' else (cx-w if ha == 'right' else cx-w/2)
            y0 = cy-h/2; b = (x0-.005, y0-.005, x0+w+.005, y0+h+.005)
            if b[0] < .01 or b[2] > .99 or b[1] < .01 or b[3] > .99: continue
            sc = .5*P.busy(b)+8*P.overlap(b)+3*d
            if best is None or sc < best[0]: best = (sc, cx, cy, ha, b)
    if best is None: return
    _, cx, cy, ha, b = best; P.boxes.append(b)
    ex = b[0]+.004 if ha == 'left' else (b[2]-.004 if ha == 'right' else (b[0]+b[2])/2)
    ey = cy if ha != 'center' else (b[1] if cy > pt[1] else b[3])
    ax.plot([pt[0], ex], [pt[1], ey], transform=ax.transAxes, color=INK, lw=.7, zorder=20, solid_capstyle='round',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
    ax.plot([pt[0]], [pt[1]], 'o', transform=ax.transAxes, ms=4.2, mfc=INK, mec=BG, mew=1.2, zorder=21)
    ax.text(cx, cy, text, transform=ax.transAxes, fontproperties=F(fp, size), color=INK, ha=ha, va='center', zorder=22,
            linespacing=1.25, bbox=dict(boxstyle='square,pad=0.35', fc=BG, ec=LN, lw=.5))


def densest(mask, q=20):
    if not mask.any(): return None
    C = ndimage.uniform_filter(mask.astype(float), q*2+1, mode='constant')
    r, c = np.unravel_index(np.argmax(C), C.shape)
    return r, c


def rc_frac(study, r, c): return (c+.5)/study.g.nx, 1-(r+.5)/study.g.ny


def utm_path(study, gm):
    u = study.g.utm; out = []
    for p in _coords(gm):
        a = np.array(p[0]); x, y = u.fwd(a[:, 0], a[:, 1]); out.append(np.column_stack([x, y]))
    return out


def spaced(s): return s.upper()


def neighbour_labels(ax, study, R, P, fonts, water):
    g = study.g; e = g.extent
    if not getattr(study, 'neigh', None): return
    nyc, nxc = 60, 90
    xs = e[0]+(np.arange(nxc)+.5)*(e[1]-e[0])/nxc; ys = e[3]-(np.arange(nyc)+.5)*(e[3]-e[2])/nyc
    XX, YY = np.meshgrid(xs, ys); pts = np.column_stack([XX.ravel(), YY.ravel()])
    wq = water[::max(1, g.ny//nyc), ::max(1, g.nx//nxc)][:nyc, :nxc]
    for f in study.neigh['features']:
        name = f['properties'].get('ADM2_NAME', '')
        if name == R['regency']: continue
        inside = np.zeros(len(pts), bool)
        for poly in utm_path(study, f['geometry']):
            inside |= Path(poly).contains_points(pts)
            ax.plot(poly[:, 0], poly[:, 1], color='#a9a49a', lw=.6, zorder=4)
        M = inside.reshape(nyc, nxc)
        if wq.shape == M.shape: M &= ~(wq > .5)
        if M.mean() < .02: continue
        D = ndimage.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]
        r, c = np.unravel_index(np.argmax(D), D.shape)
        fx, fy = (c+.5)/nxc, 1-(r+.5)/nyc
        lab = spaced(name.replace('Kota ', ''))
        w = len(lab)*9*.55/72/ax.figure.get_size_inches()[0]/ax.get_position().width
        b = (fx-w/2, fy-.012, fx+w/2, fy+.012)
        if P.overlap(b) > 0: continue
        P.boxes.append(b)
        ax.text(fx, fy, lab, transform=ax.transAxes, fontproperties=F(fonts['light'], 9), color='#7d7a72', ha='center', va='center',
                zorder=12, path_effects=halo(2.2))


def sea_label(ax, study, R, P, fonts, water):
    name = R['setup'].get('sea_name')
    if not name or water.mean() < .04: return
    q = max(1, study.g.nx//120); W_ = coarse(water.astype(float), q) > .9
    for b in P.boxes:
        ny, nx = W_.shape
        W_[int((1-b[3])*ny):int(np.ceil((1-b[1])*ny)), int(b[0]*nx):int(np.ceil(b[2]*nx))] = False
    if not W_.any(): return
    D = ndimage.distance_transform_edt(np.pad(W_, 1))[1:-1, 1:-1]
    r, c = np.unravel_index(np.argmax(D), D.shape)
    fx, fy = (c+.5)/W_.shape[1], 1-(r+.5)/W_.shape[0]
    ax.text(fx, fy, spaced(name), transform=ax.transAxes, fontproperties=F(fonts['italic'], 15), color=SEA_INK, ha='center', va='center', zorder=12)
    P.boxes.append((fx-.08, fy-.02, fx+.08, fy+.02))


def graticule(ax, g, fonts, step_min=10):
    e = g.extent; u = g.utm; st = step_min/60
    corners = [u.inv(x, y) for x in e[:2] for y in e[2:]]
    lons = [c[0] for c in corners]; lats = [c[1] for c in corners]
    lon_t = np.arange(math.ceil(min(lons)/st)*st, max(lons), st); lat_t = np.arange(math.ceil(min(lats)/st)*st, max(lats), st)
    for lo in lon_t:
        xs, ys = u.fwd(np.full(80, lo), np.linspace(min(lats)-.1, max(lats)+.1, 80)); ax.plot(xs, ys, color='white', lw=.5, alpha=.55, zorder=3)
        xt = np.interp(e[3], ys, xs); xb = np.interp(e[2], ys, xs)
        if not (e[0]+(e[1]-e[0])*.03 < xt < e[1]-(e[1]-e[0])*.03): continue
        ax.annotate(_dms(lo, 'E', 'W'), (xt, e[3]), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    fontproperties=F(fonts['sans'], 7.5), color=MU, annotation_clip=False)
        for yv, sgn in ((e[3], -1), (e[2], 1)): ax.plot([xt if sgn < 0 else xb]*2, [yv, yv+sgn*(e[3]-e[2])*.008], color=INK, lw=.7, zorder=30)
    for la in lat_t:
        xs, ys = u.fwd(np.linspace(min(lons)-.1, max(lons)+.1, 80), np.full(80, la)); ax.plot(xs, ys, color='white', lw=.5, alpha=.55, zorder=3)
        yl = np.interp(e[0], xs, ys); yr = np.interp(e[1], xs, ys)
        if not (e[2]+(e[3]-e[2])*.04 < yl < e[3]-(e[3]-e[2])*.04): continue
        ax.annotate(_dms(la, 'N', 'S'), (e[0], yl), xytext=(-4, 0), textcoords='offset points', ha='right', va='center', rotation=90,
                    fontproperties=F(fonts['sans'], 7.5), color=MU, annotation_clip=False)
        for xv, yv, sgn in ((e[0], yl, 1), (e[1], yr, -1)): ax.plot([xv, xv+sgn*(e[1]-e[0])*.006], [yv, yv], color=INK, lw=.7, zorder=30)
    ax.set_xlim(e[:2]); ax.set_ylim(e[2:]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(.9)


def outline(ax, study, lw=1.0):
    for poly in utm_path(study, study.outline):
        ax.plot(poly[:, 0], poly[:, 1], color='white', lw=lw+2.4, zorder=8, solid_joinstyle='round')
        ax.plot(poly[:, 0], poly[:, 1], color=INK, lw=lw, zorder=9, solid_joinstyle='round')


def inset(fig, ax, study, R, box, fonts):
    x0, y0, x1, y1 = box
    ia = ax.inset_axes([x0, y0, x1-x0, y1-y0], zorder=17); ia.set_facecolor(SEA)
    if getattr(study, 'java', None):
        for f in study.java['features']:
            hit = f['properties'].get('ADM1_NAME') == R['province']
            _poly(ia, f['geometry'], fc='#e2ddd1' if hit else '#f4f2ec', ec='#a39e93', lw=.3)
    if getattr(study, 'neigh', None):
        for f in study.neigh['features']:
            if f['properties'].get('ADM2_NAME') == R['regency']: _poly(ia, f['geometry'], fc=ACC, ec=ACC, lw=.4, zorder=5)
    ia.set_xlim(105.0, 114.8); ia.set_ylim(-8.9, -5.7); ia.set_aspect('equal', adjustable='datalim')
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values(): sp.set_color(INK); sp.set_linewidth(.7)
    ia.text(.04, .08, 'JAVA', transform=ia.transAxes, fontproperties=F(fonts['light'], 7.5), color='#7d7a72', va='bottom')


def legend_hist(fig, rect, study, k, lo, hi, fonts):
    a = fig.add_axes(rect); a.set_facecolor('none')
    u, v = season_hist(study, k)
    if len(u):
        cols = [DATE_CMAP(np.clip((d-lo)/(hi-lo), 0, 1)) for d in u]
        a.bar(u, v, width=study.s.bin_days*.86, color=cols, linewidth=0)
        i = int(np.argmax(v)); a.text(u[i], v[i]*1.04, '%s ha' % fmt(v[i]), fontproperties=F(fonts['sans'], 7.5), color=MU, ha='center', va='bottom')
        a.set_ylim(0, v.max()*1.22)
        a0, a1 = min(u.min(), lo)-8, max(u.max(), hi)+8
    else:
        a0, a1 = lo, hi
    a.set_xlim(a0, a1)
    ms = []
    for y_ in (2024, 2025):
        for m in range(12):
            d = int((np.datetime64('%d-%02d-01' % (y_, m+1))-REF).astype(int))
            if a0 <= d <= a1: ms.append((d, m, y_))
    a.set_xticks([d for d, _, _ in ms])
    a.set_xticklabels([MONTHS[m]+('\n%d' % y_ if (j == 0 or m == 0) else '') for j, (_, m, y_) in enumerate(ms)], fontproperties=F(fonts['sans'], 8), color=MU, ha='left', va='top')
    a.tick_params(axis='x', length=4, color=MU, pad=2); a.set_yticks([])
    for sp in ('top', 'right', 'left'): a.spines[sp].set_visible(False)
    a.spines['bottom'].set_color(MU); a.spines['bottom'].set_linewidth(.6)
    return a


SHEET_TEXT = {
    'date1': ('Rice Planting Calendar, {r} Regency',
              'First rice season of the 2024/25 crop year: the date each field was flooded and transplanted, read from Sentinel-1 radar'),
    'date2': ('Second Rice Season, {r} Regency',
              'Transplanting date of the second rice crop of the 2024/25 crop year, read from Sentinel-1 radar'),
    'seasons': ('Rice Cropping Intensity, {r} Regency',
                'Number of rice crops each field carried in the 2024/25 crop year, counted from flooding and regrowth in Sentinel-1 radar'),
    'validation': ('Accuracy Check of the Rice Map, {r} Regency',
                   '{n} random rice pixels checked against Sentinel-2: did the field turn green 35 to 100 days after the radar said it was planted?'),
}


def sheet_note(R, study, kind):
    A, s = study.A, R['setup']; rice = A['nn'] >= 1
    if kind == 'date1':
        d = A['d1'][rice & (A['d1'] > -999)]
        return ('Rice was detected on %s ha. Half of the first-season area had been transplanted by %s; '
                'the earliest tenth went in before %s and the last tenth after %s.'
                % (fmt(R['rice_ha']), day_label(np.median(d)), day_label(np.percentile(d, 10)), day_label(np.percentile(d, 90)))) if d.size else ''
    if kind == 'date2':
        d = A['d2'][rice & (A['d2'] > -999)]
        return ('%s ha carried a second crop, transplanted mostly between %s and %s. Grey fields carried a single crop.'
                % (fmt(d.size*s['res']**2/1e4), day_label(np.percentile(d, 10)), day_label(np.percentile(d, 90)))) if d.size else ''
    if kind == 'seasons':
        return ('Average cropping index %.2f: %s ha of rice fields produced %s ha of harvested area in the crop year.'
                % (R['cropping_index'], fmt(R['rice_ha']), fmt(R['harvest_ha'])))
    v = R['validation']['s2']; L = R['validation']['lbs']
    rc = (' The radar caught %d%% of the crops that Sentinel-2 NDVI shows at these pixels.' % round(v['recall']*100)) if v.get('ndvi_crops') else ''
    t = ('%d%% of the %s crop seasons with a cloud-free Sentinel-2 view were confirmed. Mapped rice area %s ha, %+.0f%% against '
         'the official paddy-field baseline of %s ha.' % (round(v['rate']*100), fmt(v['seasons_checked']), fmt(L['mapped_ha']), L['diff_pct'], fmt(L['reference_ha'])))+rc
    if 'bps' in R['validation']:
        b = R['validation']['bps']; t += ' Harvested area %+.0f%% against BPS.' % b['diff_pct']
    return t


def map_sheet(R, study, path, kind='date1', fonts=None, dpi=None):
    fonts = fonts or load_fonts()
    A, g, s = study.A, study.g, R['setup']
    W, H = 16.54, 11.69; mx = .42; head = .95; fmin = 2.1
    mw = W-2*mx; mh = mw*g.ny/g.nx
    if mh > H-head-fmin: mh = H-head-fmin; mw = mh*g.nx/g.ny
    foot = H-head-mh
    fig = plt.figure(figsize=(W, H), dpi=dpi or s['dpi']); fig.patch.set_facecolor(BG)
    xl = (W-mw)/2
    ax = fig.add_axes([xl/W, foot/H, mw/W, mh/H]); e = g.extent
    base, water = base_rgb(study)
    ax.imshow(base, extent=e, interpolation='bilinear', zorder=1)
    rice = A['nn'] >= 1; inreg = A['inreg'] == 1
    lay = np.zeros(rice.shape+(4,), np.float32)
    if kind in ('date1', 'date2'):
        k = 1 if kind == 'date1' else 2
        lo, hi = date_range(study, k); d = A['d%d' % k]; m = rice & (d > -999)
        lay[m] = DATE_CMAP(np.clip((d[m]-lo)/(hi-lo), 0, 1)); lay[m, 3] = 1
        if k == 2: lay[rice & ~m] = (*matplotlib.colors.to_rgb('#bdb7a9'), .9)
    elif kind == 'seasons':
        for k_, c in SEASON_COLS.items(): lay[A['nn'] == k_] = (*matplotlib.colors.to_rgb(c), 1)
    elif kind == 'validation':
        lay[rice] = (*matplotlib.colors.to_rgb('#bccf9f'), .9)
    ax.imshow(lay, extent=e, interpolation='nearest', zorder=5)
    outline(ax, study)
    graticule(ax, g, fonts)
    q = max(1, g.nx//120)
    P = Placer(coarse(inreg.astype(float), q)*.7+coarse((~water & ~inreg).astype(float), q)*.15+coarse(rice.astype(float), q)*.6, mw/mh)
    if getattr(study, 'java', None):
        _, ib = P.corner(2.45/mw, 1.0/mh)
        inset(fig, ax, study, R, ib, fonts)
    sea_label(ax, study, R, P, fonts, water)
    neighbour_labels(ax, study, R, P, fonts, water)
    if kind == 'validation':
        sty = {'confirmed': dict(marker='o', mfc='#2d6a4f', mec='white', ms=4.2), 'partly': dict(marker='o', mfc='#d9a441', mec='white', ms=4.2),
               'rejected': dict(marker='X', mfc=ACC, mec='white', ms=5.5), 'none': dict(marker='o', mfc='none', mec='#8a857b', ms=3.6)}
        for lon, lat, st in R['validation'].get('points', []):
            X, Y = g.utm.fwd(lon, lat); ax.plot([X], [Y], linestyle='none', zorder=11, mew=.5, **sty[st])
    if kind == 'date1' and 'field_block' in R:
        fb = R['field_block']; r0, c0, kk = fb['row'], fb['col'], fb['size_px']
        X0, Y1 = g.x0+c0*g.res, g.y1-r0*g.res
        ax.add_patch(Rectangle((X0, Y1-kk*g.res), kk*g.res, kk*g.res, fc='none', ec=INK, lw=1.1, zorder=14,
                               path_effects=[pe.withStroke(linewidth=3, foreground=BG)]))
        P.boxes.append(((c0)/g.nx, 1-(r0+kk)/g.ny, (c0+kk)/g.nx, 1-r0/g.ny))
        callout(ax, P, ((c0+kk)/g.nx, 1-(r0+kk/2)/g.ny), '%d × %d km block shown\nin the time-series video' % (s['block_km'], s['block_km']),
                fonts['sans'], 9, mw, mh, dists=(.03, .05, .08))
    if kind in ('date1', 'date2'):
        k = 1 if kind == 'date1' else 2; d = A['d%d' % k]; m = rice & (d > -999)
        if m.sum() > 500:
            e10, e90 = np.percentile(d[m], 10), np.percentile(d[m], 90)
            for sel, lab in ((m & (d <= e10), 'Planted first\n%s' % part_month(np.median(d[m & (d <= e10)]))),
                             (m & (d >= e90), 'Planted last\n%s' % part_month(np.median(d[m & (d >= e90)])))):
                rc = densest(sel)
                if rc: callout(ax, P, rc_frac(study, *rc), lab, fonts['medium'], 10, mw, mh)
    if kind == 'seasons':
        for k_, lab in ((3, 'Three crops a year'), (1, 'Mostly one crop a year')):
            sel = A['nn'] == k_
            rc = densest(sel)
            if rc and sel.sum() > 300: callout(ax, P, rc_frac(study, *rc), lab, fonts['medium'], 10, mw, mh)
    t1, t2 = SHEET_TEXT[kind]
    t1 = t1.format(r=R['regency']); t2 = t2.format(n=fmt(len(R['validation'].get('points', []))))
    fig.text(xl/W, 1-.3/H, t1, fontproperties=F(fonts['title'], 25), color=INK, va='top')
    yt = foot-.42; yb = .66; c2 = xl+5.35; c3 = W-xl-2.75
    for xv in (c2-.3, c3-.3):
        fig.lines.append(matplotlib.lines.Line2D([xv/W]*2, [yb/H, yt/H], transform=fig.transFigure, color=LN, lw=.6))
    head_ = lambda x, y, t: fig.text(x/W, y/H, t, fontproperties=F(fonts['medium'], 9.5), color=INK, va='top')
    if kind in ('date1', 'date2'):
        k = 1 if kind == 'date1' else 2; lo, hi = date_range(study, k)
        head_(xl, yt, 'Transplanting date, with area transplanted every 12 days')
        legend_hist(fig, [xl/W, (yb+.36)/H, 4.6/W, (yt-yb-.68)/H], study, k, lo, hi, fonts)
        if k == 2:
            fig.patches.append(Rectangle(((xl+3.75)/W, (yt-.42)/H), .2/W, .14/H, transform=fig.transFigure, fc='#bdb7a9', ec='none'))
            fig.text((xl+4.02)/W, (yt-.35)/H, 'single crop', fontproperties=F(fonts['sans'], 9), color=INK, va='center')
    elif kind == 'seasons':
        head_(xl, yt, 'Rice crops in the 2024/25 crop year')
        tot = max(sum(R['seasons_ha'].get(k_, R['seasons_ha'].get(str(k_), 0)) for k_ in (1, 2, 3)), 1)
        y2 = yt-.36
        for k_, lab in ((1, 'One crop'), (2, 'Two crops'), (3, 'Three crops')):
            ha = R['seasons_ha'].get(k_, R['seasons_ha'].get(str(k_), 0))
            fig.patches.append(Rectangle((xl/W, (y2-.17)/H), .3/W, .17/H, transform=fig.transFigure, fc=SEASON_COLS[k_], ec=INK, lw=.4))
            fig.text((xl+.42)/W, (y2-.085)/H, lab, fontproperties=F(fonts['sans'], 9.5), color=INK, va='center')
            fig.patches.append(Rectangle(((xl+1.45)/W, (y2-.14)/H), 2.4*ha/tot/W, .11/H, transform=fig.transFigure, fc=SEASON_COLS[k_], ec='none'))
            fig.text((xl+1.45+2.4*ha/tot+.08)/W, (y2-.085)/H, '%s ha' % fmt(ha), fontproperties=F(fonts['sans'], 9), color=MU, va='center')
            y2 -= .28
    elif kind == 'validation':
        head_(xl, yt, 'Sample pixels')
        cnt = {}
        for p in R['validation'].get('points', []): cnt[p[2]] = cnt.get(p[2], 0)+1
        y2 = yt-.26
        for key, lab, st in (('confirmed', 'Confirmed: NDVI peaked after the radar planting date', dict(marker='o', mfc='#2d6a4f', mec='white', ms=6.5)),
                             ('partly', 'One season confirmed, another not', dict(marker='o', mfc='#d9a441', mec='white', ms=6.5)),
                             ('rejected', 'Not confirmed by Sentinel-2', dict(marker='X', mfc=ACC, mec='white', ms=8)),
                             ('none', 'No cloud-free Sentinel-2 view', dict(marker='o', mfc='none', mec='#8a857b', ms=6))):
            fig.lines.append(matplotlib.lines.Line2D([(xl+.1)/W], [(y2-.08)/H], transform=fig.transFigure, linestyle='none', mew=.8, **st))
            fig.text((xl+.3)/W, (y2-.08)/H, lab, fontproperties=F(fonts['sans'], 9.5), color=INK, va='center')
            fig.text((xl+4.6)/W, (y2-.08)/H, fmt(cnt.get(key, 0)), fontproperties=F(fonts['medium'], 9.5), color=INK, va='center', ha='right')
            y2 -= .21
    head_(c2, yt, 'Summary')
    fig.text(c2/W, (yt-.3)/H, textwrap.fill(sheet_note(R, study, kind), 84), fontproperties=F(fonts['sans'], 9.5), color=INK, va='top', linespacing=1.45)
    fig.lines.append(matplotlib.lines.Line2D([xl/W, (W-xl)/W], [(yb-.14)/H]*2, transform=fig.transFigure, color=LN, lw=.6))
    fig.text(xl/W, .3/H, 'Data: Copernicus Sentinel-1 GRD and Sentinel-2 L2A (ESA), ESA WorldCover 2021, FAO GAUL 2015. Rice detected at %d m inside '
             'cropland and herbaceous wetland. Validation: Sentinel-2 NDVI, official paddy-field baseline (ATR/BPN).' % s['res'],
             fontproperties=F(fonts['sans'], 7.8), color=MU, va='bottom')
    xr = e[1]-e[0]; scale = xr/(mw*.0254)
    seg = 10*1000/xr*mw; sx = W-xl-seg; sy = (yt+yb)/2-.12
    for i in range(5):
        fig.patches.append(Rectangle(((sx+i*seg/5)/W, sy/H), seg/5/W, .07/H, transform=fig.transFigure, fc=INK if i % 2 == 0 else BG, ec=INK, lw=.6))
    for t in (0, 5, 10):
        fig.text((sx+seg*t/10)/W, (sy+.13)/H, ('%d km' % t) if t == 10 else '%d' % t, fontproperties=F(fonts['sans'], 8.5), color=INK, ha='center', va='bottom')
    nxa = c3; nya = sy-.05
    fig.patches.append(Polygon([[nxa/W, (nya+.48)/H], [(nxa-.1)/W, nya/H], [nxa/W, (nya+.12)/H]], closed=True, transform=fig.transFigure, fc=INK, ec=INK, lw=.5))
    fig.patches.append(Polygon([[nxa/W, (nya+.48)/H], [(nxa+.1)/W, nya/H], [nxa/W, (nya+.12)/H]], closed=True, transform=fig.transFigure, fc=BG, ec=INK, lw=.5))
    fig.text(nxa/W, (nya+.52)/H, 'N', fontproperties=F(fonts['medium'], 9), color=INK, ha='center', va='bottom')
    fig.text((W-xl)/W, .3/H, 'Scale 1:%s at A3.  WGS 84 / UTM zone %d%s' % (fmt(round(scale/5000)*5000), g.utm.zone, 'S' if g.utm.south else 'N'),
             fontproperties=F(fonts['sans'], 7.8), color=MU, ha='right', va='bottom')
    fig.savefig(path, facecolor=BG); plt.close(fig)
    import gc; gc.collect()


def signature_axes(ax, ex, s, fonts, small=False):
    fs = 7.5 if small else 8.5
    off = np.array(ex['vh_off']); vh = np.array([np.nan if v is None else v for v in ex['vh']])
    ax.axhspan(-30, s['flood_db'], color='#dde8ee', zorder=0)
    ax.plot(off, vh, color=INK, lw=1.3, zorder=3)
    for d in ex['dates']:
        i = int(np.argmin(np.abs(off-d))); ax.plot(off[i], vh[i], 'o', ms=6 if small else 7.5, mfc=ORANGE, mec=INK, mew=.6, zorder=4)
    if not small and ex.get('ndvi'):
        a2 = ax.twinx(); nv = np.array([np.nan if v is None else v for v in ex['ndvi']]); no = np.array(ex['ndvi_off'])
        ok = np.isfinite(nv); xo, yo = no[ok], nv[ok]
        brk = np.where(np.diff(xo) > 35)[0]+1
        for seg_x, seg_y in zip(np.split(xo, brk), np.split(yo, brk)): a2.plot(seg_x, seg_y, '-', color='#5f9e3b', lw=.9, alpha=.85)
        a2.plot(xo, yo, 'o', color='#5f9e3b', ms=3)
        a2.set_ylim(-.3, 1); a2.set_ylabel('Sentinel-2 NDVI', color='#5f9e3b', fontproperties=F(fonts['sans'], 8.5)); a2.tick_params(colors='#5f9e3b', labelsize=8)
        a2.spines['top'].set_visible(False)
    ticks = [int((np.datetime64(x)-REF).astype(int)) for x in ('2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01', '2026-01-01')]
    ax.set_xticks(ticks); ax.set_xticklabels(['Oct', 'Jan 2025', 'Apr', 'Jul', 'Oct', 'Jan 2026'], fontproperties=F(fonts['sans'], fs))
    ax.set_ylim(-27, -10); ax.set_xlim(off.min(), off.max())
    ax.tick_params(labelsize=fs, colors=MU)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(LN)


def signatures(R, path, fonts=None, dpi=200):
    fonts = fonts or load_fonts()
    ex = R.get('examples', []); s = R['setup']
    if not ex: return
    fig, axs = plt.subplots(len(ex), 1, figsize=(11, 2.6*len(ex)+1.1), dpi=dpi, sharex=True); fig.patch.set_facecolor(BG)
    axs = np.atleast_1d(axs)
    fig.text(.06, .985, 'Radar and Optical Record of Three Rice Fields, %s' % R['regency'], fontproperties=F(fonts['title'], 15), color=INK, va='top')
    fig.text(.06, .985-.32/fig.get_size_inches()[1], 'Black line: Sentinel-1 VH backscatter. Blue band: flooded field. Orange dots: detected transplanting. Green: Sentinel-2 NDVI on clear days.',
             fontproperties=F(fonts['sans'], 9.5), color=MU, va='top')
    for ax, x in zip(axs, ex):
        ax.set_facecolor(BG); signature_axes(ax, x, s, fonts)
        ax.set_ylabel('VH (dB)', fontproperties=F(fonts['sans'], 8.5), color=MU)
        ax.set_title('%d crop%s (%.4f° S, %.4f° E): planted %s' % (x['n'], '' if x['n'] == 1 else 's', -x['lat'], x['lon'],
                     ', '.join(day_label(d) for d in x['dates'])), loc='left', fontproperties=F(fonts['medium'], 9.5), color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1-.75/fig.get_size_inches()[1])); fig.savefig(path, facecolor=BG); plt.close(fig)


def validation_chart(R, path, fonts=None, dpi=200):
    fonts = fonts or load_fonts()
    V = R['validation']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), dpi=dpi); fig.patch.set_facecolor(BG)
    ax = axs[0]; v = V['s2']
    ax.bar([0, 1, 2], [v['confirmed'], v['rejected'], v['no_clear_view']], color=['#2d6a4f', ACC, '#bdb7a9'], width=.6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['confirmed', 'not confirmed', 'no clear view'], fontproperties=F(fonts['sans'], 9))
    ax.set_title('Sentinel-2 check of %s detected seasons\n%.0f%% confirmed%s' % (fmt(v['seasons_checked']+v['no_clear_view']), v['rate']*100,
                 ('; radar caught %.0f%% of NDVI-visible crops' % (v['recall']*100)) if v.get('ndvi_crops') else ''),
                 loc='left', fontproperties=F(fonts['medium'], 10))
    ax = axs[1]; L = V['lbs']
    names = ['Mapped rice area', 'Paddy-field baseline (ATR/BPN)']; vals = [L['mapped_ha'], L['reference_ha']]; cols = ['#7ea94a', '#8f877a']
    if 'bps' in V:
        names += ['Mapped harvest 2025', 'BPS harvested area 2025']; vals += [V['bps']['mapped_ha'], V['bps']['reference_ha']]; cols += ['#2d6a4f', '#5d5850']
    ax.barh(range(len(vals))[::-1], vals, color=cols, height=.6)
    for i, val in zip(range(len(vals))[::-1], vals): ax.text(val, i, '  %s ha' % fmt(val), va='center', fontproperties=F(fonts['sans'], 9))
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(names, fontproperties=F(fonts['sans'], 9)); ax.set_xticks([])
    ax.set_xlim(0, max(vals)*1.35)
    t = 'Rice area against official statistics\nextent %+.0f%% vs paddy-field baseline' % L['diff_pct']
    if 'bps' in V: t += ', harvest %+.0f%% vs BPS' % V['bps']['diff_pct']
    ax.set_title(t, loc='left', fontproperties=F(fonts['medium'], 10))
    ax = axs[2]
    pb = {int(k): v for k, v in R['planting_by_bin'].items()}; ks = sorted(pb)
    ax.bar(ks, [pb[k] for k in ks], width=R['setup']['bin_days']*.85, color='#2f7f9e')
    tk = [int((np.datetime64(x)-REF).astype(int)) for x in ('2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01')]
    ax.set_xticks(tk); ax.set_xticklabels(['Oct 2024', 'Jan 2025', 'Apr', 'Jul'], fontproperties=F(fonts['sans'], 9))
    ax.set_title('Area transplanted every 12 days (ha)\nwet-season and dry-season planting waves', loc='left', fontproperties=F(fonts['medium'], 10))
    for a in axs:
        a.set_facecolor(BG)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
        a.tick_params(colors=MU)
    fig.tight_layout(); fig.savefig(path, facecolor=BG); plt.close(fig)


def _ffmpeg(path, W, H, fps, crf=18):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(fps),
                             '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                            stdin=subprocess.PIPE)


def _web(path, web_path):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-vf', 'scale=1280:-2', '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', web_path], check=True)


def _write(proc, prev, cur, hold, fade):
    if prev is not None:
        for j in range(1, fade+1): proc.stdin.write((prev*(1-j/(fade+1))+cur*(j/(fade+1))).astype(np.uint8).tobytes())
    for _ in range(hold): proc.stdin.write(cur.tobytes())


def crop_age(study, t, dd=None, rice=None):
    A = study.A; L = study.s.field_days
    dd = dd if dd is not None else [A['d%d' % k] for k in (1, 2, 3)]
    rice = rice if rice is not None else A['nn'] >= 1
    age = np.full(rice.shape, np.nan)
    for d in dd:
        a = t-d; ok = rice & (d > -999) & (a >= 0) & (a < L)
        age[ok] = np.where(np.isnan(age[ok]), a[ok], np.minimum(age[ok], a[ok]))
    return age


def wave_video(R, study, path, fonts=None, fps=24, hold=10, fade=6, web_path=None):
    fonts = fonts or load_fonts()
    import copy
    s = R['setup']; Wp, Hp = 1920, 1080; W, H = 19.2, 10.8
    g0 = study.g; mh = H-.9; mw = W-.45-.5-3.4-.4
    n = min(g0.nx, int(round(g0.ny*mw/mh)))
    cols = np.where((study.A['inreg'] == 1).any(0))[0]
    cc = int((cols.min()+cols.max())/2) if cols.size else g0.nx//2
    c0 = int(np.clip(cc-n//2, 0, g0.nx-n))
    study = copy.copy(study)
    study.A = {k: (v[:, c0:c0+n] if getattr(v, 'ndim', 0) == 2 else v) for k, v in study.A.items()}
    study.g = Grid(g0.utm, g0.x0+c0*g0.res, g0.y1, g0.res, n, g0.ny)
    mw = mh*n/g0.ny
    A, g = study.A, study.g
    idx = [i for i in range(len(study.bins)) if study.doff[i] >= study.doff[study.target[0]]-24]
    px = s['res']**2/1e4; L = s['field_days']
    dd = [A['d%d' % k] for k in (1, 2, 3)]; rice = A['nn'] >= 1
    area = np.array([np.isfinite(crop_age(study, study.doff[i], dd, rice)).sum()*px for i in idx])
    step = max(1, int(np.ceil(max(g.nx, g.ny)/1700)))
    if step > 1:
        study = copy.copy(study)
        study.A = {k: (v[::step, ::step] if getattr(v, 'ndim', 0) == 2 else v) for k, v in A.items()}
        ny_, nx_ = study.A['inreg'].shape
        study.g = Grid(g.utm, g.x0, g.y1, g.res*step, nx_, ny_)
        A, g = study.A, study.g
        dd = [A['d%d' % k] for k in (1, 2, 3)]; rice = A['nn'] >= 1
    base, water = base_rgb(study); e = g.extent
    base8 = (base*255).astype(np.uint8); del base
    fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([.45/W, .45/H, mw/W, mh/H])
    ax.imshow(base8, extent=e, interpolation='bilinear', zorder=1)
    lay = np.zeros(rice.shape+(4,), np.uint8)
    im = ax.imshow(lay, extent=e, interpolation='nearest', zorder=5)
    outline(ax, study, .8)
    ax.set_xlim(e[:2]); ax.set_ylim(e[2:]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(INK); sp.set_linewidth(.8)
    P = Placer(coarse((A['inreg'] == 1).astype(float), max(1, g.nx//120)), mw/mh)
    sea_label(ax, study, R, P, fonts, water); neighbour_labels(ax, study, R, P, fonts, water)
    rx = (.45+mw+.5)/W; rw = 1-rx-.4/W
    fig.text(rx, 1-.45/H, 'Rice Planting Wave,\n%s Regency' % R['regency'], fontproperties=F(fonts['title'], 24), color=INK, va='top', linespacing=.95)
    t_day = fig.text(rx, 1-1.75/H, '', fontproperties=F(fonts['title'], 38), color=INK, va='top')
    t_year = fig.text(rx, 1-2.45/H, '', fontproperties=F(fonts['serif'], 22), color=MU, va='top')
    t_area = fig.text(rx, 1-3.35/H, '', fontproperties=F(fonts['medium'], 15), color=INK, va='top')
    sa = fig.add_axes([rx, 1-5.05/H, rw, 1.25/H]); sa.set_facecolor('none')
    sa.fill_between(range(len(idx)), area, color='#9cbc7a', alpha=.45, lw=0); sa.plot(range(len(idx)), area, color='#4e8a3a', lw=1.3)
    dot, = sa.plot([0], [area[0]], 'o', ms=6, mfc=INK, mec=BG, mew=1.2)
    sa.set_xlim(0, len(idx)-1); sa.set_ylim(0, max(area.max(), 1)*1.15); sa.set_yticks([])
    tk = [j for j, ii in enumerate(idx) if str(study.bins[ii])[5:7] in ('01', '04', '07', '10') and int(str(study.bins[ii])[8:10]) <= 12]
    sa.set_xticks(tk); sa.set_xticklabels([MONTHS[int(str(study.bins[idx[j]])[5:7])-1] for j in tk], fontproperties=F(fonts['sans'], 10), color=MU)
    for sp in ('top', 'right', 'left'): sa.spines[sp].set_visible(False)
    sa.spines['bottom'].set_color(LN); sa.tick_params(length=0)
    cb = fig.add_axes([rx, 1-6.05/H, rw, .16/H]); cb.imshow(np.linspace(0, 1, 256)[None], cmap=AGE_CMAP, aspect='auto'); cb.set_axis_off()
    for txt, xa, ha in (('just planted', 0, 'left'), ('growing', .55, 'center'), ('ripening', 1, 'right')):
        fig.text(rx+rw*xa, 1-6.12/H, txt, fontproperties=F(fonts['sans'], 11), color=INK, ha=ha, va='top')
    fig.text(rx, 1-6.7/H, textwrap.fill('A field lights up orange on the date Sentinel-1 radar saw it flooded and transplanted, turns green '
                                        'as the crop grows and gold about %d days later, near harvest.' % L, 40),
             fontproperties=F(fonts['sans'], 11), color='#45494b', va='top', linespacing=1.45)
    fig.text(rx, .45/H, 'Contains modified Copernicus\nSentinel-1 data 2024–2026', fontproperties=F(fonts['sans'], 9.5), color=MU, va='bottom', linespacing=1.4)
    proc = _ffmpeg(path, Wp, Hp, fps); prev = None
    for n_, i in enumerate(idx):
        age = crop_age(study, study.doff[i], dd, rice); ok = np.isfinite(age)
        lay[:] = 0; lay[ok] = AGE_CMAP(age[ok]/L, bytes=True); im.set_data(lay)
        tt = str(study.bins[i])
        t_day.set_text('%d %s' % (int(tt[8:10]), MONTHS_LONG[int(tt[5:7])-1])); t_year.set_text(tt[:4])
        t_area.set_text('%s ha under rice' % fmt(area[n_])); dot.set_data([n_], [area[n_]])
        fig.canvas.draw(); cur = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        _write(proc, prev, cur, hold if n_ < len(idx)-1 else fps*3, fade); prev = cur.astype(np.float32)
    plt.close(fig)
    proc.stdin.close(); proc.wait()
    if web_path: _web(path, web_path)


def radar_rgb(v, overlay=None):
    x = np.clip((v/100-RADAR_LO)/(RADAR_HI-RADAR_LO), 0, 1)
    rgb = RADAR_CMAP(x)[..., :3]
    rgb[v < -9000] = matplotlib.colors.to_rgb('#d9d4c8')
    if overlay is not None and overlay.any():
        rgb[overlay] = rgb[overlay]*.25+np.array(matplotlib.colors.to_rgb(ORANGE))*.75
    return rgb


def site_overlay(study, i, S=None):
    S = S or study.site
    if i not in study.target: return None
    m = np.zeros(S['rice'].shape, bool)
    for d in S['dates']: m |= (d == study.doff[i]) & S['rice']
    return m


def _panel_marks(a, g, fonts, km=1, north=False, scale=False, label=None):
    n = g.nx
    if scale:
        px = km*1000/g.res; x0, y0 = n*.05, n*.94
        a.add_patch(Rectangle((x0, y0-n*.006), px, n*.012, fc='white', ec=INK, lw=.6, zorder=6))
        a.text(x0+px/2, y0-n*.02, '%d km' % km, color='white', fontproperties=F(fonts['medium'], 10), ha='center', va='bottom', zorder=6,
               path_effects=[pe.withStroke(linewidth=2, foreground=INK)])
    if north:
        x, y = n*.95, n*.05
        a.add_patch(Polygon([[x, y], [x-n*.016, y+n*.045], [x, y+n*.034], [x+n*.016, y+n*.045]], closed=True, fc='white', ec=INK, lw=.7, zorder=6))
        a.text(x, y+n*.052, 'N', color='white', fontproperties=F(fonts['medium'], 10), ha='center', va='top', zorder=6,
               path_effects=[pe.withStroke(linewidth=2, foreground=INK)])
    if label:
        a.text(n*.03, n*.03, label, color='white', fontproperties=F(fonts['medium'], 11), ha='left', va='top', zorder=6,
               path_effects=[pe.withStroke(linewidth=2.4, foreground=INK)])


def series_axes(ax, study, fonts, cur=None, picks=(), fs=10, S=None):
    S = S or study.site; st = S['stats']; T = len(study.bins); x = np.arange(T)
    w = st['water']
    ax.fill_between(x, 0, np.nan_to_num(w), color=WATER, alpha=.18, lw=0)
    ax.plot(x, w, color=WATER, lw=1.6)
    pl = np.nan_to_num(st['planted']); ax.bar(x, pl, width=.62, color=ORANGE, lw=0)
    gr = st['green']; ok = np.isfinite(gr)
    ax.plot(x, gr, '-', color=GREEN, lw=1, alpha=.7); ax.plot(x[ok], gr[ok], 'o', color=GREEN, ms=3.8 if fs < 11 else 5, mec=BG, mew=.7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([], [], color=WATER, lw=1.8), Patch(fc=ORANGE), Line2D([], [], color=GREEN, lw=1, marker='o', ms=5, mec=BG)]
    ax.legend(hd, ['fields under water, Sentinel-1 radar', 'transplanted in the 12-day period', 'green fields on cloud-free days, Sentinel-2'],
              loc='lower left', bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, prop=F(fonts['sans'], fs), handlelength=1.5,
              columnspacing=2.4, borderaxespad=0, handletextpad=.6, labelcolor=INK)
    if cur is not None:
        ax.axvline(cur, color=INK, lw=1)
    for p in picks:
        ax.axvline(p, color=INK, lw=.6, ls=(0, (2, 2)), alpha=.7)
    seen = set(); ticks = []
    for i, b in enumerate(study.bins):
        k = str(b)[:7]
        if str(b)[5:7] in ('01', '04', '07', '10') and k not in seen: seen.add(k); ticks.append(i)
    ax.set_xticks(ticks)
    ax.set_xticklabels([('%s %s' % (MONTHS[int(str(study.bins[i])[5:7])-1], str(study.bins[i])[:4])) if str(study.bins[i])[5:7] == '01'
                        else MONTHS[int(str(study.bins[i])[5:7])-1] for i in ticks], fontproperties=F(fonts['sans'], fs), color=MU)
    ax.set_ylim(0, 105); ax.set_xlim(-.6, T-.4)
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(['0', '50', '100%'], fontproperties=F(fonts['sans'], fs-1), color=MU)
    ax.grid(axis='y', color=LN, lw=.5); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(MU); ax.spines['bottom'].set_linewidth(.6); ax.tick_params(length=0)


def field_frame(R, study, i, fonts, W=19.2, H=10.8):
    S = study.site; g = S['grid']; st = S['stats']; b = study.bins[i]
    fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor(BG)
    p = 7.3; x1 = .45; x2 = x1+p+.2; top = H-.85; y0 = top-p
    a1 = fig.add_axes([x1/W, y0/H, p/W, p/H]); a2 = fig.add_axes([x2/W, y0/H, p/W, p/H])
    if st['s2'][i]: a1.imshow(S['rgb'][i], interpolation='lanczos')
    else:
        a1.imshow(np.full(S['rice'].shape+(3,), .9))
        a1.text(g.nx/2, g.ny/2, 'no Sentinel-2 image in this period', ha='center', va='center', fontproperties=F(fonts['italic'], 15), color=MU)
    a2.imshow(radar_rgb(S['vh'][i], site_overlay(study, i)), interpolation='lanczos')
    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(INK); sp.set_linewidth(.8)
    _panel_marks(a1, g, fonts, north=True); _panel_marks(a2, g, fonts, scale=True)
    ly = (top+.12)/H
    fig.text(x1/W, ly, 'Sentinel-2 optical', fontproperties=F(fonts['medium'], 12), color=INK, va='bottom')
    cl = st['cloud'][i]
    if st['s2'][i] and np.isfinite(cl):
        fig.text((x1+p)/W, ly, '%d%% cloud' % round(cl), fontproperties=F(fonts['sans'], 11), color=MU, va='bottom', ha='right')
    fig.text(x2/W, ly, 'Sentinel-1 radar, VH polarisation', fontproperties=F(fonts['medium'], 12), color=INK, va='bottom')
    fig.text((x2+p)/W, ly, '%d × %d km' % (S['km'], S['km']), fontproperties=F(fonts['sans'], 11), color=MU, va='bottom', ha='right')
    rx = (x2+p+.5)/W; rw = 1-rx-.45/W
    y = (top+.38)/H
    fig.text(rx, y, 'Rice Fields of\n%s' % R['regency'], fontproperties=F(fonts['title'], 22), color=INK, va='top', linespacing=.95)
    tt = str(b)
    fig.text(rx, y-1.3/H, '%d %s' % (int(tt[8:10]), MONTHS_LONG[int(tt[5:7])-1]), fontproperties=F(fonts['title'], 34), color=INK, va='top')
    fig.text(rx, y-1.93/H, tt[:4], fontproperties=F(fonts['serif'], 20), color=MU, va='top')
    yy = y-2.85/H
    cb = fig.add_axes([rx, yy-.15/H, rw, .15/H]); cb.imshow(np.linspace(0, 1, 256)[None], cmap=RADAR_CMAP, aspect='auto'); cb.set_axis_off()
    yy -= .22/H
    for t_, xa, ha in (('flooded', 0, 'left'), ('bare soil', .48, 'center'), ('growing rice', 1, 'right')):
        fig.text(rx+rw*xa, yy, t_, fontproperties=F(fonts['sans'], 10.5), color=INK, ha=ha, va='top')
    yy -= .5/H
    fig.patches.append(Rectangle((rx, yy-.15/H), .3/W, .15/H, transform=fig.transFigure, fc=ORANGE, ec='none'))
    fig.text(rx+.42/W, yy-.075/H, 'transplanted in this 12-day period', fontproperties=F(fonts['sans'], 10.5), color=INK, va='center')
    yy -= .55/H
    fig.text(rx, yy, textwrap.fill('Clouds hide these fields for weeks in the rainy season. Radar is taken through cloud, '
                                   'so every 12 days shows where water stood and where rice was growing.', 36),
             fontproperties=F(fonts['sans'], 10.5), color='#45494b', va='top', linespacing=1.45)
    cy0, ch = .62, 1.2
    ax = fig.add_axes([x1/W, cy0/H, (x2+p-x1)/W, ch/H]); ax.set_facecolor('none')
    series_axes(ax, study, fonts, cur=i, fs=10.5)
    w = st['water'][i]
    if np.isfinite(w):
        fig.text(rx, (cy0+ch+.32)/H, '%d%%' % round(w), fontproperties=F(fonts['title'], 26), color=WATER, va='top')
        fig.text(rx, (cy0+ch-.12)/H, 'of the rice fields in this\nblock stood under water', fontproperties=F(fonts['sans'], 10.5), color='#45494b', va='top', linespacing=1.35)
    fig.text(rx, (cy0-.27)/H, 'Contains modified Copernicus\nSentinel data 2024–2026', fontproperties=F(fonts['sans'], 9), color=MU, va='bottom', linespacing=1.35)
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return img


def field_timelapse(R, study, path, fonts=None, fps=24, hold=14, fade=6, web_path=None):
    fonts = fonts or load_fonts()
    proc = _ffmpeg(path, 1920, 1080, fps)
    st = study.site['stats']; prev = None
    idx = [i for i in range(len(study.bins)) if st['s1'][i]]
    for k, i in enumerate(idx):
        cur = field_frame(R, study, i, fonts)
        if k % 5 == 4: import gc; gc.collect()
        _write(proc, prev, cur, hold if k < len(idx)-1 else fps*3, fade); prev = cur.astype(np.float32)
    proc.stdin.close(); proc.wait()
    if web_path: _web(path, web_path)


def strip_picks(study, n=6):
    st = study.site['stats']; t = study.target
    first = t[0]; last = min(t[-1]+3, len(study.bins)-1)
    out = []
    for w in np.linspace(first, last, n).round().astype(int):
        cand = [j for j in range(w-1, w+2) if 0 <= j < len(study.bins) and st['s1'][j] and j not in out]
        if not cand: continue
        cl = [st['cloud'][j] if st['s2'][j] and np.isfinite(st['cloud'][j]) else 101 for j in cand]
        out.append(cand[int(np.argmin(cl))])
    return out


def field_strip(R, study, path, fonts=None, dpi=200):
    fonts = fonts or load_fonts()
    S = study.site; g = S['grid']; st = S['stats']
    picks = strip_picks(study); n = len(picks)
    W, H = 18.0, 9.75
    fig = plt.figure(figsize=(W, H), dpi=dpi); fig.patch.set_facecolor(BG)
    ml, mr, gap = .75, .35, .1; pw = (W-ml-mr-(n-1)*gap)/n
    yt = H-1.3
    fig.text(ml/W, 1-.38/H, 'One Crop Year in the Rice Fields of %s' % R['regency'], fontproperties=F(fonts['title'], 26), color=INK, va='top')
    for r_, lab in ((0, 'Sentinel-2 optical'), (1, 'Sentinel-1 radar')):
        fig.text((ml-.22)/W, (yt-pw/2-r_*(pw+gap))/H, lab, fontproperties=F(fonts['medium'], 11), color=INK, rotation=90, ha='center', va='center')
    for k, i in enumerate(picks):
        x = ml+k*(pw+gap)
        a1 = fig.add_axes([x/W, (yt-pw)/H, pw/W, pw/H]); a2 = fig.add_axes([x/W, (yt-2*pw-gap)/H, pw/W, pw/H])
        if st['s2'][i]: a1.imshow(S['rgb'][i], interpolation='lanczos')
        else: a1.imshow(np.full(S['rice'].shape+(3,), .87))
        a2.imshow(radar_rgb(S['vh'][i], site_overlay(study, i)), interpolation='lanczos')
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color(INK); sp.set_linewidth(.7)
        fig.text(x/W, (yt+.1)/H, bin_label(study.bins[i]), fontproperties=F(fonts['title'], 14), color=INK, va='bottom')
        cl = st['cloud'][i]
        if st['s2'][i] and np.isfinite(cl):
            fig.text((x+pw)/W, (yt+.1)/H, '%d%% cloud' % round(cl), fontproperties=F(fonts['sans'], 9.5), color=MU, va='bottom', ha='right')
        w = st['water'][i]
        if np.isfinite(w):
            fig.text(x/W, (yt-2*pw-gap-.12)/H, '%d%% of fields under water' % round(w), fontproperties=F(fonts['sans'], 10), color=WATER, va='top')
        if k == 0: _panel_marks(a1, g, fonts, north=True); _panel_marks(a2, g, fonts, scale=True)
    ly = yt-2*pw-gap-.62; bx1 = W-mr-1.0; bx0 = bx1-2.2; by = ly-.2
    cb = fig.add_axes([bx0/W, by/H, (bx1-bx0)/W, .11/H]); cb.imshow(np.linspace(0, 1, 256)[None], cmap=RADAR_CMAP, aspect='auto'); cb.set_axis_off()
    fig.text((bx0-.08)/W, (by+.055)/H, 'radar:  flooded', fontproperties=F(fonts['sans'], 9.5), color=INK, ha='right', va='center')
    fig.text((bx1+.08)/W, (by+.055)/H, 'growing rice', fontproperties=F(fonts['sans'], 9.5), color=INK, ha='left', va='center')
    ax = fig.add_axes([ml/W, .55/H, (W-ml-mr)/W, (ly-.32-.55)/H]); ax.set_facecolor('none')
    series_axes(ax, study, fonts, picks=picks, fs=9.5)
    fig.text(1-mr/W, .12/H, 'Contains modified Copernicus Sentinel data 2024–2026',
             fontproperties=F(fonts['sans'], 8.5), color=MU, ha='right', va='bottom')
    fig.savefig(path, dpi=dpi, facecolor=BG); plt.close(fig)




def regency_frame(R, study, i, fonts, W=19.2, H=10.8):
    S = study.regts; g = S['grid']; st = S['stats']; e = g.extent; inreg = S['inreg']
    fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor(BG)
    mx = .45; gap = .3; pw = (W-2*mx-gap)/2; ph = pw*g.ny/g.nx; top = H-1.25
    paper = np.array(matplotlib.colors.to_rgb(BG))
    dim = lambda im: np.where(inreg[..., None], im, im*.45+paper*.55)
    a1 = fig.add_axes([mx/W, (top-ph)/H, pw/W, ph/H]); a2 = fig.add_axes([(mx+pw+gap)/W, (top-ph)/H, pw/W, ph/H])
    if st['s2'][i]: a1.imshow(dim(S['rgb'][i]/255.), extent=e, interpolation='lanczos')
    else:
        a1.imshow(np.full(inreg.shape+(3,), .9), extent=e)
        a1.text(.5, .5, 'no Sentinel-2 image in this period', transform=a1.transAxes, ha='center', va='center', fontproperties=F(fonts['italic'], 15), color=MU)
    a2.imshow(dim(radar_rgb(S['vh'][i], site_overlay(study, i, S))), extent=e, interpolation='lanczos')
    for a in (a1, a2):
        outline(a, study, .7)
        a.set_xlim(e[:2]); a.set_ylim(e[2:]); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(INK); sp.set_linewidth(.8)
    km = 10; x0 = e[0]+(e[1]-e[0])*.03; y0 = e[2]+(e[3]-e[2])*.06
    a2.add_patch(Rectangle((x0, y0), km*1000, (e[3]-e[2])*.012, fc='white', ec=INK, lw=.6, zorder=12))
    a2.text(x0+km*500, y0+(e[3]-e[2])*.03, '%d km' % km, ha='center', va='bottom', color='white', fontproperties=F(fonts['medium'], 10), zorder=12,
            path_effects=[pe.withStroke(linewidth=2, foreground=INK)])
    a1.annotate('N', xy=(.965, .93), xytext=(.965, .8), xycoords='axes fraction', ha='center', va='top', color='white', fontproperties=F(fonts['medium'], 10),
                arrowprops=dict(arrowstyle='-|>', color='white', lw=1.4), path_effects=[pe.withStroke(linewidth=2, foreground=INK)])
    ly = (top+.12)/H
    fig.text(mx/W, ly, 'Sentinel-2 optical', fontproperties=F(fonts['medium'], 12), color=INK, va='bottom')
    cl = st['cloud'][i]
    if st['s2'][i] and np.isfinite(cl):
        fig.text((mx+pw)/W, ly, '%d%% of the regency under cloud' % round(cl), fontproperties=F(fonts['sans'], 11), color=MU, va='bottom', ha='right')
    fig.text((mx+pw+gap)/W, ly, 'Sentinel-1 radar, VH polarisation', fontproperties=F(fonts['medium'], 12), color=INK, va='bottom')
    w = st['water'][i]
    if np.isfinite(w):
        fig.text((W-mx)/W, ly, '%d%% of rice fields under water' % round(w), fontproperties=F(fonts['sans'], 11), color=WATER, va='bottom', ha='right')
    fig.text(mx/W, 1-.35/H, 'Rice Fields of %s Regency' % R['regency'], fontproperties=F(fonts['title'], 24), color=INK, va='top')
    tt = str(study.bins[i])
    fig.text((W-mx)/W, 1-.3/H, '%d %s %s' % (int(tt[8:10]), MONTHS_LONG[int(tt[5:7])-1], tt[:4]), fontproperties=F(fonts['title'], 28), color=INK, va='top', ha='right')
    yl = top-ph-.42
    cb = fig.add_axes([(W-mx-2.2-1.05)/W, (yl-.06)/H, 2.2/W, .12/H]); cb.imshow(np.linspace(0, 1, 256)[None], cmap=RADAR_CMAP, aspect='auto'); cb.set_axis_off()
    fig.text((W-mx-2.2-1.13)/W, yl/H, 'radar:  flooded', fontproperties=F(fonts['sans'], 10), color=INK, ha='right', va='center')
    fig.text((W-mx-1.0)/W, yl/H, 'growing rice', fontproperties=F(fonts['sans'], 10), color=INK, ha='left', va='center')
    ax = fig.add_axes([mx/W, .75/H, (W-2*mx)/W, (yl-.35-.75)/H]); ax.set_facecolor('none')
    series_axes(ax, study, fonts, cur=i, fs=10.5, S=S)
    fig.text((W-mx)/W, .2/H, 'Contains modified Copernicus Sentinel data 2024–2026', fontproperties=F(fonts['sans'], 9), color=MU, ha='right', va='bottom')
    fig.canvas.draw(); img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return img


def regency_timelapse(R, study, path, fonts=None, fps=24, hold=14, fade=6, web_path=None):
    import gc
    fonts = fonts or load_fonts()
    proc = _ffmpeg(path, 1920, 1080, fps)
    st = study.regts['stats']; prev = None
    idx = [i for i in range(len(study.bins)) if st['s1'][i]]
    for k, i in enumerate(idx):
        cur = regency_frame(R, study, i, fonts)
        if k % 5 == 4: gc.collect()
        _write(proc, prev, cur, hold if k < len(idx)-1 else fps*3, fade); prev = cur.astype(np.float32)
    proc.stdin.close(); proc.wait()
    if web_path: _web(path, web_path)
def report_en(R):
    V = R['validation']; s = R['setup']; v = V['s2']; S1 = R.get('s1', {}); L = []
    sh = lambda k: R['seasons_ha'][k] if k in R['seasons_ha'] else R['seasons_ha'][str(k)]
    L += ['RICE PLANTING CALENDAR FROM SENTINEL-1 RADAR', '%s Regency, %s' % (R['regency'], {'Jawa Barat': 'West Java', 'Jawa Tengah': 'Central Java', 'Jawa Timur': 'East Java'}.get(R['province'], R['province'])), '',
          '1. METHOD',
          '- Sentinel-1 GRD, VH polarisation, %s; median per %d days, %s to %s; 20 m spatial and 3-period temporal smoothing.'
          % (S1.get('primary', R['orbit_pass']), s['bin_days'], s['start'], s['end'])]
    if S1.get('fill_tracks'):
        L.append('- Missing periods filled from %s, corrected for the mean backscatter difference between tracks.' % ', '.join(S1['fill_tracks']))
    if S1.get('periods_with_data') is not None:
        L.append('- Periods with radar data: %d of %d; gaps of up to 3 periods interpolated.' % (S1['periods_with_data'], S1['periods']))
    L += ['- A season is a flooded field (VH <= %.1f dB, local minimum) followed by a VH rise of >= %.1f dB to >= %.1f dB within %d-%d days.'
          % (s['flood_db'], s['rise_db'], s['peak_db'], s['rise_from']*s['bin_days'], s['rise_to']*s['bin_days']),
          '- Seasons counted when transplanted between %s and %s; a season more than %d days after the first is treated as next year\'s repeat.'
          % (s['window'][0], s['window'][1], s['repeat_days']),
          '- Candidate land: ESA WorldCover 2021 cropland, grassland and herbaceous wetland inside the regency boundary (FAO GAUL).', '',
          '2. RESULTS',
          '- Rice area detected: %s ha.' % fmt(R['rice_ha']),
          '- One crop %s ha, two crops %s ha, three crops %s ha; cropping index %.2f.' % (fmt(sh(1)), fmt(sh(2)), fmt(sh(3)), R['cropping_index']),
          '- Harvested area 2025 (transplanting date + %d days): %s ha.' % (s['field_days'], fmt(R['harvest_2025_ha'])), '',
          '3. VALIDATION',
          '- Sentinel-2, %d random rice pixels: %s of %s detected seasons confirmed (%.0f%%), median NDVI peak %s days after transplanting; %s seasons without a clear view.'
          % (v['points'], fmt(v['confirmed']), fmt(v['confirmed']+v['rejected']), v['rate']*100, '%.0f' % v['median_peak_lag_days'] if v['median_peak_lag_days'] else 'n/a', fmt(v['no_clear_view'])),
          '- Crops visible in Sentinel-2 NDVI caught by the radar: %s of %s (%.0f%%).' % (fmt(v.get('ndvi_crops_caught', 0)), fmt(v.get('ndvi_crops', 0)), v.get('recall', 0)*100),
          '- Official paddy-field baseline (ATR/BPN) %s ha: difference %+.1f%%.' % (fmt(V['lbs']['reference_ha']), V['lbs']['diff_pct'])]
    if 'bps' in V:
        L.append('- BPS harvested area %s ha: difference %+.1f%%.' % (fmt(V['bps']['reference_ha']), V['bps']['diff_pct']))
    else:
        L.append('- BPS harvested area not set (Setup.bps_harvest_ha).')
    L.append('- Quality control: ' + ('PASSED.' if not R['qc'] else 'WARNINGS:'))
    for q in R['qc']: L.append('    ! ' + q)
    L += ['', '4. LIMITATIONS',
          '- Plots smaller than about 0.1 ha and fields next to settlements are mixed in 30 m pixels.',
          '- Rice sown without flooding (dry direct seeding) can be missed; fish ponds can be confused with rice when they carry seasonal vegetation.',
          '- Harvest month is an estimate (transplanting date + %d days), not an observed harvest.' % s['field_days']]
    return '\n'.join(L)

fonts = load_fonts()
out = 'rice_calendar'

setup = Setup(
    regency='Indramayu',
    province='Jawa Barat',
    lbs_ha=126088,
    bps_harvest_ha=None,
)


study = RiceCalendar(ee, setup)
R = study.run()
study.save(R, out)
with open(f'{out}/07_report.txt', 'w', encoding='utf-8') as fh: fh.write(report_en(R))
print(report_en(R))


import gc
study = RiceCalendar(ee, setup)
R = study.restore(out)
for name, kind in (('01_map_planting_calendar', 'date1'), ('02_map_second_season', 'date2'),
                   ('03_map_cropping_intensity', 'seasons'), ('04_map_accuracy_check', 'validation')):
    map_sheet(R, study, f'{out}/{name}.png', kind, fonts=fonts); gc.collect(); print(name)
signatures(R, f'{out}/05_radar_signatures.png', fonts=fonts)
validation_chart(R, f'{out}/06_validation_chart.png', fonts=fonts)
wave_video(R, study, f'{out}/11_planting_wave_1080p.mp4', fonts=fonts, web_path=f'{out}/12_planting_wave_web.mp4')


study = RiceCalendar(ee, setup)
R = study.restore(out)
study.radar()
study.site_series()
field_timelapse(R, study, f'{out}/13_field_timelapse_1080p.mp4', fonts=fonts, web_path=f'{out}/14_field_timelapse_web.mp4')
field_strip(R, study, f'{out}/15_field_one_year.png', fonts=fonts)
np.savez_compressed(f'{out}/16_field_series.npz', **study.site_arrays())
study.site = None


import gc
gc.collect()
if 'study' not in globals(): study = RiceCalendar(ee, setup); R = study.restore(out)
if not hasattr(study, 'col'): study.radar()
study.regency_series()
regency_timelapse(R, study, f'{out}/17_regency_timelapse_1080p.mp4', fonts=fonts, web_path=f'{out}/18_regency_timelapse_web.mp4')
study.regts = None
gc.collect()


import shutil
shutil.make_archive(out, 'zip', out)
try:
    from google.colab import files; files.download(out + '.zip')
except ImportError:
    pass
