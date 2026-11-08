PROJECT = 'YOUR_EE_PROJECT'
RUN = ['palm', 'forest', 'urban']

COMMON = dict(BOX_KM=(6, 4), RES=10, DETAIL_M=(600, 400), S2=['2019-01-01', '2020-12-31'], GEDI=['2019-04-01', '2021-12-31'],
              CLOUD_SCORE=0.6, S2_MAX_IMAGES=15, INSET_DEG=4, AUTHOR='Ficking Alindra Wijayadarma', DPI=200, SITE_PICK=0)

MODES = {
    'palm':   dict(AUTO_SITE=True, NAME=None, MIN_TARGET=0.6,
                   SEARCH_BOX=[116.9, 0.6, 117.7, 1.3]),
    'forest': dict(AUTO_SITE=True, NAME=None, MIN_TARGET=0.8,
                   SEARCH_BOX=[110.0, -1.35, 110.35, -1.05]),
    'urban':  dict(AUTO_SITE=False, LON=106.8280, LAT=-6.2150, BUILT_FRACTION=0.3, BUILDING_PRESENCE=0.5,
                   NAME='Sudirman – Kuningan – Menteng – Tebet, DKI Jakarta'),
}

import ee
ee.Authenticate()
ee.Initialize(project=PROJECT)
print('Earth Engine OK')

import math, json, io, os, datetime, base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from scipy import ndimage

_A = 6378137.0; _F = 1/298.257223563; _E2 = _F*(2-_F); _EP2 = _E2/(1-_E2); _K0 = 0.9996
def ll2utm(lon, lat, zone, south):
    lam0 = math.radians(zone*6-183); phi = np.radians(lat); lam = np.radians(lon)
    N = _A/np.sqrt(1-_E2*np.sin(phi)**2); T = np.tan(phi)**2; C = _EP2*np.cos(phi)**2; A = np.cos(phi)*(lam-lam0)
    M = _A*((1-_E2/4-3*_E2**2/64-5*_E2**3/256)*phi-(3*_E2/8+3*_E2**2/32+45*_E2**3/1024)*np.sin(2*phi)
            +(15*_E2**2/256+45*_E2**3/1024)*np.sin(4*phi)-(35*_E2**3/3072)*np.sin(6*phi))
    x = _K0*N*(A+(1-T+C)*A**3/6+(5-18*T+T**2+72*C-58*_EP2)*A**5/120)+500000
    y = _K0*(M+N*np.tan(phi)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24+(61-58*T+T**2+600*C-330*_EP2)*A**6/720))
    return x, (y+10000000 if south else y)

def utm2ll(x, y, zone, south):
    lon, lat = zone*6-183.0, (-1.0 if south else 1.0)
    for _ in range(30):
        x0, y0 = ll2utm(lon, lat, zone, south); d = 1e-7
        x1, y1 = ll2utm(lon+d, lat, zone, south); x2, y2 = ll2utm(lon, lat+d, zone, south)
        J = np.array([[(x1-x0)/d, (x2-x0)/d], [(y1-y0)/d, (y2-y0)/d]])
        dl = np.linalg.solve(J, np.array([x-x0, y-y0])); lon += dl[0]; lat += dl[1]
        if abs(dl).max() < 1e-12: break
    return float(lon), float(lat)

def make_rect(lon, lat, w_km, h_km, res, zone=None):

    zone = zone or int((lon+180)//6)+1; south = lat < 0
    cx, cy = ll2utm(lon, lat, zone, south)
    nx = int(round(w_km*1000/res)); ny = int(round(h_km*1000/res))
    x0 = math.floor((cx-nx*res/2)/res)*res; y1 = math.ceil((cy+ny*res/2)/res)*res
    g = dict(zone=zone, south=south, epsg=(32700 if south else 32600)+zone, res=res, x0=x0, y1=y1, nx=nx, ny=ny,
             cx=float(cx), cy=float(cy), extent=[x0, x0+nx*res, y1-ny*res, y1])
    corners = [utm2ll(x, y, zone, south) for x, y in [(g['extent'][0], g['extent'][2]), (g['extent'][1], g['extent'][3]),
                                                      (g['extent'][0], g['extent'][3]), (g['extent'][1], g['extent'][2])]]
    g['bbox'] = [min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners), max(c[1] for c in corners)]
    return g

def _pixels(ee, img, g, r0, c0, h, w):
    req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
           'grid': {'dimensions': {'width': w, 'height': h},
                    'affineTransform': {'scaleX': g['res'], 'shearX': 0, 'translateX': g['x0']+c0*g['res'],
                                        'shearY': 0, 'scaleY': -g['res'], 'translateY': g['y1']-r0*g['res']},
                    'crsCode': 'EPSG:%d' % g['epsg']}}
    return ee.data.computePixels(req)

def fetch_grid(ee, img, g, names, dtype=np.float32, tile=500):

    out = {k: np.zeros((g['ny'], g['nx']), dtype) for k in names}
    for r0 in range(0, g['ny'], tile):
        for c0 in range(0, g['nx'], tile):
            h, w = min(tile, g['ny']-r0), min(tile, g['nx']-c0)
            for attempt in range(3):
                try:
                    a = _pixels(ee, img, g, r0, c0, h, w); break
                except Exception as e:
                    if attempt == 2: raise
                    msg = str(e).lower()
                    if tile > 16 and any(k in msg for k in ('memory', 'timed out', 'too many', 'limit', 'too large')):
                        return fetch_grid(ee, img, g, names, dtype, tile//2)
            for k in names: out[k][r0:r0+h, c0:c0+w] = np.asarray(a[k])
    return out

def _cmap(cols): return LinearSegmentedColormap.from_list('h', cols)
PROFILES = {
    'palm': dict(title='OIL PALM HEIGHT ON TERRAIN', short='Oil palm', obj='canopy', hmax=25,
                 cols=['#f6f1d1', '#cfe6a8', '#7fbf6a', '#2e8b57', '#14532d', '#062e1a'],
                 classes=[(0, 3, 'Under 3 m', 'Open, newly planted or replanted'), (3, 8, '3 – 8 m', 'Young palms'),
                          (8, 13, '8 – 13 m', 'Mature palms'), (13, 99, 'Over 13 m', 'Tall, old palms or trees')],
                 target='Oil palm', boundary='Oil palm boundary (BIOPAMA v1.2)', sub={1: 'industrial', 2: 'smallholder'},
                 hsrc='Top canopy height above ground (m), Meta 1 m, maximum per 10 m cell', href='GEDI lidar', vex=3,
                 data='FABDEM v1.2 (terrain, contours) · Meta / WRI Canopy Height 1 m ·\nBIOPAMA Global Oil Palm v1.2 · GEDI L2A (lidar check).\nCanopy heights reflect imagery mostly from 2018–2020.'),
    'forest': dict(title='FOREST CANOPY HEIGHT ON TERRAIN', short='Forest', obj='canopy', hmax=40,
                   cols=['#f4f1de', '#c9e3a6', '#7fbf6a', '#2f8f4e', '#145a32', '#06301a'],
                   classes=[(0, 10, 'Under 10 m', 'Gaps, regrowth or shrub'), (10, 20, '10 – 20 m', 'Young or degraded forest'),
                            (20, 30, '20 – 30 m', 'Mature forest'), (30, 999, 'Over 30 m', 'Tall emergent canopy')],
                   target='Forest', boundary='Forest 2020 (EC JRC GFC2020 V4), oil palm excluded', sub={1: 'forest'},
                   hsrc='Top canopy height above ground (m), Meta 1 m, maximum per 10 m cell', href='GEDI lidar', vex=2,
                   data='FABDEM v1.2 (terrain, contours) · Meta / WRI Canopy Height 1 m ·\nEC JRC Global Forest Cover 2020 V4 · GEDI L2A (lidar check).\nCanopy heights reflect imagery mostly from 2018–2020.'),
    'urban': dict(title='BUILDING HEIGHT ON TERRAIN', short='Urban', obj='building', hmax=100, gamma=.5,
                  cols=['#f3efe6', '#f6d8a8', '#f2a65a', '#d9603b', '#8c2d04', '#3d0c02'],
                  classes=[(0, 6, 'Under 6 m', '1 – 2 storeys, houses'), (6, 12, '6 – 12 m', '3 – 4 storeys'),
                           (12, 40, '12 – 40 m', '5 – 13 storeys, mid-rise'), (40, 999, 'Over 40 m', 'High-rise towers')],
                  target='Buildings', boundary='Built-up cells (Open Buildings 2.5D, presence ≥ 0.5)', sub={1: 'buildings'},
                  hsrc='Building height above ground (m), Google Open Buildings 2.5D, 10 m mean', href='GHSL built height', vex=2,
                  data='FABDEM v1.2 (terrain, contours) · Google Open Buildings 2.5D Temporal ·\nMeta / WRI Canopy Height 1 m (trees) · JRC GHSL Built Height 2018 (check).\nBuilding heights are model estimates from Sentinel-2, capped at 100 m.'),
}
def prof(cfg): return PROFILES[cfg['MODE']]
def cmap_of(cfg): return _cmap(prof(cfg)['cols'])
def hnorm(cfg):
    P = prof(cfg)
    return matplotlib.colors.PowerNorm(P.get('gamma', 1), 0, P['hmax'], clip=True) if P.get('gamma', 1) != 1 else Normalize(0, P['hmax'], clip=True)
def hcol(cfg, h):
    return cmap_of(cfg)(np.asarray(hnorm(cfg)(np.clip(np.nan_to_num(h), 0, None))))
def class_color(cfg, lo, hi):
    h = prof(cfg)['hmax']; mid = (lo+min(hi, h*1.2))/2
    return matplotlib.colors.to_hex(hcol(cfg, mid))

def _mosaic(ee, col_id, region, band=0):
    col = ee.ImageCollection(col_id).filterBounds(region)
    return col.mosaic().select([band]).setDefaultProjection(col.first().select([band]).projection())

def sources(ee, region):
    dtm = _mosaic(ee, 'projects/sat-io/open-datasets/FABDEM', region)
    meta = _mosaic(ee, 'projects/sat-io/open-datasets/facebook/meta-canopy-height', region)
    eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').select([0])
    return dtm, meta, eth

def open_buildings(ee, region):
    col = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1').filterBounds(region)
    last = col.aggregate_max('system:time_start')
    yc = col.filter(ee.Filter.eq('system:time_start', last))
    proj = yc.first().select('building_height').projection()
    m = yc.mosaic().setDefaultProjection(proj)
    return m.select('building_height'), m.select('building_presence'), ee.Date(last).get('year')

def find_sites(ee, cfg, top=5):

    lon0, lat0, lon1, lat1 = cfg['SEARCH_BOX']; res = cfg.get('SEARCH_RES', 300)
    clon, clat = (lon0+lon1)/2, (lat0+lat1)/2
    wkm = (lon1-lon0)*111.32*math.cos(math.radians(clat)); hkm = (lat1-lat0)*110.57
    g = make_rect(clon, clat, wkm, hkm, res)
    region = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    dtm, meta, eth = sources(ee, region)
    if cfg['MODE'] == 'palm':
        palm = _mosaic(ee, 'BIOPAMA/GlobalOilPalm/v1_2', region, 'classification')
        tgt = palm.gte(1).unmask(0).setDefaultProjection(palm.projection())
    else:
        jrc = ee.Image('JRC/GFC2020/V4').select('Map')
        tgt = jrc.eq(1).unmask(0).setDefaultProjection(jrc.projection())
    frac = tgt.reduceResolution(ee.Reducer.mean(), maxPixels=2048).rename('t')
    zm = dtm.reduceResolution(ee.Reducer.mean(), maxPixels=2048).rename('zm')
    A = fetch_grid(ee, ee.Image.cat([frac, zm]).unmask(-9999).toFloat(), g, ['t', 'zm'], tile=64)
    return rank_sites(A, g, cfg, top)

def rank_sites(A, g, cfg, top=5):
    kx = max(1, int(round(cfg['BOX_KM'][0]*1000/g['res']))); ky = max(1, int(round(cfg['BOX_KM'][1]*1000/g['res'])))
    nod = (A['zm'] < -9000) | (A['t'] < 0)
    p = np.where(nod, 0, A['t'])
    pm = ndimage.uniform_filter(p, (ky, kx), mode='constant')
    z = np.where(nod, 0, A['zm']).astype(float)
    m1 = ndimage.uniform_filter(z, (ky, kx), mode='constant'); m2 = ndimage.uniform_filter(z*z, (ky, kx), mode='constant')
    relief = 4*np.sqrt(np.clip(m2-m1*m1, 0, None))
    holes = ndimage.uniform_filter(nod.astype(float), (ky, kx), mode='constant', cval=1)
    score = np.where((pm >= cfg['MIN_TARGET']) & (holes < .02), pm*relief, 0)
    score[:ky//2+1] = 0; score[-(ky//2+1):] = 0; score[:, :kx//2+1] = 0; score[:, -(kx//2+1):] = 0
    out = []
    for _ in range(top):
        r, c = np.unravel_index(np.argmax(score), score.shape)
        if score[r, c] <= 0: break
        lon, lat = utm2ll(g['x0']+(c+.5)*g['res'], g['y1']-(r+.5)*g['res'], g['zone'], g['south'])
        out.append(dict(lon=round(lon, 4), lat=round(lat, 4), target_pct=round(float(pm[r, c])*100, 1),
                        relief_m=round(float(relief[r, c])), score=round(float(score[r, c]), 1)))
        score[max(0, r-ky):r+ky+1, max(0, c-kx):c+kx+1] = 0
    return out

def fetch(ee, cfg):
    mode = cfg['MODE']
    g = make_rect(cfg['LON'], cfg['LAT'], cfg['BOX_KM'][0], cfg['BOX_KM'][1], cfg.get('RES', 10))
    b = g['bbox']; pad = .01
    region = ee.Geometry.Rectangle([b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad])
    dtm, meta, eth = sources(ee, region)
    print('  [1/6] terrain (FABDEM) & canopy height (Meta 1 m -> 10 m, ETH 10 m) ...')
    lay = ee.Image.cat([dtm.resample('bilinear').rename('dtm'),
                        meta.reduceResolution(ee.Reducer.mean(), maxPixels=2048).rename('chm'),
                        meta.reduceResolution(ee.Reducer.max(), maxPixels=2048).rename('top'),
                        eth.rename('eth')]).unmask(-9999).toFloat()
    D = fetch_grid(ee, lay, g, ['dtm', 'chm', 'top', 'eth'], tile=200)
    D['dtm'] = despike(D['dtm'])
    print('  [2/6] Sentinel-2 %s – %s ...' % tuple(cfg['S2']))
    cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    bands = ['B4', 'B3', 'B2', 'B8']
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(*cfg['S2'])
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)).sort('CLOUDY_PIXEL_PERCENTAGE').limit(cfg.get('S2_MAX_IMAGES', 15))
           .select(bands).linkCollection(cs, ['cs']))
    clean = col.map(lambda im: im.updateMask(im.select('cs').gte(cfg['CLOUD_SCORE']))).select(bands).median()
    s2 = clean.unmask(col.select(bands).median()).clamp(0, 10000).toInt16().rename(['R', 'G', 'B', 'N'])
    D.update(fetch_grid(ee, s2, g, ['R', 'G', 'B', 'N'], np.int16))
    print('  [3/6] target layer (%s) ...' % prof(cfg)['target'])
    if mode == 'palm':
        palm = _mosaic(ee, 'BIOPAMA/GlobalOilPalm/v1_2', region, 'classification')
        D['tclass'] = fetch_grid(ee, palm.unmask(0).toInt16().rename('t'), g, ['t'], np.int16)['t']
        D['obj'] = D['top'].copy()
    elif mode == 'forest':
        palm = _mosaic(ee, 'BIOPAMA/GlobalOilPalm/v1_2', region, 'classification')
        jrc = ee.Image('JRC/GFC2020/V4').select('Map').eq(1).unmask(0)
        fc = jrc.And(palm.unmask(0).eq(0)).toInt16().rename('t')
        D['tclass'] = fetch_grid(ee, fc, g, ['t'], np.int16)['t']
        D['obj'] = D['top'].copy()
    else:
        bh, bp, yr = open_buildings(ee, region)
        D['ob_year'] = yr.getInfo()
        isb = bp.gte(cfg.get('BUILDING_PRESENCE', .5))
        ob = ee.Image.cat([bh.updateMask(isb).reduceResolution(ee.Reducer.mean(), maxPixels=4096).rename('bh'),
                          isb.unmask(0).setDefaultProjection(bp.projection()).reduceResolution(ee.Reducer.mean(), maxPixels=4096).rename('bf')])
        D.update(fetch_grid(ee, ob.unmask(-9999).toFloat(), g, ['bh', 'bf'], tile=100))
        ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_H/2018').select('built_height')
        D['ghsl'] = fetch_grid(ee, ghsl.unmask(-9999).toFloat().rename('h'), g, ['h'])['h']
        D['tclass'] = ((D['bf'] >= cfg.get('BUILT_FRACTION', .3)) & (D['bh'] > 0)).astype(np.int16)
        D['obj'] = np.where(D['tclass'] == 1, D['bh'], np.where(D['chm'] > -9000, D['chm'], -9999)).astype(np.float32)
    print('  [4/6] 1 m detail window ...')
    dx, dy = cfg['DETAIL_M']; lon_d, lat_d = cfg.get('DETAIL_LON'), cfg.get('DETAIL_LAT')
    if not lon_d: lon_d, lat_d = detail_centre(D, g, dx, dy)
    g1 = make_rect(lon_d, lat_d, dx/1000, dy/1000, 1, zone=g['zone'])
    if mode == 'urban':
        det = bh.updateMask(isb).unmask(0).rename('d')
    else:
        det = meta.unmask(-9999).rename('d')
    D['detail'] = fetch_grid(ee, det.toFloat(), g1, ['d'], tile=200)['d']; D['g1'] = g1
    print('  [5/6] GEDI lidar footprints ...')
    D['gedi_grid'] = gedi_grid(ee, cfg, region, g) if mode != 'urban' else None
    print('  [6/6] place name & location inset ...')
    pt = ee.Geometry.Point([cfg['LON'], cfg['LAT']])
    try:
        f = ee.FeatureCollection('FAO/GAUL/2015/level2').filterBounds(pt).first().getInfo()['properties']
        D['place'] = dict(adm2=f.get('ADM2_NAME'), adm1=f.get('ADM1_NAME'), adm0=f.get('ADM0_NAME'))
    except Exception:
        D['place'] = {}
    box = ee.Geometry.Rectangle([cfg['LON']-cfg['INSET_DEG'], cfg['LAT']-cfg['INSET_DEG'], cfg['LON']+cfg['INSET_DEG'], cfg['LAT']+cfg['INSET_DEG']])
    land = (ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filterBounds(box)
            .map(lambda ft: ft.setGeometry(ft.geometry().intersection(box, 2000).simplify(2000))))
    D['inset'] = land.getInfo()
    return D, g

def despike(z, tol=30):

    ok = z > -9000; zf = np.where(ok, z, np.nanmedian(np.where(ok, z, np.nan)))
    med = ndimage.median_filter(zf, 5); bad = ok & (np.abs(zf-med) > tol)
    return np.where(bad, med, z).astype(np.float32)

def tea_proxy(D, g, cfg):

    chm = D['chm']; R = D['R'].astype(float); N = D['N'].astype(float)
    ndvi = (N-R)/np.maximum(N+R, 1)
    m = (chm >= .3) & (chm <= 3) & (ndvi >= cfg.get('NDVI_MIN', .6)) & np.isin(D['wc'], [10, 20, 30, 40])
    m = ndimage.binary_opening(m, iterations=1); m = ndimage.binary_closing(m, iterations=2)
    lab, n = ndimage.label(m)
    if n:
        sz = ndimage.sum(m, lab, range(1, n+1))*(g['res']**2)/1e4
        m = np.isin(lab, np.where(sz >= cfg.get('MIN_PATCH_HA', 2))[0]+1)
    return m.astype(np.int16)

def detail_centre(D, g, dx, dy):

    kx, ky = max(1, int(dx/g['res'])), max(1, int(dy/g['res']))
    f = ndimage.uniform_filter((D['tclass'] >= 1).astype(float), (ky, kx), mode='constant')
    yy, xx = np.mgrid[0:g['ny'], 0:g['nx']]
    d = np.hypot((yy-g['ny']/2)/g['ny'], (xx-g['nx']/2)/g['nx'])
    m = np.zeros_like(f, bool); m[ky//2+1:-(ky//2+1), kx//2+1:-(kx//2+1)] = True
    score = np.where(m, f - .3*d, -9)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return utm2ll(g['x0']+(c+.5)*g['res'], g['y1']-(r+.5)*g['res'], g['zone'], g['south'])

def gedi_grid(ee, cfg, region, g):
    col = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(region).filterDate(*cfg['GEDI'])
    if col.size().getInfo() == 0: return None
    def q(im):
        ok = im.select('quality_flag').eq(1).And(im.select('degrade_flag').eq(0)).And(im.select('sensitivity').gt(0.95))
        return im.select('rh98').updateMask(ok)
    gedi = col.map(q).mosaic().unmask(-1).toFloat().rename('gedi')
    return fetch_grid(ee, gedi, g, ['gedi'], tile=500)['gedi']

def zrange(dtm):
    return float(np.nanpercentile(dtm, .5)), float(np.nanpercentile(dtm, 99.5))

def contour_interval(A, target=14):
    lo, hi = zrange(A['dtm']); r = hi-lo
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if r/v <= target: return v
    return 500

def contour_mask(dtm, ci, up=2):
    z = ndimage.zoom(ndimage.gaussian_filter(np.nan_to_num(dtm, nan=np.nanmean(dtm)), 1.2), up, order=1)
    k = np.floor(z/ci); edge = np.zeros(z.shape, bool)
    edge[:-1] |= k[:-1] != k[1:]; edge[:, :-1] |= k[:, :-1] != k[:, 1:]
    kk = np.floor(z/(ci*5)); idx = np.zeros(z.shape, bool)
    idx[:-1] |= kk[:-1] != kk[1:]; idx[:, :-1] |= kk[:, :-1] != kk[:, 1:]
    idx = ndimage.binary_dilation(idx)
    return np.where(idx, 1.0, np.where(edge, .7, 0.0))

def slope_deg(dtm, res):
    gy, gx = np.gradient(dtm, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def analyse(D, g, cfg):
    P = prof(cfg); res = g['res']; ha = res*res/1e4
    valid = (D['dtm'] > -9000) & (D['obj'] > -9000)
    dtm = np.where(D['dtm'] > -9000, D['dtm'], np.nan)
    dtm_f = np.where(np.isnan(dtm), np.nanmean(dtm), dtm)
    obj = np.where(valid, np.clip(D['obj'], 0, None), np.nan)
    chm = np.where(D['chm'] > -9000, np.clip(D['chm'], 0, None), np.nan)
    sl = slope_deg(ndimage.gaussian_filter(dtm_f, 1), res)
    T = (D['tclass'] >= 1) & valid
    med = lambda a: float(np.nanmedian(a)) if a.size and not np.all(np.isnan(a)) else float('nan')
    mean = lambda a: float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else float('nan')
    S = dict(mode=cfg['MODE'], box_km=list(cfg['BOX_KM']), area_ha=valid.sum()*ha,
             elev_min=zrange(dtm)[0], elev_max=zrange(dtm)[1], elev_mean=float(np.nanmean(dtm)),
             target_ha=T.sum()*ha, target_pct=T.sum()/max(valid.sum(), 1)*100,
             h_mean=mean(obj[T]), h_median=med(obj[T]), h_p90=float(np.nanpercentile(obj[T], 90)) if T.any() else float('nan'),
             h_other=mean(chm[valid & ~T]), slope_mean=mean(sl[T]),
             slope15_ha=(T & (sl > 15)).sum()*ha, slope25_ha=(T & (sl > 25)).sum()*ha)
    for code, name in P['sub'].items():
        m = (D['tclass'] == code) & valid
        S['sub_%s' % name] = dict(ha=m.sum()*ha, h_mean=mean(obj[m]))
    if cfg['MODE'] == 'urban':
        tree = valid & ~T & (chm >= 3)
        S['tree_cover_pct'] = tree.sum()/max((valid & ~T).sum(), 1)*100
        S['ob_year'] = D.get('ob_year')
    S['classes'] = []
    for lo, hi, l, mng in P['classes']:
        a = float((T & (obj >= lo) & (obj < hi)).sum()*ha)
        S['classes'].append(dict(label=l, meaning=mng, color=class_color(cfg, lo, hi), ha=a, pct=a/max(S['target_ha'], 1e-9)*100))
    lo_, hi_ = int(g['ny']*.12), int(g['ny']*.88)
    rows = [(np.nanmax(dtm[r][T[r]])-np.nanmin(dtm[r][T[r]])) if (lo_ <= r < hi_ and T[r].sum() > 20) else -1 for r in range(g['ny'])]
    r = int(np.argmax(rows)) if max(rows) > 0 else g['ny']//2
    Tr = dict(row=r, x=(np.arange(g['nx'])+.5)*res/1000, dtm=dtm[r], obj=obj[r], target=T[r])
    A = dict(S=S, T=Tr, slope=sl, obj=obj, chm=chm, dtm=dtm, target=T, valid=valid)
    if cfg['MODE'] == 'urban':
        A['V'] = validate_ghsl(D, A, g)
    else:
        D['gedi'] = gedi_points(D, g, A); A['V'] = validate_gedi(D['gedi'])
    return A

def gedi_points(D, g, A):
    G = D.get('gedi_grid')
    if G is None: return []
    lab, n = ndimage.label(G >= 0)
    if not n: return []
    h30 = ndimage.percentile_filter(np.nan_to_num(A['obj']), 90, size=3)
    eth = np.where(D['eth'] > -9000, D['eth'], np.nan); eth30 = ndimage.uniform_filter(np.nan_to_num(eth), 3)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        mm = lab[sl] == i; rr, cc = np.nonzero(mm); r = int(rr.mean())+sl[0].start; c = int(cc.mean())+sl[1].start
        out.append(dict(ref=float(G[r, c]), model=float(h30[r, c]), alt=None if np.isnan(eth[r, c]) else float(eth30[r, c]),
                        target=bool(A['target'][r, c]), slope=float(A['slope'][r, c])))
    return out

def _stats(x, y):
    d = y-x
    return dict(n=int(len(x)), bias=float(d.mean()), rmse=float(np.sqrt((d**2).mean())), mae=float(np.abs(d).mean()),
                r=float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float('nan'),
                x=x.tolist(), y=y.tolist())

def validate_gedi(pts, max_slope=20):
    pts = [p for p in pts if p['target'] and p['slope'] <= max_slope and 0 <= p['ref'] < 60]
    out = dict(ref='GEDI L2A RH98 lidar', n=len(pts), max_slope=max_slope, unit='footprint')
    if len(pts) < 10: return out
    x = np.array([p['ref'] for p in pts]); out['model'] = _stats(x, np.array([p['model'] for p in pts]))
    alt = [(p['ref'], p['alt']) for p in pts if p['alt'] is not None]
    if len(alt) >= 10: out['alt'] = _stats(np.array([a[0] for a in alt]), np.array([a[1] for a in alt]))
    out['names'] = dict(model='Meta / WRI 1 m (top height within 30 m of footprint)', alt='ETH 10 m (2020)')
    return out

def validate_ghsl(D, A, g, k=10):

    ny, nx = (g['ny']//k)*k, (g['nx']//k)*k
    T = A['target'][:ny, :nx]; H = np.where(T, np.nan_to_num(A['obj'][:ny, :nx]), 0); G = D['ghsl'][:ny, :nx]
    blk = lambda a: a.reshape(ny//k, k, nx//k, k)
    cnt = blk(T.astype(float)).sum((1, 3)); hs = blk(H).sum((1, 3))
    gm = np.where(blk(G) > -9000, blk(G), np.nan); gv = np.nanmean(gm.reshape(ny//k, k, nx//k, k), (1, 3))
    ok = (cnt >= k*k*.2) & ~np.isnan(gv) & (gv > 0)
    out = dict(ref='JRC GHSL built height 2018 (100 m)', n=int(ok.sum()), unit='100 m block')
    if ok.sum() >= 10:
        out['model'] = _stats(gv[ok], (hs/np.maximum(cnt, 1))[ok])
        out['names'] = dict(model='Open Buildings 2.5D (mean building height per 100 m)')
    return out

def rgb(D, gain=2.8):
    im = np.dstack([D['R'], D['G'], D['B']]).astype(float)/10000
    return np.clip(im*gain, 0, 1)**(1/1.15)

def hillshade(dtm, res, vexag=3):
    ls = LightSource(azdeg=315, altdeg=40)
    return ls.hillshade(np.nan_to_num(dtm, nan=np.nanmean(dtm)), vert_exag=vexag, dx=res, dy=res)

def shaded(A, g, cfg):
    P = prof(cfg); cm = cmap_of(cfg)
    hs = hillshade(A['dtm']+np.nan_to_num(A['obj'])*0.3, g['res'])
    col = hcol(cfg, A['obj'])[..., :3]
    if cfg['MODE'] == 'urban':
        ground = np.array(matplotlib.colors.to_rgb('#e9e6df')); green = np.array(matplotlib.colors.to_rgb('#9cc58a'))
        t = np.clip(np.nan_to_num(A['chm'])/15, 0, 1)[..., None]
        col = np.where(A['target'][..., None], col, ground*(1-t)+green*t)
    out = col*(.55+.45*hs[..., None]); out[~A['valid']] = 1
    return out

def _scalebar(ax, ext, color='k'):
    w = ext[1]-ext[0]; L = max([v for v in (250, 500, 1000, 2000, 5000) if v <= w/5] or [250])
    x0 = ext[0]+w*.04; y0 = ext[2]+(ext[3]-ext[2])*.05; h = (ext[3]-ext[2])*.012
    for i in range(4):
        ax.add_patch(Rectangle((x0+i*L/4, y0), L/4, h, fc='k' if i % 2 == 0 else 'w', ec='k', lw=.8, zorder=20))
    for i, t in enumerate([0, L/2, L]):
        ax.text(x0+t, y0+h*2, ('%g' % (t/1000)) + (' km' if i == 2 else ''), ha='center', va='bottom', fontsize=8, zorder=21,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='w')])

def _north(ax, ext):
    w = ext[1]-ext[0]; h = ext[3]-ext[2]; x = ext[1]-w*.04; y = ext[2]+h*.05
    ax.annotate('', xy=(x, y+h*.08), xytext=(x, y), zorder=21, arrowprops=dict(facecolor='k', edgecolor='w', width=4, headwidth=12, headlength=12))
    ax.text(x, y+h*.095, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=21,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='w')])

def _utm_ticks(ax, ext, step=1000):
    xt = np.arange(math.ceil(ext[0]/step)*step, ext[1], step); yt = np.arange(math.ceil(ext[2]/step)*step, ext[3], step)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels(['%d mE' % v for v in xt], fontsize=6.5); ax.set_yticklabels(['%d mN' % v for v in yt], fontsize=6.5, rotation=90, va='center')
    ax.grid(color='k', alpha=.18, lw=.5, ls='--'); ax.tick_params(length=3, top=True, right=True)

def _polys(geom):
    if not geom: return []
    t = geom.get('type')
    if t == 'Polygon': return [geom['coordinates']]
    if t == 'MultiPolygon': return list(geom['coordinates'])
    if t == 'GeometryCollection': return [p for gg in geom.get('geometries', []) for p in _polys(gg)]
    return []

def place_name(D, cfg):
    p = D.get('place') or {}
    if cfg.get('NAME'): return cfg['NAME']
    lead = {'palm': 'Oil palm on rolling terrain', 'tea': 'Tea-growing highlands', 'forest': 'Tropical forest', 'urban': 'High-rise and low-rise city'}[cfg['MODE']]
    if p.get('adm2'): return '%s, %s, %s' % (lead, p['adm2'], p.get('adm1', ''))
    return '%s (%.3f, %.3f)' % (lead, cfg['LON'], cfg['LAT'])

def _contours(ax, A, g, lw=(.35, .9), labels=True, outline=True):
    xs = g['x0']+(np.arange(g['nx'])+.5)*g['res']; ys = g['y1']-(np.arange(g['ny'])+.5)*g['res']
    ci = contour_interval(A); zs = ndimage.gaussian_filter(np.nan_to_num(A['dtm'], nan=np.nanmean(A['dtm'])), 1.2)
    lv = np.arange(math.floor(np.nanmin(A['dtm'])/ci)*ci, np.nanmax(A['dtm'])+ci, ci)
    minor = [v for v in lv if round(v/ci) % 5]; major = [v for v in lv if round(v/ci) % 5 == 0]
    if minor: ax.contour(xs, ys, zs, levels=minor, colors='#6b4f3a', linewidths=lw[0], alpha=.75, zorder=4)
    if major:
        cmj = ax.contour(xs, ys, zs, levels=major, colors='#6b4f3a', linewidths=lw[1], zorder=4)
        if labels: ax.clabel(cmj, fmt='%d', fontsize=6.5, inline=True, inline_spacing=2)
    if outline:
        ax.contour(xs, ys, ndimage.gaussian_filter(A['target'].astype(float), 1.5), levels=[.5], colors='#d62828', linewidths=.7, zorder=5)
    return ci

def _profile(ax, A, cfg, small=False):
    T = A['T']; x = T['x']; z = T['dtm']; top = z+np.nan_to_num(T['obj'])
    base = np.nanmin(z)-max(5, (np.nanmax(top)-np.nanmin(z))*.12)
    ax.fill_between(x, base, z, color='#b08968', lw=0)
    ax.fill_between(x, z, top, where=T['target'], color=class_color(cfg, prof(cfg)['hmax']*.4, prof(cfg)['hmax']*.6) if cfg['MODE'] == 'urban' else '#2a9d8f', lw=0, step='mid')
    ax.fill_between(x, z, top, where=~T['target'], color='#9aa58f', lw=0, step='mid')
    ax.plot(x, z, color='#5c4033', lw=.8); ax.set_xlim(x[0], x[-1]); ax.set_ylim(base, np.nanmax(top)+2)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7 if small else 7.5)

def map_figure(D, g, A, cfg, path):
    P = prof(cfg); S, T = A['S'], A['T']; ext = g['extent']
    fig = plt.figure(figsize=(16.5, 11.7), dpi=cfg.get('DPI', 200)); fig.patch.set_facecolor('white')
    aw = .645; ah = aw*16.5/11.7*(g['ny']/g['nx'])
    if ah > .80: ah = .80; aw = ah*11.7/16.5*(g['nx']/g['ny'])
    ax = fig.add_axes([.035, .955-ah, aw, ah])
    ax.imshow(shaded(A, g, cfg), extent=ext, interpolation='nearest')
    ci = _contours(ax, A, g, outline=cfg['MODE'] != 'urban')
    ty = g['y1']-(T['row']+.5)*g['res']
    ax.plot([ext[0], ext[1]], [ty, ty], color='w', lw=2.6, zorder=8); ax.plot([ext[0], ext[1]], [ty, ty], color='k', lw=1.1, ls=(0, (6, 3)), zorder=9)
    for x, t, ha in ((ext[0]+60, 'A', 'left'), (ext[1]-60, "A'", 'right')):
        ax.text(x, ty+70, t, fontsize=11, fontweight='bold', ha=ha, va='bottom', zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground='w')])
    e1 = D['g1']['extent']
    ax.add_patch(Rectangle((e1[0], e1[2]), e1[1]-e1[0], e1[3]-e1[2], fill=False, ec='#ffb703', lw=2, zorder=10))
    ax.text(e1[1]+60, e1[3], 'B', fontsize=11, fontweight='bold', color='k', va='top', zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground='#ffb703')])
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); _utm_ticks(ax, ext); _scalebar(ax, ext); _north(ax, ext)
    for s in ax.spines.values(): s.set_linewidth(1.4)
    pb = .06; ph = max(.10, .955-ah-.05-pb)
    axp = fig.add_axes([.035, pb, aw, ph]); _profile(axp, A, cfg); axp.set_ylabel('m a.s.l.', fontsize=8)
    axp.set_title("Profile A–A'  ·  terrain (brown) with %s and other vegetation (grey), true scale" % ('buildings (orange)' if cfg['MODE'] == 'urban' else P['short'].lower()+' canopy (teal)'), fontsize=8.5, loc='left')
    axp.text(0, -.2, 'A', transform=axp.transAxes, fontsize=10, fontweight='bold'); axp.text(1, -.2, "A'", transform=axp.transAxes, fontsize=10, fontweight='bold', ha='right')
    R = .035+aw+.03; PW = .975-R
    fig.text(R, .955, P['title'], fontsize=17, fontweight='bold', va='top')
    fig.text(R, .925, place_name(D, cfg), fontsize=11, va='top')
    fig.text(R, .905, '%g × %g km  ·  WGS 84 / UTM zone %d%s  ·  10 m grid' % (cfg['BOX_KM'][0], cfg['BOX_KM'][1], g['zone'], 'S' if g['south'] else 'N'),
             fontsize=8.5, color='#555', va='top')
    cax = fig.add_axes([R, .85, PW, .014])
    cb = fig.colorbar(plt.cm.ScalarMappable(hnorm(cfg), cmap_of(cfg)), cax=cax, orientation='horizontal')
    cb.set_label(P['hsrc'], fontsize=8.3); cb.ax.tick_params(labelsize=8)
    hs = [plt.Line2D([0], [0], color='#6b4f3a', lw=.9, label='Contour, interval %g m (index every %g m)' % (ci, ci*5)),
          (Patch(fc='#e9e6df', ec='#999', lw=.5, label='Ground, roads and open space') if cfg['MODE'] == 'urban' else Patch(fc='none', ec='#d62828', lw=1.2, label=P['boundary'])),
          Patch(fc='none', ec='#ffb703', lw=1.8, label='B  1 m detail'),
          plt.Line2D([0], [0], color='k', ls=(0, (6, 3)), lw=1.2, label="A–A'  terrain profile")]
    if cfg['MODE'] == 'urban': hs.insert(2, Patch(fc='#9cc58a', ec='none', label='Trees outside buildings (Meta canopy height)'))
    fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(R-.006, .815), frameon=False, fontsize=8.8)
    y = .66 if cfg['MODE'] == 'urban' else .68; fig.text(R, y, '%s by height' % P['target'], fontsize=11, fontweight='bold'); y -= .026
    for c in S['classes']:
        fig.patches.append(Rectangle((R, y-.003), .012, .014, transform=fig.transFigure, fc=c['color'], ec='#333', lw=.4))
        fig.text(R+.018, y, '%s  ·  %s' % (c['label'], c['meaning']), fontsize=8.6)
        fig.text(R+PW, y, '%s ha  (%.0f%%)' % (en(c['ha']), c['pct']), fontsize=8.6, ha='right', family='monospace'); y -= .022
    y -= .008
    rows = [('Elevation range', '%.0f – %.0f m' % (S['elev_min'], S['elev_max'])),
            ('%s area' % P['target'], '%s ha (%.0f%%)' % (en(S['target_ha']), S['target_pct'])),
            ('Median %s height' % P['obj'], '%.1f m' % S['h_median'])]
    if cfg['MODE'] == 'urban':
        rows.append(('Tree cover outside buildings', '%.0f%%' % S['tree_cover_pct']))
        if S.get('ob_year'): rows.append(('Building heights for year', '%s' % S['ob_year']))
    else:
        rows += [('Mean slope under %s' % P['short'].lower(), '%.1f°' % S['slope_mean']), ('On slopes over 25°', '%s ha' % en(S['slope25_ha']))]
    V = A['V']
    if 'model' in V: rows.append(('Check against %s' % P['href'], 'RMSE %.1f m, n = %d' % (V['model']['rmse'], V['model']['n'])))
    for k, v in rows:
        fig.text(R, y, k, fontsize=8.6); fig.text(R+PW, y, v, fontsize=8.6, ha='right', family='monospace'); y -= .021
    axd = fig.add_axes([R, .145, PW*.5, PW*.5*16.5/11.7*(D['g1']['ny']/D['g1']['nx'])])
    det = np.where(D['detail'] > -9000, np.clip(D['detail'], 0, None), np.nan)
    hsd = hillshade(np.nan_to_num(det), 1, vexag=1.5)
    axd.imshow(hcol(cfg, det)[..., :3]*(.5+.5*hsd[..., None]), extent=D['g1']['extent'])
    axd.set_xticks([]); axd.set_yticks([])
    for s in axd.spines.values(): s.set_edgecolor('#ffb703'); s.set_linewidth(2)
    axd.set_title('B  %s detail at 1 m' % ('Building height' if cfg['MODE'] == 'urban' else 'Canopy'), fontsize=9, loc='left')
    axi = fig.add_axes([R+PW*.54, .145, PW*.46, PW*.46*16.5/11.7*(D['g1']['ny']/D['g1']['nx'])])
    for ft in D['inset']['features']:
        for poly in _polys(ft.get('geometry')):
            if not poly or len(poly[0]) < 3: continue
            ring = np.array(poly[0]); axi.fill(ring[:, 0], ring[:, 1], fc='#e9e4d6', ec='#8a8577', lw=.4)
    d = cfg['INSET_DEG']; axi.set_xlim(cfg['LON']-d, cfg['LON']+d); axi.set_ylim(cfg['LAT']-d*.67, cfg['LAT']+d*.67)
    axi.set_facecolor('#d6e6f2'); axi.plot(cfg['LON'], cfg['LAT'], 's', mfc='#d62828', mec='k', ms=7)
    axi.set_xticks([]); axi.set_yticks([]); axi.set_title('Location', fontsize=9, loc='left')
    fig.text(R, .10, 'Data: '+P['data'], fontsize=7.2, color='#444', va='top')
    fig.text(R, .04, 'Map: %s' % cfg['AUTHOR'], fontsize=9.5)
    fig.savefig(path, facecolor='white'); plt.close(fig)

def profile_figure(D, g, A, cfg, path):
    P = prof(cfg); S = A['S']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=cfg.get('DPI', 200), gridspec_kw=dict(width_ratios=[2.1, 1]))
    _profile(a1, A, cfg)
    a1.set_xlabel("Distance along A–A' (km)  ·  true heights, no vertical exaggeration"); a1.set_ylabel('Elevation (m above sea level)')
    a1.set_title("Terrain and %s profile A–A'" % P['obj'], loc='left', fontsize=12, fontweight='bold')
    h = A['obj'][A['target']]; h = h[~np.isnan(h)]
    top = max(P['hmax']*1.2, np.percentile(h, 99) if h.size else 1)
    bins = np.linspace(0, top, 31)
    n, b, ps = a2.hist(h, bins=bins, color='#2a9d8f', edgecolor='w', lw=.4, weights=np.full(h.shape, g['res']**2/1e4))
    for p_, left in zip(ps, b[:-1]):
        for lo, hi, _, _ in P['classes']:
            if lo <= left < hi: p_.set_facecolor(class_color(cfg, lo, hi))
    a2.set_xlabel('%s height (m)' % P['obj'].capitalize()); a2.set_ylabel('%s area (ha)' % P['target'])
    a2.set_title('Height of mapped %s' % P['target'].lower(), loc='left', fontsize=12, fontweight='bold')
    if not math.isnan(S['h_median']):
        a2.axvline(S['h_median'], color='k', lw=1, ls='--'); a2.text(S['h_median'], a2.get_ylim()[1]*.97, '  median %.1f m' % S['h_median'], fontsize=8.5, va='top')
    a2.spines[['top', 'right']].set_visible(False)
    fig.text(.01, .01, '%s · %s' % (place_name(D, cfg), cfg['AUTHOR']), fontsize=8, color='#666')
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(path, facecolor='white'); plt.close(fig)

def validation_figure(A, cfg, path):
    V = A['V']; ks = [k for k in ('model', 'alt') if k in V]
    if not ks: return False
    fig, axs = plt.subplots(1, len(ks), figsize=(6.2*len(ks), 5.8), dpi=cfg.get('DPI', 200), squeeze=False)
    for ax, k in zip(axs[0], ks):
        v = V[k]; x = np.array(v['x']); y = np.array(v['y']); lim = max(prof(cfg)['hmax']*1.2, np.percentile(np.r_[x, y], 99)+2)
        ax.hexbin(x, y, gridsize=35, extent=(0, lim, 0, lim), cmap='Greens' if cfg['MODE'] != 'urban' else 'Oranges', mincnt=1, linewidths=.2)
        ax.plot([0, lim], [0, lim], color='k', lw=1, ls='--'); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.set_xlabel('%s (m)' % V['ref']); ax.set_ylabel('Mapped height (m)')
        ax.set_title(V['names'][k], loc='left', fontsize=10.5, fontweight='bold')
        ax.text(.03, .97, 'n = %d\nbias = %+.1f m\nMAE = %.1f m\nRMSE = %.1f m\nr = %.2f' % (v['n'], v['bias'], v['mae'], v['rmse'], v['r']),
                transform=ax.transAxes, va='top', fontsize=9, family='monospace', bbox=dict(fc='w', ec='#ccc'))
    extra = ' (slope ≤ %d°)' % V['max_slope'] if 'max_slope' in V else ''
    fig.suptitle('%s heights checked against %s, one point per %s%s' % (prof(cfg)['target'], V['ref'], V['unit'], extra), fontsize=11, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)
    return True

def verdict(S, V, mode=None):
    if 'model' not in V: return 'Not validated (too few reference points)', '#888888', float('nan')
    rel = V['model']['rmse']/max(S['h_median'], .1)
    if mode == 'urban':
        if rel <= .35: return 'Consistent with GHSL (model vs model)', '#2a9d8f', rel
        if rel <= .8: return 'Partly consistent with GHSL, check locally', '#e9a23b', rel
        return 'Inconsistent with GHSL: treat heights with caution', '#d62828', rel
    if rel <= .35: return 'Reliable at block / district scale', '#2a9d8f', rel
    if rel <= .8: return 'Indicative only, check in the field', '#e9a23b', rel
    return 'Not reliable: error is close to the object height', '#d62828', rel

def display_verdict(S, cfg):
    relief = max(S['elev_max']-S['elev_min'], 1); r = S['h_median']/relief
    if r < .03: return 'objects vanish at true scale, needs exaggeration', '#e9a23b', r
    if r < .15: return 'terrain dominates, 2–4× exaggeration helps', '#2a9d8f', r
    return 'objects readable even at true scale', '#2a9d8f', r

def compare_figure(results, path, author, dpi=200):
    modes = [m for m in ('palm', 'forest', 'tea', 'urban') if m in results]
    fig = plt.figure(figsize=(16.5, 11.7), dpi=dpi); fig.patch.set_facecolor('white')
    fig.text(.03, .965, 'ONE METHOD, THREE LANDSCAPES', fontsize=18, fontweight='bold', va='top')
    fig.text(.03, .935, 'Object height above terrain = surface model minus FABDEM terrain. Same 6 × 4 km window, 10 m grid and layout for every setting.',
             fontsize=10, color='#444', va='top')
    n = len(modes); W = .94/n
    for i, m in enumerate(modes):
        D, g, A, cfg = results[m]; P = prof(cfg); S, V = A['S'], A['V']; x0 = .03+i*W; w = W-.02
        fig.text(x0, .895, P['title'].title().replace('On Terrain', ''), fontsize=12.5, fontweight='bold', va='top')
        fig.text(x0, .872, place_name(D, cfg), fontsize=8.5, color='#555', va='top')
        ah = w*16.5/11.7*(g['ny']/g['nx'])
        ax = fig.add_axes([x0, .85-ah, w, ah]); ax.imshow(shaded(A, g, cfg), extent=g['extent'], interpolation='nearest')
        _contours(ax, A, g, lw=(.25, .6), labels=False, outline=m != 'urban'); ax.set_xticks([]); ax.set_yticks([])
        cax = fig.add_axes([x0, .85-ah-.022, w, .008])
        cb = fig.colorbar(plt.cm.ScalarMappable(hnorm(cfg), cmap_of(cfg)), cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=7); cb.set_label('%s height (m)' % P['obj'], fontsize=7.5, labelpad=1)
        top = .85-ah-.075
        axp = fig.add_axes([x0, top-.10, w, .10]); _profile(axp, A, cfg, small=True)
        axp.set_title("Profile A–A', true scale", fontsize=8, loc='left')
        rel_t = verdict(S, V, m)
        rows = [('Median object height', '%.1f m' % S['h_median']),
                ('Terrain relief in window', '%.0f m' % (S['elev_max']-S['elev_min'])),
                ('Object cover in window', '%.0f%%' % S['target_pct']),
                ('Height source', '1 m model' if m != 'urban' else '4 m model (Sentinel-2)'),
                ('Reference used for check', V['ref'].split(' (')[0]),
                ('Check points', '%d %ss' % (V['n'], V['unit']))]
        if 'model' in V:
            rows += [('Bias / RMSE', '%+.1f m / %.1f m' % (V['model']['bias'], V['model']['rmse'])),
                     ('Correlation r', '%.2f' % V['model']['r']),
                     ('RMSE ÷ median height', '%.0f%%' % (rel_t[2]*100))]
        y = top-.145
        for k, v in rows:
            fig.text(x0, y, k, fontsize=8.4); fig.text(x0+w, y, v, fontsize=8.4, ha='right', family='monospace'); y -= .0185
        fig.patches.append(Rectangle((x0, y-.022), w, .032, transform=fig.transFigure, fc=rel_t[1], ec='none', alpha=.18))
        fig.text(x0+.006, y-.006, 'Data: '+rel_t[0], fontsize=9, fontweight='bold', color=rel_t[1], va='center')
        dv = display_verdict(S, cfg); y -= .05
        for k, v in (('Object height ÷ terrain relief', '%.1f%%' % (dv[2]*100)), ('3D height exaggeration used', '%g×' % (cfg.get('VEXAG') or P['vex']))):
            fig.text(x0, y, k, fontsize=8.4); fig.text(x0+w, y, v, fontsize=8.4, ha='right', family='monospace'); y -= .0185
        fig.patches.append(Rectangle((x0, y-.022), w, .032, transform=fig.transFigure, fc=dv[1], ec='none', alpha=.18))
        fig.text(x0+.006, y-.006, 'Display: '+dv[0], fontsize=9, fontweight='bold', color=dv[1], va='center')
    fig.text(.03, .018, 'Data verdict: RMSE against the independent reference ÷ median object height (≤ 35%% reliable, ≤ 80%% indicative). Display verdict: median object height ÷ terrain relief.  '
             'Map: %s' % author, fontsize=8, color='#555')
    fig.savefig(path, facecolor='white'); plt.close(fig)

VERDICT_ID = {'Reliable at block / district scale': 'andal untuk skala blok/kawasan',
              'Indicative only, check in the field': 'hanya indikatif, perlu cek lapangan',
              'Not reliable: error is close to the object height': 'tidak andal, galat hampir sebesar tinggi objek',
              'Consistent with GHSL (model vs model)': 'konsisten dengan GHSL (antar-model, bukan uji akurasi)',
              'Partly consistent with GHSL, check locally': 'sebagian konsisten dengan GHSL, perlu cek lokal',
              'Inconsistent with GHSL: treat heights with caution': 'tidak konsisten dengan GHSL, tinggi gedung perlu dipakai hati-hati',
              'Not validated (too few reference points)': 'belum tervalidasi (titik pembanding kurang)'}

def compare_text(results):
    L = ['PERBANDINGAN METODE: TINGGI OBJEK DI ATAS TERRAIN PADA TIGA LANSKAP', '']
    for m in ('palm', 'forest', 'tea', 'urban'):
        if m not in results: continue
        D, g, A, cfg = results[m]; P = prof(cfg); S, V = A['S'], A['V']; v = verdict(S, V, m)
        L.append('%s — %s' % (P['title'].title().replace(' On Terrain', ''), place_name(D, cfg)))
        L.append('  tinggi objek median %s m, relief terrain %s m, tutupan objek %s%%' % (idn(S['h_median']), idn(S['elev_max']-S['elev_min'], 0), idn(S['target_pct'], 0)))
        if 'model' in V:
            L.append('  cek vs %s: n = %d, bias %s m, RMSE %s m, r = %s, RMSE/median = %s%%' % (V['ref'], V['model']['n'], sgn(V['model']['bias']),
                     idn(V['model']['rmse']), idn(V['model']['r'], 2), idn(v[2]*100, 0)))
        else:
            L.append('  cek vs %s: titik pembanding kurang (n = %d)' % (V['ref'], V['n']))
        L.append('  kesimpulan data: %s' % VERDICT_ID.get(v[0], v[0]))
        dv = display_verdict(S, cfg)
        L.append('  kesimpulan tampilan: tinggi objek = %s%% dari relief terrain → %s' % (idn(dv[2]*100), {
            'objects vanish at true scale, needs exaggeration': 'objek tak terlihat pada skala asli, 3D wajib diberi pembesaran vertikal',
            'terrain dominates, 2–4× exaggeration helps': 'terrain dominan, pembesaran 2–4× membantu objek terbaca',
            'objects readable even at true scale': 'objek sudah terbaca pada skala asli'}[dv[0]]))
        L.append('')
    L.append('CATATAN UMUM')
    rk = []
    for m in results:
        D, g, A, cfg = results[m]
        if 'model' in A['V']: rk.append((verdict(A['S'], A['V'], m)[2], prof(cfg)['short'], A['S']['h_median']))
    if rk:
        rk.sort()
        L.append('- Urutan galat relatif (RMSE ÷ tinggi median), dari yang terbaik: ' + '; '.join('%s %s%% (tinggi median %s m)' % (n, idn(r*100, 0), idn(h)) for r, n, h in rk) + '.')
        L.append('- Pola umumnya: makin tinggi objek dibanding galat data (± 2–3 m), makin bisa dipercaya hasilnya.')
    L.append('- Perkotaan: tinggi gedung dari Open Buildings adalah estimasi model dari Sentinel-2; pembanding GHSL juga berupa model, bukan pengukuran.')
    L.append('- Terrain FABDEM 30 m membatasi detail relief di ketiga lokasi; tampilan 3D memakai pembesaran vertikal yang dicantumkan.')
    return '\n'.join(L)

def en(v, d=1): return ('{:,.%df}' % d).format(v)
def idn(v, d=1): return en(v, d).replace(',', 'X').replace('.', ',').replace('X', '.')
def sgn(v): return ('+' if v >= 0 else '-')+idn(abs(v))

def write_tifs(D, g, A, cfg, folder):
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print('rasterio not installed: GeoTIFF export skipped'); return []
    tr = from_origin(g['x0'], g['y1'], g['res'], g['res']); out = []
    layers = {'dtm_fabdem': A['dtm'], '%s_height' % prof(cfg)['obj']: A['obj'], 'surface_dtm_plus_height': A['dtm']+np.nan_to_num(A['obj']),
              'slope_deg': A['slope'], 'target_mask': A['target'].astype(float)}
    for k, arr in layers.items():
        p = os.path.join(folder, '%s.tif' % k)
        with rasterio.open(p, 'w', driver='GTiff', height=g['ny'], width=g['nx'], count=1, dtype='float32', crs='EPSG:%d' % g['epsg'],
                           transform=tr, nodata=-9999, compress='deflate') as dst:
            dst.write(np.where(np.isnan(arr), -9999, arr).astype('float32'), 1)
        out.append(p)
    return out

def _png_b64(arr):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).save(b, 'PNG', optimize=True)
    return base64.b64encode(b.getvalue()).decode()

def html_3d(D, g, A, cfg, path, max_side=None):
    P = prof(cfg); max_side = max_side or (600 if cfg['MODE'] == 'urban' else 360)
    step = max(1, int(math.ceil(max(g['nx'], g['ny'])/max_side)))
    dtm = np.nan_to_num(A['dtm'], nan=np.nanmean(A['dtm']))[::step, ::step]
    obj = np.nan_to_num(A['obj'])[::step, ::step]
    ny, nx = dtm.shape; ci = contour_interval(A)
    from PIL import Image as _I
    b = io.BytesIO(); _I.fromarray((contour_mask(A['dtm'], ci)*255).astype(np.uint8), 'L').save(b, 'PNG', optimize=True)
    payload = dict(nx=nx, ny=ny, w=g['nx']*g['res'], h=g['ny']*g['res'],
                   dtm=base64.b64encode(np.round(dtm*10).astype('<i2').tobytes()).decode(),
                   chm=base64.b64encode(np.round(np.clip(obj, 0, 300)*10).astype('<i2').tobytes()).decode(),
                   sat=_png_b64(rgb(D)), hgt=_png_b64(shaded(A, g, cfg)), con=base64.b64encode(b.getvalue()).decode(), ci=ci,
                   vex=cfg.get('VEXAG') or P['vex'], title=place_name(D, cfg), elev=[round(A['S']['elev_min']), round(A['S']['elev_max'])],
                   author=cfg['AUTHOR'], obj=P['obj'], hmax=P['hmax'], gamma=P.get('gamma', 1), grad=P['cols'],
                   src='FABDEM · ' + ('Open Buildings 2.5D' if cfg['MODE'] == 'urban' else 'Meta/WRI Canopy Height') + ' · Sentinel-2')
    open(path, 'w', encoding='utf-8').write(VIEWER.replace('/*DATA*/null', json.dumps(payload)))
    return path

VIEWER = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Height on Terrain 3D</title>
<style>
:root{--bg:#141414;--panel:rgba(20,20,20,.82);--fg:#f2efe8;--mut:#a9a49a;--acc:#f2b33d;--line:rgba(255,255,255,.14)}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}
.ui{position:fixed;left:16px;top:16px;max-width:min(360px,calc(100% - 32px));background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;backdrop-filter:blur(6px)}
h1{font-size:15px;margin:0 0 2px}.sub{color:var(--mut);font-size:12px;margin-bottom:12px}
.row{display:flex;gap:6px;margin:8px 0;flex-wrap:wrap}.lab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin-top:10px}
button{flex:1;background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:inherit;font-size:13px;cursor:pointer}
button[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#1b1b1b;font-weight:600}
input[type=range]{width:100%;accent-color:var(--acc)}
.hint{font-size:11.5px;color:var(--mut);margin-top:2px}
.legend{height:8px;border-radius:4px;background:linear-gradient(90deg,#f6f1d1,#cfe6a8,#7fbf6a,#2e8b57,#14532d,#062e1a);margin-top:6px}
.lt{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;display:none;white-space:nowrap}
.foot{position:fixed;right:12px;bottom:10px;font-size:11px;color:var(--mut);text-align:right}
@media (max-width:600px){.ui{top:auto;bottom:12px;left:12px;right:12px;max-width:none;padding:12px}.sub{display:none}.foot{display:none}}
</style></head><body>
<canvas id="c"></canvas>
<div class="ui">
 <h1 id="t">Oil palm height on terrain</h1><div class="sub" id="s"></div>
 <div class="lab">Surface</div>
 <div class="row"><button id="bTer" aria-pressed="false">Terrain only</button><button id="bCan" aria-pressed="true">Terrain + <span id="objl"></span></button></div>
 <div class="lab">Colour</div>
 <div class="row"><button id="bHgt" aria-pressed="true">Height</button><button id="bSat" aria-pressed="false">Sentinel-2</button></div>
 <div id="leg"><div class="legend"></div><div class="lt"><span>0 m</span><span id="lmid"></span><span id="lmax"></span></div></div>
 <div class="lab">Contour lines</div>
 <div class="row"><button id="bCon1" aria-pressed="true">Show</button><button id="bCon0" aria-pressed="false">Hide</button></div>
 <div class="hint" id="cih"></div>
 <div class="lab">Height exaggeration <span id="vx"></span></div>
 <input id="vex" type="range" min="1" max="10" step="0.5">
 <div class="hint" id="vxh"></div>
</div>
<div id="tip"></div>
<div class="foot">Drag to rotate · scroll or pinch to zoom · right-drag to pan<br><span id="src"></span> · <span id="au"></span></div>
<script>
const D=/*DATA*/null;
const $=id=>document.getElementById(id);
$('t').textContent=D.title;$('s').textContent=(D.w/1000)+' × '+(D.h/1000)+' km · elevation '+D.elev[0]+'–'+D.elev[1]+' m';$('au').textContent=D.author;$('src').textContent=D.src;$('objl').textContent=D.obj+'s';
$('lmid').textContent=Math.round(D.hmax*Math.pow(.5,1/D.gamma))+' m';$('lmax').textContent=D.hmax+' m+';document.querySelector('.legend').style.background='linear-gradient(90deg,'+D.grad.join(',')+')';
const dec=b=>{const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Int16Array(u.buffer)};
const dtm=dec(D.dtm),chm=dec(D.chm),nx=D.nx,ny=D.ny,N=nx*ny;
let zmin=1e9;for(let i=0;i<N;i++)zmin=Math.min(zmin,dtm[i]/10);
const cv=$('c'),gl=cv.getContext('webgl',{antialias:true});
if(!gl){document.body.insertAdjacentHTML('beforeend','<p style="padding:20px">WebGL is not available in this browser.</p>')}
const S=Math.max(D.w,D.h),sx=D.w/S,sz=D.h/S;
const pos=new Float32Array(N*3),nor=new Float32Array(N*3),uv=new Float32Array(N*2);
for(let r=0;r<ny;r++)for(let c=0;c<nx;c++){const i=r*nx+c;pos[i*3]=(c/(nx-1)-.5)*sx;pos[i*3+2]=(r/(ny-1)-.5)*sz;uv[i*2]=c/(nx-1);uv[i*2+1]=r/(ny-1)}
const edge=[];for(let c=0;c<nx;c++)edge.push(c);for(let r=1;r<ny;r++)edge.push(r*nx+nx-1);for(let c=nx-2;c>=0;c--)edge.push((ny-1)*nx+c);for(let r=ny-2;r>=1;r--)edge.push(r*nx);edge.push(0);
const nE=edge.length,skP=new Float32Array(nE*2*3),skN=new Float32Array(nE*2*3);
const idx=[];for(let r=0;r<ny-1;r++)for(let c=0;c<nx-1;c++){const a=r*nx+c,b=a+1,d=a+nx,e=d+1;idx.push(a,d,b,b,d,e)}
const skI=[];for(let k=0;k<nE-1;k++){const a=k*2,b=a+1,c=a+2,d=a+3;skI.push(a,b,c,c,b,d)}
let canopy=true,useSat=false,vex=D.vex,base=-0.02;
function build(){
  const k=vex/S;
  for(let i=0;i<N;i++)pos[i*3+1]=(dtm[i]/10-zmin+(canopy?chm[i]/10:0))*k;
  nor.fill(0);
  for(let t=0;t<idx.length;t+=3){const a=idx[t]*3,b=idx[t+1]*3,c=idx[t+2]*3;
    const ux=pos[b]-pos[a],uy=pos[b+1]-pos[a+1],uz=pos[b+2]-pos[a+2],vx=pos[c]-pos[a],vy=pos[c+1]-pos[a+1],vz=pos[c+2]-pos[a+2];
    const nx_=uy*vz-uz*vy,ny_=uz*vx-ux*vz,nz_=ux*vy-uy*vx;
    for(const j of[a,b,c]){nor[j]+=nx_;nor[j+1]+=ny_;nor[j+2]+=nz_}}
  for(let i=0;i<N;i++){const l=Math.hypot(nor[i*3],nor[i*3+1],nor[i*3+2])||1;nor[i*3]/=-l;nor[i*3+1]/=-l;nor[i*3+2]/=-l;if(nor[i*3+1]<0){nor[i*3]*=-1;nor[i*3+1]*=-1;nor[i*3+2]*=-1}}
  for(let k2=0;k2<nE;k2++){const i=edge[k2];const n2=k2<nE-1?edge[k2+1]:edge[0];
    const ex=pos[n2*3]-pos[i*3],ez=pos[n2*3+2]-pos[i*3+2];let wx=ez,wz=-ex;const l=Math.hypot(wx,wz)||1;wx/=l;wz/=l;
    skP.set([pos[i*3],pos[i*3+1],pos[i*3+2],pos[i*3],base,pos[i*3+2]],k2*6);skN.set([wx,0,wz,wx,0,wz],k2*6)}
  up(bP,pos);up(bN,nor);up(sP,skP);up(sN,skN);
}
function up(b,a){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.DYNAMIC_DRAW)}
function buf(a,t=gl.ARRAY_BUFFER){const b=gl.createBuffer();gl.bindBuffer(t,b);gl.bufferData(t,a,gl.STATIC_DRAW);return b}
const ext=gl.getExtension('OES_element_index_uint');
const bP=buf(pos),bN=buf(nor),bU=buf(uv),bI=buf(new Uint32Array(idx),gl.ELEMENT_ARRAY_BUFFER),sP=buf(skP),sN=buf(skN),sI=buf(new Uint32Array(skI),gl.ELEMENT_ARRAY_BUFFER);
const VS=`attribute vec3 p;attribute vec3 n;attribute vec2 u;uniform mat4 M;varying vec3 vn;varying vec2 vu;varying float vy;
void main(){vn=n;vu=u;vy=p.y;gl_Position=M*vec4(p,1.0);}`;
const FS=`precision mediump float;varying vec3 vn;varying vec2 vu;varying float vy;uniform sampler2D T;uniform sampler2D C;uniform float side;uniform float showC;
void main(){vec3 L=normalize(vec3(-.5,.9,-.35));float d=max(dot(normalize(vn),L),0.0);
vec3 c=side>.5?vec3(.36,.30,.24)*(.55+.45*d):texture2D(T,vu).rgb*(.45+.65*d);
if(side<.5)c=mix(c,vec3(.30,.20,.12),texture2D(C,vu).r*showC*.9);gl_FragColor=vec4(c,1.0);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));gl.linkProgram(pr);gl.useProgram(pr);
const L={p:gl.getAttribLocation(pr,'p'),n:gl.getAttribLocation(pr,'n'),u:gl.getAttribLocation(pr,'u'),M:gl.getUniformLocation(pr,'M'),side:gl.getUniformLocation(pr,'side'),
 T:gl.getUniformLocation(pr,'T'),C:gl.getUniformLocation(pr,'C'),showC:gl.getUniformLocation(pr,'showC')};gl.uniform1i(L.T,0);gl.uniform1i(L.C,1);
const tex={};let ready=0;
for(const k of['sat','hgt','con']){const t=gl.createTexture(),im=new Image();im.onload=()=>{gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,im);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);ready++;draw()};im.src='data:image/png;base64,'+D[k];tex[k]=t}
let yaw=-0.6,pitch=0.62,dist=1.55,tx=0,tz=0;
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return[t/a,0,0,0,0,t,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0]}
function mul(a,b){const o=new Array(16).fill(0);for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)o[j*4+i]+=a[k*4+i]*b[j*4+k];return o}
function look(e,c){let z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];let l=Math.hypot(...z);z=z.map(v=>v/l);let x=[z[2],0,-z[0]];l=Math.hypot(...x);x=x.map(v=>v/l);
 const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
 return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]),-(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),-(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]),1]}
let MVP;
let showCon=1;
function draw(){if(ready<3)return;const w=cv.clientWidth*devicePixelRatio,h=cv.clientHeight*devicePixelRatio;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}
 gl.viewport(0,0,w,h);gl.clearColor(.08,.08,.08,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
 const dd=dist*Math.max(1,1.15/(w/h));const eye=[tx+dd*Math.cos(pitch)*Math.sin(yaw),dd*Math.sin(pitch),tz+dd*Math.cos(pitch)*Math.cos(yaw)];
 MVP=mul(persp(.75,w/h,.01,20),look(eye,[tx,0.03,tz]));gl.uniformMatrix4fv(L.M,false,new Float32Array(MVP));
 gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,tex.con);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex[useSat?'sat':'hgt']);gl.uniform1f(L.showC,showCon);
 const at=(b,l,s)=>{gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,s,gl.FLOAT,false,0,0)};
 at(bP,L.p,3);at(bN,L.n,3);at(bU,L.u,2);gl.uniform1f(L.side,0);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bI);gl.drawElements(gl.TRIANGLES,idx.length,gl.UNSIGNED_INT,0);
 gl.disableVertexAttribArray(L.u);at(sP,L.p,3);at(sN,L.n,3);gl.uniform1f(L.side,1);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,sI);gl.drawElements(gl.TRIANGLES,skI.length,gl.UNSIGNED_INT,0)}
let drag=null,pts=new Map();
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);pts.set(e.pointerId,[e.clientX,e.clientY]);drag={b:e.button,x:e.clientX,y:e.clientY,d:pts.size===2?pd():0};$('tip').style.display='none'});
cv.addEventListener('pointerup',e=>{pts.delete(e.pointerId);drag=pts.size?{b:0,x:[...pts.values()][0][0],y:[...pts.values()][0][1],d:0}:null});
function pd(){const v=[...pts.values()];return Math.hypot(v[0][0]-v[1][0],v[0][1]-v[1][1])}
cv.addEventListener('pointermove',e=>{
 if(!drag){hover(e);return}pts.set(e.pointerId,[e.clientX,e.clientY]);
 if(pts.size===2){const d=pd();if(drag.d){dist=Math.min(5,Math.max(.25,dist*drag.d/d))}drag.d=d;draw();return}
 const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.b===2||e.shiftKey){const s=dist/cv.clientHeight;tx-=(Math.cos(yaw)*dx-Math.sin(yaw)*dy)*s*.9;tz-=(-Math.sin(yaw)*dx-Math.cos(yaw)*dy)*s*.9}
 else{yaw-=dx*.006;pitch=Math.min(1.5,Math.max(.08,pitch+dy*.005))}draw()});
cv.addEventListener('wheel',e=>{e.preventDefault();dist=Math.min(5,Math.max(.25,dist*Math.exp(e.deltaY*.001)));draw()},{passive:false});
function hover(e){if(!MVP)return;const r=cv.getBoundingClientRect(),mx=(e.clientX-r.left)/r.width*2-1,my=1-(e.clientY-r.top)/r.height*2;let best=-1,bd=1e9,bz=1e9;
 const st=Math.max(1,Math.floor(Math.sqrt(N/6000)));
 for(let rr=0;rr<ny;rr+=st)for(let c=0;c<nx;c+=st){const i=rr*nx+c,x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];
  const cx=MVP[0]*x+MVP[4]*y+MVP[8]*z+MVP[12],cy=MVP[1]*x+MVP[5]*y+MVP[9]*z+MVP[13],cw=MVP[3]*x+MVP[7]*y+MVP[11]*z+MVP[15],cz=MVP[2]*x+MVP[6]*y+MVP[10]*z+MVP[14];
  if(cw<=0)continue;const d=Math.hypot(cx/cw-mx,cy/cw-my);if(d<.03&&cz/cw<bz){bz=cz/cw;best=i;bd=d}}
 const tp=$('tip');if(best<0){tp.style.display='none';return}
 tp.style.display='block';tp.style.left=(e.clientX+14)+'px';tp.style.top=(e.clientY+14)+'px';
 tp.textContent='Ground '+(dtm[best]/10).toFixed(0)+' m · '+D.obj+' '+(chm[best]/10).toFixed(1)+' m'}
const press=(a,b,v)=>{$(a).setAttribute('aria-pressed',v);$(b).setAttribute('aria-pressed',!v)};
$('bTer').onclick=()=>{canopy=false;press('bTer','bCan',true);build();draw()};$('bCan').onclick=()=>{canopy=true;press('bCan','bTer',true);build();draw()};
$('bHgt').onclick=()=>{useSat=false;press('bHgt','bSat',true);$('leg').style.visibility='visible';draw()};$('bSat').onclick=()=>{useSat=true;press('bSat','bHgt',true);$('leg').style.visibility='hidden';draw()};
$('bCon1').onclick=()=>{showCon=1;press('bCon1','bCon0',true);draw()};$('bCon0').onclick=()=>{showCon=0;press('bCon0','bCon1',true);draw()};
$('cih').textContent='Interval '+D.ci+' m, thicker line every '+(D.ci*5)+' m';
const vxText=()=>{$('vx').textContent=vex+'×';$('vxh').textContent=vex==1?'True scale: the area is '+(D.w/1000)+' km wide but only '+(D.elev[1]-D.elev[0])+' m high, so hills look almost flat.':'Heights are stretched '+vex+'× so the hills and the '+D.obj+'s are easier to see. Set to 1× for true scale.'};
$('vex').value=vex;vxText();$('vex').oninput=e=>{vex=+e.target.value;vxText();build();draw()};
addEventListener('resize',draw);build();draw();
</script></body></html>'''

def summary_text(D, g, A, cfg):
    P = prof(cfg); S, V = A['S'], A['V']; m = cfg['MODE']; L = []
    L.append({'palm': 'ANALISIS TINGGI KEBUN SAWIT DI ATAS TERRAIN', 'forest': 'ANALISIS TINGGI TAJUK HUTAN DI ATAS TERRAIN',
              'urban': 'ANALISIS TINGGI BANGUNAN DI ATAS TERRAIN'}[m])
    L.append(place_name(D, cfg))
    L.append('Pusat %.4f, %.4f · kotak %g × %g km · grid 10 m UTM zona %d%s' % (cfg['LAT'], cfg['LON'], cfg['BOX_KM'][0], cfg['BOX_KM'][1], g['zone'], 'S' if g['south'] else 'N'))
    L.append('')
    L.append('1. METODE')
    L.append('- Tinggi tanah (DTM): FABDEM v1.2, Copernicus DEM 30 m yang efek pohon dan bangunannya sudah dihapus.')
    if m == 'palm':
        L.append('- Tinggi objek: Meta/WRI Global Canopy Height 1 m (citra 2009–2020, mayoritas 2018–2020), nilai tertinggi per sel 10 m (tinggi pucuk tajuk).')
        L.append('- Batas objek: BIOPAMA Global Oil Palm v1.2 (sawit industri dan rakyat).')
    elif m == 'forest':
        L.append('- Tinggi objek: Meta/WRI Global Canopy Height 1 m, nilai tertinggi per sel 10 m (tinggi pucuk tajuk).')
        L.append('- Batas objek: hutan 2020 EC JRC GFC2020 V4 (definisi FAO), dikurangi kebun sawit BIOPAMA.')
    else:
        L.append('- Tinggi objek: Google Open Buildings 2.5D Temporal (estimasi tinggi gedung dari Sentinel-2, tahun %s), rata-rata per sel 10 m.' % S.get('ob_year'))
        L.append('- Batas objek: sel dengan ≥ %d%% piksel bangunan (presence ≥ %s). Pohon di luar gedung dari Meta Canopy Height.' % (int(cfg.get('BUILT_FRACTION', .3)*100), idn(cfg.get('BUILDING_PRESENCE', .5), 1)))
    L.append('- Permukaan 3D = DTM + tinggi objek. Kontur dan lereng dihitung dari DTM.')
    L.append('- Pembanding independen: %s.' % V['ref']); L.append('')
    L.append('2. HASIL')
    L.append('- Ketinggian tanah %s–%s m dpl (beda tinggi %s m).' % (idn(S['elev_min'], 0), idn(S['elev_max'], 0), idn(S['elev_max']-S['elev_min'], 0)))
    L.append('- %s: %s ha (%s%% kotak). Tinggi median %s m, rata-rata %s m, persentil 90 %s m.' % (P['target'], idn(S['target_ha']), idn(S['target_pct'], 0),
             idn(S['h_median']), idn(S['h_mean']), idn(S['h_p90'])))
    if m == 'palm':
        L.append('- Sawit industri %s ha (rata-rata %s m), sawit rakyat %s ha (rata-rata %s m).' % (idn(S['sub_industrial']['ha']), idn(S['sub_industrial']['h_mean']),
                 idn(S['sub_smallholder']['ha']), idn(S['sub_smallholder']['h_mean'])))
    if m == 'urban':
        L.append('- Tutupan pohon di luar bangunan %s%%.' % idn(S['tree_cover_pct'], 0))
    L.append('- Kelas tinggi:')
    for c in S['classes']: L.append('    %-12s %-32s %10s ha  (%s%%)' % (c['label'], c['meaning'], idn(c['ha']), idn(c['pct'], 0)))
    if m != 'urban':
        L.append('- Lereng rata-rata %s°. Objek di lereng > 15°: %s ha; > 25°: %s ha.' % (idn(S['slope_mean']), idn(S['slope15_ha']), idn(S['slope25_ha'])))
    if 'model' in V:
        v = V['model']; L.append('- Validasi vs %s (n = %d %s): bias %s m, MAE %s m, RMSE %s m, r = %s.' % (V['ref'], v['n'], V['unit'], sgn(v['bias']), idn(v['mae']), idn(v['rmse']), idn(v['r'], 2)))
        if 'alt' in V:
            e = V['alt']; L.append('  ETH 10 m: bias %s m, RMSE %s m, r = %s.' % (sgn(e['bias']), idn(e['rmse']), idn(e['r'], 2)))
        vd = verdict(S, V, m); L.append('- RMSE dibanding tinggi median = %s%% → %s.' % (idn(vd[2]*100, 0), VERDICT_ID.get(vd[0], vd[0])))
    else:
        L.append('- Validasi tidak dihitung: titik pembanding hanya %d.' % V['n'])
    L.append('')
    L.append('3. ARTI')
    if m == 'palm':
        L.append('- Tinggi sawit naik seiring umur: < 3 m baru tanam/replanting, 3–8 m muda, 8–13 m menghasilkan, > 13 m tua. Proxy kasar umur blok.')
        L.append('- Sawit di lereng > 25° rawan erosi dan umumnya dihindari untuk penanaman baru.')
    elif m == 'forest':
        L.append('- Tajuk < 10 m menandakan celah, semak, atau hutan yang sedang pulih; > 30 m adalah tajuk pohon emergen hutan tua.')
        L.append('- Hutan tropis tinggi (> 30 m) dikenal sering diremehkan oleh peta tinggi tajuk global; validasi GEDI menunjukkan seberapa besar.')
    else:
        hi_ = next((c for c in S['classes'] if c['label'].startswith('Over')), None); lo_ = S['classes'][0]
        L.append('- Kontras vertikal: %s ha (%s%%) bangunan rendah < 6 m (perumahan/kampung) berdampingan dengan %s ha (%s%%) gedung > 40 m.'
                 % (idn(lo_['ha']), idn(lo_['pct'], 0), idn(hi_['ha']), idn(hi_['pct'], 0)))
        L.append('- Kelas tinggi gedung memberi gambaran kepadatan vertikal dan di mana kota tumbuh ke atas, bukan melebar.')
        L.append('- Bersama tutupan pohon, data ini berguna untuk kajian panas perkotaan (urban heat island) dan ruang terbuka hijau.')
    L.append('- Profil A–A\' (skala asli) memperlihatkan tinggi objek jauh lebih kecil dari relief terrain; model 3D memakai pembesaran vertikal.')
    L.append('')
    L.append('4. KETERBATASAN')
    if m == 'urban':
        L.append('- Tinggi Open Buildings adalah estimasi model dari Sentinel-2 10 m, bukan pengukuran; gedung kecil dan rapat bisa tergabung.')
        L.append('- Nilai tinggi dibatasi maksimal 100 m, jadi menara pencakar langit (> 100 m) tercatat 100 m.')
        L.append('- GHSL 2018 juga berupa model (100 m), jadi perbandingan ini konsistensi antar-model, bukan uji akurasi absolut.')
    else:
        L.append('- CHM Meta dari citra 2009–2020 (mayoritas 2018–2020) dengan MAE global ± 2,8 m; GEDI juga punya galat, terutama di lereng.')
        L.append('- GEDI diukur 2019–2021, sedangkan citra Meta bisa lebih tua; tanaman yang tumbuh di antaranya ikut menambah selisih.')
    L.append('- FABDEM 30 m, lisensi CC BY-NC-SA 4.0 (non-komersial). Luas adalah hitungan piksel 10 m.')
    return '\n'.join(L)

import traceback
OUT = 'height_on_terrain_output'; os.makedirs(OUT, exist_ok=True)
RESULTS = {}
for mode in RUN:
    cfg = dict(COMMON, **MODES[mode], MODE=mode); folder = os.path.join(OUT, mode); os.makedirs(folder, exist_ok=True)
    print('\n========== %s ==========' % mode.upper())
    try:
        if cfg['AUTO_SITE']:
            cands = find_sites(ee, cfg)
            assert cands, 'no candidate site found'
            for i, c in enumerate(cands):
                print('  candidate [%d] %.4f, %.4f  target %5.1f%%  relief %4d m  https://www.google.com/maps/@%.4f,%.4f,6000m/data=!3m1!1e3'
                      % (i, c['lat'], c['lon'], c['target_pct'], c['relief_m'], c['lat'], c['lon']))
            cfg['LON'], cfg['LAT'] = cands[cfg['SITE_PICK']]['lon'], cands[cfg['SITE_PICK']]['lat']
        print('  site: %.4f, %.4f' % (cfg['LAT'], cfg['LON']))
        D, g = fetch(ee, cfg)
        A = analyse(D, g, cfg)
        map_figure(D, g, A, cfg, folder+'/01_map.png')
        profile_figure(D, g, A, cfg, folder+'/02_profile_histogram.png')
        validation_figure(A, cfg, folder+'/03_validation.png')
        html_3d(D, g, A, cfg, folder+'/04_3d.html')
        write_tifs(D, g, A, cfg, folder)
        txt = summary_text(D, g, A, cfg); open(folder+'/05_analisis.txt', 'w', encoding='utf-8').write(txt)
        json.dump(dict(config=cfg, stats=A['S'], validation={k: ({kk: vv for kk, vv in v.items() if kk not in ('x', 'y')} if isinstance(v, dict) else v)
                  for k, v in A['V'].items()}, place=D.get('place')), open(folder+'/06_stats.json', 'w'), indent=1, default=str)
        RESULTS[mode] = (D, g, A, cfg)
    except Exception as e:
        print('  mode %s failed: %s' % (mode, e)); traceback.print_exc()

import shutil
if len(RESULTS) >= 2:
    os.makedirs(OUT+'/comparison', exist_ok=True)
    compare_figure(RESULTS, OUT+'/comparison/00_comparison.png', COMMON['AUTHOR'], COMMON['DPI'])
    ct = compare_text(RESULTS); open(OUT+'/comparison/00_perbandingan.txt', 'w', encoding='utf-8').write(ct)
else:
    pass
shutil.make_archive(OUT, 'zip', OUT)
