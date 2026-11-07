PROJECT = 'YOUR_EE_PROJECT'
CFG = dict(
    NAME      = 'Oil palm landscape north-east of Palembang, South Sumatra',
    LON       = 104.857,
    LAT       = -2.782,
    RADIUS_KM = 5,
    CUTOFF    = 2020,
    MMU_HA    = 0.5,
    BEFORE    = ['2019-01-01', '2020-12-31'],
    AFTER     = ['2025-01-01', '2026-10-31'],
    CLOUD_SCORE = 0.6,
    INSET_DEG = 4,
    N_VERIFY  = 3,
    S2_MAX_IMAGES = 12,
    AUTHOR    = 'Ficking Alindra Wijayadarma',
    DPI       = 200)

import ee
ee.Authenticate()
ee.Initialize(project=PROJECT)
print('Earth Engine OK')

import math, json, io, os, datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Circle, FancyArrow
from matplotlib.colors import ListedColormap
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

def make_grid(lon, lat, radius_km, res=10):
    zone = int((lon+180)//6)+1; south = lat < 0
    cx, cy = ll2utm(lon, lat, zone, south)
    half = radius_km*1000 + 200
    x0 = math.floor((cx-half)/res)*res; y1 = math.ceil((cy+half)/res)*res
    while 2*half/res > 1400: res += 10
    x0 = math.floor((cx-half)/res)*res; y1 = math.ceil((cy+half)/res)*res
    n = int(round(2*half/res))
    xs = x0 + (np.arange(n)+.5)*res; ys = y1 - (np.arange(n)+.5)*res
    X, Y = np.meshgrid(xs, ys)
    inside = (X-cx)**2+(Y-cy)**2 <= (radius_km*1000)**2
    return dict(zone=zone, south=south, epsg=(32700 if south else 32600)+zone, res=res, x0=x0, y1=y1, n=n,
                cx=float(cx), cy=float(cy), radius_m=radius_km*1000, inside=inside,
                extent=[x0, x0+n*res, y1-n*res, y1])

def _pixels(ee, img, g, r0, c0, h, w):

    req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
           'grid': {'dimensions': {'width': w, 'height': h},
                    'affineTransform': {'scaleX': g['res'], 'shearX': 0, 'translateX': g['x0']+c0*g['res'],
                                        'shearY': 0, 'scaleY': -g['res'], 'translateY': g['y1']-r0*g['res']},
                    'crsCode': 'EPSG:%d' % g['epsg']}}
    return ee.data.computePixels(req)

def _fetch_tiled(ee, img, g, names, tile=520):

    n = g['n']; out = {k: np.zeros((n, n), np.int16) for k in names}
    for r0 in range(0, n, tile):
        for c0 in range(0, n, tile):
            h, w = min(tile, n-r0), min(tile, n-c0)
            for attempt in range(3):
                try:
                    a = _pixels(ee, img, g, r0, c0, h, w); break
                except Exception as e:
                    if attempt == 2: raise
                    if 'memory' in str(e).lower() and tile > 130:
                        return _fetch_tiled(ee, img, g, names, tile//2)
            for k in names: out[k][r0:r0+h, c0:c0+w] = np.asarray(a[k])
    return out

def fetch(ee, g, cfg):

    lon, lat = cfg['LON'], cfg['LAT']
    pt = ee.Geometry.Point([lon, lat]); region = pt.buffer(cfg['RADIUS_KM']*1000+500)
    gfc = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
    jrc = ee.Image('JRC/GFC2020/V4').select('Map').eq(1).unmask(0)
    palm = ee.ImageCollection('BIOPAMA/GlobalOilPalm/v1_2').filterBounds(region)
    pcls = palm.select('classification').mosaic().unmask(0)
    pyr = palm.select('planting_year').mosaic().unmask(0)
    print('  [1/3] forest, loss & oil palm layers ...')
    lay = ee.Image.cat([gfc.select('lossyear').unmask(0).rename('lossyear'),
                        gfc.select('datamask').unmask(0).rename('datamask'),
                        jrc.rename('forest2020'), pcls.rename('palm'), pyr.rename('plant_year')]).toInt16()
    D = _fetch_tiled(ee, lay, g, ['lossyear', 'datamask', 'forest2020', 'palm', 'plant_year'])
    cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    def s2(d0, d1):
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(pt).filterDate(d0, d1)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).sort('CLOUDY_PIXEL_PERCENTAGE')
               .limit(cfg.get('S2_MAX_IMAGES', 12)).select(['B4', 'B3', 'B2']).linkCollection(cs, ['cs']))
        clean = col.map(lambda im: im.updateMask(im.select('cs').gte(cfg['CLOUD_SCORE']))).select(['B4', 'B3', 'B2']).median()
        return clean.unmask(col.select(['B4', 'B3', 'B2']).median()).clamp(0, 10000).toInt16()
    for i, (pre, rng) in enumerate([('b', cfg['BEFORE']), ('a', cfg['AFTER'])]):
        print('  [%d/3] Sentinel-2 %s – %s ...' % (i+2, rng[0], rng[1]))
        img = s2(*rng).rename([pre+'R', pre+'G', pre+'B'])
        D.update(_fetch_tiled(ee, img, g, [pre+'R', pre+'G', pre+'B']))
    box = ee.Geometry.Rectangle([lon-cfg['INSET_DEG'], lat-cfg['INSET_DEG'], lon+cfg['INSET_DEG'], lat+cfg['INSET_DEG']])
    land = (ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filterBounds(box)
            .map(lambda f: f.setGeometry(f.geometry().intersection(box, 2000).simplify(2000))))
    D['inset'] = land.getInfo()
    return D

CLASS = {0: ('No data / water', '#00000000'),
         1: ('Forest 2020 still standing (JRC)', '#2e7d32'),
         2: ('Oil palm, industrial (BIOPAMA)', '#f2b33d'),
         3: ('Oil palm, smallholder (BIOPAMA)', '#c77dff'),
         4: ('Tree cover loss 2001–2020', '#9e9e9e'),
         5: ('Loss after 2020 inside oil palm (replanting)', '#40c4ff'),
         6: ('Loss after 2020, other non-forest land', '#d7b98e'),
         7: ('DEFORESTATION after 31 Dec 2020', '#ff1744')}

def analyse(D, g, cfg):
    cut = cfg['CUTOFF']; inside = g['inside'] & (D['datamask'] == 1); ha = (g['res']**2)/10000
    ly = D['lossyear'].astype(int); year = np.where(ly > 0, 2000+ly, 0)
    f20 = (D['forest2020'] == 1) & inside
    palm = (D['palm'] >= 1) & inside
    after = (year > cut) & inside; before = (year > 0) & (year <= cut) & inside
    defo_raw = after & f20

    lab, n = ndimage.label(defo_raw, structure=np.ones((3, 3)))
    sizes = ndimage.sum(defo_raw, lab, range(1, n+1))*ha if n else np.array([])
    keep = np.isin(lab, np.where(sizes >= cfg['MMU_HA'])[0]+1)
    defo = keep
    repl = after & palm & ~f20
    other = after & ~f20 & ~palm
    cls = np.zeros(inside.shape, np.uint8)
    cls[inside] = 6*0
    cls[before] = 4
    cls[palm & (D['palm'] == 1)] = 2; cls[palm & (D['palm'] == 2)] = 3
    cls[f20 & ~after] = 1
    cls[repl] = 5; cls[other] = 6
    cls[defo_raw & ~keep] = 6
    cls[defo] = 7
    cls[~inside] = 0

    lab2, n2 = ndimage.label(defo, structure=np.ones((3, 3)))
    patches = []
    for i, sl in enumerate(ndimage.find_objects(lab2), start=1):
        m = lab2[sl] == i; a = m.sum()*ha
        rr, cc = np.nonzero(m); r = rr.mean()+sl[0].start; c = cc.mean()+sl[1].start
        x = g['x0']+(c+.5)*g['res']; y = g['y1']-(r+.5)*g['res']
        lon, lat = utm2ll(x, y, g['zone'], g['south'])
        yrs = year[sl][m]; py = int(np.bincount(yrs).argmax())
        patches.append(dict(id=i, area_ha=round(float(a), 2), row=int(r), col=int(c), lon=round(lon, 5), lat=round(lat, 5), main_year=py))
    patches.sort(key=lambda p: -p['area_ha'])

    if palm.any() and defo.any():
        dist = ndimage.distance_transform_edt(~palm)*g['res']
        d_defo = dist[defo]; near_palm_ha = (d_defo <= 1000).sum()*ha; med_dist = float(np.median(d_defo))
    else:
        near_palm_ha, med_dist = 0.0, float('nan')
    yrs_all = range(2001, 2026)
    per_year = {y: dict(defo=float(((year == y) & defo).sum()*ha), repl=float(((year == y) & repl).sum()*ha),
                        other=float(((year == y) & (other | (defo_raw & ~keep))).sum()*ha),
                        before=float(((year == y) & before).sum()*ha)) for y in yrs_all}
    S = dict(study_area_ha=inside.sum()*ha, forest2020_ha=f20.sum()*ha, palm_ind_ha=((D['palm'] == 1) & inside).sum()*ha,
             palm_sh_ha=((D['palm'] == 2) & inside).sum()*ha, loss_2001_2020_ha=before.sum()*ha,
             loss_after_ha=after.sum()*ha, defo_ha=defo.sum()*ha, defo_raw_ha=defo_raw.sum()*ha,
             repl_ha=repl.sum()*ha, other_ha=other.sum()*ha, n_patches=len(patches),
             defo_pct_of_forest=(100*defo.sum()/f20.sum()) if f20.any() else 0.0,
             defo_within_1km_palm_ha=near_palm_ha, median_dist_to_palm_m=med_dist)
    S = {k: (round(float(v), 2) if isinstance(v, (float, np.floating, np.integer, int)) and not isinstance(v, bool) else v) for k, v in S.items()}
    return cls, S, patches, per_year

def rgb(D, pre, gain=2.6):
    im = np.dstack([D[pre+'R'], D[pre+'G'], D[pre+'B']]).astype(float)/10000
    im = np.clip(im*gain, 0, 1)**(1/1.15)
    return im

def scalebar(ax, g, length_m=None, y_off=.06):
    ext = g['extent']; w = ext[1]-ext[0]
    if length_m is None: length_m = [500, 1000, 2000, 2500, 5000][np.searchsorted([1500, 3000, 6000, 8000, 99e9], w/4)]
    x0 = ext[0]+w*.05; y0 = ext[2]+(ext[3]-ext[2])*y_off; h = (ext[3]-ext[2])*.012
    for i in range(4):
        ax.add_patch(Rectangle((x0+i*length_m/4, y0), length_m/4, h, fc='k' if i % 2 == 0 else 'w', ec='k', lw=.8, zorder=20))
    for i, t in enumerate([0, length_m/2, length_m]):
        ax.text(x0+t, y0+h*2.2, ('%g' % (t/1000)) + (' km' if i == 2 else ''), ha='center', va='bottom', fontsize=8,
                color='w', zorder=21, path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground='k')])

def north(ax, g):
    ext = g['extent']; w = ext[1]-ext[0]; h = ext[3]-ext[2]
    x = ext[1]-w*.06; y = ext[3]-h*.13
    ax.annotate('', xy=(x, y+h*.07), xytext=(x, y), zorder=21,
                arrowprops=dict(facecolor='w', edgecolor='k', width=4, headwidth=12, headlength=12))
    ax.text(x, y+h*.085, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold', color='w', zorder=21,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='k')])

import matplotlib.patheffects

def utm_ticks(ax, g, step=2000):
    ext = g['extent']
    xt = np.arange(math.ceil(ext[0]/step)*step, ext[1], step); yt = np.arange(math.ceil(ext[2]/step)*step, ext[3], step)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels(['%d mE' % v for v in xt], fontsize=7); ax.set_yticklabels(['%d mN' % v for v in yt], fontsize=7, rotation=90, va='center')
    ax.grid(color='w', alpha=.25, lw=.6, ls='--')
    ax.tick_params(length=3, top=True, right=True, labeltop=False, labelright=False)

def map_figure(D, g, cls, S, patches, cfg, path):
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(16.5, 11.7), dpi=cfg.get('DPI', 200))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.035, 0.06, 0.58, 0.86])
    base = rgb(D, 'a'); base[~g['inside']] = base[~g['inside']]*.35+.65*np.array([1, 1, 1])
    ax.imshow(base, extent=g['extent'], interpolation='nearest', zorder=1)
    colors = [CLASS[k][1] for k in range(8)]; colors[0] = '#00000000'
    cmap = ListedColormap(colors)
    over = np.ma.masked_where((cls == 0), cls)
    ax.imshow(over, cmap=cmap, vmin=-.5, vmax=7.5, extent=g['extent'], interpolation='nearest', alpha=.82, zorder=2)
    ax.add_patch(Circle((g['cx'], g['cy']), g['radius_m'], fill=False, ec='k', lw=1.4, zorder=5))
    for p in patches[:cfg['N_VERIFY']]:
        x = g['x0']+(p['col']+.5)*g['res']; y = g['y1']-(p['row']+.5)*g['res']
        ax.add_patch(Circle((x, y), 260, fill=False, ec='#ffeb3b', lw=1.8, zorder=6))
        ax.text(x, y+300, str(patches.index(p)+1), color='#ffeb3b', fontsize=11, fontweight='bold', zorder=7, ha='center', va='bottom',
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='k')])
    ax.set_xlim(g['extent'][:2]); ax.set_ylim(g['extent'][2:])
    utm_ticks(ax, g); scalebar(ax, g); north(ax, g)
    for s in ax.spines.values(): s.set_linewidth(1.4)

    R = 0.645
    fig.text(R, 0.94, 'DEFORESTATION SCREENING (NDPE / EUDR)', fontsize=16, fontweight='bold', va='top')
    fig.text(R, 0.905, cfg['NAME'], fontsize=13, va='top')
    fig.text(R, 0.878, 'Cut-off 31 December %d  ·  radius %g km  ·  WGS 84 / UTM zone %d%s'
             % (cfg['CUTOFF'], cfg['RADIUS_KM'], g['zone'], 'S' if g['south'] else 'N'), fontsize=9, color='#444', va='top')
    present = [k for k in range(1, 8) if (cls == k).any()]
    handles = [Patch(fc=CLASS[k][1], ec='#333', lw=.5, label=CLASS[k][0]) for k in present]
    handles.append(Patch(fc='none', ec='k', lw=1.2, label='Study area'))
    if patches: handles.append(Patch(fc='none', ec='#ffeb3b', lw=1.8, label='Largest deforestation patches (see verification)'))
    lg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(R-.005, 0.86), frameon=False, fontsize=9.5,
                    title='Legend', title_fontsize=11, alignment='left', handlelength=1.6)

    rows = [('Study area', S['study_area_ha']), ('Forest 2020 (JRC baseline)', S['forest2020_ha']),
            ('Oil palm, industrial', S['palm_ind_ha']), ('Oil palm, smallholder', S['palm_sh_ha']),
            ('Tree cover loss 2001–%d' % cfg['CUTOFF'], S['loss_2001_2020_ha']),
            ('Loss after cut-off, inside oil palm', S['repl_ha']),
            ('Loss after cut-off, other non-forest', S['other_ha']),
            ('DEFORESTATION after cut-off (≥%g ha)' % cfg['MMU_HA'], S['defo_ha'])]
    y = 0.50
    fig.text(R, y+.03, 'Mapped area (ha)', fontsize=11, fontweight='bold')
    for i, (k, v) in enumerate(rows):
        yy = y-i*.026; bold = 'DEFOREST' in k
        fig.text(R, yy, k, fontsize=9.3, fontweight='bold' if bold else 'normal', color='#c62828' if bold else '#222')
        fig.text(R+.30, yy, '{:,.1f}'.format(v), fontsize=9.3, ha='right', fontweight='bold' if bold else 'normal',
                 color='#c62828' if bold else '#222', family='DejaVu Sans Mono')
    fig.text(R, y-len(rows)*.026-.004, '%.2f%% of 2020 forest lost after cut-off  ·  %d patch(es)'
             % (S['defo_pct_of_forest'], S['n_patches']), fontsize=8.5, color='#444')

    axi = fig.add_axes([R, 0.105, 0.13, 0.135])
    def _polys(geom):
        if not geom: return []
        t = geom.get('type')
        if t == 'Polygon': return [geom['coordinates']]
        if t == 'MultiPolygon': return list(geom['coordinates'])
        if t == 'GeometryCollection': return [p for gg in geom.get('geometries', []) for p in _polys(gg)]
        return []
    for f in D['inset']['features']:
        for poly in _polys(f.get('geometry')):
            if not poly or len(poly[0]) < 3: continue
            ring = np.array(poly[0]); axi.fill(ring[:, 0], ring[:, 1], fc='#e9e4d6', ec='#8a8577', lw=.4)
    axi.plot(cfg['LON'], cfg['LAT'], marker='s', ms=7, mfc='none', mec='#c62828', mew=1.8)
    d = cfg['INSET_DEG']; axi.set_xlim(cfg['LON']-d, cfg['LON']+d); axi.set_ylim(cfg['LAT']-d, cfg['LAT']+d)
    axi.set_aspect('equal'); axi.set_facecolor('#cfe3f0'); axi.set_xticks([]); axi.set_yticks([])
    axi.set_title('Location', fontsize=8.5, loc='left')
    fig.text(R+.145, 0.235, 'Centre  %.4f°%s, %.4f°E' % (abs(cfg['LAT']), 'S' if cfg['LAT'] < 0 else 'N', cfg['LON']), fontsize=8, color='#444', va='top')
    fig.text(R+.145, 0.21, 'Base image: Sentinel-2 L2A median\n%s – %s' % tuple(cfg['AFTER']), fontsize=8, color='#444', va='top')
    fig.text(R, 0.075, 'Data: EC JRC Global Forest Cover 2020 V4 · Hansen/UMD Global Forest Change v1.13 ·\n'
             'BIOPAMA Global Oil Palm v1.2 · Copernicus Sentinel-2. Screening only: areas are pixel counts,\nnot a compliance verdict. '
             'Deforestation patches < %g ha removed (FAO minimum forest area).' % cfg['MMU_HA'], fontsize=7.2, color='#555', va='top')
    fig.text(R, 0.022, 'Map: %s · %s' % (cfg['AUTHOR'], datetime.date.today().strftime('%d %B %Y')), fontsize=8.5, va='top')
    fig.savefig(path, dpi=cfg.get('DPI', 200)); plt.close(fig)

def chart_figure(per_year, cfg, path):
    yrs = sorted(per_year); x = np.arange(len(yrs))
    B = np.array([per_year[y]['before'] for y in yrs]); Dd = np.array([per_year[y]['defo'] for y in yrs])
    Rp = np.array([per_year[y]['repl'] for y in yrs]); O = np.array([per_year[y]['other'] for y in yrs])
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=cfg.get('DPI', 200))
    ax.bar(x, B, color='#9e9e9e', label='Tree cover loss up to cut-off')
    ax.bar(x, O, color='#d7b98e', label='After cut-off: other non-forest')
    ax.bar(x, Rp, bottom=O, color='#40c4ff', label='After cut-off: inside oil palm (replanting)')
    ax.bar(x, Dd, bottom=O+Rp, color='#ff1744', label='After cut-off: DEFORESTATION of 2020 forest')
    ax.axvline(yrs.index(cfg['CUTOFF'])+.5, color='k', ls='--', lw=1); ax.text(yrs.index(cfg['CUTOFF'])+.4, ax.get_ylim()[1]*.98, 'cut-off ', fontsize=8, va='top', ha='right')
    ax.set_xticks(x); ax.set_xticklabels(yrs, rotation=90, fontsize=8); ax.set_ylabel('hectares (mapped)')
    ax.set_title('Annual tree cover loss by context — ' + cfg['NAME'], fontsize=11, loc='left')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.text(.01, .01, 'Hansen/UMD GFC v1.13 changed method from 2011; compare individual years with caution.', fontsize=7, color='#666')
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(path); plt.close(fig)

def verify_figure(D, g, cls, patches, cfg, path):
    P = patches[:cfg['N_VERIFY']]
    if not P: return False
    fig, axs = plt.subplots(len(P), 3, figsize=(12, 3.9*len(P)), dpi=cfg.get('DPI', 200), squeeze=False)
    before, after = rgb(D, 'b'), rgb(D, 'a'); w = 45
    for i, p in enumerate(P):
        r0, r1 = max(0, p['row']-w), min(g['n'], p['row']+w); c0, c1 = max(0, p['col']-w), min(g['n'], p['col']+w)
        m = (cls[r0:r1, c0:c1] == 7)
        for j, (im, t) in enumerate([(before, 'Before  %s – %s' % tuple(cfg['BEFORE'])), (after, 'After  %s – %s' % tuple(cfg['AFTER']))]):
            axs[i, j].imshow(im[r0:r1, c0:c1]); axs[i, j].contour(m, levels=[.5], colors=['#ffeb3b'], linewidths=1.2)
            axs[i, j].set_title(t, fontsize=9); axs[i, j].axis('off')
        sub = np.ma.masked_where(cls[r0:r1, c0:c1] == 0, cls[r0:r1, c0:c1])
        axs[i, 2].imshow(after[r0:r1, c0:c1]*.6+.4)
        axs[i, 2].imshow(sub, cmap=ListedColormap([CLASS[k][1] for k in range(8)]), vmin=-.5, vmax=7.5, alpha=.85)
        axs[i, 2].set_title('Classes', fontsize=9); axs[i, 2].axis('off')
        axs[i, 0].text(-0.04, .5, 'Patch %d\n%.2f ha\nmain year %d\n%.5f, %.5f' % (i+1, p['area_ha'], p['main_year'], p['lat'], p['lon']),
                       transform=axs[i, 0].transAxes, ha='right', va='center', fontsize=9)
    fig.suptitle('Visual verification of the largest deforestation patches (yellow outline). '
                 'Confirm that the BEFORE image shows natural forest and the AFTER image shows clearing.', fontsize=10, x=.02, ha='left')
    fig.tight_layout(rect=(.1, 0, 1, .97)); fig.savefig(path); plt.close(fig)
    return True

def fmt(v, d=1): return ('{:,.%df}' % d).format(v).replace(',', 'X').replace('.', ',').replace('X', '.')
def summary_text(S, patches, per_year, cfg, g, verified_note=True):
    cut = cfg['CUTOFF']; L = []
    L.append('ANALISIS SCREENING DEFORESTASI (NDPE / EUDR) — %s' % cfg['NAME'])
    L.append('Wilayah kajian: lingkaran radius %g km (%s ha) berpusat di %s°%s, %s°BT. Cut-off: 31 Desember %d.'
             % (cfg['RADIUS_KM'], fmt(S['study_area_ha'], 0), fmt(abs(cfg['LAT']), 4), 'LS' if cfg['LAT'] < 0 else 'LU', fmt(cfg['LON'], 4), cut))
    L.append('')
    L.append('1. Kondisi awal (2020). Hutan menurut peta acuan JRC 2020 seluas %s ha (%s%% wilayah kajian). '
             'Kebun sawit terpetakan (BIOPAMA) seluas %s ha: %s ha industri dan %s ha rakyat.'
             % (fmt(S['forest2020_ha']), fmt(100*S['forest2020_ha']/max(S['study_area_ha'], 1e-9)), fmt(S['palm_ind_ha']+S['palm_sh_ha']),
                fmt(S['palm_ind_ha']), fmt(S['palm_sh_ha'])))
    L.append('2. Riwayat sebelum cut-off. Tutupan pohon yang hilang 2001–%d seluas %s ha. Ini mencakup pembukaan lahan lama '
             'maupun penebangan kebun; tidak dihitung sebagai pelanggaran cut-off.' % (cut, fmt(S['loss_2001_2020_ha'])))
    if S['defo_ha'] > 0:
        top = patches[0]
        yrs = [y for y in sorted(per_year) if y > cut and per_year[y]['defo'] > 0]
        peak = max(yrs, key=lambda y: per_year[y]['defo']) if yrs else None
        L.append('3. Temuan utama. Terdeteksi %s ha hutan 2020 yang hilang setelah cut-off (%s%% dari hutan 2020), '
                 'dalam %d petak ≥ %s ha. Petak terbesar %s ha di koordinat %.5f, %.5f (tahun %d).%s'
                 % (fmt(S['defo_ha']), fmt(S['defo_pct_of_forest'], 2), S['n_patches'], fmt(cfg['MMU_HA']), fmt(top['area_ha'], 2),
                    top['lat'], top['lon'], top['main_year'], (' Kehilangan terbesar terjadi pada %d.' % peak) if peak else ''))
        if S['defo_within_1km_palm_ha'] > 0:
            L.append('   %s ha di antaranya berada ≤ 1 km dari kebun sawit terpetakan (median jarak %s m) — indikasi risiko '
                     'ekspansi kebun yang perlu dicek lapangan dan dokumen izin (HGU/IUP).'
                     % (fmt(S['defo_within_1km_palm_ha']), fmt(S['median_dist_to_palm_m'], 0)))
        else:
            L.append('   Tidak ada petak deforestasi dalam radius 1 km dari kebun sawit terpetakan, sehingga keterkaitan langsung '
                     'dengan sawit lemah; penyebab lain (pertanian, tambang, permukiman) perlu dicek.')
        L.append('   Status: TERINDIKASI berisiko, belum terbukti. Wajib verifikasi visual (lihat gambar verifikasi) dan lapangan.')
    else:
        L.append('3. Temuan utama. Tidak terdeteksi kehilangan hutan 2020 setelah cut-off (petak ≥ %s ha). '
                 'Pada tingkat screening ini, wilayah kajian berisiko rendah terhadap deforestasi pasca cut-off.' % fmt(cfg['MMU_HA']))
        if S['defo_raw_ha'] > 0:
            L.append('   Ada %s ha piksel tersebar (< %s ha per petak) yang dianggap derau peta, bukan deforestasi.'
                     % (fmt(S['defo_raw_ha']), fmt(cfg['MMU_HA'])))
    L.append('4. Kehilangan tutupan pohon pasca cut-off yang BUKAN deforestasi: %s ha di dalam kebun sawit (umumnya peremajaan/replanting) '
             'dan %s ha di lahan non-hutan lain (mis. kebun karet/akasia, semak). Keduanya tidak melanggar cut-off karena lahannya '
             'bukan hutan pada 2020.' % (fmt(S['repl_ha']), fmt(S['other_ha'])))
    L.append('')
    L.append('KETERBATASAN')
    L.append('- Luas dihitung dari jumlah piksel (screening). UMD menyatakan hitungan piksel Hansen bukan estimasi luas tanpa bias; '
             'angka resmi memerlukan sampling akurasi berstrata.')
    L.append('- Hansen mendeteksi hilangnya tajuk > 5 m; tebang pilih ringan dan degradasi bisa tidak terdeteksi, dan awan/asap dapat '
             'menggeser tahun deteksi.')
    L.append('- Peta hutan JRC 2020 V4 tidak mengikat secara hukum; BIOPAMA hanya memetakan sawit bertajuk rapat (2016–2021), '
             'sehingga kebun muda/baru belum tercatat.')
    L.append('- Komponen No Peat dan No Exploitation tidak dinilai (perlu peta gambut resmi dan data sosial).')
    return '\n'.join(L)

g = make_grid(CFG['LON'], CFG['LAT'], CFG['RADIUS_KM'])
print('Grid: %d x %d piksel @ %d m, EPSG:%d' % (g['n'], g['n'], g['res'], g['epsg']))
D = fetch(ee, g, CFG)
for k in ['forest2020', 'palm', 'lossyear', 'datamask']:
    print(k, 'unique values:', np.unique(D[k])[:30])
assert (D['aR'] > 0).mean() > 0.5, 'Sentinel-2 AFTER composite is mostly empty'

os.makedirs('ndpe_output', exist_ok=True)
cls, S, patches, per_year = analyse(D, g, CFG)
map_figure(D, g, cls, S, patches, CFG, 'ndpe_output/01_map_ndpe_screening.png')
chart_figure(per_year, CFG, 'ndpe_output/02_chart_loss_per_year.png')
has_v = verify_figure(D, g, cls, patches, CFG, 'ndpe_output/03_verification_before_after.png')
if not has_v: print('No deforestation patch >= %g ha' % CFG['MMU_HA'])

import csv, shutil
txt = summary_text(S, patches, per_year, CFG, g)
print(txt)
open('ndpe_output/04_analisis.txt', 'w', encoding='utf-8').write(txt)
json.dump(dict(config=CFG, stats=S, patches=patches[:50]), open('ndpe_output/05_stats.json', 'w'), indent=1, default=float)
with open('ndpe_output/06_loss_per_year.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['year', 'loss_up_to_cutoff_ha', 'deforestation_ha', 'replanting_in_oil_palm_ha', 'other_nonforest_ha'])
    for y in sorted(per_year): w.writerow([y, per_year[y]['before'], per_year[y]['defo'], per_year[y]['repl'], per_year[y]['other']])
with open('ndpe_output/07_deforestation_patches.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['rank', 'area_ha', 'main_year', 'lat', 'lon'])
    for i, p in enumerate(patches, 1): w.writerow([i, p['area_ha'], p['main_year'], p['lat'], p['lon']])
shutil.make_archive('ndpe_output', 'zip', 'ndpe_output')
try:
    from google.colab import files; files.download('ndpe_output.zip')
except Exception: pass
