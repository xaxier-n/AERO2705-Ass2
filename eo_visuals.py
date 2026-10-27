"""Block A presentation and notebook scaffolding; eo_toolkit N2.4; explicit repeat-design teaching model.
Matplotlib/NumPy/pandas core, optional ipywidgets. No map network calls.
Figures return Figure objects; callbacks display/close them to avoid duplicates.
"""
from copy import deepcopy
from functools import lru_cache
import gzip
import html
import io
import json
import math
from pathlib import Path
import tempfile
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Arc
import eo_toolkit as eo

__version__ = 'V1.9'
BLUE, ORANGE, GREEN, INK = '#1f6fb2', '#c74924', '#197760', '#233544'
MEAN = 'mean potential-access estimate'
METRICS = [MEAN, 'maximum acquisition gap', 'request-to-acquisition wait',
           'delivery latency', 'regional completion time']
MAP_PATH = Path(__file__).resolve().parent/'data'/'coastlines_50m.json.gz'
LIMIT = 'Simplified latitude-circle sizing only; no illumination, cloud, tasking, area completion or latency.'
METRIC_GUIDANCE = {
    'maximum acquisition gap': ('Maximum acquisition gap not calculated',
        'A mean does not establish the longest gap. Evaluate target access events with illumination and operating constraints.'),
    'request-to-acquisition wait': ('Request waiting time not calculated',
        'Waiting depends on request arrival relative to eligible opportunities. Evaluate a request time or a specified distribution of request times.'),
    'delivery latency': ('Delivery latency not calculated',
        'Acquisition timing alone is insufficient. Include tasking, eligible acquisition, downlink and processing in the delivery timeline.'),
    'regional completion time': ('Regional completion time not calculated',
        'A representative target is not the whole region. Define its boundary and evaluate the union of usable acquisitions over time.')}
REFERENCE = dict(h_km=500., inc_deg=97.4, ltan_hours=10.5, gsd_m=5.,
                 nominal_swath_km=60., fov_deg=6.9, cross_track_pixels=12000,
                 bands=5, bits=12, agility_deg=30., stations=1,
                 target_lat_deg=-34., target_lon_deg=151.)


def reference():
    """Fresh copy so an editable variant cannot mutate the locked baseline."""
    return deepcopy(REFERENCE)


def new_session():
    return dict(rows=[], mission={}, notes={}, controls={}, token=uuid.uuid4().hex[:10])


def show(fig):
    """One static notebook image, independent of the active Matplotlib backend."""
    from IPython.display import Image, display
    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer,format='png',dpi=fig.dpi,bbox_inches='tight',
                        facecolor='white',transparent=False)
            picture=Image(data=buffer.getvalue())
    finally:
        # Deregister before publishing so an end-of-cell flush cannot repeat it.
        plt.close(fig)
    display(picture)


def cards(items):
    """Display short metric cards. All dynamic text is escaped."""
    from IPython.display import HTML, display
    cells=''.join('<div style="flex:1;min-width:150px;padding:12px;border:1px solid #b9cbd6;'
                  'border-top:4px solid '+color+';border-radius:6px;background:#f5f8fa;color:#233544">'
                  '<div style="font-size:13px;color:#233544">'+html.escape(label)+'</div>'
                  '<div style="font-size:24px;font-weight:600;color:#233544">'+html.escape(str(value))+'</div></div>'
                  for label,value,color in items)
    display(HTML('<div style="display:flex;flex-wrap:wrap;gap:10px;margin:12px 0">'+cells+'</div>'))


@lru_cache(maxsize=2)
def _coastlines(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        data=json.load(f)
    segments=[]
    for line in data['lines']:
        p=np.asarray(line,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2 or not np.all(np.isfinite(p)):
            raise ValueError('Invalid coastline coordinates')
        # Never connect across the dateline; replicate only the visible segments.
        for q in np.split(p,np.flatnonzero(np.abs(np.diff(p[:,0]))>180)+1):
            if len(q)>1:segments.append(q)
    return segments


def coastlines(ax, path=None):
    """Local context layer, preserving caller limits; no auto-download on failure."""
    xlo,xhi=ax.get_xlim();ylo,yhi=ax.get_ylim()
    try:
        lines=_coastlines(str(MAP_PATH if path is None else Path(path)))
    except (OSError,ValueError,KeyError,TypeError) as err:
        ax.text(.02,.02,'Coastline asset unavailable — grid only.\nRe-extract the supplied tutorial bundle.',
                transform=ax.transAxes,fontsize=8,color=INK,
                bbox=dict(facecolor='white',alpha=.95,edgecolor='#ccd6dd'),zorder=20)
        return False
    visible=[]
    for p in lines:
        if p[:,1].max()<ylo or p[:,1].min()>yhi:continue
        for k in range(math.ceil((xlo-p[:,0].max())/360),math.floor((xhi-p[:,0].min())/360)+1):
            visible.append(p+[k*360,0])
    ax.add_collection(LineCollection(visible,colors='#536a78',linewidths=.65,zorder=3))
    ax.set_xlim(xlo,xhi);ax.set_ylim(ylo,yhi)
    return True


def case_result(config, *, nominal=False):
    """Geometry first; supported mean or a reason. No deadline calculation."""
    c=config
    lat=eo._number('target latitude [deg]',c['target_lat_deg'],-90,90)
    eo._number('target longitude [deg]',c['target_lon_deg'],-180,180)
    orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    width=c['nominal_swath_km'] if nominal else eo.access_width(c['h_km'],c['agility_deg'],fov_deg=c['fov_deg'])
    limit=eo.max_latitude(orb,width)
    out=dict(width_km=width,latitude_bound_deg=limit,mean_days=None,geographic_reach=abs(lat)<=limit,
             reason='',model=eo.CIRCLE_MODEL,metric=MEAN)
    try:
        out['mean_days']=eo.revisit_mean(lat,width,orb)['t_days']
        out['reason']='Mean estimate available within the teaching model'
    except eo.ModelScopeError as err:out['reason']=str(err)
    return out


def _signed_distance(h,angle):
    return np.sign(angle)*eo.ground_reach(h,abs(angle))


def pointing_values(roll, config=None):
    roll=eo._number('pointing limit [deg]',roll,0,30)
    c=reference() if config is None else config
    h,fov=c['h_km'],c['fov_deg']
    left,right=(_signed_distance(h,roll+s*fov/2) for s in (-1,1))
    return dict(roll=roll,centre_km=eo.ground_reach(h,roll),left_km=left,right_km=right,
                strip_km=right-left,width_km=2*right,nadir_km=eo.nadir_swath(h,fov))


def pointing_figure(roll=None, latitude=None, target_days=2., *, config=None):
    """Same fixed axes for each setting; full-FOV geometry plus mean constraint."""
    c=reference() if config is None else deepcopy(config)
    roll=c['agility_deg'] if roll is None else roll
    latitude=c['target_lat_deg'] if latitude is None else latitude
    v=pointing_values(roll,c)
    h,fov=c['h_km'],c['fov_deg'];scale=h/500
    lat=eo._number('target latitude',latitude,-70,70)
    target=eo._number('target mean [days]',target_days,1,40)
    fig,(ax,curve)=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={'width_ratios':[1.05,1]},layout='constrained')
    def surface(d):
        g=np.asarray(d)/eo.R_E
        return np.column_stack([eo.R_E*np.sin(g),eo.R_E*np.cos(g)-eo.R_E])
    earth=surface(np.linspace(-460*scale,460*scale,300))
    ax.fill_between(earth[:,0],-145*scale,earth[:,1],color='#edf3f4')
    ax.plot(*earth.T,color='#809ba3',lw=1.5)
    sat=np.array([0,h]);nadir=surface([-v['nadir_km']/2,v['nadir_km']/2])
    selected=surface([v['left_km'],v['right_km']])
    ax.add_patch(Polygon(np.vstack([sat,nadir]),color=BLUE,alpha=.12))
    ax.add_patch(Polygon(np.vstack([sat,selected]),color=GREEN,alpha=.2))
    ax.plot([0,0],[0,h],ls=':',color=BLUE,lw=1)
    centre=surface([v['centre_km']])[0]
    ax.plot([0,centre[0]],[h,centre[1]],color=GREEN,lw=1.5)
    for pt in selected:ax.plot([0,pt[0]],[h,pt[1]],color=GREEN,lw=.8,alpha=.7)
    # Potential envelope is an arc; the selected strip is a small part of it.
    envelope=surface(np.linspace(-v['width_km']/2,v['width_km']/2,100))
    ax.plot(envelope[:,0],envelope[:,1]-8,color=ORANGE,lw=4,alpha=.55)
    ax.plot(nadir[:,0],nadir[:,1]+5,color=BLUE,lw=6,solid_capstyle='butt')
    arc=surface(np.linspace(v['left_km'],v['right_km'],30))
    ax.plot(arc[:,0],arc[:,1]+10,color=GREEN,lw=6,solid_capstyle='butt')
    ax.scatter(0,h,marker='s',s=65,color=INK,zorder=5)
    ax.text(-20*scale,535*scale,'Spacecraft',ha='center',fontsize=10,color=INK)
    ax.annotate('',(-70*scale,0),(-70*scale,h),arrowprops=dict(arrowstyle='<->',color=INK))
    ax.text(-88*scale,245*scale,f'h = {h:g} km',rotation=90,ha='right',va='center',fontsize=9)
    if roll>0:
        ax.add_patch(Arc((0,h),190*scale,190*scale,theta1=270,theta2=270+roll,color=GREEN,lw=1.5))
    ax.text(27*scale,366*scale,f'η = {roll:g}°',color=GREEN,fontsize=10)
    ax.annotate(f'Selected strip\n{v["strip_km"]:.1f} km',xy=(centre[0],centre[1]+12),
                xytext=(200*scale,155*scale),ha='center',fontsize=10,color=GREEN,
                arrowprops=dict(arrowstyle='-',color=GREEN))
    ax.annotate(f'Nadir strip ≈{v["nadir_km"]:.1f} km',xy=(0,4),xytext=(-430*scale,105*scale),fontsize=9,color=BLUE,
                arrowprops=dict(arrowstyle='-',color=BLUE))
    ax.annotate('',(0,-47),(centre[0],-47),arrowprops=dict(arrowstyle='<->',color=GREEN))
    ax.text(centre[0]/2,-68,f'd(η) = {v["centre_km"]:.1f} km',ha='center',fontsize=9,color=GREEN)
    ax.text(0,-120,f'Potential envelope W = {v["width_km"]:.1f} km',ha='center',fontsize=10,color=ORANGE)
    ax.set_xlim(-470*scale,470*scale);ax.set_ylim(-155*scale,585*scale);ax.set_aspect('equal');ax.axis('off')
    ax.set_title('1  Pointing selects one strip',loc='left',color=INK,weight='bold',fontsize=12)
    orb=eo.orbit_from_altitude(h,c['inc_deg'])
    angles=np.linspace(0,30,121)
    means=np.array([eo.revisit_mean(lat,eo.access_width(h,e,fov_deg=fov),orb)['t_days'] for e in angles])
    current=eo.revisit_mean(lat,v['width_km'],orb)['t_days']
    curve.plot(angles,means,color=ORANGE,lw=2.5,label='Mean sizing estimate')
    curve.scatter([roll],[current],color=GREEN,s=65,zorder=5,label=f'Selected: {current:.2f} d')
    curve.axhline(target,color=INK,ls='--',label=f'Target mean: {target:g} d')
    curve.axvline(30,color='#738390',ls=':')
    curve.set(xlim=(0,32),ylim=(0,45),xlabel='Permitted pointing magnitude η [deg]',ylabel='Mean potential access [days]')
    curve.set_title('2  Does the mean meet the target?',loc='left',color=INK,weight='bold',fontsize=12)
    inverse=eo.eta_for_revisit(target,lat,orb,fov_deg=fov,eta_cap=30)
    angle=inverse['eta_deg']
    message=(f'Required η = {angle:.1f}° — '+('within ±30°' if inverse['feasible'] else 'outside ±30°')) if np.isfinite(angle) else inverse['reason']
    curve.text(.04,.92,message,transform=curve.transAxes,va='top',fontsize=10,
               bbox=dict(facecolor='#f3f6f8',edgecolor='none',pad=6))
    curve.grid(alpha=.18);curve.legend(loc='upper right',bbox_to_anchor=(1,.81),fontsize=9)
    fig.suptitle(f'Fixed sensor · full FOV {fov:g}° · latitude {lat:+g}°',fontsize=11,color=INK)
    fig.supxlabel(f'Geometry uses {v["nadir_km"]:.3f} km at zero roll; nominal reference accounting uses 60 km.\nW is possible pointing access, not one acquired image. Mean ≠ deadline.',fontsize=9,color=INK)
    return fig


def map_figure(days=1., mode='potential', latitude=-34., longitude=151., roll=30., *, map_path=None, show_descending=True):
    """Ascending strips and optional grey descending outlines; arbitrary epoch."""
    if mode not in ('nadir','potential'):raise ValueError('Choose nadir or potential')
    days=eo._number('map duration [days]',days,.001,36)
    v=pointing_values(roll);width=60. if mode=='nadir' else v['width_km']
    fig,ax=plt.subplots(figsize=(10.5,5.5),layout='constrained')
    color=BLUE if mode=='nadir' else ORANGE
    eo.plot_coverage(eo.orbit_from_altitude(500,97.4),days,width,(latitude,longitude),ax=ax,
                     alpha=.3 if mode=='potential' else .5,color=color,
                     distinguish_branches=True,show_descending=show_descending,
                     title=f'{days:g} days · '+('nominal nadir strips' if mode=='nadir' else f'potential envelope at ±{roll:g}°'))
    ax.set_facecolor('#f4f8fa');coastlines(ax,map_path)
    ax.annotate('Representative site',(eo._wrap(longitude),latitude),xytext=(9,9),textcoords='offset points',fontsize=9,
                color=INK,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=color,lw=6,alpha=.6,
                   label=f'Ascending · LTAN 10:30 · {width:.1f} km '+('strip' if mode=='nadir' else 'possible band'))]
    if show_descending:
        handles.append(Line2D([0],[0],color='#8a939a',lw=.8,alpha=.65,label='Descending · LTDN 22:30 · return-pass outlines'))
    handles.append(Line2D([0],[0],color='#536a78',lw=1,label='Coastline context'))
    ax.legend(handles=handles,loc='lower left',fontsize=9,framealpha=.95)
    fig.supxlabel('Ascending: reference imaging branch; arrows show travel direction. Branch selection is not a daylight filter.\nOne nominal strip/revolution in the mean/circle calculations; this display does not change those estimates.\nNatural Earth 1:50 million · no sunlight/tasking or unique-area calculation',fontsize=9,color=INK)
    return fig


def assessment(config, metric=MEAN, required_days=None):
    """Formative evidence for a candidate; never an overall mission verdict."""
    if metric not in METRICS:raise ValueError('Choose one of the listed temporal metrics')
    if required_days is not None:
        required_days=eo._number('requirement [days]',required_days,positive=True)
    result=case_result(config)
    mean=result['mean_days']
    margin=None
    if not result['geographic_reach']:
        status='Outside geographic reach'
        next_check='Reconsider orbit inclination or the target; confirm reach before computing target access times.'
    elif mean is None:
        status='Statistics outside teaching scope'
        next_check='Use an appropriate target-access model for this latitude/orbit; do not extrapolate the mean curve.'
    elif metric!=MEAN:
        status=METRIC_GUIDANCE[metric][0]
        next_check=METRIC_GUIDANCE[metric][1]
    elif required_days is None:
        status='Mean calculated; requirement not supplied'
        next_check='Specify and justify a mean-interval requirement before comparing candidates.'
    else:
        margin=required_days-mean
        status='Mean meets requirement within model' if margin>=0 else 'Mean exceeds requirement within model'
        if margin>=0:
            next_check='Test eligible target acquisitions and the other mission constraints before claiming operational feasibility.'
        elif config['agility_deg']<30:
            next_check='Compare another pointing setting within the reference ±30° limit, keeping the requirement fixed. Then consider image-quality costs.'
        else:
            next_check='The reference pointing cap is reached. Investigate constellation arrangements or a different orbit with a suitable access model; justify any requirement revision.'
    requested_note=METRIC_GUIDANCE[metric][1] if metric!=MEAN else 'A mean comparison does not establish a deadline, full regional coverage or usable imagery.'
    return dict(result=result,status=status,metric=metric,required_days=required_days,
                difference_days=margin,next_check=next_check,limitation=requested_note)


def feedback(session_config,metric=MEAN,required_days=None):
    """Plain, metric-specific evidence and next action; inputs remain student choices."""
    from IPython.display import HTML,display
    a=assessment(session_config,metric,required_days);r=a['result'];c=session_config
    reach=(f'Latitude within ±{r["latitude_bound_deg"]:.1f}° reach bound. This does not establish a target pass.'
           if r['geographic_reach'] else f'Target latitude lies outside ±{r["latitude_bound_deg"]:.1f}° reach bound.')
    sections=[('Candidate',f'{c["h_km"]:g} km, inclination {c["inc_deg"]:g}°, full FOV {c["fov_deg"]:g}°, pointing ±{c["agility_deg"]:g}°; target {c["target_lat_deg"]:+g}°, {c["target_lon_deg"]:+g}°.'),
              ('Reach and model',reach+' '+r['reason']),
              ('What this establishes',a['status']+'. '+a['limitation']),
              ('Next investigation',a['next_check'])]
    blocks=''.join('<div style="padding:10px;background:#f5f8fa;color:#233544;border:1px solid #dbe3e9;border-radius:5px">'
        '<strong style="color:#233544">'+html.escape(title)+'</strong><p style="margin:5px 0 0;color:#233544">'+html.escape(body)+'</p></div>' for title,body in sections)
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;margin:10px 0">'+blocks+'</div>'))
    return a


def reach_figure(latitude=80.,longitude=151.,roll=30., *, metric=MEAN,required_days=None,config=None):
    """Full-width world context plus a supported mean curve and requirement check.

    Boundary lines are not coverage areas. No swath/area completion is inferred.
    """
    c=reference() if config is None else deepcopy(config)
    c.update(target_lat_deg=latitude,target_lon_deg=longitude,agility_deg=roll)
    a=assessment(c,metric,required_days);result=a['result'];bound=result['latitude_bound_deg']
    orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    fig=plt.figure(figsize=(12,8.4),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[2.3,1.25],width_ratios=[1.35,1])
    ax=fig.add_subplot(grid[0,:]);curve=fig.add_subplot(grid[1,0]);info=fig.add_subplot(grid[1,1]);info.axis('off')
    ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='Longitude [deg]',ylabel='Latitude [deg]')
    ax.set_facecolor('#f4f8fa')
    for j,y in enumerate([-bound,bound]):
        ax.axhline(y,color=ORANGE,ls='--',lw=1.3,label=f'Latitude reach bound ±{bound:.1f}°' if j==0 else None)
    for j,y in enumerate([-70,70]):
        ax.axhline(y,color=BLUE,ls=':',lw=1.3,label='Latitude-circle support cutoff ±70°' if j==0 else None)
    coastlines(ax);ax.plot(longitude,latitude,'k*',ms=12,zorder=6,label='Selected target');ax.grid(alpha=.15)
    ax.set_title('World context: boundaries are not mapped coverage',loc='left',fontsize=12,color=INK)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=9,frameon=False)
    lats=np.linspace(-90,90,181);means=[]
    for lat in lats:
        try:value=eo.revisit_mean(lat,result['width_km'],orb)['t_days']
        except eo.ModelScopeError:value=np.nan
        means.append(value)
    means=np.array(means)
    curve.plot(lats,means,color=ORANGE,lw=2,label=f'Mean estimate · ±{roll:g}° pointing')
    for y in [-70,70]:curve.axvline(y,color=BLUE,ls=':',lw=1)
    for y in [-bound,bound]:curve.axvline(y,color=ORANGE,ls='--',lw=1)
    curve.axvline(latitude,color=INK,alpha=.35,lw=1)
    if result['mean_days'] is not None:
        curve.scatter([latitude],[result['mean_days']],s=65,color=GREEN,zorder=6,label='Selected latitude')
    elif not np.any(np.isfinite(means)):
        curve.text(.5,.5,'No supported statistical curve for this orbit',transform=curve.transAxes,ha='center',wrap=True)
    if metric==MEAN and required_days is not None:
        curve.axhline(required_days,color=INK,ls='--',label=f'Required mean ≤{required_days:g} d')
    # Stable scale for the chosen orbit/sensor, including the zero-pointing case.
    try:top=eo.revisit_mean(0,eo.access_width(c['h_km'],0,fov_deg=c['fov_deg']),orb)['t_days']*1.06
    except eo.ModelScopeError:top=45
    curve.set(xlim=(-90,90),ylim=(0,max(45,top,(required_days or 0)*1.15 if metric==MEAN else 0)),
              xlabel='Target latitude [deg]',ylabel='Mean interval [days]')
    curve.set_title('Supported estimate changes with latitude and pointing',fontsize=10,loc='left',color=INK)
    curve.grid(alpha=.15);curve.legend(fontsize=8,loc='upper right')
    import textwrap
    info.text(0,.98,'Result versus requirement',va='top',fontsize=12,weight='bold',color=INK)
    wanted=f'≤{required_days:g} days' if required_days is not None else 'not specified'
    mean_text=f'{result["mean_days"]:.2f} days' if result['mean_days'] is not None else 'not available within model'
    info.text(0,.81,textwrap.fill(f'Requested: {metric} · {wanted}',47),va='top',fontsize=10,color=INK)
    info.text(0,.60,f'Calculated mean: {mean_text}',va='top',fontsize=10,color=BLUE)
    info.text(0,.43,textwrap.fill(a['status'],44),va='top',fontsize=11,color=GREEN if a['difference_days'] is not None and a['difference_days']>=0 else ORANGE)
    if a['difference_days'] is not None:
        delta=a['difference_days'];prefix='Headroom' if delta>=0 else 'Exceeds limit by'
        info.text(0,.20,f'{prefix}: {abs(delta):.2f} days',va='top',fontsize=11,color=INK)
        info.text(0,.06,'Comparison uses unrounded values.',va='top',fontsize=9,color=INK)
    else:info.text(0,.17,('This metric has not been calculated.' if metric!=MEAN else 'No supported comparison is available.')+'\nRead the explanation and next investigation below.',va='top',fontsize=9,color=INK)
    fig.supxlabel('Lines mark necessary reach and model-support limits, not swath coverage. Longitude is context for this mean model.\nThe requirement changes the comparison, not the spacecraft. Natural Earth coastlines · no illumination or area-union calculation',fontsize=9,color=INK)
    return fig


def return_figure(roll=None,latitude=None, *,config=None):
    """Simplified latitude-circle events for explicit worked inputs; no new slider."""
    c=reference() if config is None else deepcopy(config)
    roll=c['agility_deg'] if roll is None else roll
    latitude=c['target_lat_deg'] if latitude is None else latitude
    orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    widths=[c['nominal_swath_km'],eo.access_width(c['h_km'],roll,fov_deg=c['fov_deg'])]
    fig,axs=plt.subplots(2,1,figsize=(11,4),sharex=True,layout='constrained')
    for ax,width,label,color in zip(axs,widths,['Nominal nadir',f'Pointing envelope ±{roll:g}°'],[BLUE,ORANGE]):
        stats=eo.circle_return_stats(latitude,width,orb)
        times=np.r_[0,np.cumsum(stats['waits'])];hits=times[times<=120]
        ax.eventplot(hits,lineoffsets=0,linelengths=.5,colors=color,linewidths=2)
        ax.set(ylim=(-.7,.9),yticks=[],xlim=(0,120))
        ax.set_title(f'{label} · sample mean {stats["mean_days"]:.2f} d · largest sampled interval {stats["max_sampled_interval_days"]:.2f} d',loc='left',fontsize=10)
        request=20.;future=times[times>=request]
        if len(future):
            next_hit=future[0]
            ax.plot(request,-.38,marker='^',color=INK,ms=6)
            ax.annotate('',(request,-.38),(min(next_hit,120),-.38),arrowprops=dict(arrowstyle='<->',color=INK))
            ax.text((request+min(next_hit,120))/2,-.65,f'Illustrative request wait: {next_hit-request:.1f} d',ha='center',fontsize=9)
        ax.grid(axis='x',alpha=.2)
    axs[-1].set_xlabel('Days from a notional latitude-circle hit at t = 0')
    fig.suptitle('Hits, intervals and a request are different objects',fontsize=13,color=INK)
    years=stats['sample_duration_days']/365.24
    fig.supxlabel(f'Simplified latitude-circle model · {c["h_km"]:g} km / {c["inc_deg"]:g}° · target latitude {latitude:+g}°\nFirst 120 days shown; 200,000 revolutions sampled (~{years:.1f} years), not operating lifetime or a deadline guarantee.',fontsize=9)
    return fig


def sampled_interval_figure(roll=15.,latitude=-34., *,config=None):
    """Explore E6 detail at one selected setting; sampled extrema, not bounds."""
    c=reference() if config is None else deepcopy(config)
    orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    width=eo.access_width(c['h_km'],roll,fov_deg=c['fov_deg'])
    stats=eo.circle_return_stats(latitude,width,orb)
    minimum=min(stats['waits']);maximum=stats['max_sampled_interval_days']
    sizing=eo.revisit_mean(latitude,width,orb)['t_days']
    times=np.r_[0,np.cumsum(stats['waits'])]
    fig,ax=plt.subplots(figsize=(11,2.4),layout='constrained')
    ax.eventplot(times[times<=60],lineoffsets=0,linelengths=.5,colors=ORANGE)
    ax.set(xlim=(0,60),ylim=(-.6,.7),yticks=[],xlabel='First 60 days from a notional hit; ticks are model opportunities')
    ax.set_title(f'Selected ±{roll:g}° · sampled min / mean / max: {minimum:.2f} / {stats["mean_days"]:.2f} / {maximum:.2f} days',loc='left',fontsize=11,color=INK)
    ax.grid(axis='x',alpha=.2)
    fig.supxlabel(f'Analytical sizing mean: {sizing:.2f} days. Sample statistics use 200,000 revolutions.\nSimplified latitude-circle model; sampled extrema are not guaranteed best/worst mission performance.',fontsize=9,color=INK)
    return fig


def sampled_range_figure():
    """Optional extension only; distinguish samples from analytical sizing."""
    c=reference();orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    angles=np.arange(31);minimum=[];maximum=[];mean=[];sizing=[]
    for angle in angles:
        width=eo.access_width(c['h_km'],angle,fov_deg=c['fov_deg'])
        stat=eo.circle_return_stats(c['target_lat_deg'],width,orb)
        minimum.append(min(stat['waits']));maximum.append(stat['max_sampled_interval_days']);mean.append(stat['mean_days'])
        sizing.append(eo.revisit_mean(c['target_lat_deg'],width,orb)['t_days'])
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.fill_between(angles,minimum,maximum,color=ORANGE,alpha=.15,label='Sampled min–max (not guaranteed bounds)')
    ax.plot(angles,sizing,color=ORANGE,lw=2,label='Analytical sizing mean')
    ax.plot(angles,mean,'.',color=INK,label='Sample mean')
    ax.set(xlabel='Permitted pointing magnitude [deg]',ylabel='Interval [days]',xlim=(0,30),ylim=(0,max(maximum)*1.15))
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.supxlabel('Simplified latitude-circle model · locked reference / latitude −34° · 200,000 revolutions per setting\nThe shaded region is sampled interval variability, not uncertainty in the mean or a deadline guarantee.',fontsize=9)
    return fig


def run_orbit_experiment(session,h_km=700.,latitude=-34.,roll=30.,required_days=3.,interpretation=''):
    """Optional circular SSO candidate; derive inclination, preserve reference."""
    h=eo._number('extension altitude [km]',h_km,200,1500)
    roll=eo._number('pointing limit [deg]',roll,0,30)
    c=reference();c.update(h_km=h,inc_deg=eo.sso_inclination(h),target_lat_deg=latitude,agility_deg=roll)
    c['nominal_swath_km']=eo.nadir_swath(h,c['fov_deg'])
    c['gsd_m']=None  # Ground sampling is not re-evaluated in this orbit extension.
    row=record_case(session,'Optional: circular SSO',c,required=required_days,notes=interpretation)
    orb=eo.orbit_from_altitude(h,c['inc_deg'])
    cards([('Derived SSO inclination',f'{c["inc_deg"]:.3f}°',BLUE),('Orbital period',f'{orb["T"]/60:.2f} min',INK),
           ('Nadir strip with fixed FOV',f'{eo.nadir_swath(h,c["fov_deg"]):.1f} km',ORANGE)])
    show(reach_figure(latitude,c['target_lon_deg'],roll,required_days=required_days,config=c))
    feedback(c,required_days=required_days);show_scorecard(session)
    print('Fixed angular FOV: swath is recalculated at the new altitude. Candidate GSD is not evaluated; it is saved as unset.')
    print('Inclination uses the circular first-order J2 SSO relation. LTAN is nominal metadata, not an epoch-to-Sun calculation.')
    return row


def record_case(session, name, config, *, required=None, metric=MEAN, notes='', nominal=False):
    """Append a revision when evidence changes; exact reruns preserve the record."""
    if metric not in METRICS:raise ValueError('Choose one of the listed temporal metrics')
    if required is not None:required=eo._number('requirement [days]',required,positive=True)
    result=case_result(config,nominal=nominal)
    payload=dict(inputs=deepcopy(config),nominal_accounting=nominal,requirement_metric=metric,
                 requirement_days=required,interpretation=notes)
    existing=[r for r in session['rows'] if r['mission_id']==name and r['tension']=='T1']
    if existing and existing[-1]['configuration']==payload:
        return existing[-1]
    revision=1+max([r['revision'] for r in existing],default=0)
    eo.add_row(session['rows'],tension='T1',name=name,lecture='Lectures 1–2',quantity=MEAN,unit='days',
               mission_id=name,revision=revision,mine=result['mean_days'],required=required,comparison='<=',
               required_quantity=metric,required_unit='days',configuration=payload,
               evidence_status='evaluated' if result['mean_days'] is not None else 'not established',
               assumptions=LIMIT+' '+result['reason'],fix=notes,bills='Image quality (T2); geolocation (T4)')
    return session['rows'][-1]


def scorecard(session):
    """Compact latest-revision display; all revisions remain in the saved rows."""
    frame=eo.scorecard_frame(session['rows'])
    latest={}
    for _,r in frame.iterrows():latest[(r['mission_id'],r['tension'],r['quantity'])]=r
    rows=[]
    for r in latest.values():
        c=r['configuration']['inputs']
        status=r['status']
        if status.startswith('not established') and r['required_quantity'] in METRIC_GUIDANCE:
            status='not established: '+METRIC_GUIDANCE[r['required_quantity']][0].lower()
        if r['required'] is None or pd.isna(r['required']):
            status='Result recorded; requirement unset' if r['mine'] is not None and not pd.isna(r['mine']) else status
        rows.append({'Case':r['name'],'Rev':r['revision'],'Latitude °':c['target_lat_deg'],
                     'Pointing °':c['agility_deg'],'Mean estimate [d]':r['mine'],
                     'Requirement metric':r['required_quantity'],'Limit [d]':r['required'],'Evidence':status})
    return pd.DataFrame(rows)


def show_scorecard(session):
    from IPython.display import HTML, display
    table=scorecard(session)
    if table.empty:
        print('No evidence recorded yet. Run W1 to start the scorecard.')
        return
    # Notebook renderers may style th/td directly. Pair colours on each cell;
    # inherited dark text over a transparent header fails in dark editors.
    headers=''.join('<th scope="col" style="padding:8px;text-align:left;border-bottom:2px solid #91a9b8;background:#e4edf3;color:#233544">'+html.escape(c)+'</th>' for c in table.columns)
    body=[]
    for _,row in table.iterrows():
        cells=[]
        for column,value in row.items():
            if pd.isna(value):text='—'
            elif isinstance(value,(int,float,np.number)):
                text=f'{value:.2f}' if column in ('Mean estimate [d]','Limit [d]') else f'{value:g}'
            else:text=str(value)
            color='#f5f8fa'
            if column=='Evidence':
                color='#e5f4ed' if text.startswith('meets') else '#fff0e7' if text.startswith('does not meet') else '#edf1f5'
            cells.append('<td style="padding:8px;text-align:left;vertical-align:top;border-bottom:1px solid #dbe3e9;color:#233544;background:'+color+'">'+html.escape(text)+'</td>')
        body.append('<tr>'+''.join(cells)+'</tr>')
    display(HTML('<div style="overflow-x:auto;margin:12px 0"><table style="border-collapse:collapse;font-size:13px;width:100%;background:#f5f8fa;color:#233544">'
                 '<thead><tr>'+headers+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'))
    print('Display rounding only. Decisions use full precision; all revisions are saved.')


def baseline(session):
    c=reference();orb=eo.orbit_from_altitude(c['h_km'],c['inc_deg'])
    drift=eo.ground_track_shift(orb)
    cards([('Orbit period',f'{orb["T"]/60:.2f} min',BLUE),('Revolutions / solar day',f'{orb["revs_solar"]:.2f}',INK),
           ('Equatorial track shift',f'{drift["d_km"]:,.0f} km west',ORANGE)])
    c['agility_deg']=0;record_case(session,'Worked: nadir',c,nominal=True)
    c['agility_deg']=30;record_case(session,'Worked: agile',c)
    # Hidden numerical tolerance is a check on the supplied example, not a mission allowance.
    checks=[np.isclose(orb['T']/60,94.61,atol=.01,rtol=0),
            np.isclose(case_result(c)['width_km'],673,atol=.5,rtol=0)]
    print('Reference self-check: '+('OK — rounded lecture values reproduced.' if all(checks) else 'CHECK — restore the locked inputs/toolkit.'))
    show_scorecard(session)


def run_pointing(session,roll=15.,target_days=3.,prediction='',interpretation='',show_samples=False):
    show(pointing_figure(roll,-34,target_days))
    if show_samples:show(sampled_interval_figure(roll))
    c=reference();c['agility_deg']=roll
    row=record_case(session,'Your turn: pointing',c,required=target_days,
                    notes=f'Prediction: {prediction}\nInterpretation: {interpretation}')
    session['controls']['pointing']=dict(roll=roll,target_days=target_days,prediction=prediction,interpretation=interpretation,show_samples=show_samples)
    show_scorecard(session)
    return row


def run_target(session,latitude=80.,longitude=151.,roll=30.,metric=MEAN,required_days=3.,prediction=''):
    c=reference();c.update(target_lat_deg=latitude,target_lon_deg=longitude,agility_deg=roll)
    assessment(c,metric,required_days) # Validate the requirement before plotting or recording.
    show(reach_figure(latitude,longitude,roll,metric=metric,required_days=required_days))
    row=record_case(session,'Your turn: location',c,required=required_days,metric=metric,notes=prediction)
    session['controls']['target']=dict(latitude=latitude,longitude=longitude,roll=roll,metric=metric,required_days=required_days,prediction=prediction)
    feedback(c,metric,required_days);show_scorecard(session)
    return row


def run_map(days=1.,mode='potential',latitude=-34.,longitude=151.,roll=30.,show_descending=True):
    show(map_figure(days,mode,latitude,longitude,roll,show_descending=show_descending))


def constellation_cases(plane_separation=15.):
    """Controlled three-case comparison; all angles at a common initial epoch.

    The last case has adjacent RAAN offsets separated by plane_separation.
    Local mean node times are nominal SSO conventions relative to LTAN 10:30,
    not sunlight calculations for the arbitrary map epoch.
    """
    separation=eo._number('adjacent plane separation [deg]',plane_separation,15,120)
    cases=[dict(name='One satellite',raan_deg=[0.],phase_deg=[0.]),
           dict(name='Three in one plane',raan_deg=[0.,0.,0.],phase_deg=[0.,120.,240.]),
           dict(name='Three different planes',raan_deg=[-separation,0.,separation],phase_deg=[0.,0.,0.])]
    for c in cases:
        c['ltan_hours']=[(REFERENCE['ltan_hours']+r/15)%24 for r in c['raan_deg']]
        c['ltdn_hours']=[(h+12)%24 for h in c['ltan_hours']]
    return cases


def _clock(hours):
    minutes=round(hours*60)%1440
    return f'{minutes//60:02d}:{minutes%60:02d}'


def constellation_figure(days=1.,plane_separation=15.):
    """Geometric opportunities only: same sensor, site, time and map window."""
    days=eo._number('elapsed time [days]',days,.001,10)
    cases=constellation_cases(plane_separation)
    orb=eo.orbit_from_altitude(REFERENCE['h_km'],REFERENCE['inc_deg'])
    fig=plt.figure(figsize=(12.6,6.3),layout='constrained')
    grid=fig.add_gridspec(2,3,height_ratios=[4,1.45])
    for j,c in enumerate(cases):
        ax=fig.add_subplot(grid[0,j]);info=fig.add_subplot(grid[1,j]);info.axis('off')
        options={'phase_in_plane':True} if j==1 else {'raan_offsets_deg':c['raan_deg']}
        eo.plot_coverage(orb,days,60,(-34,151),ax=ax,n_sats=len(c['raan_deg']),
                         alpha=.5,color=ORANGE,title=c['name'],distinguish_branches=True,**options)
        ax.set_facecolor('#f4f8fa');coastlines(ax)
        if j>0:ax.set_ylabel('')
        ax.tick_params(labelsize=9)
        def angles(values):return ', '.join(f'{v:g}°' for v in values)
        def times(values):return ' / '.join(_clock(v) for v in values)
        info.text(0,.98,'Initial phase u₀: '+angles(c['phase_deg']),va='top',fontsize=9,color=INK)
        info.text(0,.73,'RAAN offsets ΔΩ: '+angles(c['raan_deg']),va='top',fontsize=9,color=INK)
        info.text(0,.43,'LTAN  '+times(c['ltan_hours']),va='top',fontsize=10,color=BLUE)
        info.text(0,.16,'LTDN  '+times(c['ltdn_hours']),va='top',fontsize=10,color=GREEN)
    duration=f'{days:g} day' + ('' if days==1 else 's')
    fig.suptitle(f'Change the arrangement · {duration} · nominal 60 km nadir strips',fontsize=13,color=INK)
    fig.supxlabel('Ascending: spacecraft-coloured fills. Descending: grey outlines. Arrows show travel direction.\nBoth branches are geometric context; grey does not mean unusable in every plane. Check LTAN and LTDN.\nSame orbit/site · arbitrary epoch · no sunlight or usable-image revisit calculation · Natural Earth coastlines',fontsize=9,color=INK)
    return fig


def run_constellation(session,days=1.,plane_separation=15.,prediction='',interpretation=''):
    """Save a qualitative comparison alongside the scorecard, without inventing a mean."""
    fig=constellation_figure(days,plane_separation)
    cases=constellation_cases(plane_separation)
    show(fig)
    controls=dict(days=days,plane_separation=plane_separation,prediction=prediction,interpretation=interpretation)
    payload=dict(controls=deepcopy(controls),cases=cases,reference=reference(),
                 plotted_strip_km=60.,metric='geometric opportunity patterns',
                 limitation='No acquired imagery, unique-area metric or target revisit gain evaluated.')
    history=session['notes'].setdefault('constellation_history',[])
    if not history or history[-1]!=payload:history.append(payload)
    session['controls']['constellation']=controls
    period=eo.orbit_from_altitude(500,97.4)['T']
    print(f'120° orbital phasing gives {period/3/60:.2f} minutes between crossings of a common orbital phase.')
    print('Those crossings occur at different ground longitudes; this is not the revisit time of one target.')
    print(f'Comparison saved alongside the scorecard (revision {len(history)}). No numerical revisit gain has been assigned.')
    return payload


def _track_lines(ax, orb, times, color, *, descending=False, **kwargs):
    """Draw centre lines, split at dateline and latitude-turning branches."""
    q, velocity = eo._ground_state(orb, times, earth_rotation_deg_s=eo.REPEAT_OMEGA_E)
    lon = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1)))
    keep = velocity[:, 2] < 0 if descending else velocity[:, 2] >= 0
    cut = np.r_[False, np.abs(np.diff(lon)) > 180]
    lon = np.where(keep & ~cut, lon, np.nan)
    ax.plot(lon, lat, color=color, **kwargs)


def repeat_figure(h_km=500., *, show_fixed=True, show_descending=True):
    """Global two-cycle comparison and matched local node comparisons."""
    adjusted = eo.repeat_diagnostics(h_km)
    fixed = eo.repeat_diagnostics(h_km, adjust_sso=False)
    orb = adjusted['orbit']; sol = adjusted['solution']
    fig = plt.figure(figsize=(12, 9.5), layout='constrained', facecolor='white')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.7, 1, .3])
    world = fig.add_subplot(grid[0, :])
    world.set(xlim=(-180, 180), ylim=(-90, 90), xlabel='Longitude [°]', ylabel='Latitude [°]')
    world.set_facecolor('#f4f8fa'); coastlines(world)
    # Identical sampling relative to each five-day window makes coincidence visible.
    t = np.linspace(0, sol['days']*eo.D_SOLAR, 30001)
    for start, color, label, linestyle, width in [(0, BLUE, 'Days 0–5', '-', .65),
                                                 (sol['days']*eo.D_SOLAR, ORANGE, 'Days 5–10', '--', .65)]:
        if show_descending:
            _track_lines(world, orb, t+start, '#8a939a', descending=True, lw=.35, alpha=.25)
        _track_lines(world, orb, t+start, color, lw=width, ls=linestyle, alpha=.8, label=label)
    world.legend(loc='lower left', fontsize=9, framealpha=1)
    world.set_title(f'Candidate with SSO adjustment · h = {h_km:.3f} km · i = {orb["inc_deg"]:.5f}°\n'
                    'Ground-track centre lines; grey descending tracks are context, not acquisitions', fontsize=11, color=INK)
    cases = [adjusted, fixed] if show_fixed else [adjusted]
    axes = [fig.add_subplot(grid[1, k] if show_fixed else grid[1, :]) for k in range(len(cases))]
    extent = max(35., max(abs(c['node_error_km']) for c in cases)+25)
    for ax, case in zip(axes, cases):
        o = case['orbit']
        # Same argument of latitude near the first and 77th ascending nodes.
        local_t = np.linspace(-4, 4, 200)/o['n']
        for offset, color, label, style in [(0, BLUE, 'Initial node', '-'),
                 (sol['revolutions']*o['T'], ORANGE, 'After 76 revolutions', '--')]:
            q, _ = eo._ground_state(o, local_t+offset, earth_rotation_deg_s=eo.REPEAT_OMEGA_E)
            x = eo.R_E*np.arctan2(q[:, 1], q[:, 0])
            lat = np.degrees(np.arcsin(q[:, 2]))
            ax.plot(x, lat, color=color, ls=style, lw=2.2 if offset==0 else 1.6, label=label)
        heading = 'SSO inclination adjusted' if case['adjusted_sso'] else 'Inclination held at 97.4°'
        ax.set_title(f'{heading} · i = {o["inc_deg"]:.5f}°\n'
                     f'Node error {case["node_error_km"]:+.3f} km', fontsize=10, color=INK)
        ax.set(xlim=(-extent, extent), ylim=(-4, 4), xlabel='Longitude offset × Earth radius [km]', ylabel='Latitude [°]')
        ax.axhline(0, color=INK, lw=.5, alpha=.4); ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc='lower left'); ax.set_facecolor('white')
    gap = fig.add_subplot(grid[2, :])
    spacing = 2*np.pi*eo.R_E/sol['revolutions']
    strip = eo.nadir_swath(sol['h_km'], REFERENCE['fov_deg'])
    for centre in [0, spacing]:
        gap.axvspan(centre-strip/2, centre+strip/2, color=BLUE, alpha=.3)
        gap.axvline(centre, color=BLUE, lw=1.5)
    gap.text(spacing/2, .5, 'Repeating sparse tracks can preserve gaps', ha='center', va='center', color=INK)
    gap.set(xlim=(-70, spacing+70), ylim=(0, 1), yticks=[], xticks=[0, spacing],
            xticklabels=['Track A', 'Adjacent track'])
    gap.set_title(f'At the calculated repeat: {spacing:.0f} km equatorial spacing versus ≈{strip:.0f} km nadir width\n'
                  'Width comparison only, not a projected footprint or area-coverage calculation', fontsize=9, color=INK)
    fig.suptitle('W2 · Trim altitude, then check the pattern', fontsize=14, color=INK)
    fig.supxlabel('The locked reference remains 500 km / 97.4°. Both panels use the same candidate altitude.\n'
                  'Local panels compare equivalent node crossings; away from the solution, 76 revolutions do not take exactly five days.\n'
                  'Keplerian period + J2 node, consistent solar clock. Repeat of a pattern ≠ coverage of every target.', fontsize=9, color=INK)
    return fig


def _save_evidence(session, key, payload):
    history = session['notes'].setdefault(key, [])
    if not history or history[-1] != payload:
        history.append(deepcopy(payload))
    return len(history)


def run_repeat(session, h_km=500., show_fixed=True, show_descending=True, interpretation=''):
    """Separate repeat candidate, saved as supporting evidence, never a revisit row."""
    adjusted = eo.repeat_diagnostics(h_km)
    fixed = eo.repeat_diagnostics(h_km, adjust_sso=False)
    show(repeat_figure(h_km, show_fixed=show_fixed, show_descending=show_descending))
    sol = adjusted['solution']
    cards([('Calculated repeat altitude', f'{sol["h_km"]:.3f} km (≈506 km)', BLUE),
           ('76 revolutions minus 5 days', f'{adjusted["time_error_s"]:+.2f} s', INK),
           ('Candidate SSO inclination', f'{adjusted["orbit"]["inc_deg"]:.5f}°', GREEN)])
    print(f'Fixed 97.4° comparison: nominal LTAN drift {fixed["ltan_drift_min_per_day"]:+.5f} min/day; '
          'it is not exactly Sun-synchronous at this altitude.')
    print('SSO adjustment retains nominal LTAN 10:30 in the model; neither map computes sunlight or acquisitions.')
    print('506 km is a rounded display value. Use Calculated repeat to insert the full-precision solution.')
    print('Physical repeat design requires the perturbed nodal period, orbit tolerances and maintenance.')
    payload = dict(adjusted=adjusted, fixed_inclination=fixed, interpretation=interpretation,
                   evidence='Orbit-pattern closure only; no target revisit, coverage guarantee or mission verdict.')
    revision = _save_evidence(session, 'repeat_history', payload)
    session['controls']['repeat'] = dict(h_km=h_km, show_fixed=show_fixed, show_descending=show_descending, interpretation=interpretation)
    print(f'Repeat evidence saved alongside the scorecard (revision {revision}); numerical mean rows unchanged.')
    return payload


def phasing_cases():
    """Reference-orbit analogues, not operational mission ephemerides."""
    return [dict(name='Equal phasing', phases=[0., 120., 240.]),
            dict(name='Uneven phasing', phases=[0., 36., 180.])]


def phasing_figure():
    cases = phasing_cases(); period = eo.orbit_from_altitude(500, 97.4)['T']/60
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), layout='constrained', facecolor='white')
    theta = np.linspace(0, 2*np.pi, 240)
    for k, c in enumerate(cases):
        ax = axes[0, k]; phases = np.array(c['phases'])
        ax.plot(np.cos(theta), np.sin(theta), color='#8a939a')
        ax.scatter([0], [0], s=350, color=BLUE, alpha=.3)
        angles = np.radians(phases)
        for j, angle in enumerate(angles):
            ax.scatter(np.cos(angle), np.sin(angle), s=80, color=[BLUE, ORANGE, GREEN][j], zorder=4)
            ax.text(1.28*np.cos(angle), 1.28*np.sin(angle), f'{phases[j]:g}°', ha='center', va='center', color=INK)
        ax.set(xlim=(-1.65, 1.65), ylim=(-1.5, 1.5), aspect='equal'); ax.axis('off')
        ax.set_title(c['name']+' · same plane / nominal LTAN 10:30', fontsize=11)
        # Actual forward times to a common orbital phase, starting with the phase-0 satellite.
        times = np.sort(np.mod(-phases, 360)/360*period)
        end = np.r_[times, period]
        timeline = axes[1, k]
        timeline.vlines(times, 0, 1, color=ORANGE, lw=3)
        timeline.axvline(period, color=INK, ls='--', lw=1)
        for a, b in zip(end[:-1], end[1:]):
            timeline.annotate('', (a, .45), (b, .45), arrowprops=dict(arrowstyle='<->', color=INK))
            timeline.text((a+b)/2, .58, f'{b-a:.2f} min', ha='center', fontsize=9, color=INK)
        timeline.set(xlim=(-2, period+2), ylim=(0, 1.1), yticks=[], xlabel='Time to a common orbital phase [min]')
        timeline.set_title('Crossing intervals within one revolution', fontsize=10)
    fig.suptitle('Optional E7 · Same spacecraft count, different spacing', fontsize=14, color=INK)
    fig.supxlabel('Schematic reference-orbit analogues, not Sentinel-2 ephemerides. All spacecraft remain in one plane.\n'
                  'Earth rotates between crossings: these intervals are NOT repeat acquisitions of one ground target.', fontsize=9, color=INK)
    return fig


def run_phasing(session, interpretation=''):
    show(phasing_figure())
    payload = dict(cases=phasing_cases(), reference=reference(), interpretation=interpretation,
                   limitation='Common orbital-phase intervals only, not target revisit; no optimal arrangement claimed.')
    revision = _save_evidence(session, 'phasing_history', payload)
    print(f'Optional phasing evidence saved (revision {revision}). Same-plane phasing preserves nominal LTAN; '
          'changing RAAN in Y2 changes nominal node time.')
    return payload


def scheduling_segments(region='Europe', days=1.):
    """Toy tasking rule selecting ascending nadir-track samples by a bounding box.

    Eligible means geometric branch plus this illustrative policy, NOT daylight,
    sensor scheduling feasibility, a real mission plan or a unique-area result.
    """
    boxes = {'Europe': (-12, 35, 35, 70), 'Australia': (110, 155, -45, -10)}
    if region not in boxes: raise ValueError('Choose Europe or Australia')
    days = eo._number('schedule duration [days]', days, .25, 5)
    orb = eo.orbit_from_altitude(500, 97.4)
    track = eo.ground_track(orb, days=days, n_points=int(days*12000)+1)
    t = track['t_s']; lon = track['lon_deg']; lat = track['lat_deg']
    mid = (t[:-1]+t[1:])/2
    q, vel = eo._ground_state(orb, mid)
    mlon = np.degrees(np.arctan2(q[:, 1], q[:, 0])); mlat = np.degrees(np.arcsin(q[:, 2]))
    ascending = vel[:, 2] >= 0
    x0, x1, y0, y1 = boxes[region]
    selected = ascending & (mlon>=x0) & (mlon<=x1) & (mlat>=y0) & (mlat<=y1)
    points = np.column_stack([lon, lat])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    visible = np.abs(np.diff(lon)) < 180
    dt = np.diff(t)
    return dict(segments=segments, visible=visible, ascending=ascending, selected=selected,
                box=boxes[region], ascending_minutes=float(dt[ascending].sum()/60),
                selected_minutes=float(dt[selected].sum()/60), region=region, days=days)


def scheduling_figure(region='Europe', days=1.):
    result = scheduling_segments(region, days)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[5, 1], layout='constrained', facecolor='white')
    ax, bar = axes
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel='Longitude [°]', ylabel='Latitude [°]')
    ax.set_facecolor('#f4f8fa'); coastlines(ax)
    for mask, color, label, width in [(~result['ascending'], '#8a939a', 'Descending context', .4),
                  (result['ascending'], BLUE, 'Ascending geometric tracks', .7),
                  (result['selected'], ORANGE, 'Selected by toy schedule', 2.5)]:
        ax.add_collection(LineCollection(result['segments'][mask & result['visible']], colors=color, linewidths=width, label=label, zorder=4))
    from matplotlib.patches import Rectangle
    x0,x1,y0,y1 = result['box']
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=ORANGE,ls='--',lw=1.5,zorder=5))
    ax.legend(loc='lower left', fontsize=9, framealpha=1)
    ax.set_title(f'Optional E8 · Access ≠ acquisition · {days:g} '+('day' if days==1 else 'days')+f' · priority box: {region}', fontsize=13)
    bar.barh([1,0], [result['ascending_minutes'], result['selected_minutes']], color=[BLUE,ORANGE])
    bar.set(yticks=[1,0], yticklabels=['Ascending branch', 'Selected by rule'], xlabel='Sampled track time [minutes]')
    bar.set_xlim(0, result['ascending_minutes']*1.25)
    for y, val in [(1,result['ascending_minutes']), (0,result['selected_minutes'])]:
        bar.text(val+result['ascending_minutes']*.02, y, f'{val:.1f}', va='center', color=INK)
    fig.supxlabel('Toy rule: select ascending nadir-track segments whose centres lie inside the box. These are centre lines, not swath areas.\n'
                  'Blue = geometry; orange = assumed selections, not verified acquisitions. No sunlight, clouds, capacity or real mission schedule.\n'
                  'Track time is not area coverage or target revisit. Boundaries and times are sampled; coastlines are bundled offline.', fontsize=9, color=INK)
    return fig


def run_scheduling(session, region='Europe', days=1., interpretation=''):
    result = scheduling_segments(region, days)
    show(scheduling_figure(region, days))
    payload = {key:result[key] for key in ['region','days','box','ascending_minutes','selected_minutes']}
    payload['box'] = list(payload['box'])
    payload.update(interpretation=interpretation, limitation='Illustrative branch/region selection only; not real acquisitions or usable imagery.')
    revision = _save_evidence(session, 'scheduling_history', payload)
    session['controls']['scheduling'] = dict(region=region, days=days, interpretation=interpretation)
    print(f'Optional schedule evidence saved (revision {revision}). Changing the priority box does not change the orbit.')
    return payload


def widgets_available():
    try:import ipywidgets
    except ImportError:return False
    return True


def _panel(callback,controls,label='Run comparison',presets=()):
    """Button-triggered work; no recalculation on drag and no hidden initial run."""
    from IPython.display import display
    from IPython.utils.capture import capture_output
    import ipywidgets as w
    button=w.Button(description=label,button_style='primary',layout=w.Layout(width='210px'))
    output=w.Output()
    def run(_):
        # Capture locally and replace the widget state, rather than relying on
        # front-end message routing/clear_output during a button callback.
        button.disabled=True
        try:
            with plt.ioff(),capture_output() as captured:
                try:callback(**{key:value.value for key,value in controls.items()})
                except (ValueError,KeyError,TypeError) as err:print('Check inputs:',err)
            records=[dict(output_type='display_data',data=item.data,metadata=item.metadata)
                     for item in captured.outputs]
            for name,value in [('stdout',captured.stdout),('stderr',captured.stderr)]:
                if value:records.append(dict(output_type='stream',name=name,text=value))
            output.outputs=tuple(records)
        finally:
            button.disabled=False
    button.on_click(run)
    preset_buttons=[]
    for name,values in presets:
        preset=w.Button(description=name,layout=w.Layout(width='190px'))
        def choose(_, values=values):
            for key,value in values.items():controls[key].value=value
            run(None)
        preset.on_click(choose);preset_buttons.append(preset)
    panel=w.VBox([*controls.values(),w.HBox(preset_buttons),button,output])
    # Controls are returned for meaningful callback tests; never serialize widgets.
    display(panel)
    return dict(panel=panel,controls=controls,button=button,output=output,presets=preset_buttons)



def uses_widgets(interface='widgets'):
    """Explicit code mode also works when widgets import but cannot render."""
    if interface not in ('widgets', 'code'):
        raise ValueError("INTERFACE must be 'widgets' or 'code'")
    return interface == 'widgets' and widgets_available()


def activity_inputs(kind, session, changes=None):
    """Fresh defaults, overlaid by saved inputs and explicit student changes."""
    defaults = {
        'repeat': dict(h_km=eo.repeat_solution()['h_km'], show_fixed=True,
                       show_descending=True, interpretation=''),
        'map': dict(days=1., mode='potential', show_descending=True),
        'pointing': dict(roll=15., target_days=3., prediction='', interpretation=''),
        'intervals': dict(roll=15.),
        'constellation': dict(days=1., plane_separation=15., prediction='', interpretation=''),
        'target': dict(latitude=80., longitude=151., roll=30., metric=MEAN, required_days=3., prediction=''),
        'scheduling': dict(region='Europe', days=1., interpretation=''),
        'mission': blank_mission(),
    }
    if kind not in defaults:
        raise ValueError('Unknown activity: ' + str(kind))
    values = defaults[kind]
    allowed = set(values)
    saved = session['mission'] if kind == 'mission' else session['controls'].get(kind, {})
    values.update(deepcopy(saved))
    if kind == 'pointing':
        # Older JSON may request a second plot; sampling now belongs to Explore E6.
        values.pop('show_samples', None)
    changes = {} if changes is None else changes
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError('Unknown input(s): ' + ', '.join(sorted(unknown)))
    values.update(deepcopy(changes))
    return values


def _run_map_activity(session, **inputs):
    result = run_map(**inputs)
    session['controls']['map'] = deepcopy(inputs)
    return result


def run_intervals(session, roll=15.):
    """E6 selected-setting view; preserve core rows and record only its control."""
    eo._number('pointing [deg]', roll, 0, 30)
    show(sampled_interval_figure(roll))
    session['controls']['intervals'] = dict(roll=roll)
    return dict(roll=roll)


def activity(kind, session, interface='widgets', changes=None):
    """One activity output: initialise widgets once, or run the same code directly.

    Empty changes retain saved inputs. Widget updates replace the initial result.
    Only plain inputs/results are recorded; no widget objects enter the session.
    """
    values = activity_inputs(kind, session, changes)
    if uses_widgets(interface):
        panel = mission_panel(session, inputs=values) if kind == 'mission' else explorer(kind, session, inputs=values)
        panel['button'].click()
        return panel
    if kind == 'mission':
        return run_mission(session, values)
    runners = dict(repeat=run_repeat, pointing=run_pointing, constellation=run_constellation,
                   target=run_target, scheduling=run_scheduling, map=_run_map_activity, intervals=run_intervals)
    return runners[kind](session, **values)


def explorer(kind,session,inputs=None):
    if not widgets_available():
        print("Widgets unavailable. Use activity(..., interface='code') or the run_* functions.")
        return None
    import ipywidgets as w
    c = activity_inputs(kind, session, inputs)
    style={'description_width':'150px'};layout=w.Layout(width='95%',max_width='650px')
    def slider(desc,v,lo,hi,step=1):
        # A legitimate saved/code input must not be silently clamped on reopening.
        return w.FloatSlider(description=desc,value=v,min=min(lo,v),max=max(hi,v),step=step,
                             continuous_update=False,style=style,layout=layout)
    def text(desc,v=''):return w.Textarea(description=desc,value=v,style=style,layout=layout)
    if kind=='repeat':
        altitude=slider('Candidate altitude [km]',c['h_km'],min(495,c['h_km']),max(515,c['h_km']),.025)
        altitude.readout_format='.3f'
        controls=dict(h_km=altitude,
                      show_fixed=w.Checkbox(value=c['show_fixed'],description='Compare fixed 97.4° inclination below',indent=False,layout=layout),
                      show_descending=w.Checkbox(value=c['show_descending'],description='Show descending context (grey)',indent=False,layout=layout),
                      interpretation=text('Explain / gaps?',c['interpretation']))
        return _panel(lambda **kw:run_repeat(session,**kw),controls,'Update repeat map',
                      presets=[('Reference altitude',{'h_km':500.}),('Calculated repeat',{'h_km':eo.repeat_solution()['h_km']})])
    if kind=='scheduling':
        controls=dict(region=w.Dropdown(description='Priority box',options=['Europe','Australia'],value=c['region'],style=style,layout=layout),
                      days=slider('Elapsed days',c['days'],.25,5,.25),interpretation=text('Explain / next check',c['interpretation']))
        return _panel(lambda **kw:run_scheduling(session,**kw),controls,'Apply toy schedule')
    if kind=='pointing':
        controls=dict(roll=slider('Pointing limit [°]',c['roll'],0,30),target_days=slider('Target mean [days]',c['target_days'],1,40,.5),
                      prediction=text('Predict before run',c['prediction']),interpretation=text('Explain after run',c['interpretation']))
        return _panel(lambda **kw:run_pointing(session,**kw),controls)
    if kind=='intervals':
        return _panel(lambda **kw:run_intervals(session,**kw),
                      dict(roll=slider('Pointing limit [°]',c['roll'],0,30)), 'Update intervals')
    if kind=='map':
        controls=dict(days=w.SelectionSlider(description='Elapsed time [days]',options=sorted({.001,.25,1.,3.,10.,36.,c['days']}),value=c['days'],continuous_update=False,style=style,layout=layout),
                      mode=w.Dropdown(description='Layer',options=[('Potential access','potential'),('Nadir strips','nadir')],value=c['mode'],style=style,layout=layout),
                      show_descending=w.Checkbox(value=c['show_descending'],description='Show descending return passes',indent=False,layout=layout))
        return _panel(lambda **kw:_run_map_activity(session,**kw),controls,'Update map')
    if kind=='constellation':
        controls=dict(days=w.SelectionSlider(description='Elapsed time [days]',options=sorted({1.,3.,10.,c['days']}),
                                             value=c['days'],continuous_update=False,style=style,layout=layout),
                      plane_separation=slider('Plane spacing [°]',c['plane_separation'],15,120,15),
                      prediction=text('Predict before run',c['prediction']),
                      interpretation=text('Explain / next check',c['interpretation']))
        return _panel(lambda **kw:run_constellation(session,**kw),controls,'Compare arrangements')
    if kind=='target':
        controls=dict(latitude=slider('Target latitude [°]',c['latitude'],-90,90),longitude=slider('Longitude (context)',c['longitude'],-180,180),
                      roll=slider('Pointing limit [°]',c['roll'],0,30),metric=w.Dropdown(description='Required metric',options=METRICS,value=c['metric'],style=style,layout=layout),
                      required_days=slider('Requirement [days]',c['required_days'],1,40,.5),prediction=text('Predict / explain',c['prediction']))
        return _panel(lambda **kw:run_target(session,**kw),controls,'Check target')
    raise ValueError('Unknown explorer')


MISSION_FIELDS=('domain','purpose','latitude','longitude','metric','required_days','rationale','roll','evidence','next_check')


def blank_mission():
    return dict.fromkeys(MISSION_FIELDS,'')


def run_mission(session,inputs):
    """Scaffold only. No mission numbers/requirement/answer are supplied by default."""
    if not any(str(v).strip() for v in inputs.values()):
        print('Choose a provisional observation need, then enter your inputs. No final mission choice is required.')
        return None
    # Preserve unfinished drafts as well as completed evidence.
    session['mission']=deepcopy(inputs)
    missing=[f for f in MISSION_FIELDS[:8] if inputs.get(f) is None or not str(inputs.get(f,'')).strip()]
    if missing:
        print('Draft saved. Still needed:',', '.join(missing));return None
    try:
        vals={k:float(inputs[k]) for k in ['latitude','longitude','required_days','roll']}
        eo._number('pointing [deg]',vals['roll'],0,30)
        metric=inputs['metric']
        if metric not in METRICS:raise ValueError('Select a listed required metric')
        c=reference();c.update(target_lat_deg=vals['latitude'],target_lon_deg=vals['longitude'],agility_deg=vals['roll'])
        # Requirement validation precedes any drawing or scorecard mutation.
        eo._number('requirement [days]',vals['required_days'],positive=True)
        result=case_result(c)
    except (ValueError,TypeError) as err:
        print('Draft saved. Check inputs:',err);return None
    notes='\n'.join(f'{k}: {inputs.get(k,"")}' for k in ['purpose','rationale','evidence','next_check'])
    row=record_case(session,'Your mission: '+str(inputs['domain']).strip(),c,required=vals['required_days'],metric=metric,notes=notes)
    show(reach_figure(vals['latitude'],vals['longitude'],vals['roll'],metric=metric,required_days=vals['required_days']))
    feedback(c,metric,vals['required_days'])
    print('Candidate: reference orbit/sensor, with your target and pointing choice. Other architectures need their own inputs/model review.')
    if not str(inputs.get('evidence','')).strip() or not str(inputs.get('next_check','')).strip():
        print('Result recorded. Finish your evidence statement and next check, then run again to save them.')
    show_scorecard(session)
    return row


def mission_panel(session,inputs=None):
    if not widgets_available():
        print("Use activity('mission', session, interface='code').");return None
    import ipywidgets as w
    m=activity_inputs('mission',session,inputs);style={'description_width':'160px'};layout=w.Layout(width='95%',max_width='750px')
    labels=dict(domain='Provisional domain',purpose='Observation purpose',latitude='Latitude [°]',longitude='Longitude [°]',
                metric='Required metric',required_days='Threshold [days]',rationale='Why this requirement?',roll='Pointing [0–30°]',
                evidence='Evidence statement',next_check='Next calculation')
    controls={}
    for k in MISSION_FIELDS:
        value='' if m.get(k) is None else str(m.get(k,''))
        if k=='metric':controls[k]=w.Dropdown(description=labels[k],options=[('Choose a metric','')]+[(v,v) for v in METRICS],value=value,style=style,layout=layout)
        else:
            cls=w.Textarea if k in ['purpose','rationale','evidence','next_check'] else w.Text
            controls[k]=cls(description=labels[k],value=value,style=style,layout=layout)
    return _panel(lambda **kw:run_mission(session,kw),controls,'Run / record my case')


def save_session(session,path=None):
    path=Path(path) if path is not None else Path(f'coverage_scorecard_{session["token"]}.json')
    metadata={k:deepcopy(v) for k,v in session.items() if k!='rows'}
    eo.save_scorecard(session['rows'],path,mission=metadata)
    return path


def restore_session(session,path):
    """Replace the in-memory session only after the complete record validates."""
    record=eo.load_scorecard(path);meta=record['mission']
    if not all(k in meta for k in ['mission','controls','notes','token']):
        raise ValueError('Use a saved Block A session JSON, including its mission and controls')
    restored=dict(meta,rows=record['rows'])
    if not all(isinstance(restored[k],dict) for k in ['mission','controls','notes']):raise ValueError('Invalid session metadata')
    # Ensure display can consume the record before replacing the existing session.
    scorecard(restored)
    session.clear();session.update(restored)
    return session


def persistence_panel(session):
    if not widgets_available():
        print('Use save_session / restore_session in the cells above.');return None
    import ipywidgets as w
    from IPython.display import display,FileLink
    upload=w.FileUpload(accept='.json',multiple=False,description='Choose saved JSON')
    save=w.Button(description='Save + download',button_style='success')
    restore=w.Button(description='Restore chosen JSON',button_style='warning')
    output=w.Output()
    def save_clicked(_):
        with output:
            output.clear_output(wait=True);p=save_session(session);print('Saved',p)
            try:
                from google.colab import files
                files.download(str(p))
            except ImportError:display(FileLink(str(p)))
    def restore_clicked(_):
        with output:
            output.clear_output(wait=True)
            if not upload.value:print('Choose your saved JSON first.');return
            value=upload.value
            item=next(iter(value.values())) if isinstance(value,dict) else value[0]
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'record.json';p.write_bytes(bytes(item['content']));restore_session(session,p)
                print('Restored. Rerun explorer/form cells to refresh their controls before making edits.')
                show_scorecard(session)
            except (ValueError,KeyError,TypeError,OSError) as err:print('Could not restore; current session kept:',err)
    save.on_click(save_clicked);restore.on_click(restore_clicked)
    display(w.VBox([w.HBox([save,upload,restore]),output]))
    return dict(save=save,upload=upload,restore=restore,output=output)
