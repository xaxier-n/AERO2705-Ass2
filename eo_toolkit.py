"""AERO2705 shared Earth-observation teaching toolkit (N2.3).

Students may use these functions in their assignment. Check inputs, units,
model and interpretation; implementing the helpers is not an exercise.

MODELS: spherical Earth (R_E=6378 km), circular Keplerian motion and first-order
secular J2 node precession. Ground tracks use Keplerian argument-of-latitude
rate, not a full J2 nodal-period model. No drag, manoeuvres, epoch-to-Sun model,
illumination, weather or acquisition scheduling. No propagator accuracy claim.
Circularity alone implies neither an exact repeating track nor uniform light.

revisit_mean and circle_return_stats use the simplified latitude-circle
TEACHING MODEL (a circle around Earth at the target latitude):
one nominal strip per revolution, its width treated as longitude extent along
a parallel. They do not integrate the plotted tracks. Their explicit teaching
scope is 200–1500 km, track latitude reach >=75 degrees, target |latitude|<=70,
and a band wholly below the track's latitude turning point. These conservative
support limits are not accuracy guarantees. Other missions require a suitable
access model, not extrapolation of these formulae. Geometry/orbit functions
have broader domains. In particular GEO can use geometry, but not strip-rate
revisit. Eccentric orbits require a separate propagator and coverage analysis.

ELLIPTICAL ADAPTATION: see README.md / STUDENT_README.md for the function map.
The dynamics workflow supports circular orbits only. Review/replace
orbit_from_altitude, _orbit, sso_inclination, _ground_state, ground_track,
ground_track_shift, node_crossings, swath_polygons, plot_coverage, max_latitude,
_check_reachable, _circle_scope, revisit_mean, circle_return_stats/revisit_stats
and eta_for_revisit for an elliptical trajectory. Review the notebook's
constant-speed pass duration, data_generated inputs and altitude-dependent
optical performance helpers. Instantaneous spherical access_geometry,
ground_reach, nadir_swath/access_width and earth_angular_radius may be reused
at the appropriate instantaneous height/angle; they do not propagate an ellipse.

Return statistics are finite, phase-dependent circle-model samples. Changing
period/clock can strongly change return patterns near repeat resonances; an
accounting cancellation does not preserve their tails. A sample maximum is
not a worst-case mission guarantee, request wait, area completion or latency.

Angles are degrees at the API; distances km unless a name explicitly says m.
Off-nadir angles are NONNEGATIVE MAGNITUDES: left/right have equal spherical
geometry. FOV means the full cross-track angular field, never a width in km.

N2.2 migration: use circle_return_stats and max_sampled_interval_days; the old
revisit_stats name is deprecated. Use explicit raan_offsets_deg for fleets.
The legacy v_ground key remains a rotation-neglecting approximation; a new
v_ground_no_rotation key makes that meaning explicit. Scorecard comparisons
require matching quantity/unit metadata and an explicit comparison operator.
"""
import json
import math
from pathlib import Path
import warnings
import numpy as np

__version__ = "N2.4"
MU = 398600.0           # km^3/s^2
R_E = 6378.0            # km, adopted spherical radius
J2 = 1.08263e-3
D_SIDEREAL = 86164.0    # s, rounded teaching constant
D_SOLAR = 86400.0       # s
D_YEAR = 365.24 * D_SOLAR
OMEGA_E = 360.0 / D_SIDEREAL  # deg/s
# Internally consistent solar/year clock for the repeat-design experiment only.
# Legacy drift/coverage examples retain their rounded sidereal-day convention.
REPEAT_OMEGA_E = 360.0 / D_SOLAR + 360.0 / D_YEAR
CIRCLE_MAX_LATITUDE = 70.0
CIRCLE_MODEL = "longitude-circle: one nominal strip per revolution"


class ModelScopeError(ValueError):
    """Valid physical inputs, but outside the selected teaching model's scope."""


def _number(name, value, lower=None, upper=None, positive=False):
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if positive and x <= 0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and x < lower or upper is not None and x > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return x


def _integer(name, value, minimum=1):
    x = _number(name, value, lower=minimum)
    if not x.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(x)


def _orbit(orb):
    for key in ('h', 'T', 'n', 'revs_solar'):
        _number(f"orbit {key}", orb[key], positive=True)
    _number('inclination [deg]', orb['inc_deg'], 0, 180)
    _number('node rate [deg/day]', orb['raan_dot'])
    return orb


def orbit_from_altitude(h, inc_deg=97.4):
    """Circular orbit: h [km], inclination [deg]. T [s], v [km/s], n [deg/s].

    revs_solar counts revolutions, not usable target passes. raan_dot [deg/day]
    is first-order J2. v_ground_no_rotation [km/s] neglects Earth rotation;
    v_ground is a compatibility alias for that approximation.
    """
    h = _number('altitude [km]', h, positive=True)
    inc_deg = _number('inclination [deg]', inc_deg, 0, 180)
    a = R_E + h
    v = np.sqrt(MU / a)
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    n_rad = np.sqrt(MU / a**3)
    rate = np.degrees(-1.5*n_rad*J2*(R_E/a)**2*np.cos(np.radians(inc_deg))) * D_SOLAR
    speed = v * R_E / a
    return dict(a=a, T=T, v=v, n=np.degrees(n_rad), v_ground=speed,
                v_ground_no_rotation=speed, revs_solar=D_SOLAR/T,
                raan_dot=rate, h=h, inc_deg=inc_deg)


def sso_inclination(h):
    """First-order circular J2 SSO inclination [deg]; fail if no solution exists."""
    h = _number('altitude [km]', h, positive=True)
    a = R_E+h
    cos_i = -(2*np.pi/D_YEAR)/(1.5*np.sqrt(MU/a**3)*J2*(R_E/a)**2)
    if abs(cos_i) > 1:
        raise ModelScopeError('No circular J2 Sun-synchronous inclination at this altitude')
    return np.degrees(np.arccos(cos_i))


def ground_track_shift(orb, raan_dot=None):
    """Westward shift per Keplerian revolution [deg, equatorial km].

    Relative rotation is omega_E - node_rate. An exact ideal SSO gives a solar
    clock; rounded inclination/year/day constants give a small mismatch.
    A zero node rate is a hypothetical comparison, except at exactly polar i.
    """
    _orbit(orb)
    om = orb['raan_dot'] if raan_dot is None else _number('node rate', raan_dot)
    relative = OMEGA_E - om/D_SOLAR
    dlon = relative*orb['T']
    return dict(dlon_deg=dlon, d_km=np.radians(dlon)*R_E, raan_dot=om,
                clock_s=360/relative if relative != 0 else np.inf)


def parallel_circumference(lat_deg):
    """Spherical parallel circumference [km], including zero at either pole."""
    lat = _number('latitude [deg]', lat_deg, -90, 90)
    return 0.0 if abs(lat) == 90 else 2*np.pi*R_E*np.cos(np.radians(lat))


def earth_angular_radius(h):
    """Earth's angular radius [deg] at positive altitude h [km]."""
    h = _number('altitude [km]', h, positive=True)
    return np.degrees(np.arcsin(R_E/(R_E+h)))


def access_geometry(h, eta_deg=None, eps_deg=None):
    """Visible spherical ray intersection from exactly one angle magnitude.

    h [km]; eta in [0,rho] or elevation eps in [0,90], both degrees.
    Returns rho, eta, gamma, eps [deg], R_s and ground_reach [km].
    Stable at zenith/nadir and the tangent horizon; ground distance is an arc.
    """
    h = _number('altitude [km]', h, positive=True)
    if (eta_deg is None) == (eps_deg is None):
        raise ValueError('Give exactly one of eta_deg or eps_deg')
    a = R_E+h
    rho = earth_angular_radius(h)
    if eta_deg is not None:
        eta = _number('off-nadir magnitude [deg]', eta_deg, 0, rho)
        e = np.radians(eta)
        # Rationalised quadratic root avoids subtraction near nadir.
        disc = max(0.0, (R_E-a*np.sin(e))*(R_E+a*np.sin(e)))
        if eta == rho:
            disc = 0.0
        slant = h*(2*R_E+h)/(a*np.cos(e)+np.sqrt(disc))
        gamma = np.degrees(np.arctan2(slant*np.sin(e), a-slant*np.cos(e)))
        eps = max(0.0, 90-eta-gamma)
    else:
        eps = _number('elevation [deg]', eps_deg, 0, 90)
        e = np.radians(eps)
        if eps == 90:
            eta, gamma, slant = 0.0, 0.0, h
        else:
            b = R_E*np.sin(e)
            slant = h*(2*R_E+h)/(np.sqrt(b*b+h*(2*R_E+h))+b)
            gamma = np.degrees(np.arctan2(slant*np.cos(e), R_E+slant*np.sin(e)))
            eta = max(0.0, 90-eps-gamma)
    return dict(rho=rho, eta=eta, gamma=gamma, eps=eps, R_s=slant,
                ground_reach=R_E*np.radians(gamma))


def ground_reach(h, eta_deg):
    """Surface distance [km] to a nonnegative off-nadir angle [deg]."""
    return access_geometry(h, eta_deg=eta_deg)['ground_reach']


def nadir_swath(h, fov_deg):
    """Spherical zero-roll strip width [km] for a full positive FOV [deg]."""
    return access_width(h, 0, fov_deg=fov_deg)


def access_width(h, eta_max_deg, *, fov_deg):
    """Symmetric potential envelope [km]: 2*d(roll_limit + full_FOV/2).

    A finite strip broadens off nadir. This union of possible pointing choices
    is not simultaneously acquired. All strip edges must be below the horizon.
    """
    rho = earth_angular_radius(h)
    fov = _number('full FOV [deg]', fov_deg, positive=True)
    roll = _number('roll magnitude [deg]', eta_max_deg, lower=0)
    edge = roll+fov/2
    if edge >= rho:
        raise ValueError(f'Strip edge {edge:.3f} deg must be below horizon {rho:.3f} deg')
    return 2*ground_reach(h, edge)


def max_latitude(orb, W_km):
    """Geometric latitude bound [deg] for a symmetric spherical band.

    Necessary reach condition, not a temporal coverage prediction.
    """
    _orbit(orb)
    width = _number('band width [km]', W_km, lower=0, upper=2*np.pi*R_E)
    track_max = min(orb['inc_deg'], 180-orb['inc_deg'])
    return min(90.0, track_max+np.degrees(width/(2*R_E)))


def _check_reachable(lat_deg, W_km, orb):
    lat = _number('latitude [deg]', lat_deg, -90, 90)
    limit = max_latitude(orb, W_km)
    if abs(lat) > limit:
        raise ModelScopeError(f'Target latitude {lat:+.1f} deg is unreachable: '
                              f'this band reaches {limit:.1f} deg. No coverage.')


def _circle_scope(lat_deg, orb, W_km=None):
    _orbit(orb)
    lat = _number('latitude [deg]', lat_deg, -90, 90)
    track_max = min(orb['inc_deg'], 180-orb['inc_deg'])
    if W_km is not None:
        _number('band width [km]', W_km, positive=True)
        _check_reachable(lat, W_km, orb)
    if not 200 <= orb['h'] <= 1500 or track_max < 75 or abs(lat) > CIRCLE_MAX_LATITUDE:
        raise ModelScopeError('Circle-model teaching scope: h=200–1500 km, track '
                              'latitude reach >=75 deg, target |latitude|<=70 deg. '
                              'Use a suitable target-access model for this case.')
    if W_km is not None and abs(lat)+np.degrees(W_km/(2*R_E)) >= track_max:
        raise ModelScopeError('Band approaches the orbit latitude turning point; '
                              'longitude-circle statistics are not supported here.')


def revisit_mean(lat_deg, W_km, orb):
    """C/(N*W) mean potential-access sizing estimate [days], with model label.

    One nominal strip/revolution along a latitude parallel. Not unique-area
    completion, actual pass rate, deadline, maximum gap or request waiting time.
    See module docstring for the explicit, conservative teaching scope.
    """
    _circle_scope(lat_deg, orb, W_km)
    C = parallel_circumference(lat_deg)
    return dict(t_days=C/(orb['revs_solar']*W_km), C=C, N=orb['revs_solar'],
                W=W_km, model=CIRCLE_MODEL, metric='mean potential-access estimate')


def circle_return_stats(lat_deg, W_km, orb, n_orbits=200000, raan_dot=None):
    """Finite return intervals in the longitude-circle illustration [days].

    Not a propagation of ground_track. Starts a notional target at longitude
    zero with a hit at t=0. Intervals are between consecutive hits; boundary
    waiting times are excluded. n_orbits is a sample count, not a proof of a
    global maximum. No physical longitude, epoch, sunlight or tasking model.
    Histogram values are exact integer-orbit intervals expressed in days.
    """
    _circle_scope(lat_deg, orb, W_km)
    n_orbits = _integer('n_orbits', n_orbits, 2)
    dlon = ground_track_shift(orb, raan_dot)['dlon_deg']
    half_ang = np.degrees(W_km/(2*R_E*np.cos(np.radians(lat_deg))))
    n = np.arange(n_orbits)
    lon = (-n*dlon) % 360
    hits = np.flatnonzero(np.minimum(lon, 360-lon) <= half_ang)
    if len(hits) < 2:
        raise ModelScopeError('Fewer than two hits in this finite circle sample; '
                              'no return interval can be estimated. Increase n_orbits.')
    counts_orbits = np.diff(hits)
    vals, counts = np.unique(counts_orbits, return_counts=True)
    waits = counts_orbits*orb['T']/D_SOLAR
    return dict(mean_days=waits.mean(), max_sampled_interval_days=waits.max(),
                n_hits=len(hits), waits=waits, interval_orbits=vals,
                values=vals*orb['T']/D_SOLAR, percent=100*counts/counts.sum(),
                sample_duration_days=(n_orbits-1)*orb['T']/D_SOLAR,
                n_orbits=n_orbits, first_hit_day=hits[0]*orb['T']/D_SOLAR,
                last_hit_day=hits[-1]*orb['T']/D_SOLAR,
                model=CIRCLE_MODEL, metric='finite sampled hit-to-hit intervals')


def revisit_stats(*args, **kwargs):
    """Deprecated name. Use circle_return_stats; no worst_days key is returned."""
    warnings.warn('Use circle_return_stats and max_sampled_interval_days; '
                  'these are not mission worst-case statistics.', DeprecationWarning,
                  stacklevel=2)
    return circle_return_stats(*args, **kwargs)


def eta_for_revisit(target_days, lat_deg, orb, *, fov_deg, eta_cap=None):
    """Invert the circle mean model for required boresight [deg].

    W_km/W_required_km are the required width; W_achieved_km is the width at
    the returned angle (at least the nadir strip). With a cap, a required angle
    beyond it is returned for interpretation, but feasible is False. Geographic
    failure returns NaN angle with a reason; unsupported models raise clearly.
    """
    target = _number('target mean interval [days]', target_days, positive=True)
    lat = _number('latitude [deg]', lat_deg, -90, 90)
    _orbit(orb)
    base = access_width(orb['h'], 0, fov_deg=fov_deg)
    ceiling = earth_angular_radius(orb['h'])-fov_deg/2-1e-7
    cap = ceiling if eta_cap is None else _number('agility cap [deg]', eta_cap, 0, 90)
    cap = min(cap, ceiling)
    allowed_width = access_width(orb['h'], cap, fov_deg=fov_deg)
    needed = parallel_circumference(lat)/(orb['revs_solar']*target)
    result = dict(eta_deg=np.nan, eta_edge_deg=np.nan, W_km=needed,
                  W_required_km=needed, W_achieved_km=np.nan, feasible=False,
                  reason='', model=CIRCLE_MODEL, metric='mean potential-access estimate')
    if abs(lat) > max_latitude(orb, allowed_width):
        result['reason'] = 'Target is outside the geographic reach of the permitted envelope'
        return result
    _circle_scope(lat, orb)
    if needed < base:
        eta = 0.0
    elif needed > access_width(orb['h'], ceiling, fov_deg=fov_deg):
        result['reason'] = 'Required width exceeds geometric horizon limit'
        return result
    else:
        lo, hi = 0.0, ceiling
        for _ in range(70):
            mid = (lo+hi)/2
            if access_width(orb['h'], mid, fov_deg=fov_deg) < needed:
                lo = mid
            else:
                hi = mid
        eta = hi
    achieved = access_width(orb['h'], eta, fov_deg=fov_deg)
    _circle_scope(lat, orb, achieved)
    result.update(eta_deg=eta, eta_edge_deg=eta+fov_deg/2,
                  W_achieved_km=achieved, feasible=eta <= cap,
                  reason='Meets the target within the circle sizing model' if eta <= cap
                  else 'Required pointing exceeds the permitted agility cap')
    return result


# Circular mean-element positions and illustrative footprint plotting.
def _wrap(x):
    return (x+180) % 360-180


def _ground_state(orb, t_s, raan0_deg=0, u0_deg=0, *, earth_rotation_deg_s=OMEGA_E):
    """Earth-fixed unit position and time derivative, consistent with ground_track."""
    u = np.radians(orb['n']*t_s+u0_deg)
    i = np.radians(orb['inc_deg'])
    w = np.radians(orb['raan_dot']/D_SOLAR-earth_rotation_deg_s)
    a = np.radians(raan0_deg)+w*t_s
    cu, su, ca, sa, ci, si = np.cos(u), np.sin(u), np.cos(a), np.sin(a), np.cos(i), np.sin(i)
    q = np.column_stack([ca*cu-sa*ci*su, sa*cu+ca*ci*su, si*su])
    du = np.column_stack([-ca*su-sa*ci*cu, -sa*su+ca*ci*cu, si*cu])
    rot = np.column_stack([-q[:,1], q[:,0], np.zeros_like(u)])
    return q, du*np.radians(orb['n'])+w*rot


def ground_track(orb, days=1.0, raan0_deg=0.0, u0_deg=0.0, n_points=None, *, earth_rotation_deg_s=OMEGA_E):
    """Circular/J2-node track; geocentric lat/lon [deg], t_s [s].

    earth_rotation_deg_s defaults to the rounded sidereal teaching constant;
    the repeat exercise supplies a solar/year-consistent rotation explicitly.
    RAAN is relative to the adopted zero Greenwich orientation at t=0, not an
    epoch-to-Sun conversion. No LTAN/illumination inference. Metadata supports
    unambiguous node-crossing measurement; v_ground_kms is Earth-fixed speed.
    """
    _orbit(orb)
    days = _number('duration [days]', days, positive=True)
    raan0_deg = _number('initial RAAN [deg]', raan0_deg)
    u0_deg = _number('initial argument of latitude [deg]', u0_deg)
    count = max(2, int(np.ceil(days*3000))+1) if n_points is None else _integer('n_points', n_points, 2)
    t = np.linspace(0, days*D_SOLAR, count)
    rotation = _number('Earth rotation [deg/s]', earth_rotation_deg_s, positive=True)
    q, qdot = _ground_state(orb, t, raan0_deg, u0_deg, earth_rotation_deg_s=rotation)
    return dict(lat_deg=np.degrees(np.arcsin(np.clip(q[:,2], -1, 1))),
                lon_deg=_wrap(np.degrees(np.arctan2(q[:,1], q[:,0]))), t_s=t,
                v_ground_kms=R_E*np.linalg.norm(qdot, axis=1),
                period_s=orb['T'], westward_shift_deg=(rotation-orb['raan_dot']/D_SOLAR)*orb['T'],
                earth_rotation_deg_s=rotation,
                inclination_deg=orb['inc_deg'])


def repeat_solution(revolutions=76, days=5):
    """Ideal SSO repeat estimate: coprime j revolutions in k mean solar days.

    Keplerian period, spherical Earth, first-order J2 node rate only. This is
    not a perturbed nodal-period solution or an operational maintained orbit.
    """
    j, k = _integer('revolutions', revolutions), _integer('repeat days', days)
    if math.gcd(j, k) != 1:
        raise ValueError('Use a coprime repeat ratio (reduce revolutions/days first)')
    period = k*D_SOLAR/j
    altitude = (MU*(period/(2*np.pi))**2)**(1/3)-R_E
    _number('repeat altitude [km]', altitude, 200, 1500)
    return dict(revolutions=j, days=k, h_km=float(altitude), period_s=period,
                inc_deg=float(sso_inclination(altitude)))


def repeat_diagnostics(h_km, revolutions=76, days=5, *, adjust_sso=True, fixed_inc_deg=97.4):
    """Signed equatorial node misclosure after j revolutions; not target revisit.

    Positive error means east of the original ascending node. Also reports the
    time error relative to k solar days. Neither is a mission acceptance test.
    """
    solution = repeat_solution(revolutions, days)
    h = _number('repeat candidate altitude [km]', h_km, 200, 1500)
    inc = sso_inclination(h) if adjust_sso else _number('fixed inclination [deg]', fixed_inc_deg, 0, 180)
    orb = orbit_from_altitude(h, inc)
    j, k = solution['revolutions'], solution['days']
    relative = REPEAT_OMEGA_E-orb['raan_dot']/D_SOLAR
    error = float(_wrap(360*k-j*relative*orb['T']))
    return dict(orbit=orb, solution=solution, adjusted_sso=bool(adjust_sso),
                node_error_deg=error, node_error_km=float(np.radians(error)*R_E),
                time_error_s=float(j*orb['T']-k*D_SOLAR),
                ltan_drift_min_per_day=float(4*(orb['raan_dot']-360*D_SOLAR/D_YEAR)),
                model='Keplerian period + J2 node; consistent mean-solar/year clock; no nodal-period correction')


def node_crossings(track):
    """Unwrapped longitudes of ascending crossings, with model-rate-aware turns.

    Requires metadata from ground_track and sample spacing <= period/20.
    Longitude wrap integer uses the known model shift; subdegree crossing
    positions are interpolated from samples. Not an independent propagator.
    An epoch exactly on the ascending node is not counted.
    """
    if 'period_s' not in track or 'westward_shift_deg' not in track:
        raise ValueError('Use the complete track returned by ground_track, including metadata')
    la, lo, t = (np.asarray(track[k], dtype=float) for k in ('lat_deg','lon_deg','t_s'))
    if la.ndim != 1 or lo.shape != la.shape or t.shape != la.shape or len(t) < 2:
        raise ValueError('Track arrays must have equal one-dimensional shapes and >=2 points')
    if not all(np.all(np.isfinite(a)) for a in (la,lo,t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Track requires finite coordinates and strictly increasing time')
    if np.max(np.diff(t)) > track['period_s']/20:
        raise ValueError('Track too coarsely sampled for node detection; use >=20 samples/orbit')
    if np.max(np.abs(la)) < 1e-9:
        raise ModelScopeError('Ascending nodes are undefined for an equatorial track')
    j = np.flatnonzero((la[:-1] < 0) & (la[1:] >= 0))
    if not len(j):
        return np.array([])
    f = -la[j]/(la[j+1]-la[j])
    cross = lo[j]+f*_wrap(lo[j+1]-lo[j])
    times = t[j]+f*(t[j+1]-t[j])
    prediction = cross[0]-np.rint((times-times[0])/track['period_s'])*track['westward_shift_deg']
    return cross+360*np.rint((prediction-cross)/360)


def _offset(lat, lon, bearing, d_km):
    """Great-circle destination from degree coordinates, distance km."""
    d = np.asarray(d_km)/R_E
    la, lo, b = np.radians(lat), np.radians(lon), np.radians(bearing)
    la2 = np.arcsin(np.clip(np.sin(la)*np.cos(d)+np.cos(la)*np.sin(d)*np.cos(b), -1, 1))
    lo2 = lo+np.arctan2(np.sin(b)*np.sin(d)*np.cos(la), np.cos(d)-np.sin(la)*np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def swath_polygons(orb, days, swath_km, raan0_deg=0.0, u0_deg=0.0, pts_per_pass=140, *, branch='both'):
    """Illustrative symmetric Earth-fixed cross-track strips as lon/lat polygons.

    Optional ascending/descending selection is geometric, not a daylight filter.
    Exact requested time endpoints;
    curved edges are sampled. Supported for LEO h<=1500 km and bands that do
    not reach either pole; a cylindrical polygon is unsuitable at a pole.
    Increase pts_per_pass to check visual convergence. This is not an image
    acquisition or area-union calculation.
    """
    _orbit(orb)
    days = _number('duration [days]', days, lower=0)
    width = _number('strip width [km]', swath_km, positive=True)
    raan0_deg = _number('initial RAAN [deg]', raan0_deg)
    u0_deg = _number('initial phase [deg]', u0_deg)
    pts = _integer('pts_per_pass', pts_per_pass, 3)
    if branch not in ('both','ascending','descending'):
        raise ValueError('Choose both, ascending or descending branches')
    if orb['h'] > 1500 or max_latitude(orb, width) >= 90:
        raise ModelScopeError('Regional strip polygons support LEO h<=1500 km and '
                              'bands that do not intersect a pole; use another map model')
    if days == 0:
        return []
    duration, half = days*D_SOLAR, orb['T']/2
    polys = []
    delta = width/(2*R_E)
    if branch=='both':
        # Preserve the original unfiltered polygon segmentation for API callers.
        bounds=np.r_[np.arange(0,duration,half),duration]
    else:
        # Latitude changes direction at u=90+180*k, not at the equator.
        # Include the initial spacecraft phase, also for phased constellations.
        first=math.floor((u0_deg-90)/180)+1
        last=math.ceil((u0_deg+orb['n']*duration-90)/180)
        turns=(90+180*np.arange(first,last)-u0_deg)/orb['n']
        bounds=np.r_[0,turns[(turns>0)&(turns<duration)],duration]
    for start,end in zip(bounds[:-1],bounds[1:]):
        if end <= start:
            continue
        if branch!='both':
            _,velocity=_ground_state(orb,np.array([(start+end)/2]),raan0_deg,u0_deg)
            ascending=velocity[0,2]>0
            if ascending != (branch=='ascending'):continue
        t = np.linspace(start, end, max(2, int(np.ceil((end-start)/half*(pts-1)))+1))
        q, qdot = _ground_state(orb, t, raan0_deg, u0_deg)
        normal = np.cross(q, qdot)
        norm = np.linalg.norm(normal, axis=1)
        if np.any(norm < 1e-10):
            raise ModelScopeError('Stationary track: cross-track strip direction is undefined')
        normal /= norm[:,None]
        centre_lon = np.degrees(np.unwrap(np.arctan2(q[:,1], q[:,0])))
        edges = []
        for sign in (-1,1):
            edge = q*np.cos(delta)+sign*normal*np.sin(delta)
            lat = np.degrees(np.arcsin(np.clip(edge[:,2], -1, 1)))
            lon = centre_lon+_wrap(np.degrees(np.arctan2(edge[:,1], edge[:,0]))-centre_lon)
            edges.append(np.column_stack([lon,lat]))
        p = np.vstack([edges[0], edges[1][::-1]])
        p[:,0] -= 360*np.round(p[:,0].mean()/360)
        polys.append(p)
    return polys


def plot_coverage(orb, days, swath_km, target, ax=None, half_lon=18.0,
                  aspect=0.6, n_sats=1, raan_spread_deg=None, phase_in_plane=False,
                  alpha=0.5, color='#E64626', title='', show_track=False,
                  *, raan_offsets_deg=None, pts_per_pass=140,
                  distinguish_branches=False, show_descending=True):
    """Regional geometric strips, not a record of acquired imagery.

    target=(lat,lon) degrees. Use the nadir strip S or explicitly label W as
    potential access. Both branches are plotted with arbitrary zero epoch.
    Prefer an explicit raan_offsets_deg list for different planes. Deprecated
    raan_spread_deg means FIRST-TO-LAST span (inclusive endpoints), not spacing.
    Same-plane phasing uses equal phase angles; rejects conflicting controls.
    No network access, basemap data or illumination model.
    distinguish_branches fills ascending strips and outlines descending strips
    in grey, with direction arrows. show_descending only affects that display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    _orbit(orb)
    days = _number('duration [days]', days, lower=0)
    n_sats = _integer('number of satellites', n_sats)
    half_lon = _number('longitude half-window [deg]', half_lon, 0, 180, positive=True)
    aspect = _number('latitude/longitude window ratio', aspect, positive=True)
    alpha = _number('opacity', alpha, 0, 1)
    tlat = _number('target latitude [deg]', target[0], -90, 90)
    tlon = _wrap(_number('target longitude [deg]', target[1]))
    if raan_offsets_deg is not None and raan_spread_deg is not None:
        raise ValueError('Give explicit RAAN offsets OR the legacy span, not both')
    if phase_in_plane and (raan_offsets_deg is not None or raan_spread_deg is not None):
        raise ValueError('Choose same-plane phasing OR different-plane offsets')
    if raan_offsets_deg is not None:
        if len(raan_offsets_deg) != n_sats:
            raise ValueError('One RAAN offset is required per satellite')
        offsets = [_number('RAAN offset [deg]', x) for x in raan_offsets_deg]
    elif raan_spread_deg is not None:
        span = _number('first-to-last RAAN span [deg]', raan_spread_deg, 0, 360)
        warnings.warn('Use explicit raan_offsets_deg; legacy span includes both endpoints',
                      DeprecationWarning, stacklevel=2)
        offsets = np.linspace(0, span, n_sats)
    else:
        offsets = [0.0]*n_sats
    if ax is None:
        _, ax = plt.subplots(figsize=(9,4.5))
    cols = ['#E64626','#1f6fb2','#2e8b57','#8156a8']
    xlo, xhi = tlon-half_lon, tlon+half_lon
    for k in range(n_sats):
        raan0, u0 = (0,360*k/n_sats) if phase_in_plane else (offsets[k],0)
        c = cols[k % len(cols)] if n_sats > 1 else color
        branches=['ascending','descending'] if distinguish_branches and show_descending else ['ascending'] if distinguish_branches else ['both']
        for branch in branches:
            visible=[]
            kwargs={} if branch=='both' else {'branch':branch}
            for p in swath_polygons(orb, days, swath_km, raan0, u0, pts_per_pass,**kwargs):
                plo, phi = p[:,0].min(), p[:,0].max()
                for j in range(math.ceil((xlo-phi)/360), math.floor((xhi-plo)/360)+1):
                    visible.append(p+[360*j,0])
            descending=branch=='descending'
            if visible:
                collection=PolyCollection(visible,facecolors='none' if descending else c,
                    edgecolors='#8a939a' if descending else 'none',linewidths=.65,
                    alpha=.55 if descending else alpha,zorder=1 if descending else 2)
                collection.set_gid(f'{branch}-satellite-{k}')
                ax.add_collection(collection)
            if distinguish_branches:
                # One arrow per visible branch/spacecraft; chronological centres.
                for p in visible:
                    n=len(p)//2;centres=(p[:n]+p[n:][::-1])/2
                    inside=np.flatnonzero((centres[:,0]>xlo+1)&(centres[:,0]<xhi-1)&
                        (abs(centres[:,1]-tlat)<half_lon*aspect-1))
                    if len(inside)<3:continue
                    m=inside[len(inside)//2]
                    arrow=ax.annotate('',xy=centres[m+1],xytext=centres[m-1],
                        arrowprops=dict(arrowstyle='->',color='#8a939a' if descending else c,
                                        lw=1,alpha=.7 if descending else .9),zorder=5)
                    arrow.set_gid(f'{branch}-direction-{k}')
                    break
        if show_track and days > 0:
            tr = ground_track(orb, days, raan0, u0)
            for j in (-1,0,1):
                ax.plot(tr['lon_deg']+360*j, tr['lat_deg'], '.', ms=.4, color=c, alpha=.5)
    ax.plot(tlon,tlat,'k*',ms=14,zorder=6)
    ax.set_xlim(xlo,xhi)
    ax.set_ylim(max(-90,tlat-half_lon*aspect), min(90,tlat+half_lon*aspect))
    ax.grid(alpha=.25)
    ax.set_xlabel('longitude [deg]'); ax.set_ylabel('latitude [deg]')
    ax.set_title(title or 'Geometric strips: both branches; no illumination/tasking filter', fontsize=10)
    return ax


# Block B: validated dimensional helpers; no new Week 8 GSD exercise.
def gsd_from_optics(h_km, p_um, f_m):
    """Pinhole nadir sampling: GSD=h*p/f [m]; h km, pitch micrometres, f metres."""
    h = _number('altitude [km]',h_km,positive=True)*1e3
    p = _number('pixel pitch [um]',p_um,positive=True)*1e-6
    f = _number('focal length [m]',f_m,positive=True)
    return dict(gsd_m=h*p/f,h_m=h,p_m=p,f_m=f)


def focal_length_for(h_km, p_um, gsd_m):
    """Pinhole focal length [m] for a positive target GSD [m]."""
    return (_number('altitude [km]',h_km,positive=True)*1e3 *
            _number('pixel pitch [um]',p_um,positive=True)*1e-6 /
            _number('GSD [m]',gsd_m,positive=True))


def swath_from_optics(h_km, p_um, f_m, n_px):
    """Flat-ground full FOV [deg], detector width [mm], swath [km].

    Algebraically consistent with n_px*GSD under the same pinhole assumptions;
    not independent optical validation or an exact spherical finite-FOV swath.
    """
    gsd = gsd_from_optics(h_km,p_um,f_m)
    n = _integer('cross-track pixels',n_px)
    w = n*gsd['p_m']
    fov = 2*np.degrees(np.arctan(w/(2*gsd['f_m'])))
    return dict(S_km=n*gsd['gsd_m']/1e3,fov_deg=fov,w_det_mm=w*1e3)


def swath_from_detector(gsd_m, n_px):
    """Nominal uniform nadir swath [km]=n_px*GSD; shared flat-ground assumptions."""
    return dict(S_km=_number('GSD [m]',gsd_m,positive=True)*_integer('pixels',n_px)/1e3)


def diffraction_limit(h_km, D_m, wavelength_nm=550.0):
    """Rayleigh separation 1.22*lambda*h/D [m]; not Airy diameter/system resolution."""
    h = _number('altitude [km]',h_km,positive=True)
    D = _number('aperture [m]',D_m,positive=True)
    wavelength = _number('wavelength [nm]',wavelength_nm,positive=True)
    return dict(x_diff_m=1.22*wavelength*1e-9*h*1e3/D,wavelength_nm=wavelength,D_m=D)


def aperture_for(x_diff_m, h_km, wavelength_nm=550.0):
    """Aperture [m] for a specified Rayleigh separation [m]."""
    x = _number('Rayleigh separation [m]',x_diff_m,positive=True)
    return diffraction_limit(h_km,1,wavelength_nm)['x_diff_m']/x


def resolution_regime(gsd_m, x_diff_m):
    """Diagnostic scale comparison, NOT a rigorous resolution classifier.

    Sampling, contrast/MTF, optics and detector integration all matter. The
    blur/GSD ratio alone does not imply that additional sampling buys nothing.
    """
    r = _number('Rayleigh separation [m]',x_diff_m,positive=True)/_number('GSD [m]',gsd_m,positive=True)
    label = 'equal scales' if r == 1 else ('sampling pitch larger' if r < 1 else 'Rayleigh separation larger')
    return dict(regime=label,ratio=r,metric='Rayleigh separation / sampling pitch')


def off_nadir_gsd(h_km, gsd_nadir_m, eta_deg):
    """Local differential angular-pixel stretches for pure cross-track roll.

    Across-track includes the surface tilt; the orthogonal angular dimension
    scales with range. A pushbroom along-track sample ALSO depends on line
    timing and ground motion: this function does not determine that timing.
    """
    gsd = _number('nadir GSD [m]',gsd_nadir_m,positive=True)
    g = access_geometry(h_km,eta_deg=eta_deg)
    if g['eps'] <= 1e-10:
        raise ModelScopeError('Pixel stretch is singular at the horizon')
    along = gsd*g['R_s']/h_km
    cross = along/np.sin(np.radians(g['eps']))
    return dict(eta=g['eta'],eps=g['eps'],R_s=g['R_s'],gsd_along_m=along,
                gsd_cross_m=cross,area_ratio=along*cross/gsd**2)


def effective_revisit(t_geometric_days, p_clear):
    """Reciprocal successful-access-rate estimate t/p [days].

    Assumes p is a representative long-run success fraction of the relevant
    opportunities. Not a general request waiting time or deadline guarantee.
    Serial correlation and varying observation conditions require more detail.
    p=0 explicitly returns infinity: there are no successful opportunities.
    """
    t = _number('geometric mean interval [days]',t_geometric_days,positive=True)
    p = _number('clear fraction',p_clear,0,1)
    return dict(t_eff_days=t/p if p else np.inf,t_geometric_days=t,p_clear=p,
                metric='reciprocal successful-access-rate estimate')


def p_all_cloudy(n_passes, p_clear):
    """(1-p)^n for independent trials with identical p; no universal bound.

    Correlated cloud needs a dependence model. n=0 returns 1 (no opportunities).
    """
    n = _integer('number of opportunities',n_passes,0)
    p = _number('clear probability',p_clear,0,1)
    return (1-p)**n


def data_generated(v_ground_kms, gsd_m, n_px, n_bands, bits, duty_min_per_day):
    """Raw pushbroom line rate, bit/s and decimal GB/day (no compression/overhead).

    Use a speed consistent with the ground spacing/line-rate assumption.
    No downlink, storage, data availability or acquisition schedule is modelled.
    """
    v = _number('ground speed [km/s]',v_ground_kms,lower=0)
    gsd = _number('along-track sample spacing [m]',gsd_m,positive=True)
    px,bands,b = (_integer(n,x) for n,x in [('pixels',n_px),('bands',n_bands),('bits per sample',bits)])
    duty = _number('duty [min/day]',duty_min_per_day,0,1440)
    line_rate = v*1e3/gsd
    rate = line_rate*px*bands*b
    return dict(line_rate_per_s=line_rate,R_gen_bit_s=rate,GB_per_day=rate*duty*60/8/1e9)


# Scorecard core: plain data, repeatable updates, explicit evidence comparison.
SCORECARD_FIELDS = ['tension','name','lecture','quantity','unit','reference',
                    'case_study','mine','required','comparison','required_quantity',
                    'required_unit','evidence_status','margin','status','fix','bills',
                    'assumptions','configuration','mission_id','revision']


def new_scorecard():
    """Create a new empty record explicitly; do not call when loading an existing one."""
    return []


def _plain(value):
    """Normalise NumPy scalars/arrays and missing NaN values to portable data."""
    if isinstance(value,np.ndarray): return _plain(value.tolist())
    if isinstance(value,np.generic): return _plain(value.item())
    if isinstance(value,float):
        if np.isnan(value): return None
        if not np.isfinite(value):
            raise ValueError('Store unavailable/infinite results as mine=None with evidence_status')
    if isinstance(value,dict): return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)): return value
    raise ValueError(f'Cannot save {type(value).__name__} as plain scorecard data')


def add_row(scorecard, **kwargs):
    """Upsert evidence by mission/revision/tension/quantity; preserve omitted fields.

    Use a new mission_id/revision to preserve a changed mission's earlier work.
    To compare, provide required_quantity/unit matching quantity/unit and an
    explicit comparison ('<=', '>=', '=='). Missing metadata is not a pass.
    """
    bad = set(kwargs)-set(SCORECARD_FIELDS)
    if bad: raise ValueError(f'Unknown scorecard fields: {sorted(bad)}')
    if not kwargs.get('tension') or not kwargs.get('quantity'):
        raise ValueError('A scorecard row requires tension and quantity')
    if kwargs.get('comparison') not in (None,'<=','>=','=='):
        raise ValueError('comparison must be <=, >= or ==')
    if kwargs.get('evidence_status') not in (None,'evaluated','not evaluated','not established'):
        raise ValueError('Unknown evidence_status')
    fields = ('mission_id','revision','tension','quantity')
    key = tuple(kwargs.get(f) for f in fields)
    for row in scorecard:
        if tuple(row.get(f) for f in fields) == key:
            row.update(_plain(kwargs)); return scorecard
    scorecard.append(_plain({f:kwargs.get(f) for f in SCORECARD_FIELDS}))
    return scorecard


def _assessment(row):
    if row.get('evidence_status') in ('not evaluated','not established'):
        return None,row['evidence_status']
    mine,needed = row.get('mine'),row.get('required')
    if mine is None or needed is None: return None,'not evaluated'
    if (not row.get('quantity') or not row.get('unit') or
        row.get('required_quantity') != row['quantity'] or row.get('required_unit') != row['unit']):
        return None,'not established: metric/unit mismatch or unspecified'
    op = row.get('comparison')
    if op not in ('<=','>=','=='): return None,'not established: comparison unspecified'
    numeric = isinstance(mine,(float,int)) and isinstance(needed,(float,int)) and not isinstance(mine,bool) and not isinstance(needed,bool)
    if numeric and not (np.isfinite(mine) and np.isfinite(needed)): return None,'not evaluated'
    if op != '==' and not numeric: return None,'not established: numeric bound required'
    passed = mine <= needed if op == '<=' else mine >= needed if op == '>=' else mine == needed
    margin = None
    if numeric and mine > 0 and needed > 0 and op != '==':
        margin = needed/mine if op == '<=' else mine/needed
    return margin,'meets within model' if passed else 'does not meet within model'


def scorecard_frame(scorecard):
    """DataFrame with decisions based on UNROUNDED matching metrics and bounds.

    margin is optional and unrounded, not an integration score. Use status for
    the decision; display precision must never relax the mission requirement.
    """
    import pandas as pd
    rows = _plain(scorecard)
    for r in rows: r['margin'],r['status'] = _assessment(r)
    return pd.DataFrame(rows,columns=SCORECARD_FIELDS)


def scorecard_literal(scorecard, name='SCORECARD'):
    """Portable Python literal of native values; no np.float64 or widget objects."""
    import keyword,pprint
    if not isinstance(name,str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError('Choose a valid Python variable name')
    return f'{name} = '+pprint.pformat(_plain(scorecard),width=78,sort_dicts=False)


def save_scorecard(scorecard, path, *, mission=None):
    """Save schema/toolkit versions, mission inputs and evidence as local JSON.

    Download/copy the file out of the Colab runtime to keep it across sessions.
    This function does not mount Drive or access the network.
    """
    record = dict(schema_version=1,toolkit_version=__version__,mission=_plain(mission or {}),rows=_plain(scorecard))
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return record


def load_scorecard(path):
    """Read the saved record (rows and mission metadata); reject unknown schemas."""
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or not isinstance(record.get('rows'),list) or not isinstance(record.get('mission'),dict):
        raise ValueError('Unsupported or malformed scorecard record')
    rows = []
    for row in record['rows']:
        if not isinstance(row,dict): raise ValueError('Each scorecard row must be an object')
        add_row(rows,**row)
    record['rows'] = rows
    return _plain(record)
