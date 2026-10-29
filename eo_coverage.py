"""Lecture 2 calculation foundation, version C1.0.

Standard-library-only functions. Times are elapsed seconds, angles degrees,
distances km. No absolute UTC/LTAN conversion, sunlight, clouds or scheduling.

Three distinct models:
* supplied finite access windows: exact interval arithmetic on that evidence;
* longitude-circle events: the instantaneous p73 teaching model;
* circular Kepler motion + secular J2 node, spherical Earth, NADIR CIRCULAR
  ground footprint: finite geometric windows, NOT the reference line scanner.
  Footprint radius is an explicit assumption, not inferred along-track FOV.

Regional results are area-weighted cell-centre estimates, not exact polygon
unions. Model/horizon/grid metadata must accompany any compliance statement.
"""
from dataclasses import dataclass
import math

__version__ = "C1.0"
R_E_KM = 6378.0
MU_KM3_S2 = 398600.0
J2 = 1.08263e-3
DAY_S = 86400.0
SIDEREAL_DAY_S = 86164.0


def _finite(value, name):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(value, name):
    number = _positive(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _horizon(horizon_s):
    lo, hi = (_finite(x, "horizon") for x in horizon_s)
    if hi <= lo:
        raise ValueError("horizon end must exceed start")
    return lo, hi


@dataclass(frozen=True)
class AccessWindow:
    """Eligible continuous geometric window [start,end), not an acquisition.

    Set censored flags when the generator's analysis boundary truncates access.
    asset/source identify provenance; ineligible windows are excluded by metrics.
    """
    start_s: float
    end_s: float
    asset: str = "supplied"
    eligible: bool = True
    left_censored: bool = False
    right_censored: bool = False

    def __post_init__(self):
        start = _finite(self.start_s, "window start")
        end = _finite(self.end_s, "window end")
        if end <= start:
            raise ValueError("finite windows need end > start; use event_intervals for points")
        if not isinstance(self.asset, str) or not self.asset:
            raise ValueError("asset must be a nonempty string")
        for key in ("eligible", "left_censored", "right_censored"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)


def merge_access_windows(windows, horizon_s):
    """Clip eligible windows, then union overlap and adjacency (no tolerance).

    Clipping does not create a known access start: left censorship is retained.
    Input order/duplicates/asset labels do not affect temporal union coverage.
    """
    lo, hi = _horizon(horizon_s)
    clipped = []
    for w in windows:
        if not isinstance(w, AccessWindow):
            raise ValueError("windows must contain AccessWindow objects")
        if not w.eligible or w.end_s <= lo or w.start_s >= hi:
            continue
        clipped.append(AccessWindow(max(lo, w.start_s), min(hi, w.end_s),
                                    asset="union", left_censored=w.left_censored or w.start_s < lo,
                                    right_censored=w.right_censored or w.end_s > hi))
    clipped.sort(key=lambda w: (w.start_s, w.end_s))
    merged = []
    for w in clipped:
        if not merged or w.start_s > merged[-1].end_s:
            merged.append(w)
            continue
        old = merged.pop()
        left = old.left_censored or (w.start_s == old.start_s and w.left_censored)
        right = (old.right_censored if old.end_s > w.end_s else
                 w.right_censored if w.end_s > old.end_s else
                 old.right_censored or w.right_censored)
        merged.append(AccessWindow(old.start_s, max(old.end_s, w.end_s),
                                   asset="union", left_censored=left, right_censored=right))
    return merged


def access_metrics(windows, horizon_s):
    """Metrics of union access, with no extrapolation or cyclic wrap.

    max_no_access_in_window_s includes clipped boundary gaps; the full lengths
    of those boundary gaps may be larger. Mean revisit excludes a clipped start.
    It describes starts of union windows, NOT image count or acquisition cadence.
    None denotes an unavailable inter-access statistic, never zero.
    """
    lo, hi = _horizon(horizon_s)
    merged = merge_access_windows(windows, (lo, hi))
    revisit = [b.start_s-a.start_s for a, b in zip(merged, merged[1:])
               if not a.left_censored and not b.left_censored]
    internal = [b.start_s - a.end_s for a, b in zip(merged, merged[1:])]
    gaps = []
    cursor = lo
    for w in merged:
        if w.start_s > cursor:
            gaps.append(dict(start_s=cursor, end_s=w.start_s,
                             duration_s=w.start_s-cursor, boundary_censored=cursor == lo))
        cursor = w.end_s
    if cursor < hi:
        gaps.append(dict(start_s=cursor, end_s=hi, duration_s=hi-cursor,
                         boundary_censored=True))
    covered = sum(w.end_s-w.start_s for w in merged)
    return dict(horizon_s=[lo, hi], merged_windows=merged,
                access_duration_s=covered, time_coverage_fraction=covered/(hi-lo),
                revisit_intervals_s=revisit,
                mean_revisit_s=sum(revisit)/len(revisit) if revisit else None,
                complete_no_access_gaps_s=internal,
                max_complete_no_access_gap_s=max(internal) if internal else None,
                no_access_intervals=gaps,
                max_no_access_in_window_s=max((g["duration_s"] for g in gaps), default=0.0),
                scope="finite-window union access; no cyclic or all-time claim")


def request_waits(windows, requests_s, horizon_s, *, minimum_duration_s=0.0):
    """Earliest feasible start at/after each request, allowed inside a window.

    A required duration must fit contiguously in union access. This assumes
    handover across assets is allowed; per-image no-handover tasking is not modelled.
    Requests are in [horizon start,end). End-only contact cannot start an image.
    No observed feasible start returns None and a censored status, not infinity.
    """
    lo, hi = _horizon(horizon_s)
    duration = _finite(minimum_duration_s, "minimum duration")
    if duration < 0:
        raise ValueError("minimum duration must be nonnegative")
    merged = merge_access_windows(windows, (lo, hi))
    result = []
    for request in requests_s:
        q = _finite(request, "request")
        if not lo <= q < hi:
            raise ValueError("request must be inside [horizon start,end)")
        start = next((max(q, w.start_s) for w in merged
                      if max(q, w.start_s) < w.end_s
                      and max(q, w.start_s)+duration <= w.end_s), None)
        result.append(dict(request_s=q, next_start_s=start,
                           wait_s=None if start is None else start-q,
                           status="not_observed_before_horizon_end" if start is None else "observed"))
    return result


def event_intervals(event_times_s):
    """Instantaneous hit intervals; no finite-duration gap or boundary inference."""
    times = [_finite(x, "event time") for x in event_times_s]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("event times must be strictly increasing")
    intervals = [b-a for a, b in zip(times, times[1:])]
    return dict(intervals_s=intervals,
                mean_interval_s=sum(intervals)/len(intervals) if intervals else None,
                max_sampled_interval_s=max(intervals) if intervals else None)


def circle_access_events(*, period_s, latitude_deg, width_km=60.0,
                         n_orbits=200000, phase_turns=0.0):
    """p73 circle rotation only: one crossing/orbit, day=86400 s.

    No real longitude, inclination/reach, start epoch, duration or Sun model.
    Latitude changes parallel circumference; width is measured ALONG that parallel.
    phase_turns specifies the target relative to the initial crossing (modulo 1).
    """
    period = _positive(period_s, "period")
    lat = _finite(latitude_deg, "latitude")
    if not -90 < lat < 90:
        raise ValueError("circle latitude must be strictly between the poles")
    width = _positive(width_km, "width")
    count = _integer(n_orbits, "n_orbits")
    phase = _finite(phase_turns, "phase") % 1
    circumference = 2*math.pi*R_E_KM*math.cos(math.radians(lat))
    if width > circumference:
        raise ValueError("strip width exceeds the parallel circumference")
    half = width/circumference/2
    indices = [n for n in range(count)
               if abs((n*period/DAY_S-phase+.5) % 1-.5) <= half]
    return dict(orbit_indices=indices, event_times_s=[n*period for n in indices],
                sample_horizon_s=[0.0, (count-1)*period],
                period_s=period, latitude_deg=lat, width_km=width, phase_turns=phase,
                model="instantaneous longitude-circle crossings, one per orbit")


def _bisect(function, lo, hi, tolerance):
    flo = function(lo)
    if flo == 0:
        return lo
    if function(hi) == 0:
        return hi
    while hi-lo > tolerance:
        mid = (lo+hi)/2
        fmid = function(mid)
        if (flo > 0) == (fmid > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return (lo+hi)/2


def nadir_cap_access(*, latitude_deg, longitude_deg, horizon_s,
                     altitude_km=500.0, inclination_deg=97.4,
                     raan0_deg=0.0, argument0_deg=0.0, footprint_radius_km=30.0,
                     branch="ascending", max_step_s=60.0, root_tolerance_s=0.001):
    """Generate finite windows for an explicit circular nadir ground footprint.

    Geometric model: sphere, circular Kepler rate, first-order secular J2 RAAN,
    Earth rotation 360/86164 deg/s, Greenwich angle zero at elapsed t=0.
    No UTC/LTAN, sunlight, weather, real scanner or tasking is implied.

    Bracket extrema of target alignment as well as access roots so a short pass
    can be found even when no coarse sample is inside the footprint. Resolve
    branch boundaries analytically. Root precision is not event completeness:
    halve max_step_s and check events, especially near tangencies. Zero-duration
    tangencies are excluded. Supported step <= T/32; no force-model accuracy claim.
    """
    lo, hi = _horizon(horizon_s)
    lat = _finite(latitude_deg, "latitude")
    inc = _finite(inclination_deg, "inclination")
    if not -90 <= lat <= 90 or not 0 <= inc <= 180:
        raise ValueError("invalid latitude or inclination")
    lon = math.radians(_finite(longitude_deg, "longitude"))
    altitude = _positive(altitude_km, "altitude")
    radius = _positive(footprint_radius_km, "footprint radius")
    a = R_E_KM+altitude
    horizon_radius = R_E_KM*math.acos(R_E_KM/a)
    if radius >= horizon_radius:
        raise ValueError("footprint must lie inside the geometric horizon")
    if branch not in ("ascending", "descending", "both"):
        raise ValueError("branch must be ascending, descending or both")
    if branch != "both" and (inc == 0 or inc == 180):
        raise ValueError("equatorial orbit has no ascending/descending branch")
    n = math.sqrt(MU_KM3_S2/a**3)
    period = 2*math.pi/n
    step = _positive(max_step_s, "max_step_s")
    tolerance = _positive(root_tolerance_s, "root_tolerance_s")
    if step > period/32 or tolerance >= step:
        raise ValueError("require root tolerance < max_step <= period/32")
    i = math.radians(inc)
    u0 = math.radians(_finite(argument0_deg, "argument0"))
    node0 = math.radians(_finite(raan0_deg, "RAAN0"))
    node_rate = -1.5*n*J2*(R_E_KM/a)**2*math.cos(i)
    w = node_rate-2*math.pi/SIDEREAL_DAY_S
    phi = math.radians(lat)
    target = (math.cos(phi)*math.cos(lon), math.cos(phi)*math.sin(lon), math.sin(phi))
    ci, si = math.cos(i), math.sin(i)
    threshold = math.cos(radius/R_E_KM)

    def alignment(t, derivative=False):
        u, node = u0+n*t, node0+w*t
        cu, su, ca, sa = math.cos(u), math.sin(u), math.cos(node), math.sin(node)
        q = (ca*cu-sa*ci*su, sa*cu+ca*ci*su, si*su)
        if derivative:
            q = (n*(-ca*su-sa*ci*cu)-w*q[1],
                 n*(-sa*su+ca*ci*cu)+w*q[0], n*si*cu)
        return sum(x*y for x, y in zip(q, target))-(0.0 if derivative else threshold)

    count = max(1, math.ceil((hi-lo)/step))
    coarse = [lo+(hi-lo)*k/count for k in range(count+1)]
    cuts = set(coarse)
    # All bracketed extrema, not only peaks: split into monotonic pieces.
    previous_t, previous_d = coarse[0], alignment(coarse[0], True)
    for t in coarse[1:]:
        d = alignment(t, True)
        if d == 0:
            cuts.add(t)
        elif previous_d*d < 0:
            cuts.add(_bisect(lambda x: alignment(x, True), previous_t, t, tolerance))
        previous_t, previous_d = t, d
    if branch != "both":
        first = math.ceil((u0+n*lo-math.pi/2)/math.pi)
        last = math.floor((u0+n*hi-math.pi/2)/math.pi)
        cuts.update((math.pi/2+k*math.pi-u0)/n for k in range(first, last+1))
    ordered = sorted(cuts)
    roots = set(ordered)
    for left, right in zip(ordered, ordered[1:]):
        if alignment(left)*alignment(right) < 0:
            roots.add(_bisect(alignment, left, right, tolerance))
    ordered = sorted(roots)
    segments = []
    for left, right in zip(ordered, ordered[1:]):
        mid = (left+right)/2
        ascending = math.cos(u0+n*mid) > 0
        eligible_branch = branch == "both" or ascending == (branch == "ascending")
        if alignment(mid) > 0 and eligible_branch:
            segments.append(AccessWindow(left, right, asset="single_nadir_cap",
                                         left_censored=left == lo and alignment(lo) > 0,
                                         right_censored=right == hi and alignment(hi) > 0))
    windows = merge_access_windows(segments, (lo, hi))
    return dict(windows=windows, horizon_s=[lo, hi],
                inputs=dict(latitude_deg=lat, longitude_deg=math.degrees(lon),
                            altitude_km=altitude, inclination_deg=inc, raan0_deg=raan0_deg,
                            argument0_deg=argument0_deg, footprint_radius_km=radius, branch=branch),
                period_s=period, node_rate_deg_day=math.degrees(node_rate)*DAY_S,
                max_step_s=step, root_tolerance_s=tolerance, module_version=__version__,
                model="circular Kepler + secular J2 node; spherical circular nadir footprint",
                epoch="elapsed t=0, Greenwich angle zero; not an absolute UTC or solar epoch",
                exclusions=["illumination", "cloud", "tasking", "physical scanner geometry",
                            "drag", "short-period J2", "manoeuvres"])


def spherical_grid(latitude_bounds_deg, longitude_bounds_deg, *, rows, columns):
    """Equal-angle rectangular region cells with exact spherical cell areas.

    Longitudes are unwrapped increasing bounds (e.g. 170,190 crosses the date
    line); width <=360 degrees. Cell-centre access approximates the whole cell.
    """
    south, north = (_finite(x, "latitude bound") for x in latitude_bounds_deg)
    west, east = (_finite(x, "longitude bound") for x in longitude_bounds_deg)
    if not -90 <= south < north <= 90 or not 0 < east-west <= 360:
        raise ValueError("invalid rectangular region bounds")
    rows, columns = _integer(rows, "rows"), _integer(columns, "columns")
    cells = []
    for row in range(rows):
        bottom, top = south+(north-south)*row/rows, south+(north-south)*(row+1)/rows
        area = R_E_KM**2*math.radians((east-west)/columns)*(math.sin(math.radians(top))-math.sin(math.radians(bottom)))
        for col in range(columns):
            cells.append(dict(latitude_deg=(bottom+top)/2,
                              longitude_deg=(west+(east-west)*(col+.5)/columns+180) % 360-180,
                              area_km2=area))
    return cells


def _region(point_windows, area_weights, horizon_s):
    windows = [merge_access_windows(w, horizon_s) for w in point_windows]
    weights = [_positive(w, "area weight") for w in area_weights]
    if not weights or len(windows) != len(weights):
        raise ValueError("provide matching nonempty point-window and area-weight lists")
    return windows, weights


def regional_metrics(point_windows, area_weights, horizon_s, *, deadline_s, required_fraction=1.0):
    """Cumulative unique cell coverage from horizon start to deadline [start,end).

    Any positive-duration eligible access counts a cell as reached, not imaged.
    completion_elapsed_s is the infimum time reaching required_fraction (first
    access onset); footprints are cell-centre estimates, never exact area unions.
    """
    lo, hi = _horizon(horizon_s)
    deadline = _finite(deadline_s, "deadline")
    required = _finite(required_fraction, "required fraction")
    if not lo < deadline <= hi or not 0 < required <= 1:
        raise ValueError("deadline must lie in (start,end]; fraction in (0,1]")
    windows, weights = _region(point_windows, area_weights, (lo, hi))
    total = math.fsum(weights)
    reached = [bool(w) and w[0].start_s < deadline for w in windows]
    fraction = math.fsum(a for a, hit in zip(weights, reached) if hit)/total
    first = sorted((w[0].start_s, a) for w, a in zip(windows, weights) if w)
    accumulated, completion = [], None
    for t, a in first:
        accumulated.append(a)
        if math.fsum(accumulated)/total >= required:
            completion = t-lo
            break
    worst_gap = max(access_metrics(w, (lo, hi))["max_no_access_in_window_s"] for w in windows)
    return dict(covered_fraction=fraction, required_fraction=required,
                meets_fraction_by_deadline=fraction >= required,
                completion_elapsed_s=completion, deadline_s=deadline,
                cell_reached=reached, region_max_no_access_in_window_s=worst_gap,
                cell_count=len(weights), area_weight_sum=total,
                scope="fixed-window, area-weighted cell-centre geometric coverage")


def rolling_regional_coverage(point_windows, area_weights, horizon_s, *, window_s):
    """Worst covered fraction over ALL contained rolling windows on input grid.

    For finite union windows coverage changes only at access-end or
    access-start minus rolling duration. Evaluate those breakpoints and interiors.
    This is exact for supplied cell windows, not for continuous regional area.
    Each cell needs positive overlap with [rolling start,rolling end).
    """
    lo, hi = _horizon(horizon_s)
    duration = _positive(window_s, "rolling window duration")
    if duration > hi-lo:
        raise ValueError("rolling duration exceeds analysis horizon")
    windows, weights = _region(point_windows, area_weights, (lo, hi))
    cuts = {lo, hi-duration}
    for point in windows:
        for w in point:
            cuts.update(t for t in (w.end_s, w.start_s-duration) if lo <= t <= hi-duration)
    ordered = sorted(cuts)
    starts = ordered + [(a+b)/2 for a, b in zip(ordered, ordered[1:])]
    total = math.fsum(weights)
    def fraction(t):
        return math.fsum(a for a, point in zip(weights, windows)
                         if any(w.start_s < t+duration and w.end_s > t for w in point))/total
    worst = min(starts, key=lambda t: (fraction(t), t))
    return dict(worst_covered_fraction=fraction(worst), worst_window_start_s=worst,
                window_s=duration, evaluated_start_count=len(starts),
                scope="all contained rolling windows on supplied area-weighted cell grid")
