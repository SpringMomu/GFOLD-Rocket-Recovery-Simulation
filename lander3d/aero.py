"""Aerodynamics for the 3D booster: engineering model + live 3-D CFD coupling.

``AeroModel`` (always on)
    A semi-empirical engineering model of the kind used in flight
    simulators and preliminary design (slender-body theory + Jorgensen
    crossflow + Mach-dependent drag + grid fins).  It is valid at the real
    flight Reynolds and Mach numbers and produces the forces and moments
    that act on the vehicle every physics step.

    Component build-up as in a launch-vehicle aerodynamic database: body
    alone (axial + slender-body potential + crossflow normal force),
    grid-fin increments (each panel in its local flow, with deflection,
    deployment and the shadowing of leeward panels in the body wake at
    angle of attack), deployed legs, and the propulsive-aerodynamic
    interaction of the engine (retro-propulsion: the plume ahead of an
    engines-first booster shields it, and the aerodynamic axial force falls
    with the thrust coefficient C_T = T / (q A_ref); nose-first, the plume
    fills the base and removes the base drag).  The propellant leaving the
    nozzle also damps pitch/yaw rates (jet damping, lander3d/vehicle.py).

``ExternalCFD`` (live 3-D CFD, optional)
    The browser runs an incompressible 3-D flow solver on the GPU
    (web3d/fluid.js: by default a 64x64x96 grid of 1 m cells whose axes are
    fixed to east/north/up and which translates with the booster) and
    continuously posts the integrated body surface-pressure force
    coefficient and the centre of pressure of its side force to /api/cfd3d.
    The grid fins sit in that grid as porous flow-turning panels (deflected
    and folded as flown) and the engine as a jet whose strength follows the
    thrust coefficient, so both act on the body flow.  With ``cfd_forces``
    enabled the coefficients anchor the database the way CFD anchors an
    engineering aero deck: the CFD (incompressible) body coefficients are
    compared with the model at the same angle, thrust coefficient and
    Mach -> 0, and the ratios (bounded to 0.5-2x, low-pass filtered) scale
    the model's body axial / normal force at the flight Mach number; the CFD
    centre of pressure moves the model's, by at most 15 % of the length.
    The grid fins stay a separate increment (the CFD integrates the body
    skin only).  Stale data (> 1 s) falls back to the model automatically.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .vehicle import cross3, vnorm


# ------------------------------------------------------------ engineering
def _transonic(mach: float, low: float, peak: float, high: float) -> float:
    """Smooth drag rise: low below M0.8, peak near M1.1, decaying supersonic."""
    if mach < 0.8:
        return low
    if mach < 1.1:
        t = (mach - 0.8) / 0.3
        return low + (peak - low) * t * t * (3 - 2 * t)
    return high + (peak - high) * math.exp(-(mach - 1.1) / 0.6)


def fin_normal_coeff(alpha: float, cn_alpha: float) -> float:
    """Normal-force coefficient of one set of grid-fin cell walls.

    Linear (cn_alpha) at small angles, peaking at 45 deg -- grid fins keep
    working to large angles instead of stalling like a planar fin."""
    return cn_alpha * math.sin(alpha) * math.cos(alpha)


def fin_mach_factor(mach: float) -> float:
    """Grid fins lose part of their effectiveness around Mach 1 (the cells
    choke) and recover supersonically."""
    return 1.0 - 0.35 * math.exp(-((mach - 1.05) / 0.22) ** 2)


# Retro-propulsion (engines-first, engine burning): share of the jet-off
# axial force left at thrust coefficient C_T.  Trend of central-nozzle
# retro-propulsion wind-tunnel tests: the drag is kept at small C_T and falls
# steeply once the plume stands off the base (C_T ~ 1).
SRP_FLOOR = 0.2
SRP_CT = 0.6
BASE_DRAG = 0.12          # base-drag part of the nose-first axial coefficient, filled by a burning plume
# Grid fins in the body wake: a leeward panel loses up to this share of its
# force at large angle of attack.
FIN_SHADOW = 0.45
CFD_TAU = 0.5             # low-pass of the CFD anchoring ratios [s]


def srp_axial_factor(ct: float) -> float:
    """Jet-on / jet-off axial force of an engines-first booster."""
    if ct <= 0.0:
        return 1.0
    return SRP_FLOOR + (1.0 - SRP_FLOOR) / (1.0 + (ct / SRP_CT) ** 1.5)


def srp_axial_factor_many(ct):
    """Vectorised :func:`srp_axial_factor`."""
    c = np.maximum(np.asarray(ct, float), 0.0)
    return SRP_FLOOR + (1.0 - SRP_FLOOR) / (1.0 + (c / SRP_CT) ** 1.5)


def fin_shadow_strength(sin_alpha):
    """Share of a fully leeward panel's force lost in the body wake at total
    angle of attack alpha: none along the axis, all of FIN_SHADOW beyond ~30 deg."""
    return FIN_SHADOW * np.minimum(1.0, np.asarray(sin_alpha, float) / 0.5)


def _transonic_many(mach, low: float, peak: float, high: float):
    """Vectorised :func:`_transonic` (same piecewise curve)."""
    m = np.asarray(mach, float)
    mid = np.clip((m - 0.8) / 0.3, 0.0, 1.0)
    rise = low + (peak - low) * mid * mid * (3.0 - 2.0 * mid)
    fall = high + (peak - high) * np.exp(-(np.maximum(m, 1.1) - 1.1) / 0.6)
    return np.where(m < 0.8, low, np.where(m < 1.1, rise, fall))


def fin_normal_coeff_many(alpha, cn_alpha: float):
    """Vectorised :func:`fin_normal_coeff`."""
    a = np.asarray(alpha, float)
    return cn_alpha * np.sin(a) * np.cos(a)


def fin_mach_factor_many(mach):
    """Vectorised :func:`fin_mach_factor`."""
    return 1.0 - 0.35 * np.exp(-((np.asarray(mach, float) - 1.05) / 0.22) ** 2)


def grid_fin_forces(spec, R, zc, omega_b, v_rel_w, rho, a_snd, defl, deploy):
    """Force and moment of the four grid fins.

    Each panel sits at radius fin_r_cp and height grid_fin_z and is hinged
    about its radial axis e_r.  The cells point along c = cos(d) e_z -
    sin(d) e_t (d: deflection, e_t the tangential direction).  The local
    air velocity relative to the panel (including the body rotation, which
    gives the fins their damping) is split into components along the cells
    and across them; each set of cell walls turns the crossflow back into
    the cells and receives a force along the crossflow (lift), and the panel
    has a drag along the air velocity.

    Returns (force_world, moment_world_about_com, force_body_per_fin)."""
    F_w = np.zeros(3)
    M_w = np.zeros(3)
    if deploy <= 0.01 or rho <= 0.0:
        return F_w, M_w, None
    Rt = R.T
    v_b = Rt @ v_rel_w
    zf = spec.grid_fin_z - zc
    S = spec.fin_area * deploy
    F_b = np.zeros(3)
    M_b = np.zeros(3)
    speed = float(np.linalg.norm(v_rel_w))
    fM = fin_mach_factor(speed / max(a_snd, 1.0))
    # Body wake: the crossflow (where the air goes across the body) leaves
    # the leeward panels in the separated wake of the cylinder.
    cross = -np.array([v_b[0], v_b[1], 0.0])
    cross_n = float(np.linalg.norm(cross))
    shadow = float(fin_shadow_strength(cross_n / speed)) if speed > 1e-6 else 0.0
    cross_dir = cross / cross_n if cross_n > 1e-9 else cross
    for i, az in enumerate(spec.fin_azimuth_deg):
        ph = math.radians(az)
        cph, sph = math.cos(ph), math.sin(ph)
        e_r = np.array([cph, sph, 0.0])
        e_t = np.array([-sph, cph, 0.0])
        p_b = spec.fin_r_cp * e_r + np.array([0.0, 0.0, zf])
        v_loc = v_b + cross3(omega_b, p_b)
        w = -v_loc                                   # air velocity relative to the panel
        w2 = float(np.dot(w, w))
        if w2 < 1e-6:
            continue
        d = float(defl[i])
        cd, sd = math.cos(d), math.sin(d)
        c = np.array([0.0, 0.0, cd]) - sd * e_t      # cell axis
        n_t = cd * e_t + np.array([0.0, 0.0, sd])    # across the cells, tangential walls
        wc, wt, wr = float(np.dot(w, c)), float(np.dot(w, n_t)), float(np.dot(w, e_r))
        q_loc = 0.5 * rho * w2 * (1.0 - shadow * max(0.0, float(np.dot(e_r, cross_dir))))
        a_t = math.atan2(wt, abs(wc))
        a_r = math.atan2(wr, abs(wc))
        f = q_loc * S * (fM * (fin_normal_coeff(a_t, spec.fin_cn_alpha) * n_t
                               + fin_normal_coeff(a_r, spec.fin_cn_alpha) * e_r)
                         + spec.fin_cd0 * w / math.sqrt(w2))
        F_b += f
        M_b += cross3(p_b, f)
    return R @ F_b, R @ M_b, F_b


def fin_static_parts(spec, u, axis, alpha, q, mach, deploy):
    """The four grid fins at zero deflection (for trajectory prediction,
    where the roll angle is unknown): identical to the per-fin model for any
    roll angle at small angle of attack.  Returns (drag, normal) forces."""
    if deploy <= 0.01:
        return np.zeros(3), np.zeros(3)
    c = float(np.dot(u, axis))
    perp = u - c * axis
    pn = float(np.linalg.norm(perp))
    S = spec.fin_area * deploy * (1.0 - float(fin_shadow_strength(pn)) / math.pi)
    drag = -u * q * 4.0 * S * spec.fin_cd0
    normal = np.zeros(3)
    if pn > 1e-9:
        normal = -(perp / pn) * q * 4.0 * S * fin_mach_factor(mach) * fin_normal_coeff(alpha, spec.fin_cn_alpha)
    return drag, normal


def fin_static_force(spec, u, axis, alpha, q, mach, deploy):
    """Sum of the four grid fins at zero deflection (see ``fin_static_parts``)."""
    drag, normal = fin_static_parts(spec, u, axis, alpha, q, mach, deploy)
    return drag + normal


class AeroModel:
    """Forces and moments on the booster from the relative wind."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.cfd: ExternalCFD | None = None
        self.cfd_forces = False
        self.last: dict = {}
        self._anchor = None          # (time, k_axial, k_normal, dz_cp): filtered CFD anchoring

    def coefficients(self, alpha: float, mach: float, tail_first: bool, ct: float = 0.0):
        """Axial and normal force coefficients (reference: frontal area).

        ``ct`` is the thrust coefficient T / (q A_ref) of the burning engine
        (propulsive-aerodynamic interaction, see the module docstring)."""
        s = self.spec
        a_ref = math.pi * (s.diameter / 2) ** 2
        a_plan = s.length * s.diameter
        sa, cabs = math.sin(alpha), abs(math.cos(alpha))
        if tail_first:
            ca0 = _transonic(mach, 0.95, 1.45, 1.10)  # engine section / base leads
            # the plume stands ahead of the base only while the flow is near the axis
            ca0 *= 1.0 - (1.0 - srp_axial_factor(ct)) * cabs
        else:
            ca0 = _transonic(mach, 0.62, 1.05, 0.85)  # flat interstage leads
            ca0 -= BASE_DRAG * min(1.0, max(0.0, ct))
        pg = 1.0 / math.sqrt(max(0.2, 1.0 - min(mach, 0.85) ** 2))
        cn_pot = 2.0 * sa * cabs * pg * 0.5  # blunt ends carry about half the ideal slope
        cdc = _transonic(mach, 1.2, 1.75, 1.45)
        cn_visc = 0.62 * cdc * (a_plan / a_ref) * sa * sa
        ca = ca0 * cabs * cabs
        return ca, cn_pot, cn_visc

    def quick_force(self, v_rel, axis, rho, a_snd, legs=0.0, fins=1.0, trim=None, thrust=0.0):
        """Force only (no moments) for trajectory prediction in guidance.

        The fins are taken at zero deflection.  In flight the attitude loop
        deflects them to cancel the aerodynamic moment of the body, and their
        normal force then cancels most of the body's lift (at 8 kPa, 15 deg
        off the flow: -2.7 m/s^2 at zero deflection, -0.4 m/s^2 flown).
        ``trim(theta, q)`` -- ``theta`` the angle between the axis and the
        engines-first direction, ``q`` the dynamic pressure -- returns the
        factor by which the trimmed booster's normal force (body + fins)
        differs from this untrimmed one (``Autopilot3D._lift_trim``).
        ``thrust`` (N): the engine thrust, for the plume interaction."""
        s = self.spec
        speed = float(vnorm(v_rel))
        if speed < 0.3 or rho <= 0.0:
            return np.zeros(3)
        u = v_rel / speed
        if axis is None:
            axis = -u  # engines-first, aligned with the flow
        c = float(np.dot(u, axis))
        alpha = math.acos(min(1.0, abs(c)))
        mach = speed / a_snd
        q = 0.5 * rho * speed * speed
        a_ref = math.pi * (s.diameter / 2) ** 2
        ct = float(thrust) / (q * a_ref) if thrust > 0.0 else 0.0
        ca, cn_pot, cn_visc = self.coefficients(alpha, mach, c < 0.0, ct)
        perp = u - c * axis
        pn = float(vnorm(perp))
        f = -math.copysign(1.0, c) * q * a_ref * ca * axis
        f_n = np.zeros(3)
        if pn > 1e-6:
            f_n = -(perp / pn) * q * a_ref * (cn_pot + cn_visc)
        f_drag, f_fin_n = fin_static_parts(s, u, axis, alpha, q, mach, fins)
        f_n = f_n + f_fin_n
        if trim is not None:
            f_n = f_n * trim(math.acos(max(-1.0, min(1.0, -c))), q)
        f += f_n + f_drag
        f += -u * q * 4.0 * legs * 0.9
        return f

    def quick_force_many(self, v_rel, axis, rho, a_snd, legs=0.0, fins=1.0, trim=None, thrust=0.0):
        """Vectorised :meth:`quick_force` over the leading axis (no CFD blend).

        ``v_rel`` and ``axis`` are ``(n, 3)``; ``rho`` / ``a_snd`` / ``thrust``
        are scalars or ``(n,)``; ``legs`` and ``fins`` are scalars.  ``trim`` is a
        vectorised ``kappa(theta, q)``.  Returns the forces ``(n, 3)``."""
        s = self.spec
        vr = np.asarray(v_rel, float).reshape(-1, 3)
        ax = np.broadcast_to(np.asarray(axis, float), vr.shape)
        n = vr.shape[0]
        rho = np.broadcast_to(np.asarray(rho, float), (n,))
        a_snd = np.asarray(a_snd, float) * np.ones(n)
        speed = np.sqrt(np.sum(vr * vr, axis=1))
        live = (speed >= 0.3) & (rho > 0.0)
        if not np.any(live):
            return np.zeros((n, 3))
        sl = np.where(live, speed, 1.0)
        u = vr / sl[:, None]
        c = np.sum(u * ax, axis=1)
        alpha = np.arccos(np.minimum(1.0, np.abs(c)))
        mach = sl / np.maximum(a_snd, 1e-9)
        q = 0.5 * rho * sl * sl
        a_ref = math.pi * (s.diameter / 2) ** 2
        a_plan = s.length * s.diameter
        sa, cabs = np.sin(alpha), np.abs(np.cos(alpha))
        ct = np.broadcast_to(np.asarray(thrust, float), (n,)) / np.maximum(q * a_ref, 1e-9)
        ca0 = np.where(c < 0.0,
                       _transonic_many(mach, 0.95, 1.45, 1.10) * (1.0 - (1.0 - srp_axial_factor_many(ct)) * cabs),
                       _transonic_many(mach, 0.62, 1.05, 0.85) - BASE_DRAG * np.clip(ct, 0.0, 1.0))
        pg = 1.0 / np.sqrt(np.maximum(0.2, 1.0 - np.minimum(mach, 0.85) ** 2))
        cn_pot = 2.0 * sa * cabs * pg * 0.5
        cdc = _transonic_many(mach, 1.2, 1.75, 1.45)
        cn_visc = 0.62 * cdc * (a_plan / a_ref) * sa * sa
        ca = ca0 * cabs * cabs
        perp = u - c[:, None] * ax
        pn = np.sqrt(np.sum(perp * perp, axis=1))
        pn_safe = np.maximum(pn, 1e-12)
        f = -np.copysign(1.0, c)[:, None] * (q * a_ref * ca)[:, None] * ax
        f_n = np.where((pn > 1e-6)[:, None], -(perp / pn_safe[:, None]) * (q * a_ref * (cn_pot + cn_visc))[:, None], 0.0)
        if fins > 0.01:
            S = s.fin_area * fins * (1.0 - fin_shadow_strength(np.minimum(pn, 1.0)) / math.pi)
            f_n = f_n - (perp / pn_safe[:, None]) * (q * 4.0 * S * fin_mach_factor_many(mach)
                                                     * fin_normal_coeff_many(alpha, s.fin_cn_alpha))[:, None]
            f = f - u * (q * 4.0 * S * s.fin_cd0)[:, None]
        if trim is not None:
            theta = np.arccos(np.clip(-c, -1.0, 1.0))
            f_n = f_n * trim(theta, q)[:, None]
        f = f + f_n - u * (q * 4.0 * float(legs) * 0.9)[:, None]
        return np.where(live[:, None], f, 0.0)

    def __call__(self, vehicle, v_rel, rho, a_snd, com_world):
        s = self.spec
        R = vehicle.R
        axis = R[:, 2]
        speed = float(vnorm(v_rel))
        _, zc, _, _ = vehicle.mass_properties()
        if speed < 0.3:
            self.last = {"mach": 0.0, "q": 0.0, "alpha": 0.0, "cp_z": zc, "drag": 0.0, "normal": 0.0,
                         "ca": 0.0, "cn": 0.0, "source": "MODEL"}
            return np.zeros(3), np.zeros(3), self.last
        u = v_rel / speed
        c = float(np.dot(u, axis))            # >0 nose-first, <0 engine-first
        tail_first = c < 0.0
        alpha = math.acos(min(1.0, abs(c)))    # total angle of attack w.r.t. the leading end
        mach = speed / a_snd
        q = 0.5 * rho * speed * speed
        a_ref = math.pi * (s.diameter / 2) ** 2
        thrust = float(getattr(vehicle, "last", {}).get("thrust_mag", 0.0))     # attitude probes: engine off
        ct = thrust / (q * a_ref) if q > 1e-6 else 0.0

        ca, cn_pot, cn_visc = self.coefficients(alpha, mach, tail_first, ct)
        z_lead = 0.0 if tail_first else s.length  # potential lift sits at the leading end
        z_mid = s.length * 0.5
        cn_m = cn_pot + cn_visc
        z_n = (cn_pot * z_lead + cn_visc * z_mid) / cn_m if cn_m > 1e-9 else z_mid
        source = "MODEL"
        anchor = self._cfd_anchor(axis, alpha, tail_first, ct) if self.cfd_forces and self.cfd is not None else None
        if anchor is not None:
            k_a, k_n, dz = anchor
            ca *= k_a
            cn_pot *= k_n
            cn_visc *= k_n
            z_n += dz
            source = "CFD"

        # Directions: axial force opposes the axial flow, normal force opposes
        # the crossflow.
        perp = u - c * axis
        perp_n = float(vnorm(perp))
        f_axial = -math.copysign(1.0, c) * q * a_ref * ca * axis
        n_dir = np.zeros(3)
        if perp_n > 1e-6:
            n_dir = perp / perp_n
            f_normal = -n_dir * q * a_ref * (cn_pot + cn_visc)
        else:
            f_normal = np.zeros(3)

        # Deployed legs add drag near the base.
        leg_area = 4.0 * vehicle.legs
        f_legs = -u * q * leg_area * 0.9

        forces = [(f_axial, z_mid), (f_normal, z_n), (f_legs, 2.0)]
        total = np.zeros(3)
        moment = np.zeros(3)
        for f, zb in forces:
            r = R @ np.array([0.0, 0.0, zb - zc])
            total += f
            moment += cross3(r, f)
        # Body aerodynamic damping (pitch/yaw of the long body).  The grid
        # fins' damping comes from their own model (local flow includes the
        # rotation).
        omega_w = R @ vehicle.omega
        w_axial = float(np.dot(omega_w, axis))
        w_perp = omega_w - w_axial * axis
        moment += -q * a_ref * s.length ** 2 * 2.2 * w_perp / max(speed, 5.0)
        moment += -q * 1.0 * (s.diameter * 0.8) ** 2 * 0.4 * w_axial * axis / max(speed, 5.0)
        moment_body = moment.copy()
        # Grid fins: four hinged lattice panels (deflection = control).
        f_fins, m_fins, _ = grid_fin_forces(s, R, zc, vehicle.omega, v_rel, rho, a_snd,
                                            vehicle.fin_defl, float(vehicle.fins_deploy))
        total += f_fins
        moment += m_fins

        normal_mag = float(vnorm(f_normal + (f_fins - np.dot(f_fins, axis) * axis)))
        cp_z = zc
        if normal_mag > 1e-3:
            # Centre of pressure: where the normal force would act to give the
            # same moment about the CoM.
            m_perp = vnorm(cross3(axis, cross3(moment, axis)))
            sign = 1.0 if np.dot(cross3(axis, -n_dir), moment) >= 0 else -1.0
            cp_z = zc + sign * m_perp / normal_mag
        drag = float(-np.dot(total, u))
        self.last = {"mach": mach, "q": q, "alpha": math.degrees(alpha), "tail_first": tail_first, "cp_z": cp_z,
                     "com_z": zc, "drag": drag, "normal": normal_mag, "ca": ca, "cn": cn_pot + cn_visc,
                     "source": source, "force": total, "n_dir": n_dir, "ct": ct,
                     "force_fins": f_fins, "moment_fins": m_fins, "moment_body": moment_body}
        return total, moment, self.last

    def _cfd_anchor(self, axis, alpha, tail_first, ct):
        """CFD anchoring of the body coefficients: (k_axial, k_normal, dz_cp)
        from the latest fresh CFD sample, low-pass filtered (wall clock, like
        the CFD stream itself); None (model only) when the CFD data is stale."""
        cfd = self.cfd.coefficients(axis)
        if cfd is None:
            self._anchor = None
            return None
        ca_c, cn_c, zcp_c = cfd
        if not (math.isfinite(ca_c) and math.isfinite(cn_c)):
            return None if self._anchor is None else self._anchor[1:]
        s = self.spec
        # The CFD is incompressible: compare with the model at Mach -> 0.
        ca_r, pot_r, visc_r = self.coefficients(alpha, 0.0, tail_first, ct)
        cn_r = pot_r + visc_r
        prev = self._anchor
        k_a = float(np.clip(abs(ca_c) / ca_r, 0.5, 2.0)) if ca_r > 0.05 else (1.0 if prev is None else prev[1])
        # The normal-force ratio is meaningless near zero angle of attack.
        k_n = float(np.clip(cn_c / cn_r, 0.5, 2.0)) if cn_r > 0.05 else (1.0 if prev is None else prev[2])
        dz = 0.0
        if zcp_c is not None and cn_r > 0.05 and math.isfinite(zcp_c):
            z_lead = 0.0 if tail_first else s.length
            z_model = (pot_r * z_lead + visc_r * s.length * 0.5) / cn_r
            dz = float(np.clip(zcp_c - z_model, -0.15 * s.length, 0.15 * s.length))
        elif prev is not None:
            dz = prev[3]
        t = time.monotonic()
        if prev is None:
            self._anchor = (t, k_a, k_n, dz)
        else:
            f = 1.0 - math.exp(-max(0.0, t - prev[0]) / CFD_TAU)
            self._anchor = (t, prev[1] + f * (k_a - prev[1]), prev[2] + f * (k_n - prev[2]), prev[3] + f * (dz - prev[3]))
        return self._anchor[1:]


# ------------------------------------------------------------ 3-D GPU CFD
class ExternalCFD:
    """Latest force coefficients from the browser's 3-D GPU flow solver.

    ``update`` receives the body pressure-force coefficient vector in the
    world frame (F / (q * A_frontal)) and the centre of pressure of its side
    force (m from the base, optional); ``coefficients(axis)`` splits it into
    axial and normal parts for the current attitude (with the centre of
    pressure), or returns None when the data is older than ``max_age``
    seconds (then the engineering model is used).
    """

    def __init__(self, max_age: float = 1.0) -> None:
        self.c_world = None
        self.zcp = None
        self.stamp = 0.0
        self.max_age = max_age
        self.lock = threading.Lock()

    def update(self, c_world, zcp=None) -> None:
        c = np.asarray(c_world, float).reshape(3)
        if not np.all(np.isfinite(c)):
            return
        try:
            z = float(zcp) if zcp is not None else None
        except (TypeError, ValueError):
            z = None
        with self.lock:
            self.c_world = np.clip(c, -20.0, 20.0)
            self.zcp = z if z is not None and math.isfinite(z) else None
            self.stamp = time.monotonic()

    @property
    def fresh(self) -> bool:
        return self.c_world is not None and time.monotonic() - self.stamp < self.max_age

    def coefficients(self, axis):
        with self.lock:
            if self.c_world is None or time.monotonic() - self.stamp > self.max_age:
                return None
            c = self.c_world.copy()
            zcp = self.zcp
        ca = float(-np.dot(c, axis))
        cn = float(vnorm(c - np.dot(c, axis) * np.asarray(axis)))
        return ca, cn, zcp
