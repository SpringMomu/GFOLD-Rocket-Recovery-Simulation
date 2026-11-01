"""Falcon-9-style recovery autopilot for the 3D booster.

Coast -> (boost-back) -> one landing burn straight to leg contact.  The burn is
planned by the G-FOLD planner (``planner.py``); attitude control and actuator
allocation are in ``attitude.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import threading
import time

import numpy as np

from common.conic import golden_section_search

from .aero import grid_fin_forces
from .attitude import FIN_MIN_EFF, attitude_control
from .planner import (
    KeepOutBox,
    Plan3D,
    PlanConfig3D,
    Planner3D,
    along_path,
    detour_path,
    detour_paths,
    separating_planes,
)
from .vehicle import G0, atmosphere, atmosphere_many, fin_mix

# Landing-burn tilt budget vs height (m -> deg): upright at contact.  It is
# measured from the wind trim (Autopilot3D._trim_xy: in a cross wind the
# booster holds its place leaning into the wind), which fades out over the
# last TRIM_FADE metres.  (Holding the planner to this budget too was tried:
# it then gives up the last-seconds corrections the tracker can still
# partly fly, and lands further off in wind.)
TILT_BUDGET = ([0.3, 1.5, 3.0, 8.0, 15.0, 40.0], [0.6, 2.0, 5.0, 12.0, 20.0, 75.0])
TRIM_FADE = (0.5, 5.5)          # m: upright below the first height, full trim above the second
SLEW_DEG = ([2.0, 30.0, 300.0], [8.0, 15.0, 25.0])   # thrust-direction slew-rate limit (deg/s) by height
# Angle of attack (thrust vs the oncoming flow) vs dynamic pressure (Pa ->
# deg): only a fallback without an aerodynamic model.  With one, the limit is
# physical -- the attitude the TVC can hold against the aerodynamic moment
# (Autopilot3D._aoa_table).
Q_AOA_BUDGET = ([300.0, 800.0, 1500.0, 2500.0, 8000.0], [115.0, 90.0, 60.0, 30.0, 12.0])
AOA_Q_GRID = (150.0, 300.0, 500.0, 800.0, 1200.0, 2000.0, 3500.0, 6000.0, 10000.0, 20000.0)


class _AttitudeProbe:
    """The booster at another attitude, for the aerodynamic model only."""

    def __init__(self, veh, R):
        self.R = R
        self._mp = veh.mass_properties()
        self.legs = veh.legs
        self.omega = np.zeros(3)
        self.fin_defl = np.zeros(4)
        self.fins_deploy = 1.0

    def mass_properties(self):
        return self._mp


class Autopilot3D:
    """Falcon-9 style recovery: coast -> (boost-back) -> one landing burn to touchdown.

    * COAST: engine off, engines-first, flying a small angle of attack whose
      body/grid-fin lift steers the booster (zero-effort-miss guidance).
      The aim point is not the drag-only impact but ``_burn_miss``: where
      an upright landing burn lit at the predicted ignition would end.  The
      glide thus removes the horizontal error and the burn only needs a
      small, steady tilt.  An ignition predictor on the drag-aware coast
      decides the latest safe moment to light the engine.
    * BOOSTBACK: only when the ballistic impact point is far from the pad.
      The velocity change is found by shooting on the coast so that the
      same aim point lands on the pad, i.e. the booster really comes back
      over the pad; when the pad is out of reach, or the fall would arrive
      too shallow, the burn also lofts the trajectory (more time, steeper).
    * LANDING BURN: at ignition the touchdown time is locked and the 3-D
      G-FOLD SOCP (Planner3D) is re-solved every 0.2 s with a smooth
      objective; if the pad becomes unreachable the time moves to the
      earliest reachable one (never simply later).  The plan ends on the pad
      itself (base height 0, vertical, 1.2 m/s sink); the burn continues
      until a foot touches and the engine is cut -- no separate terminal
      phase.  The last ~0.6 s are flown on the final plan with a tilt limit
      that tightens with height, so the vehicle is upright at contact.
    * Grid fins: stowed for the boost-back, deployed for the coast/entry
      and the landing burn; they steer the attitude by their real
      aerodynamic forces (see attitude_control).
    """

    burn_replan_period = 0.2
    coast_predict_period = 0.25
    # Wall-clock budget of one re-plan (search included).  Over the pad the
    # plans must keep up with the 5 Hz cadence: an out-of-reach problem costs
    # a dozen extra solves, and without a cap the planner fell behind the
    # vehicle exactly when the guidance was needed most.
    plan_budget = 0.20            # s, a full-horizon re-plan
    plan_budget_short = 0.12      # s, over the last 8 s before touchdown
    plan_budget_search = 0.50     # s, a final-time / route search (at most every search_backoff)
    plan_budget_ignition = 1.00   # s, the one-off ignition search (planner thread)
    ign_prep_lead = 1.8           # s before ignition: search its plan on the planner thread
    prep_period = 0.7             # s between ignition preparations
    ign_defer_max = 1.2           # s: an immediate ignition may coast this long for its plan
    # Ignition threshold (thrust fraction a constant-deceleration burn started
    # now would need).  The later the ignition, the less gravity loss; the
    # threshold adapts to how much of the 0.85 plan budget the burn will
    # really have to spare (see _ignition_threshold).
    ignition_fraction = 0.76      # straight-down burn: ~10 % below the plan cap
    ignition_lateral_cut = 0.14   # less when the burn must also divert sideways
    ignition_short = 0.48         # short hover-slam burns (t_burn <= 2 s) ...
    ignition_mid = 0.62           # ... 5 s; full threshold from 8 s on
    burn_fraction_nominal = 0.85
    minimum_throttle = 0.20
    divert_tilt_deg = 65.0
    touchdown_speed = 1.2
    gate_height = 0.0             # G-FOLD plan ends on the pad (base height 0) ...
    gate_speed = 1.2              # ... vertical, at the touchdown sink rate
    low_rcs_height = 15.0         # RCS assists TVC close to the ground
    sink_gain = 0.5               # final approach: sink <= 1.2 + 0.5 h (m/s): 6.2 at 10 m, 2.2 at 2 m
    sink_track_height = 15.0      # the tracker enforces the envelope below this height
    legs_deploy_height = 450.0
    # Boost-back from this predicted miss on.  A fast fall is steered by the
    # glide (body lift ~ q), but a slow one is not: e.g. 180 m off from 2 km
    # at -30 m/s, the glide removes less than half and ignition comes fast
    # (~120 m/s, ~9 kPa), where the angle-of-attack budget allows no sideways
    # thrust -- the burn would divert only once slow, near the ground, i.e.
    # hover sideways at ~80 m.  There a boost-back of ~15 m/s up high is far
    # cheaper and the burn stays nearly vertical.
    boostback_miss = 350.0
    boostback_miss_slow = 120.0   # below boostback_slow_speed[0] airspeed ...
    boostback_slow_speed = (80.0, 140.0)   # ... blending to boostback_miss at [1]
    boostback_min_alt = 900.0     # start a boost-back only above this height ...
    boostback_end_alt = 500.0     # ... and stop it below this one (the landing burn diverts the rest)
    aero_steer_max_deg = 15.0
    lift_gradient_cap = 3.0       # body lift per radian of tilt, in units of thrust accel
    boostback_done_dv = 2.0       # boost-back ends when the remaining velocity change is this small
    boostback_gamma_ok = 10.0     # deg: flight path at ignition steeper than this needs no loft
    boostback_loft_max = 80.0     # m/s of extra climb the boost-back may add for a longer, steeper fall
    boostback_solve_period = 0.3
    # Hop: low (below boostback_min_alt), slow and far off.  The landing burn
    # would fly the divert as a long traverse at its lateral-speed limit
    # (~30 m/s: faster needs thrust along the motion, nose-first, which the
    # angle-of-attack budget does not allow), paying gravity loss all the
    # way -- e.g. 2 km from 600 m at rest: ~75 s and ~3.4 t, about all the
    # propellant.  A lofted boost-back instead (engine off on the arc, then
    # the normal landing burn) is far cheaper.
    hop_miss = 500.0              # m of predicted miss ...
    hop_min_alt = 150.0           # ... from this height up (lower: no room to turn and climb)
    hop_end_frac = 0.7            # a hop ends if it sinks below this fraction of its start height
    # Any boost-back thrusting nose-first (along the motion) above this
    # airspeed keeps its thrust within nose_first_aoa_deg of the flow: low
    # down, a thrust that turns away from the flow puts the booster broadside
    # and the air flips it (fins stowed, RCS too weak to bring it back).
    nose_first_speed = 50.0       # m/s (~1.5 kPa low down)
    nose_first_aoa_deg = 20.0
    search_backoff = 1.0          # s between final-time searches (doubled after failures, up to x4)
    # Free final time: the touchdown time minimises J = propellant +
    # time_weight * t_f (PlanConfig3D.time_weight) -- a golden-section search
    # at ignition and whenever the plan in hand misses, a step either side
    # (retime_step of the time to go) every search_backoff otherwise.
    time_weight = 1.0             # propellant units (m/s) per second of flight
    retime_step = 0.08
    retime_freeze = 5.0           # s to go: no more retiming (unless lost)
    time_search_evals = 9
    ignition_refine = 2               # re-solves about the first plan at ignition (successive convexification)
    # Tower catch: the burn ends with the collar this high above the arm tops,
    # sinking this slowly; there the booster holds, centred, until the arms
    # have closed round the barrel, and then settles onto them.
    catch_gate_height = 1.2
    catch_gate_speed = 0.5
    catch_settle_speed = 0.6
    catch_hold_max = 25.0         # s; then settle anyway
    catch_hold_tilt_deg = 4.0     # tilt budget about the wind trim while holding
    catch_integral_radius = 1.0   # m: the hold's integral only builds up this close to the axis
    glide_gain = 1.5
    frozen_kp = 0.5               # PD gains and limit of the feedback when the plan is frozen (replan off); the re-planned
    frozen_kd = 1.6               # flight keeps 0.5 / 1.6 / 5 m/s^2 (the plan itself is renewed every 0.6 s)
    frozen_fb_limit = 5.0
    frozen_net_gain = True        # a frozen plan (replan off): feedback through the net-acceleration gain of a tilt (_net_gain_feedback)
    replan_enabled = True         # continuous planning (front-end switch); off: the plan in hand is flown to touchdown, no new ones
    lift_calibrate = True         # ... and scaled by the ratio of the lift measured in flight to the lift of that model
    lift_gain_tau = 3.0           # s, memory of that ratio
    lift_trim = True              # the planner and the glide use the lift of the booster as the attitude loop trims it
    lineup_q_max = 1500.0         # Pa: above this dynamic pressure the glide goes on until ignition (no line-up)
    brake_floor = True            # a plan that is not being renewed, and the plan-less law, never command less vertical thrust than stopping the fall takes
    ground_legs_height = 30.0     # m: a given-up catch puts the legs out below this height (the feet go down first)
    catch_far_distance = 25.0     # m from the catch axis beyond which a catch with a stale plan puts the legs out
    brake_floor_age = 1.0         # s without a new plan after which the one in hand is held to that floor
    brake_margin = 1.1            # the floor asks this much more deceleration than the bare stopping need
    brake_floor_limit = 0.97      # ... and never more than this share of the available thrust
    plan_stale_after = 5.0        # s without a new plan (nothing solves): the one in hand is not trusted any more
    fallback_pull_max = 2.0       # m/s^2: pull towards the pad of the plan-less fallback law
    # Wind profile for the predictions: the wind measured now, extrapolated
    # to other heights with a log-law profile whose shear strength k (as in
    # vehicle.Wind: 0 = the same at every height, 1 = the standard log law)
    # is fitted to the winds measured on the way down; with little height
    # spread in the samples the prior k = 1 wins.
    wind_sample_period = 0.25     # s
    wind_memory = 60.0            # s, e-folding age of the samples
    wind_shear_prior_weight = 0.003   # (shapes of nearby k differ little: a stronger prior hides real shear)
    # Wind-resistance options of the plan (front-end solver panel, see
    # lander3d/settings.py; the defaults are the original behaviour):
    wind_plan_gain = 1.0          # wind the planner's predictions assume, x the extrapolated measured wind
    aero_pred = 2                 # aerodynamics in the plan: 0 none, 1 drag at the reference attitude, 2 + lift linearisation
    boostback_wind = True         # True: the boost-back / hop is aimed and judged with the wind; False: in still air (wind ignored)
    trim_gain = 1.0               # share of the wind trim (lean into the wind) used by the tilt cone and the tracker
    shear_fit = True              # fit the shear strength from the winds measured on the way down (else k = 1)
    # Wind trim: the sideways thrust acceleration that cancels the aerodynamic
    # force on the upright booster (lean into the wind).  The tracker's tilt
    # budget and the planner's tilt cone are both centred on it (fading to
    # upright over the last TRIM_FADE metres).
    trim_max_deg = 12.0
    trim_filter = 0.5             # s
    hold_bias_filter = 1.0        # s (_update_hold_bias)
    aoa_moment_margin = 0.8       # share of the gimbal's moment the aerodynamic moment may take (_aoa_table)
    aoa_refresh = 2.0             # s between recomputations of the angle-of-attack table
    q_filter = 1.0                # s, smoothing of the dynamic pressure for that limit
    # attitude loop
    k_angle = 2.0
    k_rate = 4.0
    max_rate = math.radians(30.0)

    def __init__(self, aero_model=None) -> None:
        self.planner = Planner3D()
        self.aero = aero_model
        self.async_plan = False        # re-plans on a background thread (server / browser)
        self._plan_cond = threading.Condition()
        self._plan_job = None          # job waiting for the planning thread
        self._plan_done = None         # finished job waiting to be committed
        self._plan_thread = None
        self._plan_closed = False
        self._plan_gen = 0             # bumped on reset: results of older jobs are dropped
        self.plan_errors = 0           # planning-thread exceptions (shown in the state)
        self._ignition_searching = False   # the asynchronous ignition search is in flight
        self._prep = None              # searched ignition plan (final time + route) for the predicted state
        self._next_prep = 0.0
        self._ignite_defer_until = None    # coasting a moment longer for the ignition plan
        self._plan_deadline = math.inf  # wall-clock deadline of the re-plan in progress
        self.set_target()
        self.reset()

    def set_target(self, base_z=None, rest_height=0.0, legs=True, obstacles=(), catch_link=None) -> None:
        """Where the burn ends.  Default: standing on the pad (legs).  For a
        tower catch, ``base_z`` is the world height of the booster's base when
        its catch collar rests on the arms; the whole guidance then works in
        heights above that point (``h_off`` above the pad altitude) and the
        legs stay stowed.  ``obstacles`` are keep-out boxes for the base point
        in world coordinates (``(centre, axes, half)``, e.g. the catch tower);
        the landing burn plans around them.  ``catch_link`` (a tower catch)
        reports whether the arms are closed round the barrel: the burn then
        ends in a hold just above the arms (``_catch_hold``)."""
        self.h_off = 0.0 if base_z is None else float(base_z) - float(rest_height)
        self.use_legs = legs
        self.catch_link = catch_link
        cls = type(self)
        self.gate_height = cls.catch_gate_height if catch_link is not None else cls.gate_height
        self.gate_speed = cls.catch_gate_speed if catch_link is not None else cls.gate_speed
        z0 = 0.0 if base_z is None else float(base_z)
        self._z_ground = -z0                 # the ground, in planner heights (base point)
        self.obstacles = [KeepOutBox(np.asarray(c, float) - np.array([0.0, 0.0, z0]), np.asarray(ax, float), np.asarray(hf, float))
                          for c, ax, hf in obstacles]

    # ------------------------------------------------- planning thread
    def enable_async_planning(self) -> None:
        """Run the re-plans on a background thread.

        A re-plan is tens of milliseconds of matrix assembly plus the solver
        (and ~0.4 s at a final-time search).  Done inside ``Sim3D.step`` it
        stalls the whole state stream, so the web front end freezes on every
        solve.  The thread works on a frozen copy of the autopilot and of the
        vehicle; the sim thread commits the finished plan at its next step."""
        if self._plan_thread is not None:
            return
        self._plan_closed = False
        try:
            thread = threading.Thread(target=self._plan_loop, name="autopilot-planner", daemon=True)
            thread.start()
        except Exception:      # no threads at all (Pyodide): plan synchronously
            self.async_plan = False
            self._plan_thread = None
            return
        self.async_plan = True
        self._plan_thread = thread

    def close_planning(self) -> None:
        with self._plan_cond:
            self._plan_closed = True
            self._plan_job = None
            self._plan_cond.notify_all()
        thread, self._plan_thread = self._plan_thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        self.async_plan = False

    def _plan_loop(self) -> None:
        while True:
            with self._plan_cond:
                while self._plan_job is None and not self._plan_closed:
                    self._plan_cond.wait(0.1)
                if self._plan_closed:
                    return
                job, self._plan_job = self._plan_job, None
            try:
                result = self._run_plan_job(job)
            except Exception:      # a failed plan must never stop the guidance
                self.plan_errors += 1
                result = None
            if result is not None:
                with self._plan_cond:
                    self._plan_done = result

    def _run_plan_job(self, job):
        ap = job["autopilot"]
        kind = job.get("kind", "replan")
        if kind == "ignite":
            self._run_ignition_search(ap, job)
        elif kind == "prep":
            # Before ignition: search the final time and the route for the
            # *predicted* ignition state, so that at the switch there is no
            # search left to run (a single solve from that plan's final time
            # reaches the pad; the plan itself is not flown).
            ap._plan_deadline = time.monotonic() + ap.plan_budget_search + 0.3
            ap.plan = ap._search_time(job["vehicle"], job["wind"], job["t_guess"])
        else:
            ap._replan(job["vehicle"], job["wind"])
        return {"gen": job["gen"], "kind": kind,
                "plan": ap.plan, "touchdown_time": ap.touchdown_time,
                "burn_fraction": ap.burn_fraction, "route": ap._route,
                "next_search": ap._next_search, "search_fails": ap._search_fails,
                "last_status": ap.planner.last_status, "solver_fail": ap.solver_fail}

    @staticmethod
    def _run_ignition_search(ap, job) -> None:
        """The ignition search, on the planner thread: the same final-time
        search and successive-convexification refines the synchronous path
        runs in ``_ignite``, bounded by ``plan_budget_ignition``."""
        veh, wind = job["vehicle"], job["wind"]
        ap._plan_deadline = time.monotonic() + ap.plan_budget_ignition
        plan = ap._search_time(veh, wind, job["t_est"])
        if plan is None:
            return
        ap._commit(plan)
        for _ in range(ap.ignition_refine):
            p = ap._solver(veh, wind, ap.burn_fraction, ap._keep_out(veh))(ap.plan.t_f)
            if p is None or not p.sane:
                break
            ap._commit(p)

    def _frozen_vehicle(self, veh):
        """Copy of the vehicle for the planning thread (the sim keeps stepping
        the real one in place)."""
        fv = copy.copy(veh)
        fv.pos = np.array(veh.pos, float)
        fv.vel = np.array(veh.vel, float)
        fv.q = np.array(veh.q, float)
        fv.omega = np.array(veh.omega, float)
        fv.fin_defl = np.array(veh.fin_defl, float)
        fv.last = dict(veh.last)
        fv.controls = copy.copy(veh.controls)
        return fv

    def _plan_busy(self) -> bool:
        with self._plan_cond:
            return self._plan_job is not None

    def _ignition_deferred(self) -> bool:
        """Wait for the ignition preparation before switching to the burn.

        An immediate ignition (low, slow, far off the pad) leaves no time to
        prepare a plan, and lighting with nothing that reaches the pad drops
        the guidance into the backup law for as long as the search takes --
        the vehicle then flies a route the plan never assumed (scenario 10 hit
        the tower that way).  Coasting a fraction of a second longer costs a
        couple of metres of height; the deferral is bounded, then the burn
        lights on the estimator's time regardless."""
        if not (self.async_plan and self.planner.available):
            return False
        prep = self._prep
        if prep is not None and prep["t_f"] and self.elapsed - prep["t"] <= self.ign_prep_lead:
            return False
        if self._ignite_defer_until is None:
            self._ignite_defer_until = self.elapsed + self.ign_defer_max
        return self.elapsed < self._ignite_defer_until

    def _submit_prep(self, veh, wind_now) -> None:
        """Search the ignition plan for the *predicted* ignition state while
        the booster is still coasting.

        A single solve from the state at ignition does not reach the pad in
        the catch scenarios (the route over the tower only comes out of the
        search), so without this the switch fell back to the plain brake law
        for the second or so the search took.  Only the searched final time,
        route and thrust ceiling are reused: at ignition the plan is solved
        again from the state the booster is really in."""
        ign = self._ign_state
        if ign is None or self._plan_busy() or self.elapsed < self._next_prep:
            return
        t_ign, p_ign, v_ign, t_burn = ign
        if p_ign is None or not (math.isfinite(t_ign) and t_ign > 0.0):
            return
        self._next_prep = self.elapsed + self.prep_period
        fv = self._frozen_vehicle(veh)
        fv.pos = np.array([p_ign[0], p_ign[1], p_ign[2] + fv.rest_height + self.h_off])
        fv.vel = np.array(v_ign, float)
        ap = copy.copy(self)
        ap.aero = copy.copy(self.aero) if self.aero is not None else None
        ap.plan = None
        ap._route = None
        ap._trim_xy = np.array(self._trim_xy, float)
        ap._wind_ref = (np.array(self._wind_ref[0], float), float(self._wind_ref[1]))
        job = {"gen": self._plan_gen, "kind": "prep",
               "t_guess": max(1.5, 1.05 * t_burn) + 2.0,
               "autopilot": ap, "vehicle": fv, "wind": np.array(wind_now, float)}
        with self._plan_cond:
            self._plan_job = job
            self._plan_cond.notify()

    def _out_of_time(self) -> bool:
        """Whether the re-plan in progress has spent its wall-clock budget."""
        return time.monotonic() > self._plan_deadline

    def _submit_plan(self, veh, wind_now) -> None:
        ap = copy.copy(self)          # shallow: _replan only assigns these attributes
        ap.aero = copy.copy(self.aero) if self.aero is not None else None
        ap.plan = self.plan
        ap._route = self._route
        ap._trim_xy = np.array(self._trim_xy, float)
        ap._wind_ref = (np.array(self._wind_ref[0], float), float(self._wind_ref[1]))
        job = {"gen": self._plan_gen, "autopilot": ap, "vehicle": self._frozen_vehicle(veh),
               "wind": np.array(wind_now, float)}
        with self._plan_cond:
            self._plan_job = job
            self._plan_cond.notify()

    def _apply_plan_result(self) -> bool:
        """Commit a finished background plan; ``False`` when there is none."""
        with self._plan_cond:
            result, self._plan_done = self._plan_done, None
        if result is None or result["gen"] != self._plan_gen:
            return False
        if result.get("kind") == "prep":
            # an ignition preparation: keep the searched final time and route
            # (the plan itself was solved for a predicted state -- the switch
            # re-solves it from the state the booster is really in)
            self._prep = {"t": self.elapsed,
                          "t_f": None if result["plan"] is None else float(result["plan"].t_f),
                          "route": result["route"], "burn_fraction": result["burn_fraction"]}
            return False
        if not self.burn_locked:
            return False
        if result.get("kind") == "ignite":
            # The ignition search ran while the booster flew the interim plan,
            # so its own trajectory started a few tenths of a second ago and
            # committing it would jump the reference forward by that much.
            # Only its final time, route and thrust ceiling are adopted: the
            # next re-plan (right now) rebuilds the plan from the live state
            # at that final time.
            self._ignition_searching = False
            if result["plan"] is None:
                return False
            self.touchdown_time = self.elapsed + result["plan"].t_f
            self.burn_fraction = result["burn_fraction"]
            self._route = result["route"]
            self.next_replan = self.elapsed
            self.last_status = result["last_status"]
            return True
        if result["plan"] is not None and result["plan"].created_at < self.elapsed - 2.0:
            return False                       # far too old to be worth flying
        self.plan = result["plan"]
        self.touchdown_time = result["touchdown_time"]
        self.burn_fraction = result["burn_fraction"]
        self._route = result["route"]
        self._next_search = result["next_search"]
        self._search_fails = result["search_fails"]
        self.last_status = result["last_status"]
        self.solver_fail = result["solver_fail"]
        return True

    def reset(self) -> None:
        self._plan_gen += 1
        self.plan_errors = 0
        self._ignition_searching = False
        self._prep = None
        self._next_prep = 0.0
        self._ignite_defer_until = None
        self._plan_deadline = math.inf
        with self._plan_cond:
            self._plan_job = None
            self._plan_done = None
        self.elapsed = 0.0
        self.plan: Plan3D | None = None
        self.next_replan = 0.0
        self.next_predict = 0.0
        self.burn_locked = False
        self.touchdown_time: float | None = None
        self.burn_fraction = self.burn_fraction_nominal
        self.thrust_scale = 1.0       # measured / modelled engine thrust (accelerometer estimate)
        self._wind_ref = (np.zeros(3), 10.0)
        self._wind_samples: list = []   # (time, height, wind x, wind y)
        self._next_wind_sample = 0.0
        self._next_shear_fit = 0.0
        self.wind_shear_est = 1.0
        self._trim_xy = np.zeros(2)
        self._aoa_cache = (-1e9, Q_AOA_BUDGET)
        self._trim_cache = (-1e9, None)
        self._trim_table = None
        self._trim_fn = None
        self._lift_gain = 1.0         # measured / modelled lift of the booster (_update_lift_gain)
        self._lift_num = 0.0
        self._lift_den = 0.0
        self._next_lift_fit = 0.0
        self._q_filt = 0.0
        self.solver_fail = None       # (time, status) of the last re-plan that found no solution at all
        self._hold_bias = np.zeros(2)
        self._boostback = False
        self._boostback_done = False
        self._hop = False
        self._bb_floor = self.boostback_end_alt
        self._route = None            # reference path round the obstacles while no plan gets through
        self._touched = False
        self.relax = 0                # (no relaxation levels any more; kept for the front end)
        self._next_search = 0.0
        self._search_fails = 0
        self._holding = False
        self._hold_h = 0.0
        self._hold_i = np.zeros(2)
        self._hold_t = 0.0
        self._secured = False
        self._dir_prev = None
        self._gate_xy = np.zeros(2)
        self._dir_rate = np.zeros(3)
        self._dt = 1.0 / 240.0
        self._bb_dv = None
        self._bb_v = None
        self._bb_next = 0.0
        self._bb_reserve = 0.0
        self._ign_state = None
        self._burn_miss_now = np.zeros(2)
        self._bb_impact = None
        self._bb_impact_next = 0.0
        self._brake_time = 0.0
        self.impact = None
        self.ignition_in = math.inf
        self.phase = "STANDBY"
        self.last_status = "idle"
        self.time_to_go = math.nan
        self.cmd_accel = np.zeros(3)
        self.cmd_throttle = 0.0
        self.cmd_axis = np.array([0.0, 0.0, 1.0])
        self.predicted_points: list = []
        self.visited: list = []

    # ------------------------------------------------------------ helpers
    def _ref_plan(self):
        """The plan the next one is linearised about (aerodynamics, route,
        heights, directions): the one in hand once the burn is on, unless it
        has run away (``Plan3D.sane``) -- then a straight descent again."""
        p = self.plan if self.burn_locked else None
        return p if p is not None and p.sane else None

    def _state(self, veh):
        r = np.array([veh.pos[0], veh.pos[1], veh.altitude - self.h_off])
        return r, veh.vel.copy(), veh.mass

    def _height(self, veh) -> float:
        """Height of the base above the end of the burn (the pad or the catch point)."""
        return float(veh.altitude) - self.h_off

    def _drag_model(self, veh, wind_now, t_guess):
        """Aerodynamics along the previous plan (or a decelerating guess).

        Returns ``(d0, G)``: d0 is the drag acceleration with the booster
        aligned with the flow; ``G @ u`` adds the body/grid-fin normal force
        produced by tilting the thrust vector (and the vehicle) by
        ``u_perp / sigma_ref`` against the flow.  For an engines-first
        booster this force points opposite to the tilt, which is why a plan
        that ignored it would overshoot laterally at high dynamic pressure.

        All knots are evaluated in one batch (atmosphere, wind profile and
        aerodynamic model are vectorised): the per-node Python loop used to
        be most of the matrix-assembly time.  ``_drag_model_scalar`` is the
        reference implementation it must agree with.
        """
        prev = self._ref_plan()
        now = self.elapsed
        r0, v0, m0 = self._state(veh)
        legs = max(veh.legs, 1.0 if veh.controls.legs_down else 0.0)
        s = veh.spec
        sig_now = max(3.0, veh.throttle * s.thrust_sl / m0)
        trim = self._lift_trim_many(veh)

        def model(times):
            times = np.atleast_1d(np.asarray(times, float))
            n = times.size
            t_end = float(times[-1]) + (0.5 * float(times[1] - times[0]) if n > 1 else 0.0)
            t0 = now - prev.created_at if prev is not None else 0.0
            span = max(0.0, prev.t_f - t0) if prev is not None else 0.0
            out = np.zeros((n, 3))
            G = np.zeros((n, 3, 3))
            qd = np.zeros(n)
            wd = np.zeros((n, 3))
            dq = np.zeros(n)
            if self.aero is None:
                return out, G, qd, wd, dq
            # ---- state along the reference
            h = np.empty(n)
            v = np.empty((n, 3))
            m = np.empty(n)
            sig = np.empty(n)
            u_ref = np.zeros((n, 3))
            has_ref = np.zeros(n, bool)
            guess = (times < 0.3) if prev is not None else np.ones(n, bool)
            if np.any(guess):
                tg = times[guess]
                T = max(t_guess, 0.5)
                f = np.maximum(0.0, 1.0 - tg / T)
                tt = np.minimum(tg, T)
                h[guess] = np.maximum(self.gate_height, r0[2] + v0[2] * (tt - 0.5 * tt * tt / T))
                v[guess] = v0 * f[:, None]
                m[guess] = m0
                sig[guess] = max(sig_now, 0.6 * s.thrust_sl / m0)
                if veh.throttle > 0.05:
                    u_ref[guess] = veh.last["thrust"] / m0
                    has_ref[guess] = True
            rest = ~guess
            if np.any(rest):
                tau = t0 + span * times[rest] / max(t_end, 1e-9)
                pos, vel, acc = prev.sample_many(tau)
                h[rest] = pos[:, 2]
                v[rest] = vel
                k = np.minimum((np.maximum(tau, 0.0) / prev.dt).astype(np.int64), prev.knots)
                m[rest] = np.exp(prev.z[k])
                kk = np.minimum((np.maximum(tau, 0.0) / prev.dt).astype(np.int64), prev.knots - 1)
                fr = (np.maximum(tau, 0.0) - kk * prev.dt) / prev.dt
                sig_lin = (1.0 - fr) * prev.sigma[kk] + fr * prev.sigma[kk + 1]
                sig[rest] = np.maximum(3.0, np.where(tau >= prev.t_f, prev.sigma[-1], sig_lin))
                u_ref[rest] = acc
                has_ref[rest] = True
            # ---- atmosphere, wind and the dynamic pressure at each knot
            rho, _, _, a_s = atmosphere_many(np.maximum(0.0, h + self.h_off))
            vr = v - self._wind_plan_many(h)
            sp = np.sqrt(np.sum(vr * vr, axis=1))
            qd = 0.5 * rho * sp * sp
            rho_p, _, _, _ = atmosphere_many(np.maximum(0.0, h + 5.0 + self.h_off))
            rho_m, _, _, _ = atmosphere_many(np.maximum(0.0, h - 5.0 + self.h_off))
            vp = v - self._wind_plan_many(h + 5.0)
            vm = v - self._wind_plan_many(h - 5.0)
            dq = (0.5 * rho_p * np.sum(vp * vp, axis=1) - 0.5 * rho_m * np.sum(vm * vm, axis=1)) / 10.0
            live = sp >= 3.0
            u = vr / np.maximum(sp, 1e-9)[:, None]
            wd[live] = -u[live]
            if self.aero_pred <= 0 or not np.any(live):
                return out, G, qd, wd, dq
            base = -u
            # Linearise around the reference attitude (successive
            # linearisation: each replan uses the previous plan's tilt).
            d_ref = u_ref / np.maximum(np.sqrt(np.sum(u_ref * u_ref, axis=1)), 1e-9)[:, None]
            tilt = d_ref - base * np.sum(d_ref * base, axis=1)[:, None]
            broadside = has_ref & (np.sqrt(np.sum(u_ref * u_ref, axis=1)) > 2.0) & (np.sum(d_ref * base, axis=1) < 0.2)
            # Flying (nearly) across the flow (a low divert): the body is
            # broadside and its drag is large -- take the force at that
            # attitude, without a lift gradient.
            tilt[broadside] = 0.0
            axis0 = base + tilt
            axis0 /= np.maximum(np.sqrt(np.sum(axis0 * axis0, axis=1)), 1e-9)[:, None]
            f0 = self.aero.quick_force_many(vr, axis0, rho, a_s, legs, trim=trim)
            out = f0 / m[:, None]
            if self.aero_pred == 1:
                out[~live] = 0.0
                return out, G, qd, wd, dq
            if np.any(broadside):
                fb = self.aero.quick_force_many(vr, d_ref, rho, a_s, legs, trim=trim)
                out[broadside] = fb[broadside] / m[broadside, None]
            grad = ~broadside & live
            if np.any(grad):
                e1 = np.cross(u, np.where((np.abs(u[:, 2]) < 0.9)[:, None],
                                          np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])))
                e1 /= np.maximum(np.sqrt(np.sum(e1 * e1, axis=1)), 1e-9)[:, None]
                e2 = np.cross(u, e1)
                d = math.radians(2.0)
                ax1 = axis0 + d * e1
                ax1 /= np.maximum(np.sqrt(np.sum(ax1 * ax1, axis=1)), 1e-9)[:, None]
                ax2 = axis0 + d * e2
                ax2 /= np.maximum(np.sqrt(np.sum(ax2 * ax2, axis=1)), 1e-9)[:, None]
                f1 = self.aero.quick_force_many(vr[grad], ax1[grad], rho[grad], a_s[grad], legs, trim=trim)
                f2 = self.aero.quick_force_many(vr[grad], ax2[grad], rho[grad], a_s[grad], legs, trim=trim)
                mg = m[grad, None]
                g1 = (f1 - f0[grad]) / d / mg
                g2 = (f2 - f0[grad]) / d / mg
                # Keep the net effect of a tilt in the thrust direction
                # (limits over-reliance on a local linearisation).
                cap = self.lift_gradient_cap * sig[grad]
                n1 = np.sqrt(np.sum(g1 * g1, axis=1))
                n2 = np.sqrt(np.sum(g2 * g2, axis=1))
                g1 *= np.minimum(1.0, cap / np.maximum(n1, 1e-12))[:, None]
                g2 *= np.minimum(1.0, cap / np.maximum(n2, 1e-12))[:, None]
                s_g = sig[grad, None]
                Gi = (np.einsum("ni,nj->nij", g1, e1[grad]) + np.einsum("ni,nj->nij", g2, e2[grad])) / s_g[:, :, None]
                out_g = f0[grad] / mg
                tilted = np.any(tilt[grad] != 0.0, axis=1)
                if np.any(tilted):
                    out_g[tilted] -= np.einsum("nij,nj->ni", Gi[tilted], u_ref[grad][tilted])
                out[grad] = out_g
                G[grad] = Gi
            out[~live] = 0.0
            return out, G, qd, wd, dq

        return model

    def _drag_model_scalar(self, veh, wind_now, t_guess):
        """Aerodynamics along the previous plan (or a decelerating guess).

        Returns ``(d0, G)``: d0 is the drag acceleration with the booster
        aligned with the flow; ``G @ u`` adds the body/grid-fin normal force
        produced by tilting the thrust vector (and the vehicle) by
        ``u_perp / sigma_ref`` against the flow.  For an engines-first
        booster this force points opposite to the tilt, which is why a plan
        that ignored it would overshoot laterally at high dynamic pressure.
        """
        prev = self._ref_plan()
        now = self.elapsed
        r0, v0, m0 = self._state(veh)
        legs = max(veh.legs, 1.0 if veh.controls.legs_down else 0.0)
        s = veh.spec
        sig_now = max(3.0, veh.throttle * s.thrust_sl / m0)

        def model(times):
            n = len(times)
            t_end = float(times[-1]) + (0.5 * float(times[1] - times[0]) if n > 1 else 0.0)
            t0 = now - prev.created_at if prev is not None else 0.0
            span = max(0.0, prev.t_f - t0) if prev is not None else 0.0
            out = np.zeros((n, 3))
            G = np.zeros((n, 3, 3))
            qd = np.zeros(n)
            wd = np.zeros((n, 3))
            dq = np.zeros(n)                # d(dynamic pressure)/d(height) along the reference
            if self.aero is None:
                return out, G, qd, wd, dq
            for i, t in enumerate(times):
                u_ref = None
                if prev is None or t < 0.3:
                    T = max(t_guess, 0.5)
                    f = max(0.0, 1.0 - t / T)
                    # height consistent with that velocity (its integral),
                    # not below the gate: the old guess r0 + v0 t / 2 ran far
                    # underground on a long horizon
                    tt = min(float(t), T)
                    h = max(float(self.gate_height), float(r0[2] + v0[2] * (tt - 0.5 * tt * tt / T)))
                    v, m, sig = v0 * f, m0, max(sig_now, 0.6 * s.thrust_sl / m0)
                    if t < 0.3 and veh.throttle > 0.05:
                        u_ref = veh.last["thrust"] / m0
                else:
                    # the previous plan's remainder stretched over this
                    # horizon (a retimed plan follows the same path; at the
                    # same clock times a shorter one would see the old,
                    # slower, still-high part of it)
                    tau = t0 + span * t / t_end
                    pos, vel, acc = prev.sample(tau)
                    h, v = pos[2], vel
                    k = min(int(max(tau, 0.0) / prev.dt), prev.knots)
                    m = float(math.exp(prev.z[k]))
                    sig = max(3.0, prev.sigma_at(tau))
                    u_ref = acc
                rho, _, _, a = atmosphere(max(0.0, h + self.h_off))
                vr = v - self._wind_plan(h)     # the wind where the plan will be, not the wind here
                sp = float(np.linalg.norm(vr))
                qd[i] = 0.5 * rho * sp * sp
                # lower down the wind is weaker (and the air a little denser):
                # how the dynamic pressure would change with the height
                qs = []
                for dh in (5.0, -5.0):
                    rho_d = atmosphere(max(0.0, h + dh + self.h_off))[0]
                    qs.append(0.5 * rho_d * float(np.linalg.norm(v - self._wind_plan(h + dh))) ** 2)
                dq[i] = (qs[0] - qs[1]) / 10.0
                if sp < 3.0:
                    continue
                u = vr / sp
                wd[i] = -u
                if self.aero_pred <= 0:
                    continue                # (the angle-of-attack cap above still sees the flow)
                base = -u
                # Linearise around the reference attitude (successive
                # linearisation: each replan uses the previous plan's tilt).
                tilt = np.zeros(3)
                broadside = None
                if u_ref is not None and float(np.linalg.norm(u_ref)) > 2.0:
                    d_ref = u_ref / np.linalg.norm(u_ref)
                    tilt = d_ref - base * float(np.dot(d_ref, base))
                    if float(np.dot(d_ref, base)) < 0.2:
                        # Flying (nearly) across the flow, e.g. a low divert:
                        # the body is broadside and its drag is large.  Take
                        # the force at that attitude, without a lift gradient
                        # (no useful linearisation that far from the flow).
                        tilt = np.zeros(3)
                        broadside = d_ref
                axis0 = base + tilt
                axis0 /= np.linalg.norm(axis0)
                if broadside is not None:
                    out[i] = self._quick(veh, vr, broadside, rho, a, legs) / m
                    continue
                f0 = self._quick(veh, vr, axis0, rho, a, legs)
                if self.aero_pred == 1:
                    out[i] = f0 / m         # drag at the reference attitude, no lift gradient
                    continue
                e1 = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(u, e1)
                d = math.radians(2.0)
                Gi = np.zeros((3, 3))
                for e in (e1, e2):
                    ax1 = axis0 + d * e
                    f1 = self._quick(veh, vr, ax1 / np.linalg.norm(ax1), rho, a, legs)
                    grad = (f1 - f0) / d / m                  # accel per radian of tilt along e
                    # Keep the net effect of a tilt in the thrust direction
                    # (limits over-reliance on a local linearisation).
                    gn = float(np.linalg.norm(grad))
                    if gn > self.lift_gradient_cap * sig:
                        grad *= self.lift_gradient_cap * sig / gn
                    Gi += np.outer(grad, e) / sig             # tilt along e ~ (e . u) / sigma
                out[i] = f0 / m - (Gi @ u_ref if u_ref is not None and np.any(tilt) else 0.0)
                G[i] = Gi
            return out, G, qd, wd, dq

        return model

    def _keep_out(self, veh, path=None):
        """Keep-out half-spaces for the planner, linearised along the previous
        plan if that one made it to the gate (successive convexification),
        otherwise along a reference path: ``path``, or the route chosen by
        ``_routes`` (first plan, or one stuck behind an obstacle, which would
        only reproduce itself)."""
        if not self.obstacles:
            return None
        now = self.elapsed
        boxes = self.obstacles
        prev = self._ref_plan()
        if path is not None or (prev is not None and not prev.reaches_pad):
            prev = None
        if prev is None and path is None:
            path = self._route if self._route is not None else detour_path(
                boxes, self._state(veh)[0], np.zeros(3), z_min=self._z_ground + 1.0)

        def planes(t_knots):
            t_knots = np.asarray(t_knots, float)
            t_end = max(float(t_knots[-1]), 1e-6)
            if prev is not None:
                # the previous path stretched over this horizon (see _drag_model)
                t0 = now - prev.created_at
                span = max(0.0, prev.t_f - t0)
                pts = list(prev.sample_many(t0 + span * t_knots / t_end)[0])
            else:
                pts = along_path(path, [float(t) / t_end for t in t_knots])
            return separating_planes(boxes, pts, target=np.zeros(3))

        return planes

    def _height_ref(self, veh):
        """Reference heights at the knot times (for the tilt budget): the
        previous plan stretched over the new horizon, else a straight descent."""
        prev = self._ref_plan()
        now = self.elapsed
        h0 = float(self._state(veh)[0][2])

        def heights(t_knots):
            t_knots = np.asarray(t_knots, float)
            t_end = max(float(t_knots[-1]), 1e-6)
            if prev is None:
                return list(h0 + (self.gate_height - h0) * t_knots / t_end)
            t0 = now - prev.created_at
            span = max(0.0, prev.t_f - t0)
            return list(prev.sample_many(t0 + span * t_knots / t_end)[0][:, 2])

        return heights

    def _dir_ref(self, veh):
        """Reference thrust direction at the knot times (for the planner's
        slew limit): the previous plan's, stretched over the new horizon like
        the heights; at ignition (no plan yet) the body axis."""
        prev = self._ref_plan()
        now = self.elapsed
        axis = np.asarray(veh.axis, float)

        def dirs(t_knots):
            tk = np.asarray(t_knots, float)
            if prev is None:
                return np.tile(axis, (tk.size, 1))
            t_end = max(float(tk[-1]), 1e-6)
            t0 = now - prev.created_at
            span = max(0.0, prev.t_f - t0)
            acc = prev.sample_many(t0 + span * tk / t_end)[2]
            n = np.sqrt(np.sum(acc * acc, axis=1))
            return np.where((n > 1e-6)[:, None], acc / np.maximum(n, 1e-9)[:, None], axis)

        return dirs

    def _config(self, veh, frac, keep_out=None):
        s = veh.spec
        return PlanConfig3D(
            thrust_min=self.minimum_throttle * s.thrust_sl * self.thrust_scale,
            thrust_max=frac * s.thrust_sl * self.thrust_scale, isp=s.isp_sl,
            dry_mass=s.dry_mass + 150.0, sigma_now=veh.throttle * s.thrust_sl / veh.mass,
            u_now=tuple(veh.last["thrust"] / veh.mass), engine_on=True,
            u_cmd=tuple(self.plan.sample(self.elapsed - self.plan.created_at)[2]) if self.plan is not None else None,
            max_tilt_deg=self.divert_tilt_deg,
            touchdown_speed=self.gate_speed, target_alt=self.gate_height, target_xy=tuple(self._gate_xy),
            sink_gain=self.sink_gain, keep_out=keep_out, q_aoa=self._aoa_table(veh), ground_alt=self._z_ground,
            body_length=s.length,
            tilt_trim=tuple(self._gate_trim(veh)), trim_fade=self._trim_fade(), tilt_budget=TILT_BUDGET,
            height_ref=self._height_ref(veh), slew_deg=SLEW_DEG, dir_ref=self._dir_ref(veh), time_weight=self.time_weight,
            min_sink=1.0 if self.catch_link is None else 0.0,
        )

    def _solver(self, veh, wind_now, frac, keep_out):
        """t_f -> plan from the current state (drag predicted for that horizon)."""
        r, v, m = self._state(veh)
        base = self._config(veh, frac, keep_out)

        def solve(t_f):
            if not t_f > 0.6 or self._out_of_time():
                return None
            return self.planner.solve(r, v, m, t_f, dataclasses.replace(base, drag_model=self._drag_model(veh, wind_now, t_f)),
                                      retry=not self._out_of_time())

        return solve

    @staticmethod
    def _thrust_available(veh, h):
        s = veh.spec
        _, p, _, _ = atmosphere(max(0.0, h))
        return s.thrust_vac - (s.thrust_vac - s.thrust_sl) * min(1.0, p / 101_325.0)

    # ------------------------------------------------------- predictions
    @staticmethod
    def _profile(z, k):
        """Log-law shear profile of strength k relative to the 10 m wind (vehicle.Wind)."""
        f = np.clip(np.log(np.maximum(0.5, z) / 0.05) / math.log(10.0 / 0.05), 0.25, 2.6)
        return np.maximum(0.05, 1.0 + k * (f - 1.0))

    def _wind(self, z):
        """Wind assumed at height ``z`` (planner coordinates) for the
        predictions: the wind measured now, scaled with the estimated profile."""
        w, z_now = self._wind_ref
        k = self.wind_shear_est
        shape = float(self._profile(z + self.h_off, k) / self._profile(z_now, k))
        return w * min(6.0, shape)

    def _wind_many(self, z):
        """Vectorised :meth:`_wind` over an array of heights."""
        w, z_now = self._wind_ref
        k = self.wind_shear_est
        shape = self._profile(np.asarray(z, float) + self.h_off, k) / self._profile(z_now, k)
        return w * np.minimum(6.0, shape)[:, None]

    def _wind_plan(self, z):
        """The wind the planner's predictions assume at height ``z``: the
        extrapolated measured wind times ``wind_plan_gain`` (a setting)."""
        return self._wind(z) * self.wind_plan_gain

    def _wind_plan_many(self, z):
        """Vectorised :meth:`_wind_plan` over an array of heights."""
        return self._wind_many(z) * self.wind_plan_gain

    def _update_trim(self, veh, wind_now, dt):
        """Filtered wind trim (sideways thrust acceleration that holds the
        upright booster against the aerodynamic force on it)."""
        if self.aero is None:
            return
        rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
        f = self.aero.quick_force(veh.vel - wind_now, np.array([0.0, 0.0, 1.0]), rho, a_s, veh.legs)
        target = -f[:2] / veh.mass
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(target))
        if n > cap:
            target *= cap / n
        target = target * self.trim_gain
        self._trim_xy += (target - self._trim_xy) * min(1.0, dt / self.trim_filter)

    def _update_hold_bias(self, veh, dt):
        """Filtered lateral thrust acceleration that cancels what the booster
        actually feels besides its commanded thrust (accelerometer: sensed
        specific force minus the command).  Unlike the modelled trim it
        includes the thrust lost to the gimbal holding the attitude against
        the wind's moment -- in a 15 m/s wind enough to push a fully trimmed,
        hovering booster off the catch axis at ~0.4 m/s^2.  Used over the gate
        (catch hold), where the commanded and the felt force are near steady."""
        sensed = (np.asarray(veh.last["thrust"], float) + np.asarray(veh.last["aero"], float))[:2] / veh.mass
        target = np.asarray(self.cmd_accel[:2], float) - sensed
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(target))
        if n > cap:
            target *= cap / n
        self._hold_bias += (target - self._hold_bias) * min(1.0, dt / self.hold_bias_filter)

    def _trim_fade(self):
        """Heights (above the end of the burn) over which the wind trim fades
        to upright.  A tower catch does not end in a touchdown: the booster
        holds at the gate, leaning into the wind, and only comes upright once
        the arms have closed -- fading the trim over the last metres above the
        gate would let the wind push it off just before the hold."""
        return (-2.0, -1.0) if self.catch_link is not None else TRIM_FADE

    def _gate_trim(self, veh) -> np.ndarray:
        """Wind trim (sideways thrust acceleration, m/s^2) near the end of the
        burn in the predicted wind: the planner's tilt cone is centred on it."""
        if self.aero is None:
            return np.zeros(2)
        z = self.gate_height + 2.0
        rho, _, _, a_s = atmosphere(max(0.0, z + self.h_off))
        vr = np.array([0.0, 0.0, -self.gate_speed]) - self._wind_plan(z)
        t = -self.aero.quick_force(vr, np.array([0.0, 0.0, 1.0]), rho, a_s, 1.0)[:2] / veh.mass
        cap = math.tan(math.radians(self.trim_max_deg)) * G0
        n = float(np.linalg.norm(t))
        return (t * (cap / n) if n > cap else t) * self.trim_gain

    def _update_wind_profile(self, veh, wind_now):
        """Record the measured wind and refit the shear strength (every 1 s)."""
        if self.elapsed >= self._next_wind_sample:
            self._next_wind_sample = self.elapsed + self.wind_sample_period
            self._wind_samples.append((self.elapsed, max(0.5, float(veh.altitude)), float(wind_now[0]), float(wind_now[1])))
            cut = self.elapsed - 3.0 * self.wind_memory
            while self._wind_samples and self._wind_samples[0][0] < cut:
                self._wind_samples.pop(0)
        if not self.shear_fit:
            self.wind_shear_est = 1.0
            return
        if self.elapsed < self._next_shear_fit or len(self._wind_samples) < 8:
            return
        self._next_shear_fit = self.elapsed + 1.0
        a = np.asarray(self._wind_samples)
        wt = np.exp(-(self.elapsed - a[:, 0]) / self.wind_memory)
        w = a[:, 2:4]
        scale = float(np.sum(wt * np.sum(w * w, axis=1)) / np.sum(wt)) + 1e-6
        best = None
        for k in np.linspace(0.0, 3.0, 31):
            sh = self._profile(a[:, 1], k)
            c = (wt * sh) @ w / max(1e-9, float(np.sum(wt * sh * sh)))       # 10 m wind (2-vector)
            res = w - np.outer(sh, c)
            cost = float(np.sum(wt * np.sum(res * res, axis=1)) / np.sum(wt)) / scale
            cost += self.wind_shear_prior_weight * (k - 1.0) ** 2
            if best is None or cost < best[0]:
                best = (cost, float(k))
        self.wind_shear_est = best[1]

    def _burn_need(self, veh, r, v, m, drag_up):
        """(thrust fraction needed for a landing burn started now, burn time, thrust direction)."""
        g = G0
        vz = float(v[2])
        h_eff = r[2] - self.gate_height - 0.45 * max(0.0, -vz)   # engine spool-up margin
        if vz > -(self.touchdown_speed + 2.0) and h_eff > 5.0:
            return 0.0, math.inf, np.array([0.0, 0.0, 1.0])
        if h_eff <= 0.5:
            return 9.0, 1.0, np.array([0.0, 0.0, 1.0])
        a_v = max(0.1, (vz * vz - self.gate_speed ** 2) / (2.0 * h_eff))
        t_b = max(0.5, (-vz - self.gate_speed) / a_v)
        a_lat = -(6.0 * r[:2] / t_b ** 2 + 4.0 * v[:2] / t_b) * 1.0
        need = np.array([a_lat[0], a_lat[1], a_v + g - drag_up / 3.0])
        frac = m * float(np.linalg.norm(need)) / self._thrust_available(veh, r[2])
        return frac, t_b, need / float(np.linalg.norm(need))

    def _ignition_threshold(self, t_burn, burn_dir):
        """Latest safe ignition: the required thrust fraction at which to light.

        A vertical burn lights at ``ignition_fraction``; the lateral share of
        the burn (sin of its tilt) lowers it, since the plan then also needs
        tilt and slew authority; short low burns light earlier so the burn
        lasts long enough to trim drift without a late sideways swing.
        """
        lateral = float(np.linalg.norm(burn_dir[:2]))
        ign = self.ignition_fraction - self.ignition_lateral_cut * min(1.0, lateral / 0.5)
        if t_burn < 8.0:
            ign = float(np.interp(t_burn, [2.0, 5.0, 8.0], [self.ignition_short, self.ignition_mid, ign]))
        return ign

    def _ballistic(self, veh, r, v, m, wind_now, dt=0.25, t_max=240.0, stop_on_ignition=True, blind=False):
        """Drag-aware coast prediction. Returns (points, impact_xy, t_impact, t_ignition, burn_time).

        ``blind``: in still air, and only the impact (no glide-target state)."""
        pts = [r.copy()]
        p, u = r.copy(), v.copy()
        t = 0.0
        t_ign, t_burn = math.inf, math.inf
        while t < t_max:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            wv = np.zeros(3) if blind else self._wind(p[2])
            f = self.aero.quick_force(u - wv, None, rho, a, 0.0) if self.aero is not None else np.zeros(3)
            if stop_on_ignition and not math.isfinite(t_ign):
                frac, tb, bd = self._burn_need(veh, p, u, m, max(0.0, f[2] / m))
                if frac >= self._ignition_threshold(tb, bd):
                    t_ign, t_burn = t, tb
                    ign_p, ign_v = p.copy(), u.copy()
            acc = f / m + np.array([0.0, 0.0, -G0])
            u = u + acc * dt
            p = p + u * dt
            t += dt
            if len(pts) < 400:
                pts.append(p.copy())
            if p[2] <= 0.0:
                break
        impact = p[:2].copy()
        if blind:
            return pts, impact, t, t_ign, t_burn
        # State at the predicted ignition (or at the impact): the glide target.
        self._ign_state = (t_ign, ign_p, ign_v, t_burn) if math.isfinite(t_ign) else (t, p.copy(), u.copy(), 0.0)
        self._burn_miss_now, self._brake_time = (self._burn_miss(veh, ign_p, ign_v, m, wind_now) if math.isfinite(t_ign)
                                                 else (p[:2].copy(), 0.0))
        if math.isfinite(t_ign):
            # Visual prediction: coast until ignition, then a smooth arc to the pad.
            k = int(t_ign / dt)
            pts = pts[:k + 1]
            p0, v0 = ign_p, ign_v
            T = max(t_burn, 0.5)
            v_end = np.array([0.0, 0.0, -self.touchdown_speed])
            for s in np.linspace(0.0, 1.0, 30)[1:]:
                # Cubic Hermite from (p0, v0) to (pad, touchdown velocity); the pad is the origin.
                h00, h10, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, s ** 3 - s ** 2
                pts.append(h00 * p0 + h10 * T * v0 + h11 * T * v_end)
        return pts, impact, t, t_ign, t_burn

    # ------------------------------------------------------------- replan
    def _replan(self, veh, wind_now):
        """Re-plan from the current state.  The touchdown time stays where it
        is (smooth tracking) unless the cost J = propellant + time_weight t_f
        is lower a step earlier or later; a plan that misses the gate gets a
        full search of the final time (and the route).  The optimiser alone
        decides how the booster gets there.

        A re-plan is bounded by ``plan_budget``: a search over the final time
        is a dozen solves, and when the pad is out of reach every replan would
        spend them, so the planner fell seconds behind the vehicle and the
        booster flew a stale plan into the ground.  Over budget the search
        stops and keeps the best plan it has (the caller still commits it)."""
        t_go = self.touchdown_time - self.elapsed
        self._plan_deadline = time.monotonic() + (self.plan_budget_short if t_go < 8.0 else self.plan_budget)
        solve = self._solver(veh, wind_now, self.burn_fraction, self._keep_out(veh))
        if self.plan is not None and not self.plan.sane:
            self.plan = None            # a runaway plan is no plan: fly the fallback law until a real one is found
        plan = solve(t_go)
        if plan is not None and not plan.sane:
            plan = None
        lost = plan is None or plan.terminal_position_error > 10.0
        if self.elapsed >= self._next_search and (t_go > self.retime_freeze or lost):
            if plan is None or not plan.reaches_pad:
                # A search is a dozen solves (route candidates + golden
                # section): give it its own budget -- cut in the middle it
                # returns a plan that still misses the pad, and the guidance
                # then keeps missing it (the catch scenarios must go over the
                # tower, which the search is what discovers).
                self._plan_deadline = max(self._plan_deadline, time.monotonic() + self.plan_budget_search)
                best = self._search_time(veh, wind_now, t_go)
                self._search_fails = 0 if best is not None and best.reaches_pad else min(self._search_fails + 1, 2)
                if best is not None and (plan is None or best.objective < plan.objective):
                    plan = best
            else:
                d = max(0.4, self.retime_step * t_go)
                for t in (t_go - d, t_go + d):
                    if self._out_of_time():
                        break
                    p = solve(t)
                    if p is not None and p.reaches_pad and p.objective < plan.objective:
                        plan = p
                self._search_fails = 0
            self._next_search = self.elapsed + self.search_backoff * 2 ** self._search_fails
        self.last_status = self.planner.last_status
        if plan is not None:
            self._commit(plan)
        else:
            # no solution at all this time (the solver gave up or found the
            # problem infeasible): the booster flies on along the last plan --
            # until that one has ended (a plan's end is "on the pad, sinking
            # slowly", which holds a booster that is nowhere near it in the
            # air: the endless hover) or has gone stale (a plan the vehicle
            # cannot follow, and nothing else solves): then the fallback law
            # brings it down
            if self.touchdown_time - self.elapsed > 1.0:
                # (in the last second there is nothing left to plan: the plan
                # has ended as far as the solver is concerned, that is not a
                # solver failure to show)
                self.solver_fail = (self.elapsed, self.planner.last_status)
            if self.plan is not None and (self.touchdown_time - self.elapsed <= 0.6
                                          or self.elapsed - self.plan.created_at > self.plan_stale_after):
                self.plan = None

    def solver_alert(self):
        """For the display: None, or ('fail', status) when the last re-plans
        found no solution (shown for 2 s)."""
        if not self.burn_locked or self._holding:
            # holding at the gate (or after the burn) nothing is planned:
            # the plan in hand is stale and says nothing
            return None
        if self._ignition_searching:
            # the ignition search is still running: there is no plan yet by
            # design, that is not a solver failure
            return None
        if self.solver_fail is not None and self.elapsed - self.solver_fail[0] < 2.0:
            return ("fail", self.solver_fail[1])
        # A plan that does not reach the gate is simply re-searched (reaches_pad); it
        # is no alert -- only a missing solution is shown.
        return None

    def _commit(self, plan):
        plan.created_at = self.elapsed
        self.plan = plan
        self.touchdown_time = self.elapsed + plan.t_f
        self.last_status = plan.status
        if plan.reaches_pad:
            self._route = None          # from now on along this plan

    def _search_time(self, veh, wind_now, t_guess):
        """Final time minimising J (golden section), on the best route round
        the obstacles; at full thrust if nothing reaches at the nominal one."""
        r, v, m = self._state(veh)
        t_far = self._far_time(r, v)
        lo = max(1.0, 0.35 * min(t_guess, t_far))
        hi = 1.5 * max(t_guess, t_far) + 2.0
        # No longer than the propellant lasts at minimum throttle (the engine
        # cannot go below it: a plan beyond that burns propellant that is not
        # there, and the solver then returns nonsense).
        t_cap = self._burn_time_limit(veh, m)
        if t_cap < hi:
            hi = max(3.0, t_cap)
            lo = min(lo, 0.6 * hi)
        best = None
        for frac in dict.fromkeys((self.burn_fraction, 0.97)):
            if self._out_of_time() and best is not None:
                break                   # the budget: keep what the search has
            keep_out = self._keep_out(veh, self._routes(veh, wind_now, frac, 0.5 * (lo + hi)))
            p = golden_section_search(self._solver(veh, wind_now, frac, keep_out), lo, hi, self.time_search_evals)
            if p is not None and (best is None or p.objective < best.objective):
                best = p
                if p.reaches_pad:
                    self.burn_fraction = max(self.burn_fraction, frac)
            if best is not None and best.reaches_pad:
                break
        if best is not None and self._ref_plan() is None and not self._out_of_time():
            # Found about a straight descent, not about a plan: in a strong
            # wind that reference hovers in the flow and flatters the plan
            # (it "reaches" the pad where the plan itself would not).  Check
            # it about itself, as at ignition.
            best = self._verify(veh, wind_now, best)
        if best is not None and not best.sane:
            # The best plan found is a runaway (it trades the landing speed or
            # propellant that is not there for distance): not a plan at all.
            return None
        return best

    def _verify(self, veh, wind_now, plan):
        """Check of a plan found about a straight descent: ``plan`` itself if
        it still gets there when re-solved ``ignition_refine`` times about
        itself (the reference of the aerodynamics, route and heights, at its
        own final time and thrust ceiling), else the re-solved plan that does
        not -- it never got there."""
        held = self.plan
        p = plan
        try:
            for _ in range(self.ignition_refine):
                if self._out_of_time():
                    return plan
                p.created_at = self.elapsed             # (it is the reference: samples run from now)
                self.plan = p
                q = self._solver(veh, wind_now, self.burn_fraction, self._keep_out(veh))(p.t_f)
                if q is None:
                    return plan
                p = q
                if not p.reaches_pad:
                    return p
        finally:
            self.plan = held
        return plan

    def _burn_time_limit(self, veh, m):
        """Longest burn the propellant allows at minimum throttle (with a margin)."""
        s = veh.spec
        mdot = self.minimum_throttle * s.thrust_sl * self.thrust_scale / (s.isp_sl * G0)
        return 0.92 * max(0.0, m - (s.dry_mass + 150.0)) / max(mdot, 1e-6)

    def _routes(self, veh, wind_now, frac, t_f):
        """Which way round the obstacles (over, round either side): one convex
        problem per candidate reference, the cheapest wins.  Returns the
        chosen reference path (None: no obstacles)."""
        if not self.obstacles:
            return None
        paths = detour_paths(self.obstacles, self._state(veh)[0], np.zeros(3), z_min=self._z_ground + 1.0)
        best = None
        for path in paths:
            if self._out_of_time() and best is not None:
                break
            p = self._solver(veh, wind_now, frac, self._keep_out(veh, path))(t_f)
            if p is not None and (best is None or p.objective < best[0]):
                best = (p.objective, path)
        self._route = best[1] if best is not None else paths[0]
        return self._route

    def _far_time(self, r, v):
        """A generous time to the gate from here: a vertical fall plus a
        sideways transfer at a modest acceleration."""
        h, vz = max(0.0, float(r[2]) - self.gate_height), float(v[2])
        t_fall = (vz + math.sqrt(vz * vz + 2.0 * G0 * h)) / G0
        d = float(np.linalg.norm(r[:2] - self._gate_xy))
        return t_fall + 2.0 * math.sqrt(d / 2.0) + float(np.linalg.norm(v[:2])) / 3.0

    def _publish(self):
        p = self.plan
        if p is None:
            return
        start = max(0.0, self.elapsed - p.created_at)
        n = max(2, min(200, int(max(0.0, p.t_f - start) / 0.1) + 2))
        pos, _, _ = p.sample_many(np.linspace(start, p.t_f, n))
        self.predicted_points = [(float(x), float(y), float(max(0.0, z)) + self.h_off) for x, y, z in pos]

    # ------------------------------------------------------------ command
    def command(self, veh, wind_now, dt):
        c = veh.controls
        if veh.state in ("CRASHED", "LANDED", "CAUGHT"):
            c.throttle = 0.0
            c.rcs = (0.0, 0.0, 0.0)
            c.gimbal = (0.0, 0.0)
            c.fins = (0.0, 0.0, 0.0)
            self.phase = "LANDED" if veh.state in ("LANDED", "CAUGHT") else "STANDBY"
            self._mark()
            return
        if veh.state == "ON PAD":
            c.throttle = 0.0
            self.phase = "STANDBY"
            return
        self.elapsed += dt
        self._dt = dt
        self._wind_ref = (np.asarray(wind_now, float).copy(), max(0.5, float(veh.altitude)))
        self._update_wind_profile(veh, wind_now)
        self._update_lift_gain(veh, wind_now)
        s = veh.spec
        r, v, m = self._state(veh)
        g = G0
        h = r[2]

        # Touchdown: cut the engine as soon as a foot is on the ground (or the
        # catch collar is on the arms).
        if veh.feet_contact > 0 or veh.catch_contact > 0 or self._touched:
            self._touched = True
            c.throttle = 0.0
            self.phase = "LANDING BURN"
            self._attitude(veh, np.array([0.0, 0.0, 1.0]), use_rcs=True)
            self._mark()
            return

        if h < self.legs_deploy_height and self.use_legs:
            c.legs_down = True
        if self.burn_locked:
            c.fins_deploy = True
            self._update_trim(veh, wind_now, dt)
            self._update_hold_bias(veh, dt)
            # dynamic pressure for the angle-of-attack limit, smoothed: the
            # booster's own sway and the gusts swing it by +-50 % in a strong
            # cross wind, and the limit (steep in q there) would flick on and off
            self._q_filt += (float(veh.last.get("q", 0.0)) - self._q_filt) * min(1.0, dt / self.q_filter)

        if not self.burn_locked:
            if self.async_plan:
                # pick up a finished ignition preparation, and start one when
                # the ignition is coming up (the search then costs the switch
                # nothing)
                self._apply_plan_result()
                if self.ignition_in <= self.ign_prep_lead:
                    self._submit_prep(veh, wind_now)
            self._pre_ignition(veh, r, v, m, wind_now)
            return

        # Engine performance from the measured thrust (accelerometer): a
        # weaker-than-modelled engine shrinks the planner's thrust limits, so
        # the plan does not count on braking it cannot deliver.
        t_model = veh.throttle * self._thrust_available(veh, h)
        if veh.throttle > 0.3 and t_model > 0.0:
            ratio = float(np.clip(np.linalg.norm(veh.last["thrust"]) / t_model, 0.6, 1.1))
            self.thrust_scale += (ratio - self.thrust_scale) * min(1.0, dt / 0.5)

        t_go = self.touchdown_time - self.elapsed
        # One burn all the way to leg contact: re-plan until the last ~0.6 s,
        # then fly the final plan.  Past its end the plan's reference is
        # "on the pad, sinking at touchdown speed", so a late vehicle keeps
        # descending instead of hovering.
        # Past the plan's end but still high or off the pad (the plan could
        # not be tracked): re-plan from here instead of trusting the old end.
        if self.catch_link is not None and (self._holding or (
                (t_go <= 0.6 or h <= self.gate_height + 0.2) and h < self.gate_height + 5.0
                and float(np.linalg.norm(r[:2])) < 6.0)):
            self._catch_hold(veh, r, v, dt)
            return
        stranded = t_go <= 0.6 and (h > 20.0 or float(np.linalg.norm(r[:2])) > 15.0)
        if self.async_plan:
            # The solve runs on the planner thread: this step only commits a
            # finished plan (and hands over a new one), it never waits for one.
            if self._apply_plan_result():
                self._publish()
                t_go = self.touchdown_time - self.elapsed
            if (not self._plan_busy() and self.replan_enabled and self.planner.available
                    and self.elapsed >= self.next_replan and (t_go > 0.6 or stranded)):
                self._submit_plan(veh, wind_now)
                self.next_replan = self.elapsed + float(np.interp(max(t_go, 0.0), [8.0, 30.0],
                                                                  [self.burn_replan_period, 0.6]))
        elif self.replan_enabled and self.planner.available and self.elapsed >= self.next_replan and (t_go > 0.6 or stranded):
            self._replan(veh, wind_now)
            self._publish()
            t_go = self.touchdown_time - self.elapsed
            # Long burns re-plan less often early on (each solve of a long,
            # 70-knot horizon costs ~0.1 s); 5 Hz over the last ~8 s.
            self.next_replan = self.elapsed + float(np.interp(t_go, [8.0, 30.0], [self.burn_replan_period, 0.6]))
        self.time_to_go = max(0.0, t_go)
        if t_go < 8.0 and self.use_legs:
            c.legs_down = True

        if self.plan is not None:
            self.phase = "LANDING BURN"
            tau = self.elapsed - self.plan.created_at
            p_ref, v_ref, u_ff = self.plan.sample(tau)
            kp, kd, lim = (0.5, 1.6, 5.0) if self.replan_enabled else (self.frozen_kp, self.frozen_kd, self.frozen_fb_limit)
            fb = kp * (p_ref - r) + kd * (v_ref - v)
            nfb = float(np.linalg.norm(fb))
            if nfb > lim:
                fb *= lim / nfb
            if not self.replan_enabled and self.frozen_net_gain:
                fb = self._net_gain_feedback(veh, wind_now, u_ff, fb)
            accel = u_ff + fb
            if h < self.sink_track_height:
                # Final approach envelope (same as the planner's): if sinking
                # faster than 1.2 + 0.5 h m/s, brake the excess.  Near the pad
                # the sink rate settles at touchdown speed with thrust ~ weight,
                # so cutting the engine at contact cannot bounce the vehicle.
                v_lim = -(self.gate_speed + self.sink_gain * max(0.0, h))
                if v[2] < v_lim:
                    accel[2] += 2.0 * (v_lim - v[2])
            if self.elapsed - self.plan.created_at > self.brake_floor_age:
                accel = self._braking_floor(veh, r, v, m, accel)
                if self._commit_to_ground(veh, r, v):
                    c.legs_down = True
            # Tilt budget tightens with height (upright at contact) and with
            # dynamic pressure: there the body lift of an engines-first
            # booster opposes its tilt and outweighs the thrust's sideways
            # part, so more tilt would push the wrong way (and feed back).
            max_tilt = math.radians(float(np.interp(h, *TILT_BUDGET)))
            center = self._trim_xy * float(np.interp(h, self._trim_fade(), (0.0, 1.0)))
            trim_tilt = math.atan2(float(np.linalg.norm(center)), G0)
            accel = self._aoa_limit(veh, accel, wind_now, max_tilt + trim_tilt, self._aoa_table(veh), self._q_filt)
        else:
            # No plan (none that can be trusted): a plain vertical suicide
            # burn, down wherever the booster is.  The pull towards the pad is
            # capped (a far-off pad is not worth leaving the descent for), and
            # the engine may stay off while the fall is still slower than the
            # braking profile -- at minimum throttle it out-thrusts the weight
            # and would carry the booster up.
            self.phase = "FALLBACK"
            a_max_now = s.thrust_sl / m
            a_brake = 0.35 * (a_max_now - g)        # the deceleration of the braking profile ...
            # (down to the ground, not to the end of the planned burn: a tower
            # catch's end point is 30 m up, and the legs come out here)
            vz_ref = -math.sqrt(2 * a_brake * max(float(veh.altitude), 0.0)) - self.touchdown_speed
            if self.use_legs:
                if float(veh.altitude) < self.legs_deploy_height:
                    c.legs_down = True
            elif self._commit_to_ground(veh, r, v):
                c.legs_down = True
            pull = -0.08 * r[:2]
            pn = float(np.linalg.norm(pull))
            if pn > self.fallback_pull_max:
                pull *= self.fallback_pull_max / pn
            accel = np.array([pull[0] - 0.9 * v[0], pull[1] - 0.9 * v[1], g + a_brake + 1.5 * (vz_ref - v[2])])   # ... fed forward
            accel = self._braking_floor(veh, r, v, m, accel)
            max_tilt = math.radians(35.0)
            center = None
            if self.aero is not None:
                accel = self._aoa_limit(veh, accel, wind_now, max_tilt, self._aoa_table(veh), self._q_filt)
        self._mark()
        self._fly_accel(veh, accel, max_tilt, floor=0.9 * self.minimum_throttle if h > 1.0 and (self.plan is not None or accel[2] > 0.7 * g) else None, center=center)

    def _given_up_catch(self, r) -> bool:
        """A tower catch with the booster nowhere near the tower any more
        (``catch_far_distance`` from the catch axis): with no plan to get
        back it comes down on the ground, not on the arms."""
        return (not self.use_legs) and float(np.linalg.norm(r[:2])) > self.catch_far_distance

    def _commit_to_ground(self, veh, r, v) -> bool:
        """Put the legs out for a given-up catch (``_given_up_catch``) at the
        last moment: below ``ground_legs_height`` (the feet then reach the
        ground before the hull does).  Not before: the legs cannot be folded
        again, and a booster that is only recovering from an overshoot (low,
        far off, climbing back) would lose its catch."""
        return self._given_up_catch(r) and float(veh.altitude) < self.ground_legs_height

    def _braking_floor(self, veh, r, v, m, accel):
        """The vertical thrust acceleration of the command, raised to what it
        takes to stop the fall at the gate with the thrust the engine has.

        A plan that is no longer renewed (nothing solves) lags behind the
        state, and the plan-less law is sized for a fall from higher up; the
        stopping distance does not care.  From the height ``h`` and sink
        ``vz`` a constant deceleration ``a = (vz^2 - v_gate^2) / 2 h`` brings
        it to the gate speed there; the thrust has to carry ``g + a`` less the
        drag that is helping now.  Asking for that (``brake_margin`` more, up
        to ``brake_floor_limit`` of the available thrust) keeps the booster
        from being flown into the ground with the engine still throttled
        back; a command that already brakes harder is left alone."""
        if not self.brake_floor:
            return accel
        vz = float(v[2])
        if self._given_up_catch(r):
            h_stop, v_gate = float(veh.altitude), 2.0          # down to the ground, on the legs
        else:
            h_stop, v_gate = float(r[2]) - self.gate_height, self.gate_speed
        h_eff = h_stop - 0.3 * max(0.0, -vz)                    # spool-up margin
        if vz > -(v_gate + 0.5) or h_eff < 0.3:
            return accel
        a_v = (vz * vz - v_gate ** 2) / (2.0 * max(h_eff, 0.5))
        aero = veh.last.get("aero")
        drag_up = max(0.0, float(aero[2]) / m) if aero is not None else 0.0
        need = self.brake_margin * a_v + G0 - drag_up
        cap = self.brake_floor_limit * self._thrust_available(veh, float(r[2])) * self.thrust_scale / m
        need = min(need, cap)
        if need <= float(accel[2]):
            return accel
        out = np.array(accel, float)
        out[2] = need
        return out

    def _catch_hold(self, veh, r, v, dt):
        """Tower catch: hold the collar just above the arm tops, centred on the
        catch axis, until the tower reports the arms closed round the barrel;
        then settle onto them (the engine is cut at contact).  The arms close
        only on a booster that is in their capture window and slow, so a
        booster that arrives a little off is caught after it has centred,
        instead of drifting past while they close."""
        if not self._holding:
            self._holding = True
            self._hold_t = 0.0
            self._hold_i = np.zeros(2)
            self._hold_h = self.gate_height
            self._secured = False
        self._hold_t += dt
        # latched: the arms touching the barrel can let go for a moment as
        # they push it to the centre; flipping between holding (leaning into
        # the wind) and settling (upright) on that rocks the booster
        self._secured = self._secured or bool(self.catch_link())
        settle = self._secured or self._hold_t > self.catch_hold_max or veh.propellant < 80.0
        if settle:
            vz_ref = -self.catch_settle_speed
            self._hold_h -= self.catch_settle_speed * dt
        else:
            vz_ref = 0.0
        e = -np.asarray(r[:2], float)
        if float(np.linalg.norm(e)) < self.catch_integral_radius:
            # integrate only near the axis: while it is still coming in from
            # metres off (arrival overshoot) the sum would wind up and carry
            # it past the axis, out of the capture window on the other side
            self._hold_i = np.clip(self._hold_i + e * dt, -4.0, 4.0)
        # settling: the closed arms hold it sideways -- gentle and upright
        k_xy = 0.5 if settle else 1.0
        tilt = math.radians(1.0 if settle else self.catch_hold_tilt_deg)
        # Towards the axis no faster than it can stop with half the tilt
        # budget: a saturated linear PD brakes too late and swings metres
        # past the axis in a strong wind.  Near the axis this is the plain PD
        # (0.6 e - 1.6 v).
        en = float(np.linalg.norm(e))
        v_ref = np.zeros(2)
        if en > 1e-6:
            v_ref = e / en * min(0.375 * en, math.sqrt(math.tan(tilt) * G0 * en))
        a_xy = k_xy * (1.6 * (v_ref - v[:2]) + 0.1 * self._hold_i)
        a_z = G0 + 1.2 * (self._hold_h - r[2]) + 2.0 * (vz_ref - v[2])
        self.phase = "CATCH HOLD"
        self.time_to_go = max(0.0, float(r[2]) / self.catch_settle_speed) if settle else math.nan
        self._mark()
        # Holding: lean into the wind (else it drifts out of the capture
        # window and the arms never close), centred on the measured bias.
        # Settling on closed arms: upright, they hold it sideways.
        secured = self._secured
        self._fly_accel(veh, np.array([a_xy[0], a_xy[1], a_z]), tilt, floor=0.9 * self.minimum_throttle,
                        center=None if secured else self._hold_bias)

    def _aoa_table(self, veh):
        """Angle-of-attack limit vs dynamic pressure, from physics: the
        largest angle between the thrust (body axis) and the oncoming flow,
        counted from engines-first, at which the aerodynamic moment stays
        within ``aoa_moment_margin`` of what the controls can hold: the
        gimbal (thrust ~ weight, at the current mass and centre of mass) and
        the four grid fins at full deflection (each fin in its own local
        flow, so their moment grows with the dynamic pressure like the
        aerodynamic one).  Both moments are taken from the models (per Pa at
        each angle, refreshed every ``aoa_refresh``).  A hovering booster in a cross wind leans into
        it at ~90-110 deg from the flow: what limits it there is the dynamic
        pressure, hardly the lean (leaning 20 deg adds ~20 % moment), and
        near the ground, where the wind is weaker, it can lean much more."""
        t_last, table = self._aoa_cache
        if self.aero is None or self.elapsed - t_last < self.aoa_refresh:
            return table
        s = veh.spec
        m, zc, _, _ = veh.mass_properties()
        m_gim = m * G0 * math.sin(s.gimbal_limit) * max(0.5, zc - s.gimbal_z)
        rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
        q_ref = 1000.0
        sp = math.sqrt(2.0 * q_ref / rho)
        last = getattr(self.aero, "last", None)
        angles = np.arange(0.0, 181.0, 10.0)
        k, k_fin = [], []
        w = np.array([1.0, 0.0, 0.0])                     # where the air goes, relative to the booster
        patterns = [np.array(p, float) * s.fin_max_defl for p in ((1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (1, -1, -1, 1))]
        for ang in angles:
            a = math.radians(float(ang))
            axis = np.array([math.cos(a), 0.0, math.sin(a)])
            # rotation taking the body z axis to ``axis`` (about body y)
            R = np.array([[math.sin(a), 0.0, math.cos(a)], [0.0, 1.0, 0.0], [-math.cos(a), 0.0, math.sin(a)]])
            assert np.allclose(R[:, 2], axis)
            _, mom, _ = self.aero(_AttitudeProbe(veh, R), -w * sp, rho, a_s, veh.com_world())
            k.append(float(np.linalg.norm(mom)) / q_ref)
            # what the four fins can add (pitch/yaw, best deflection pattern)
            _, m0, _ = grid_fin_forces(s, R, zc, np.zeros(3), -w * sp, rho, a_s, np.zeros(4), 1.0)
            best = 0.0
            for d in patterns:
                _, m1, _ = grid_fin_forces(s, R, zc, np.zeros(3), -w * sp, rho, a_s, d, 1.0)
                dm = m1 - m0
                best = max(best, float(np.linalg.norm(dm - axis * float(np.dot(dm, axis)))))
            k_fin.append(best / q_ref)
        if last is not None:
            self.aero.last = last
        # highest dynamic pressure each angle can be held at:
        #   K q <= margin (M_gimbal + K_fin q)
        mg = self.aoa_moment_margin
        excess = np.asarray(k) - mg * np.asarray(k_fin)
        q_max = np.where(excess > 0.0, mg * m_gim / np.maximum(excess, 1e-12), np.inf)
        q_max = np.minimum.accumulate(q_max)                # the cone from engines-first must hold throughout
        out = []
        for q in AOA_Q_GRID:
            ok = q_max >= q
            if ok.all():
                out.append(180.0)
            elif not ok[0]:
                out.append(0.0)
            else:
                j = int(np.argmin(ok))                      # first angle that cannot be held
                # interpolate in 1/q_max between the last good and first bad angle
                a0, a1 = angles[j - 1], angles[j]
                r0, r1 = 1.0 / q_max[j - 1], 1.0 / q_max[j]
                out.append(float(a0 + (a1 - a0) * (1.0 / q - r0) / max(r1 - r0, 1e-12)))
        table = (AOA_Q_GRID, tuple(out))
        self._aoa_cache = (self.elapsed, table)
        return table

    def _lift_trim(self, veh):
        """The trimmed-lift factor ``kappa(theta, q)`` (see ``AeroModel.quick_force``).

        The attitude loop cancels the aerodynamic moment of the body with the
        grid fins first (least squares on their measured effectiveness,
        commands within +-1), and only what is left with the gimbal and the
        RCS.  So the lift that really acts on the booster is that of the body
        plus the fins *as deflected for that*, not at zero deflection: it is
        taken here from the same models and the same allocation as the loop,
        per angle ``theta`` from engines-first (every ``aoa_refresh``, at a
        reference dynamic pressure: body and fins both scale with q, so the
        deflection needed does not).  Below the fins' effectiveness threshold
        the loop leaves them neutral, hence the fade with q."""
        if not self.lift_trim or self.aero is None:
            return None
        t_last, cached = self._trim_cache
        if self.elapsed - t_last < self.aoa_refresh and cached is not None:
            return cached
        s = veh.spec
        _, zc, _, _ = veh.mass_properties()
        rho, _, _, a_s = atmosphere(max(0.0, float(veh.pos[2])))
        q_ref = 1000.0
        sp = math.sqrt(2.0 * q_ref / rho)
        last = getattr(self.aero, "last", None)
        w = np.array([1.0, 0.0, 0.0])
        v_rel = -w * sp
        angles = np.arange(0.0, 181.0, 10.0)
        kappas, b_refs = [], []
        zero3 = np.zeros(3)
        for ang in angles:
            a = math.radians(float(ang))
            axis = np.array([math.cos(a), 0.0, math.sin(a)])
            R = np.array([[math.sin(a), 0.0, math.cos(a)], [0.0, 1.0, 0.0], [-math.cos(a), 0.0, math.sin(a)]])
            body = _AttitudeProbe(veh, R)
            body.fins_deploy = 0.0
            f_b, m_b, _ = self.aero(body, v_rel, rho, a_s, veh.com_world())

            def fins(u, R=R):
                d = np.clip(fin_mix(u) * s.fin_max_defl, -s.fin_max_defl, s.fin_max_defl)
                f, m, _ = grid_fin_forces(s, R, zc, zero3, v_rel, rho, a_s, d, 1.0)
                return f, R.T @ m

            f0, m0 = fins(zero3)
            B = np.zeros((3, 3))
            for j in range(3):
                e = np.zeros(3)
                e[j] = 0.5
                B[:, j] = fins(e)[1] - fins(-e)[1]
            try:
                u, *_ = np.linalg.lstsq(B, -(R.T @ m_b) - m0, rcond=1e-3)
            except np.linalg.LinAlgError:
                u = np.zeros(3)
            f_trim = f_b + fins(np.clip(u, -1.0, 1.0))[0]
            f_zero = f_b + f0
            pz = f_zero - float(np.dot(f_zero, axis)) * axis
            pt = f_trim - float(np.dot(f_trim, axis)) * axis
            den = float(np.dot(pz, pz))
            kappas.append(float(np.clip(np.dot(pt, pz) / den, -0.5, 1.5)) if den > 1e-3 * (q_ref * 10.0) ** 2 else 1.0)
            b_refs.append(float(np.abs(B).max()))
        if last is not None:
            self.aero.last = last
        kappas, b_refs = np.asarray(kappas), np.asarray(b_refs)
        self._trim_table = (angles, kappas, b_refs, q_ref)

        def kappa(theta, q):
            th = math.degrees(theta)
            b = float(np.interp(th, angles, b_refs)) * q / q_ref
            fade = float(np.clip((b - FIN_MIN_EFF) / FIN_MIN_EFF, 0.0, 1.0))
            return (1.0 + fade * (float(np.interp(th, angles, kappas)) - 1.0)) * self._lift_gain

        self._trim_cache = (self.elapsed, kappa)
        return kappa

    def _lift_trim_many(self, veh):
        """Vectorised :meth:`_lift_trim` (``theta`` and ``q`` arrays)."""
        trim = self._lift_trim(veh)
        if trim is None:
            return None
        angles, kappas, b_refs, q_ref = self._trim_table

        def kappa(theta, q):
            th = np.degrees(np.asarray(theta, float))
            b = np.interp(th, angles, b_refs) * np.asarray(q, float) / q_ref
            fade = np.clip((b - FIN_MIN_EFF) / FIN_MIN_EFF, 0.0, 1.0)
            return (1.0 + fade * (np.interp(th, angles, kappas) - 1.0)) * self._lift_gain

        return kappa

    def _update_lift_gain(self, veh, wind_now):
        """Calibrate the trimmed-lift model against the flight: the ratio
        of the lift measured now (the aerodynamic force the accelerometer
        senses, less the drag of the booster aligned with the flow) to what
        the model gives at the attitude flown, least squares with an
        exponential memory.  The attitude loop adds more fin deflection than
        the pure trim (angle and rate terms, saturation), most at high
        dynamic pressure; the model scales by what is found.  Only with the
        fins out and a lift worth measuring."""
        if not (self.lift_trim and self.lift_calibrate) or self.aero is None or self.elapsed < self._next_lift_fit:
            return
        dt = 0.05
        self._next_lift_fit = self.elapsed + dt
        trim = self._lift_trim(veh)
        aero = veh.last.get("aero")
        if trim is None or aero is None or veh.fins_deploy < 0.9:
            return
        vr = veh.vel - wind_now
        sp = float(np.linalg.norm(vr))
        if sp < 20.0:
            return
        rho, _, _, a = atmosphere(max(0.0, float(veh.pos[2])))
        q = 0.5 * rho * sp * sp
        m = veh.mass
        base = self.aero.quick_force(vr, -vr / sp, rho, a, veh.legs)
        raw = self.aero.quick_force(vr, np.asarray(veh.axis, float), rho, a, veh.legs, trim=lambda th, qq: trim(th, qq) / self._lift_gain)
        l_model = (raw - base) / m
        l_meas = (np.asarray(aero, float) - base) / m
        decay = math.exp(-dt / self.lift_gain_tau)
        self._lift_num *= decay
        self._lift_den *= decay
        n2 = float(np.dot(l_model, l_model))
        if q > 300.0 and n2 > 0.09:                 # > 0.3 m/s^2
            self._lift_num += float(np.dot(l_model, l_meas))
            self._lift_den += n2
        if self._lift_den > 0.5:
            self._lift_gain = float(np.clip(self._lift_num / self._lift_den, 0.3, 1.5))

    def _net_gain_feedback(self, veh, wind_now, u_ff, fb):
        """The sideways feedback of the PD loop, converted from a wish for net
        acceleration into a thrust change.

        Tilting the thrust by ``d`` moves the booster by ``u d`` sideways, and
        the lift of its body by ``G u d`` -- against it, and at high dynamic
        pressure by more than the thrust (the planner linearises the same
        thing: ``G``).  So a tilt changes the net sideways acceleration by
        only ``f = 1 + G`` of the thrust change: nothing at ``f`` near 0 (and
        the opposite way for ``f < 0``, where plain feedback is positive
        feedback: the booster drifts off a frozen plan until the air thins).
        The correction is divided by ``f`` (at most x2.5), faded out towards
        ``f = 0``, and reversed below it."""
        fh = np.asarray(fb[:2], float)
        n = float(np.linalg.norm(fh))
        if n < 1e-6 or self.aero is None:
            return fb
        vr = veh.vel - wind_now
        sp = float(np.linalg.norm(vr))
        if sp < 8.0:
            return fb
        e = np.array([fh[0] / n, fh[1] / n, 0.0])
        axis = np.asarray(veh.axis, float)
        e = e - float(np.dot(e, axis)) * axis
        en = float(np.linalg.norm(e))
        if en < 1e-6:
            return fb
        e /= en
        rho, _, _, a = atmosphere(max(0.0, float(veh.pos[2])))
        d = math.radians(2.0)
        a1, a2 = axis + d * e, axis - d * e
        f1 = self._quick(veh, vr, a1 / np.linalg.norm(a1), rho, a, veh.legs)
        f2 = self._quick(veh, vr, a2 / np.linalg.norm(a2), rho, a, veh.legs)
        e_h = np.array([fh[0] / n, fh[1] / n, 0.0])
        grad = float(np.dot((f1 - f2) / (2.0 * d), e_h)) / veh.mass          # m/s^2 per rad of tilt towards the correction
        sig = max(float(np.linalg.norm(u_ff)), 3.0)
        f = 1.0 + grad / sig
        s = float(np.clip(1.0 / f, -2.5, 2.5)) * min(1.0, abs(f) / 0.4) if abs(f) > 1e-3 else 0.0
        out = np.array(fb, float)
        out[:2] = fh * s
        return out

    def _quick(self, veh, vr, axis, rho, a_snd, legs):
        """``AeroModel.quick_force`` with the lift as flown (``_lift_trim``)."""
        return self.aero.quick_force(vr, axis, rho, a_snd, legs, trim=self._lift_trim(veh))

    @staticmethod
    def _aoa_limit(veh, accel, wind_now, max_tilt, table=Q_AOA_BUDGET, q_dyn=None):
        """Keep the commanded thrust within the angle-of-attack limit of the
        oncoming flow (``table``: dynamic pressure -> deg, normally
        ``_aoa_table``), but never beyond the tilt cone."""
        vr = veh.vel - wind_now
        sp = float(np.linalg.norm(vr))
        q_dyn = float(veh.last.get("q", 0.0)) if q_dyn is None else float(q_dyn)
        an = float(np.linalg.norm(accel))
        if sp < 3.0 or an < 1e-6 or q_dyn <= table[0][0]:
            return accel
        w = -vr / sp
        aoa = math.radians(float(np.interp(q_dyn, *table)))
        if aoa >= math.radians(179.0):
            return accel
        aoa = max(aoa, math.acos(float(np.clip(w[2], -1.0, 1.0))) - max_tilt + math.radians(3.0))
        d = accel / an
        ang = math.acos(float(np.clip(np.dot(d, w), -1.0, 1.0)))
        if ang <= aoa:
            return accel
        if w[2] < 0.0 and accel[2] > 0.0:
            # Rising through the air (a climb, or a crosswind stronger than
            # the sink): the flow comes from above, so no upward thrust is
            # "engines-first", and turning towards the flow would only add a
            # push downwind -- in a strong crosswind the booster then drifts
            # off with the wind.  Give up horizontal thrust only: the largest
            # share of the commanded horizontal part within the budget.
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                t = np.array([accel[0] * mid, accel[1] * mid, accel[2]])
                if math.acos(float(np.clip(np.dot(t / np.linalg.norm(t), w), -1.0, 1.0))) <= aoa:
                    lo = mid
                else:
                    hi = mid
            # ... and what it gives up does not go into a climb: rising
            # already, at most the hover vertical acceleration (a climb in a
            # strong wind only meets a stronger one)
            az = min(float(accel[2]), G0) if float(veh.vel[2]) >= 0.0 else float(accel[2])
            return np.array([accel[0] * lo, accel[1] * lo, az])
        side = d - w * float(np.dot(d, w))
        sn = float(np.linalg.norm(side))
        if sn < 1e-9:
            return accel
        d_new = w * math.cos(aoa) + side / sn * math.sin(aoa)
        if d_new[2] > 0.2 and accel[2] > 0.0:
            # Keep the commanded vertical acceleration: the limit only gives
            # up horizontal thrust (with a sideways flow, turning the thrust
            # towards it would otherwise turn lost push into lift -- a climb).
            return d_new * float(accel[2]) / float(d_new[2])
        # the part of the command along the allowed direction
        return d_new * an * max(0.0, math.cos(ang - aoa))

    def _fly_accel(self, veh, accel, max_tilt, floor=None, align_gate=0.35, center=None):
        """Fly an acceleration command.  The horizontal part is kept within
        ``max_tilt`` of ``center`` (the wind trim, or upright)."""
        c = veh.controls
        m = veh.mass
        horiz = accel[:2].copy()
        az = max(float(accel[2]), 1e-3)
        c0 = np.zeros(2) if center is None else np.asarray(center, float) * (az / G0)
        off = horiz - c0
        on = float(np.linalg.norm(off))
        lim = math.tan(max_tilt) * az
        if on > lim:
            horiz = c0 + off * (lim / on)
        accel = np.array([horiz[0], horiz[1], az])
        self.cmd_accel = accel
        mag = float(np.linalg.norm(accel))
        d = accel / mag
        # Slew-rate limit on the commanded thrust direction: a vehicle this
        # size cannot follow direction jumps between replans, and chasing
        # them is what makes it rock.  Slower near the ground.
        if self._dir_prev is not None:
            slew = math.radians(float(np.interp(self._height(veh), *SLEW_DEG))) * self._dt
            ang = math.acos(max(-1.0, min(1.0, float(np.dot(self._dir_prev, d)))))
            if ang > slew:
                axis_r = np.cross(self._dir_prev, d)
                an = float(np.linalg.norm(axis_r))
                if an > 1e-9:
                    axis_r /= an
                    d = self._dir_prev * math.cos(slew) + np.cross(axis_r, self._dir_prev) * math.sin(slew)
                    d /= np.linalg.norm(d)
        # Angular-rate feed-forward of the (slew-limited) commanded direction.
        self._dir_rate = np.zeros(3)
        if self._dir_prev is not None and self._dt > 0:
            wv = np.cross(self._dir_prev, d) / self._dt
            self._dir_rate = veh.R.T @ wv
            self._dir_rate[2] = 0.0
        self._dir_prev = d.copy()
        err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
        t_av = self._thrust_available(veh, veh.pos[2]) * self.thrust_scale
        if err < math.radians(25.0):
            # Small attitude lag: deliver the commanded component along d.
            thr = float(np.clip(m * mag / (t_av * math.cos(err)), 0.0, 1.0))
        else:
            thr = float(np.clip(m * mag / t_av, 0.0, 1.0)) * float(np.clip(math.cos(err), align_gate, 1.0))
        # ... but while the attitude still lags a sideways command, its excess
        # must not go into lift: the vertical thrust stays at most the
        # commanded vertical acceleration (no climbing on a divert).
        if veh.axis[2] > 0.2:
            thr = min(thr, m * (az + 0.5) / (t_av * float(veh.axis[2])))
        if floor is not None:
            thr = max(thr, floor)
        self.cmd_throttle = thr
        self.cmd_axis = d
        c.throttle = thr
        attitude_control(veh, d, self.k_angle, self.k_rate, self.max_rate,
                         veh.throttle < 0.25 or self._height(veh) < self.low_rcs_height, hold_roll=True)

    # ----------------------------------------------------- before ignition
    def _pre_ignition(self, veh, r, v, m, wind_now):
        c = veh.controls
        s = veh.spec
        if self.elapsed >= self.next_predict or self.impact is None:
            pts, impact, t_imp, t_ign, t_burn = self._ballistic(veh, r, v, m, wind_now)
            self.impact = impact
            self.ignition_in = t_ign
            self.predicted_points = [(float(p[0]), float(p[1]), float(max(0.0, p[2])) + self.h_off) for p in pts[:: max(1, len(pts) // 120)]]
            self.time_to_go = t_ign + t_burn if math.isfinite(t_ign) else t_imp
            self.next_predict = self.elapsed + self.coast_predict_period
        miss = float(np.linalg.norm(self.impact))

        # ---- boost-back: put the booster over the pad at ignition.
        # Aimed and judged in still air (``boostback_wind``): the speed is the
        # ground speed and the miss is the drag-only impact without wind.  The
        # glide and the landing burn take the wind out afterwards.
        wind_bb = wind_now if self.boostback_wind else np.zeros(3)
        if not self.boostback_wind and float(np.linalg.norm(wind_now)) > 0.05 and self.aero is not None:
            if self.elapsed >= self._bb_impact_next or self._bb_impact is None:
                self._bb_impact = self._ballistic(veh, r, v, m, wind_now, stop_on_ignition=False, blind=True)[1]
                self._bb_impact_next = self.elapsed + self.coast_predict_period
            miss = float(np.linalg.norm(self._bb_impact))
        airspeed = float(np.linalg.norm(v - wind_bb))
        drag_up = max(0.0, float(veh.last["aero"][2]) / m)
        frac, t_burn, burn_dir = self._burn_need(veh, r, v, m, drag_up)
        ignite_now = frac >= self._ignition_threshold(t_burn, burn_dir)
        # Boost-back only with time to come back and coast: not when the
        # landing burn is already due (then the burn itself flies the divert,
        # G-FOLD), and a running one stops once a vertical burn would be due.
        up_only = np.array([0.0, 0.0, 1.0])
        f_v, t_v, _ = self._burn_need(veh, r * up_only, v * up_only, m, drag_up)
        burn_due = f_v >= self._ignition_threshold(t_v, up_only)
        bb_miss = float(np.interp(airspeed, self.boostback_slow_speed, (self.boostback_miss_slow, self.boostback_miss)))
        start = miss > bb_miss and r[2] > self.boostback_min_alt and airspeed < 180.0 and not ignite_now
        hop = (not start and self.hop_min_alt < r[2] <= self.boostback_min_alt and airspeed < 100.0
               and miss > self.hop_miss and not ignite_now)
        if not self._boostback_done and (self._boostback or start or hop):
            if not self._boostback:
                self._hop = hop
                self._bb_floor = self.hop_end_frac * float(r[2]) if hop else self.boostback_end_alt
            self._boostback = True
            self.phase = "BOOSTBACK"
            self._mark()
            if self.elapsed >= self._bb_next or self._bb_dv is None:
                self._boostback_target(veh, r, v, m, wind_now)
                self._bb_next = self.elapsed + self.boostback_solve_period
            elif self._bb_v is not None:
                self._bb_dv = self._bb_dv - (v - self._bb_v)   # velocity gained since the last solve
                self._bb_v = v.copy()
            dv = self._bb_dv
            dvn = float(np.linalg.norm(dv)) if dv is not None else 0.0
            # Nose-first at speed (see nose_first_speed): keep within the flow
            # cone (a gravity turn); once the horizontal change still wanted
            # points back (more nose-first thrust only overshoots), or the air
            # has already turned the booster round (engines-first: the stable
            # attitude, from which the thrust cannot be pointed along the
            # motion again -- it would wait at zero throttle), the
            # engines-first coast and the landing burn take over.  "Back" is
            # against the motion through the air: in a wind the ground speed
            # passes through zero while the flow still comes the other way,
            # and the sign of the ground speed would end the boost-back
            # there (with the pad still ~1.5 km off in 15 m/s).  A hop also
            # ends once the change wanted points against the flow (braking):
            # it starts slow, so there is no retrograde phase to fly.
            d = None if dv is None or dvn <= 0.0 else dv / dvn
            braking = False
            if d is not None and airspeed > self.nose_first_speed:
                u_air = (v - wind_bb) / airspeed
                cos_f = float(np.dot(d, u_air))
                cap = math.radians(self.nose_first_aoa_deg)
                if cos_f > 0.0:
                    if float(np.dot(dv[:2], u_air[:2])) < 0.0 or float(np.dot(veh.axis, u_air)) < 0.0:
                        braking = True
                    elif cos_f < math.cos(cap):
                        side = d - u_air * cos_f
                        d = u_air * math.cos(cap) + side / float(np.linalg.norm(side)) * math.sin(cap)
                elif self._hop:
                    braking = True
            if (dv is None or dvn < self.boostback_done_dv or r[2] < self._bb_floor or burn_due
                    or veh.propellant < self._bb_reserve or braking):
                self._boostback = False
                self._boostback_done = True
            else:
                c.fins_deploy = False           # grid fins stay stowed for the boost-back
                err = math.acos(max(-1.0, min(1.0, float(np.dot(veh.axis, d)))))
                thr = float(np.clip(dvn / 8.0, 0.4, 1.0)) if err < math.radians(25.0) else 0.0
                if self._hop and thr == 0.0 and veh.axis[2] > 0.3:
                    # still turning: hold the height meanwhile (a hop starts
                    # low and slow -- falling now only steepens the flow it
                    # must then fly against)
                    thr = float(np.clip(1.1 * m * G0 / (self._thrust_available(veh, r[2]) * float(veh.axis[2])), 0.0, 1.0))
                c.throttle = thr
                self.cmd_throttle = thr
                self.cmd_axis = d
                self.cmd_accel = d * thr * s.thrust_sl / m
                self.next_predict = min(self.next_predict, self.elapsed + 0.1)
                self._attitude(veh, d, use_rcs=True)
                return

        # ---- coast, and the ignition decision.
        self.phase = "COAST"
        c.fins_deploy = True                    # grid fins out after the boost-back
        self._mark()
        # Low, slow and far off (no boost-back, too little speed to glide):
        # falling first only loses the height the divert needs -- light now
        # and let the landing burn fly it.
        divert_now = (not self._boostback_done and r[2] < self.boostback_min_alt and airspeed < 100.0
                      and miss > max(150.0, 0.5 * r[2]))
        if (ignite_now or divert_now) and not self._ignition_deferred():
            if not math.isfinite(t_burn):
                # not falling yet: fall time plus the sideways transfer
                h, vz = max(0.0, float(r[2])), float(v[2])
                t_burn = (vz + math.sqrt(vz * vz + 2.0 * G0 * h)) / G0 + 2.0 * math.sqrt(miss / 3.0)
            self._ignite(veh, r, v, m, wind_now, t_burn)
            return
        c.throttle = 0.0
        self.cmd_throttle = 0.0
        self.cmd_accel = np.zeros(3)
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        if self.ignition_in < 3.0 and float(veh.last.get("q", 0.0)) < self.lineup_q_max:
            # Line up for the landing burn: upright, plus the steady lateral
            # deceleration of _burn_miss and what removes the rest of the
            # aim-point miss (the glide has done the bulk of it).
            t_b = max(2.0, self._brake_time)
            a_lat = -v[:2] / t_b - 2.0 * self._burn_miss_now / t_b ** 2
            a_z = max(G0, frac * self._thrust_available(veh, r[2]) / m * float(burn_dir[2]))
            target = np.array([a_lat[0], a_lat[1], a_z])
            target /= float(np.linalg.norm(target))
            cap = math.radians(float(np.clip(r[2] / 40.0, 1.0, 20.0)))
            if math.acos(max(-1.0, min(1.0, float(target[2])))) > cap:
                hz = np.array([target[0], target[1], 0.0])
                hz /= max(1e-9, float(np.linalg.norm(hz)))
                target = hz * math.sin(cap) + np.array([0.0, 0.0, math.cos(cap)])
        elif sp > 25.0:
            target = self._aero_steer(veh, r, v, m, wind_now)   # engines-first, steered by body lift
        else:
            target = np.array([0.0, 0.0, 1.0])
        self.cmd_axis = target
        self._attitude(veh, target, use_rcs=True)

    def _coast(self, veh, r, v, m, wind_now, dt=0.25, t_max=200.0, blind=False):
        """Drag-aware coast to the predicted ignition: ``(t_ign, p_ign, v_ign, t_burn)``, or None
        (``blind``: in still air)."""
        if self.aero is None:
            return None
        p, u = r.copy(), v.copy()
        t = 0.0
        while t < t_max and p[2] > 0.0:
            rho, _, _, a = atmosphere(max(0.0, p[2]))
            wv = np.zeros(3) if blind else self._wind(p[2])
            f = self.aero.quick_force(u - wv, None, rho, a, 0.0)
            frac, tb, bd = self._burn_need(veh, p, u, m, max(0.0, f[2] / m))
            if frac >= self._ignition_threshold(tb, bd):
                return t, p, u, tb
            u = u + (f / m + np.array([0.0, 0.0, -G0])) * dt
            p = p + u * dt
            t += dt
        return None

    def _burn_miss(self, veh, p_ign, v_ign, m, wind_now, dt=0.2, blind=False):
        """Where a (nearly) upright landing burn lit at ``(p_ign, v_ign)`` ends.

        Returns ``(miss_xy, t_brake)``.  The burn brakes at constant
        deceleration until it meets the final-approach envelope (sink
        <= gate_speed + sink_gain h), and must be over the pad, without
        sideways speed, by then: over the braking time the horizontal speed
        is bled off at the smallest steady rate (the rest is done by the
        drag of the upright body), so the booster drifts about
        ``v_ign t_brake / 2``; the drag of a cross wind is leaned against, up
        to the wind trim's tilt.  The glide and the boost-back aim this point
        at the pad: the burn then needs only a small, steady tilt -- at high
        dynamic pressure a tilted engines-first booster gets a body lift
        *against* its tilt, so a large sideways correction with the engine
        is costly and fights itself.

        ``blind``: in still air (the boost-back), without the wind trim."""
        h0, vz0 = float(p_ign[2]) - self.gate_height, float(v_ign[2])
        a_v = max(0.5, (vz0 ** 2 - self.gate_speed ** 2) / (2.0 * max(1.0, h0)))
        # Braking time: the constant-deceleration profile sqrt(2 a_v h)
        # meets the final-approach envelope near h = 2 a_v / sink_gain^2 (~100 m).
        t_br, h, vz = 0.0, h0, vz0
        while h > 0.0 and -vz <= self.gate_speed + self.sink_gain * h and t_br < 60.0:
            vz += a_v * dt
            h += vz * dt
            t_br += dt
        if t_br < dt:                       # already below the envelope: low, short burn
            t_br = 0.6 * max(0.0, -vz0) / a_v
        t_br = max(t_br, 1.0)
        p, u = p_ign.copy(), v_ign.copy()
        up = np.array([0.0, 0.0, 1.0])
        e0 = v_ign[:2] / max(1e-6, float(np.linalg.norm(v_ign[:2])))
        t = 0.0
        while t < t_br - 1e-9:
            a_h = np.zeros(2)
            if self.aero is not None:
                rho, _, _, a = atmosphere(max(0.0, p[2]))
                wv = np.zeros(3) if blind else self._wind(p[2])
                a_h = self.aero.quick_force(u - wv, up, rho, a, 0.0)[:2] / m
                # The burn leans into the wind (the wind trim; the plan's
                # sideways-effort cost does not charge thrust that only cancels
                # drag), up to ``trim_max_deg`` of tilt of a thrust that carries
                # g + a_v vertically: only the drag beyond that drifts the
                # booster.  Without this the boost-back aims far upwind of the
                # pad to make up the whole drift of the burn (1-6 m/s^2 of
                # drag in 12-30 m/s of wind over ~25 s: hundreds of metres) --
                # in a strong wind it chases a velocity change that is not
                # needed (-580 m/s instead of -170 m/s), and the booster is
                # never over the pad at ignition.
                n_h = float(np.linalg.norm(a_h))
                cap = self.trim_gain * math.tan(math.radians(self.trim_max_deg)) * (G0 + a_v)
                if n_h > 1e-9 and not blind:
                    a_h *= max(0.0, n_h - cap) / n_h
            # steady thrust decel that stops the along-track drift by t_br
            # (drag does part of it)
            a_h -= e0 * max(0.0, float(np.dot(u[:2], e0))) / max(dt, t_br - t)
            u[:2] += a_h * dt
            u[2] += a_v * dt
            p = p + u * dt
            t += dt
        return p[:2], t_br

    def _boostback_target(self, veh, r, v, m, wind_now):
        """Velocity change that brings the booster back over the pad.

        Shooting on the drag-aware coast: Newton on the horizontal velocity
        change so that the landing burn ends on the pad (``_burn_miss`` = 0),
        and loft (extra climb) while that is out of reach or the flight path
        at ignition is shallower than ``boostback_gamma_ok``: a longer,
        steeper fall lets drag bleed the horizontal speed off.  Warm-started
        between solves (every ``boostback_solve_period``)."""
        if self._bb_dv is None:
            dv = np.zeros(3)
            if self._hop:
                # start lofted: from low down a flat shot just falls short
                # (and the booster must not sink while it turns)
                dv[2] = float(np.clip(0.5 * math.sqrt(G0 * float(np.linalg.norm(self.impact))), 20.0, self.boostback_loft_max))
        else:
            dv = self._bb_dv.copy()
        if self._bb_v is not None and self._bb_dv is not None:
            dv -= v - self._bb_v
        dv[2] = max(0.0, dv[2])

        def shoot(x):
            blind = not self.boostback_wind
            res = self._coast(veh, r, v + x, m, wind_now, blind=blind)
            return None if res is None else (self._burn_miss(veh, res[1], res[2], m, wind_now, blind=blind)[0], res)

        out = shoot(dv)
        for _ in range(2):
            if out is None or float(np.linalg.norm(out[0])) < 3.0:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(3)
                e[j] = 4.0
                oj = shoot(dv + e)
                if oj is None:
                    J = None
                    break
                J[:, j] = (oj[0] - out[0]) / 4.0
            if J is None or abs(np.linalg.det(J)) < 1e-6:
                break
            step = -np.linalg.solve(J, out[0])
            sn = float(np.linalg.norm(step))
            if sn > 80.0:
                step *= 80.0 / sn
            dv[:2] += step
            out = shoot(dv)
        if out is None:
            # No ignition predicted: fall back to the drag-only impact point.
            dv = np.zeros(3)
            dv[:2] = -self.impact / max(5.0, self.time_to_go)
            dv[2] = 0.3 * float(np.linalg.norm(dv[:2]))
            self._bb_reserve = 0.0
        else:
            v_i = out[1][2]
            # Loft (extra climb = more time) while the pad is out of reach,
            # or while the fall would still be too shallow at ignition.
            excess = float(np.linalg.norm(v_i[:2])) - math.tan(math.radians(self.boostback_gamma_ok)) * max(20.0, -float(v_i[2]))
            if float(np.linalg.norm(out[0])) > 30.0:
                excess = max(excess, 5.0)
            dv[2] = float(np.clip(dv[2] + np.clip(excess, -5.0, 5.0), 0.0, self.boostback_loft_max))
            # Keep the landing-burn propellant (with margin).
            isp_g = veh.spec.isp_sl * G0
            self._bb_reserve = 1.3 * m * (1.0 - math.exp(-(float(np.linalg.norm(v_i)) + 15.0) / isp_g)) + 250.0
        self._bb_dv = dv
        self._bb_v = v.copy()

    def _aero_steer(self, veh, r, v, m, wind_now):
        """Engines-first attitude with a small angle of attack whose body/grid-fin
        lift steers the booster over the pad (Falcon-9 style glide).

        Zero-effort-miss guidance on the drag-aware coast: the lateral
        acceleration moves ``_burn_miss`` (where an upright landing burn from
        the predicted ignition would end) onto the pad, so the glide -- not
        the burn -- removes the horizontal error."""
        vr = v - wind_now
        sp = float(np.linalg.norm(vr))
        u = vr / sp
        base = -u
        if self._ign_state is None or self.aero is None:
            return base
        t_i = self._ign_state[0]
        miss = self._burn_miss_now
        t_go = max(3.0, float(t_i))
        # A constant lateral acceleration a held until ignition moves the
        # aim point by about a (t^2/2 + t_brake t / 2).
        a_des = np.zeros(3)
        a_des[:2] = -self.glide_gain * miss / (0.5 * t_go ** 2 + 0.5 * self._brake_time * t_go)
        a_des -= u * float(np.dot(a_des, u))
        an = float(np.linalg.norm(a_des))
        if an < 0.02:
            return base
        e_a = a_des / an
        rho, _, _, a_snd = atmosphere(max(0.0, r[2]))
        l0 = float(np.dot(self._quick(veh, vr, base, rho, a_snd, veh.legs), e_a))
        a_max = math.radians(self.aero_steer_max_deg)

        def lift(alpha, e):
            axis = math.cos(alpha) * base + math.sin(alpha) * e
            return float(np.dot(self._quick(veh, vr, axis, rho, a_snd, veh.legs), e_a)) - l0

        # Engines-first, the lift points against the tilt: find the side that
        # pushes along e_a, then the angle (secant on the real lift curve).
        alpha = math.radians(5.0)
        best = max(((e, lift(alpha, e)) for e in (-e_a, e_a)), key=lambda x: x[1])
        e, l_a = best
        if l_a <= 0.0:
            return base
        for _ in range(2):
            alpha = float(np.clip(alpha * an * m / l_a, 0.0, a_max))
            if alpha >= a_max:
                break
            l_a = lift(alpha, e)
            if l_a <= 0.0:
                break
        return math.cos(alpha) * base + math.sin(alpha) * e

    def _ignite(self, veh, r, v, m, wind_now, t_burn):
        self._dir_prev = veh.axis.copy()
        self._gate_xy = np.zeros(2)          # the burn itself lands on the pad centre
        self.burn_locked = True
        self._boostback = False
        self._route = None
        self.plan = None
        # the predictor's burn time only seeds the search of the final time
        t_est = max(1.5, 1.05 * t_burn) + 2.0
        self.touchdown_time = self.elapsed + t_est
        if self.async_plan and self.planner.available:
            # Coast -> burn used to freeze the whole state stream for 0.3-0.6 s
            # while the final-time search ran.  Now the search is done during
            # the coast for the predicted ignition state (``_submit_prep``) and
            # only its final time and route are reused: one solve from the
            # state the booster is really in puts a plan in hand before the
            # engine lights, so the burn starts planned instead of dropping
            # into the plain brake law for a second.
            plan = None
            prep = self._prep
            if prep is not None and prep["t_f"] and self.elapsed - prep["t"] <= self.ign_prep_lead + 1.0:
                self._route = prep["route"]
                self.burn_fraction = prep["burn_fraction"]
                plan = self._solver(veh, wind_now, self.burn_fraction, self._keep_out(veh))(prep["t_f"])
                if plan is not None and not plan.sane:
                    plan = None             # a runaway plan is no plan (see the project rules)
            if plan is None:
                plan = self._solver(veh, wind_now, self.burn_fraction, self._keep_out(veh))(t_est)
                if plan is not None and not plan.sane:
                    plan = None
            if plan is not None:
                self._commit(plan)
            else:
                # no usable preparation: light on the estimator's time and let
                # the search fill in the plan, refines included, off-thread
                self._ignition_searching = True
                self._submit_search(veh, wind_now, t_est)
        else:
            plan = self._search_time(veh, wind_now, t_est) if self.planner.available else None
            if plan is not None:
                self._commit(plan)
                # The search linearised drag and angle of attack about a straight
                # descent (there was no plan yet) -- in a strong wind aloft that
                # reference hovers in a 30-40 m/s flow, and the plan breaks the
                # angle-of-attack cone along a path it will not fly.  Two passes
                # about the plan itself (successive convexification) before it is
                # flown.
                for _ in range(self.ignition_refine):
                    p = self._solver(veh, wind_now, self.burn_fraction, self._keep_out(veh))(self.plan.t_f)
                    if p is None or not p.sane:
                        break
                    self._commit(p)
        self._next_search = self.elapsed + self.search_backoff
        self._publish()
        self.next_replan = self.elapsed + self.burn_replan_period
        self._prep = None
        self._ignite_defer_until = None
        self.phase = "LANDING BURN"
        self._mark()
        veh.controls.throttle = self.minimum_throttle

    def _submit_search(self, veh, wind_now, t_est) -> None:
        """Hand the ignition search (final time + refinements) to the planner
        thread; ``_run_plan_job`` commits the result when it is done."""
        ap = copy.copy(self)
        ap.aero = copy.copy(self.aero) if self.aero is not None else None
        ap.plan = None                  # the search is linearised about a straight descent, as at ignition
        ap._route = None
        ap._trim_xy = np.array(self._trim_xy, float)
        ap._wind_ref = (np.array(self._wind_ref[0], float), float(self._wind_ref[1]))
        job = {"gen": self._plan_gen, "kind": "ignite", "t_est": float(t_est),
               "autopilot": ap, "vehicle": self._frozen_vehicle(veh), "wind": np.array(wind_now, float)}
        with self._plan_cond:
            self._plan_job = job
            self._plan_cond.notify()

    def _mark(self):
        if self.phase in ("COAST", "BOOSTBACK", "LANDING BURN", "LANDED") and self.phase not in self.visited:
            self.visited.append(self.phase)

    # ------------------------------------------------------------ attitude
    def _attitude(self, veh, desired_axis, use_rcs):
        """SO(3) attitude control: TVC for pitch/yaw, RCS for roll (+ low thrust)."""
        attitude_control(veh, desired_axis, self.k_angle, self.k_rate, self.max_rate, use_rcs, hold_roll=True)
