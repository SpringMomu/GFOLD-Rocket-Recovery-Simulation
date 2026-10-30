"""Catch tower ("Mechazilla"): two chopstick arms that catch the booster.

Frames as in ``vehicle.py`` (world ENU, pad centre at the origin).  The tower
stands west of the pad; its two arms pivot at the tower's front corners and
swing in the horizontal plane.  The catch point is above the pad centre, at
the height where the booster's catch collar (``CATCH_COLLAR_Z`` on the body,
just below the grid fins) rests on top of the closed arms.  Under it stands
the launch mount, which a booster that is not caught falls onto.

What is modelled
----------------
* Arm kinematics: OPEN (wide) -> READY (pre-closed, clear of the booster on
  approach) -> CLOSING (fast, then slow within ``ARM_SLOW_GAP`` of the body so
  the arms only touch it gently) -> CLOSED around the booster.  The tower,
  not the booster, decides when to close: when the collar is a few metres
  above the arms (or arrives within ``CLOSE_LEAD_TIME``) and the booster is
  inside the capture window.
* Catch collar on the arm tops: stiff point contacts with friction around
  the collar ring, on a carriage that sinks on shock absorbers (spring-damper
  with its own mass), so the booster is caught softly.  Below the collar
  the booster hangs from the arms like a pendulum (centre of mass far below).
* Arm faces against the barrel: a capped spring-damper that centres the
  booster as the arms close.
* Crashes: barrel against the tower, base/nozzle on an arm or the launch
  mount, or hitting an arm face too fast.

The booster is CAUGHT when it rests on the arms, engine off, and has settled.
"""

from __future__ import annotations

import math

import numpy as np

from .vehicle import cross3, vnorm

# --- geometry (m) ---------------------------------------------------------
TOWER_X = (-15.0, -9.0)        # tower footprint (x), |y| <= TOWER_HALF_Y
TOWER_HALF_Y = 3.0
TOWER_TOP = 48.0
ARM_TOP = 30.0                 # arm top surface when unloaded
ARM_DEPTH = 1.6                # arm height (top surface down to its bottom)
ARM_HALF_WIDTH = 0.5
ARM_LENGTH = 10.5
PIVOT_X, PIVOT_Y = -9.0, 3.0   # arm pivots at the tower's front corners (y = +/-PIVOT_Y)
MOUNT_RADIUS = 6.0             # launch mount table under the catch point
MOUNT_TOP = 8.0
CATCH_COLLAR_Z = 15.0          # body height of the catch collar (below the grid fins at 16.8)
CATCH_COLLAR_R = 2.1           # collar contact radius (hardpoints stick out 0.6 m)
COLLAR_POINTS = 16
BODY_RADIUS = 1.5
FACE_CLEARANCE = 0.1           # closed arm face to barrel

ARM_OPEN = math.radians(40.0)
ARM_READY = math.radians(22.0)
ARM_FAST_RATE = math.radians(28.0)
ARM_SLOW_SPEED = 0.35          # m/s at the face once within ARM_SLOW_GAP of the barrel
ARM_SLOW_GAP = 0.5
READY_RATE = math.radians(8.0)

READY_HEIGHT = 120.0           # collar height above the arms at which they pre-close ...
READY_RADIUS = 15.0            # ... with the booster this close to the catch axis
CLOSE_HEIGHT = 4.5             # close when the collar is this far above the arms ...
CLOSE_LEAD_TIME = 1.7          # ... or will arrive within this time
CAPTURE_ALONG = 2.5            # capture window: along the arms (|x|) ...
CAPTURE_ACROSS = 1.2           # ... and across them (|y|)

# --- dynamics ---------------------------------------------------------------
CONTACT_K, CONTACT_C, CONTACT_MU = 2.0e6, 6.0e4, 0.6
CARRIAGE_MASS = 3000.0
CARRIAGE_K, CARRIAGE_C = 2.0e5, 1.1e5
CARRIAGE_TRAVEL = 1.5
FACE_K, FACE_C, FACE_FORCE_MAX = 4.0e5, 4.0e4, 6.0e4
ARM_IMPACT_SPEED = 3.0         # m/s at the face: faster is a collision
# Keep-out boxes for the planner: margin beyond the barrel (sideways) and
# beyond the base/top of the booster (vertically).
KEEP_OUT_MARGIN = 2.0
KEEP_OUT_MARGIN_Z = 1.5
SETTLE_TIME = 0.8


def _closed_angle() -> float:
    """Arm angle at which the inner face is FACE_CLEARANCE from a centred barrel."""
    d = BODY_RADIUS + FACE_CLEARANCE + ARM_HALF_WIDTH        # catch axis to arm centreline
    return math.asin(d / math.hypot(PIVOT_X, PIVOT_Y)) - math.atan2(PIVOT_Y, -PIVOT_X)


ARM_CLOSED = _closed_angle()


class CatchTower:
    def __init__(self) -> None:
        self.angle = ARM_OPEN          # arm swing from the +x direction (mirrored for the -y arm)
        self.rate = 0.0
        self.mode = "OPEN"             # OPEN | READY | CLOSING | CLOSED
        self.sink = 0.0                # carriage travel (down) on its shock absorbers
        self.sink_rate = 0.0
        self.load = 0.0                # vertical load on the arms (N)
        self.events: list[str] = []
        self._settle = 0.0
        self._face_hit = False

    # ------------------------------------------------------------ geometry
    @property
    def arm_top(self) -> float:
        return ARM_TOP - self.sink

    @property
    def catch_base_z(self) -> float:
        """World z of the booster base (body origin) when the collar rests on the unloaded arms."""
        return ARM_TOP - CATCH_COLLAR_Z

    @staticmethod
    def keep_out_boxes(spec):
        """Obstacles for the booster's base point (world): the tower, both arms
        in their READY pose (where they wait during the approach) and the
        launch mount, each grown by the barrel radius, the body extent above
        and below the base, and a margin.  Returned as
        (centre, axes 3x3, half extents) tuples."""
        r = BODY_RADIUS + KEEP_OUT_MARGIN
        up = spec.length + KEEP_OUT_MARGIN_Z            # body above the base
        down = -spec.nozzle_exit_z + KEEP_OUT_MARGIN_Z  # nozzle below the base
        boxes = []

        def add(c, axes, half):
            boxes.append((np.asarray(c, float), np.asarray(axes, float), np.asarray(half, float)))

        def span(z0, z1):                               # obstacle z-range -> base z-range
            lo, hi = z0 - up, z1 + down
            return 0.5 * (lo + hi), 0.5 * (hi - lo)

        zc, zh = span(-100.0, TOWER_TOP)
        add(((TOWER_X[0] + TOWER_X[1]) / 2, 0.0, zc), np.eye(3),
            ((TOWER_X[1] - TOWER_X[0]) / 2 + r, TOWER_HALF_Y + r, zh))
        zc, zh = span(-100.0, MOUNT_TOP)
        add((0.0, 0.0, zc), np.eye(3), (MOUNT_RADIUS + r, MOUNT_RADIUS + r, zh))
        zc, zh = span(ARM_TOP - ARM_DEPTH, ARM_TOP)
        a = ARM_READY
        for sgn in (1.0, -1.0):
            d = np.array([math.cos(a), sgn * math.sin(a), 0.0])
            n = np.array([math.sin(a), -sgn * math.cos(a), 0.0])
            c = np.array([PIVOT_X, sgn * PIVOT_Y, 0.0]) + d * ARM_LENGTH / 2
            add((c[0], c[1], zc), np.column_stack([d, n, [0.0, 0.0, 1.0]]), (ARM_LENGTH / 2 + r, ARM_HALF_WIDTH + r, zh))
        return boxes

    def arms(self):
        """For each arm: (pivot, along-arm unit vector, inward unit normal)."""
        out = []
        for sgn in (1.0, -1.0):
            a = self.angle
            d = np.array([math.cos(a), sgn * math.sin(a), 0.0])
            n = np.array([math.sin(a), -sgn * math.cos(a), 0.0])
            out.append((np.array([PIVOT_X, sgn * PIVOT_Y, 0.0]), d, n))
        return out

    def _collar_height(self, veh) -> float:
        return float((veh.pos + veh.R @ np.array([0.0, 0.0, CATCH_COLLAR_Z]))[2]) - self.arm_top

    def _axis_at(self, veh, z):
        """Point of the body axis at world height z (None if not on the body)."""
        ax = veh.axis
        if ax[2] < 0.3:
            return None
        t = (z - float(veh.pos[2])) / float(ax[2])
        if t < -1.0 or t > veh.spec.length:
            return None
        return veh.pos + ax * t

    def face_gaps(self, veh):
        """Barrel-to-face gap of each arm at arm height (inf if not alongside)."""
        b = self._axis_at(veh, self.arm_top - 0.5 * ARM_DEPTH)
        gaps = []
        for p, d, n in self.arms():
            if b is None:
                gaps.append(math.inf)
                continue
            rel = b - p
            s = float(rel @ d)
            w = float(rel @ n)
            gaps.append(w - ARM_HALF_WIDTH - BODY_RADIUS if -BODY_RADIUS <= s <= ARM_LENGTH + BODY_RADIUS else math.inf)
        return gaps

    # ------------------------------------------------------------ control
    def update(self, dt: float, veh) -> None:
        """Arm controller (the tower's own, not the booster's)."""
        if veh.state not in ("FLYING", "CAUGHT"):
            self.rate = 0.0
            return
        h = self._collar_height(veh)
        b = self._axis_at(veh, self.arm_top)
        xy = veh.pos[:2] if b is None else b[:2]
        r = float(np.linalg.norm(xy))
        if self.mode == "OPEN" and h < READY_HEIGHT and r < READY_RADIUS and veh.vel[2] < 0.0:
            self.mode = "READY"
            self.events.append("ARMS READY")
        if self.mode == "READY":
            sink = max(0.3, -float(veh.vel[2]))
            in_window = abs(float(xy[0])) < CAPTURE_ALONG and abs(float(xy[1])) < CAPTURE_ACROSS
            if in_window and -0.3 < h < max(CLOSE_HEIGHT, sink * CLOSE_LEAD_TIME):
                self.mode = "CLOSING"
                self.events.append("ARMS CLOSING")
        target = {"OPEN": ARM_OPEN, "READY": ARM_READY}.get(self.mode, ARM_CLOSED)
        if self.mode in ("OPEN", "READY"):
            step = float(np.clip(target - self.angle, -READY_RATE * dt, READY_RATE * dt))
        else:
            # Fast until close to the barrel, then slow (the arms only touch it).
            gap = min(self.face_gaps(veh))
            lever = max(2.0, math.hypot(float(xy[0]) - PIVOT_X, abs(float(xy[1])) - PIVOT_Y))
            rate = ARM_FAST_RATE if gap > ARM_SLOW_GAP else ARM_SLOW_SPEED / lever
            if gap <= 0.02:
                rate = 0.0
            step = max(-rate * dt, target - self.angle)
            if self.mode == "CLOSING" and target - self.angle >= -1e-4:
                self.mode = "CLOSED"
        self.rate = step / dt if dt > 0 else 0.0
        self.angle += step

    # ------------------------------------------------------------ contact
    def contact(self, veh, R, com, dt, m):
        """Forces of the tower on the booster: (force_world, torque_body).

        Also integrates the carriage and flags crashes and the CAUGHT state."""
        force = np.zeros(3)
        torque_b = np.zeros(3)
        veh.catch_contact = 0
        omega_w = R @ veh.omega
        top = self.arm_top
        arms = self.arms()

        if veh.state == "FLYING" and self._collision(veh, R, arms):
            return force, torque_b

        # Catch collar on the arm tops.
        load = 0.0
        if self.mode in ("CLOSING", "CLOSED"):
            for k in range(COLLAR_POINTS):
                a = 2.0 * math.pi * k / COLLAR_POINTS
                pw = veh.pos + R @ np.array([CATCH_COLLAR_R * math.cos(a), CATCH_COLLAR_R * math.sin(a), CATCH_COLLAR_Z])
                depth = top - float(pw[2])
                if depth <= 0.0 or depth > 0.6:
                    continue
                if not any(0.0 <= float((pw - p) @ d) <= ARM_LENGTH and abs(float((pw - p) @ n)) <= ARM_HALF_WIDTH
                           for p, d, n in arms):
                    continue
                r = pw - com
                v = veh.vel + cross3(omega_w, r)
                vz_rel = float(v[2]) + self.sink_rate       # the arm top moves down at sink_rate
                fn = max(0.0, CONTACT_K * depth - CONTACT_C * vz_rel)
                vt = np.array([v[0], v[1], 0.0])
                vt_n = vnorm(vt)
                ft = -vt / vt_n * min(CONTACT_MU * fn, 1.5e4 * vt_n) if vt_n > 1e-6 else np.zeros(3)
                f = np.array([0.0, 0.0, fn]) + ft
                force += f
                torque_b += R.T @ cross3(r, f)
                load += fn
                veh.catch_contact += 1
            if veh.catch_contact and veh.touchdown is None and veh.state == "FLYING":
                self._record_catch(veh)

        # Arm faces against the barrel (they centre it while closing).
        b = self._axis_at(veh, top - 0.5 * ARM_DEPTH)
        if b is not None and self.mode in ("CLOSING", "CLOSED"):
            for sgn, (p, d, n) in zip((1.0, -1.0), arms):
                rel = b - p
                s = float(rel @ d)
                pen = ARM_HALF_WIDTH + BODY_RADIUS - float(rel @ n)
                if pen <= 0.0 or not (-BODY_RADIUS <= s <= ARM_LENGTH + BODY_RADIUS) or pen > BODY_RADIUS:
                    continue
                r = b - com
                v = veh.vel + cross3(omega_w, r)
                v_arm = self.rate * s * np.array([-math.sin(self.angle), sgn * math.cos(self.angle), 0.0])
                vn = float((v - v_arm) @ n)            # > 0: barrel moving away from the face
                if veh.state == "FLYING" and not self._face_hit and -vn > ARM_IMPACT_SPEED:
                    veh._crash("ARM COLLISION")
                    return np.zeros(3), np.zeros(3)
                self._face_hit = True
                fmag = float(np.clip(FACE_K * pen - FACE_C * vn, 0.0, FACE_FORCE_MAX))
                f = n * fmag
                force += f
                torque_b += R.T @ cross3(r, f)

        # Carriage on its shock absorbers.
        acc = (load - CARRIAGE_K * self.sink - CARRIAGE_C * self.sink_rate) / CARRIAGE_MASS
        self.sink_rate += acc * dt
        self.sink += self.sink_rate * dt
        if self.sink < 0.0:
            self.sink, self.sink_rate = 0.0, max(0.0, self.sink_rate)
        if self.sink > CARRIAGE_TRAVEL:
            self.sink, self.sink_rate = CARRIAGE_TRAVEL, min(0.0, self.sink_rate)
        self.load = load
        self._check_caught(veh, dt)
        return force, torque_b

    def _collision(self, veh, R, arms) -> bool:
        """Crash checks; True if the booster just crashed."""
        s = veh.spec
        # Barrel against the tower (axis samples against the footprint grown by the radius).
        if float(veh.pos[0]) < TOWER_X[1] + s.length + BODY_RADIUS and float(veh.pos[2]) < TOWER_TOP + BODY_RADIUS:
            for t in np.linspace(s.nozzle_exit_z, s.length, 12):
                p = veh.pos + R[:, 2] * t
                if (TOWER_X[0] - BODY_RADIUS < p[0] < TOWER_X[1] + BODY_RADIUS
                        and abs(p[1]) < TOWER_HALF_Y + BODY_RADIUS and p[2] < TOWER_TOP):
                    veh._crash("TOWER COLLISION")
                    return True
        # Base and nozzle on an arm or on the launch mount.
        base = [np.array([0.0, 0.0, s.nozzle_exit_z])]
        base += [np.array([BODY_RADIUS * math.cos(a), BODY_RADIUS * math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        top = self.arm_top
        for pb in base:
            pw = veh.pos + R @ pb
            if pw[2] < MOUNT_TOP and math.hypot(pw[0], pw[1]) < MOUNT_RADIUS:
                veh._crash("HIT LAUNCH MOUNT")
                return True
            if top - ARM_DEPTH < pw[2] < top:
                for p, d, n in arms:
                    rel = pw - p
                    if 0.0 <= float(rel @ d) <= ARM_LENGTH and abs(float(rel @ n)) <= ARM_HALF_WIDTH:
                        veh._crash("ARM COLLISION")
                        return True
        return False

    def _record_catch(self, veh):
        b = self._axis_at(veh, self.arm_top) if veh.axis[2] > 0.3 else None
        xy = veh.pos[:2] if b is None else b[:2]
        veh.touchdown = {
            "x": float(xy[0]), "y": float(xy[1]), "error": float(math.hypot(xy[0], xy[1])),
            "vz": float(veh.vel[2]), "vh": float(math.hypot(veh.vel[0], veh.vel[1])),
            "tilt": veh.tilt_deg(), "propellant": float(veh.propellant), "time": float(veh.time), "catch": True,
        }
        self.events.append("CATCH CONTACT")

    def _check_caught(self, veh, dt):
        if veh.state != "FLYING" or veh.catch_contact < 2:
            self._settle = 0.0
            return
        if (vnorm(veh.vel) < 0.25 and vnorm(veh.omega) < 0.06 and veh.throttle < 0.05
                and abs(self.sink_rate) < 0.1):
            self._settle += dt
            if self._settle > SETTLE_TIME:
                veh.state = "CAUGHT"
                veh.vel[:] = 0.0
                veh.omega[:] = 0.0
                self.sink_rate = 0.0
        else:
            self._settle = 0.0

    def snapshot(self) -> dict:
        return {"arm": round(math.degrees(self.angle), 3), "sink": round(self.sink, 4), "mode": self.mode,
                "load": round(self.load, 1),
                "geom": {"tower_x": TOWER_X, "tower_half_y": TOWER_HALF_Y, "tower_top": TOWER_TOP, "arm_top": ARM_TOP,
                         "arm_depth": ARM_DEPTH, "arm_half_width": ARM_HALF_WIDTH, "arm_length": ARM_LENGTH,
                         "pivot": [PIVOT_X, PIVOT_Y], "mount_radius": MOUNT_RADIUS, "mount_top": MOUNT_TOP,
                         "collar_z": CATCH_COLLAR_Z, "collar_r": CATCH_COLLAR_R}}
