"""
Bounded risk score R(x) in [0,1] for the fleet-recalibration study (Paper P5).
Composite of eight normalised surrogate-safety features (TTC, time headway,
DRAC, required deceleration, speed, closing speed, inverse spacing, jerk),
each mapped into [0,1] by a saturating transform and combined by a fixed
convex weight vector, so R(x) in [0,1] by construction and is monotone in
danger. The same function is applied to real NGSIM rows and to the
NGSIM-calibrated generator.
"""
import numpy as np

TTC_REF, THW_REF, DRAC_REF, V_REF = 8.0, 3.0, 4.0, 30.0
DV_REF, GAP_REF, BRAKE_REF, JERK_REF = 8.0, 60.0, 6.0, 4.0
WEIGHTS = np.array([0.22, 0.16, 0.16, 0.08, 0.12, 0.08, 0.12, 0.06])

def _sat(x):
    return x / (1.0 + x)

def risk_score(gap, v, dv, accel=None, jerk=None):
    gap = np.maximum(np.asarray(gap, float), 0.1)
    v   = np.maximum(np.asarray(v, float), 0.0)
    dv  = np.asarray(dv, float)
    approaching = np.maximum(dv, 0.0)
    if jerk is None: jerk = np.zeros_like(gap)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(approaching > 1e-3, gap / np.where(approaching > 1e-3, approaching, 1.0), np.inf)
        thw = np.where(v > 1e-3, gap / np.where(v > 1e-3, v, 1.0), np.inf)
    drac = np.where(gap > 1e-3, approaching**2 / (2.0 * gap), 0.0)
    f1 = _sat(TTC_REF / np.maximum(ttc, 1e-3))
    f2 = _sat(THW_REF / np.maximum(thw, 1e-3))
    f3 = _sat(drac / DRAC_REF)
    f4 = _sat(v / V_REF)
    f5 = _sat(approaching / DV_REF)
    f6 = _sat(GAP_REF / gap)
    f7 = _sat(drac / BRAKE_REF)
    f8 = _sat(np.abs(jerk) / JERK_REF)
    F = np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=-1)
    return np.clip(F @ WEIGHTS, 0.0, 1.0)

FEATURE_NAMES = ["invTTC","invTHW","DRAC","speed","closing","invGap","reqBrake","jerk"]
