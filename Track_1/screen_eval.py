"""Backbone-screen evaluator (Track_1/plan/plan_20260914.md sections 4 and 7).

    # one finished HELD-OUT arm X (GPU):
    python Track_1/screen_eval.py --arm w1_cnext_unet --ckpt SCREEN/w1/w1_cnext_unet_s1.pth \
        --wave 1 --lat SCREEN/lat/lat1.json [--role auto] [--family auto] [--gpu-h 2.4] [--note ..]
    # sigma_seed from the three FNO cmb3sim twins (GPU):
    python Track_1/screen_eval.py --sigma-seed --lat SCREEN/lat/lat1.json
    # create / update a row's bookkeeping fields only (no GPU):
    python Track_1/screen_eval.py --register --arm w1_ffno --status running --set job=123 ...

WHAT IT COMPUTES, all on the held-out twins' split (`model.VAL_TRAJ`: fit on val_seen, report on
val_unseen), always through the real instruments rather than re-implementations of them:

1. SINGLE MODEL.  X alone at ALPHA 0.60 with a band refit from X's own held-out errors by
   `submit_fno/fit_band.py` (--band-p 0.45, its defaults otherwise: fit on val_seen AoA 15-20,
   1.39x stretch, box k 3 -- the build's recipe), scored on the full val_unseen split and on its
   AoA 15-20 proxy; plus the ALPHA sweep {0.4, 0.6, 0.8}, band refit per ALPHA. Per block:
   rel / tke / mvpe (local errors), `sps` = the band's SPS with errors and accuracy factors
   stretched 1.39x (ens_v9's board-like convention), `sps_local` unstretched, and `wfinal` =
   final_v9.predict of the stretched subscores at a fixed time_score (a ranking number only).
2. MEMBER SETS at ALPHA 0.60, each through `ens_v9.py` (the documented instrument, its own p = 0
   band) AND through fit_band.py's shipping band on both splits:
     base = FNO x3 cmb3sim twins + ViT x2 twins; base + X; FNO x3 + X; X alone;
     role fno_member adds: the FNO control alone; ViT x2 + X; ViT x2 + FNO control.
3. PROJECTION.  `predict_board.py --anchor` for Track 1 and Track 2 against the newest recorded
   anchors (r25 / t2r12 once scribe adds them, else r24 / t2r11; recorded per row). The anchor
   package is a held-out set with its own refit band. FLOPs: A100 ms from the one-process latency
   table x 1.850 (4060-equivalent), batch-64 fp16 for Track 1 and batch-1 fp16 for Track 2, passed
   as the DIFFERENCE against xfam6's fused groups (--anchor-local-ms 0). ROLES (plan 09-14 rev.):
     partner          base + X (+1 invocation, + X's ms), FNO x3 + X in place of the ViT pair
                      (same invocations, X's ms - the ViT pair's), X alone (1 invocation).
                      Primary: base + X.                         (cnext_unet, simvp, unet_mod)
     vit_replacement  the same three; primary: FNO x3 + X.        (swin_unet)
     fno_member       X alone vs the FNO CONTROL alone, and ViT x2 + X vs ViT x2 + control, with
                      latency as if xfam6's four fused FNOs became four X (4 ms(X) - ms(FNO x4
                      fused)); primary: the in-package swap.         (ffno)
     control          the FNO control itself: a 4th fused FNO member, FNO x4, alone.   (fno)
   The recorded number is  proj = 0.5 * dFinal(rel, tke, mvpe, sps at local 1:1 transfer; time
   held at the anchor) + dFinal(time only)  -- the local accuracy delta HALVED (the proxy
   over-reads ~2x), latency charged in full -- with the per-sample-factor (containment fixed) and
   sps-unchanged variants stored beside it.
4. LEDGER.  Upserts the arm's row of leaderboard/screen.jsonl, keyed by `arm`, under a lock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

SCREEN = "/fs/scratch/PAS1187/qwu/tmp/realpde/screen"
LEDGER = os.path.join(REPO, "leaderboard", "screen.jsonl")
SIGMA_JSON = os.path.join(HERE, "plan", "screen_sigma_20260914.json")
UNET = os.path.join(HERE, "unet")
BASE_FNO = [os.path.join(UNET, n) for n in
            ("fno_r18_cmb3sim.pth", "fno_r18_cmb3sim_s2.pth", "fno_r18_cmb3sim_s3.pth")]
BASE_VIT = [os.path.join(UNET, n) for n in ("arch_vit_d384.pth", "arch_vit_cmb3sim.pth")]
FNO_CTRL = os.path.join(SCREEN, "w1", "w1_fno_ctrl_s1.pth")
ALPHA, BAND_P, F_BOARD, A100_TO_4060 = 0.60, 0.45, 1.39, 1.850
ANCHOR_INV = 2                          # xfam6: one fused FNO group + one fused ViT group
SHIP_FNO = 4                            # xfam6's FNO group size
T_FIXED = 92.297891                     # R25's time_score; only shifts the wfinal ranking column
AOA_HI = (15, 20)
KCURVE_JSON = os.path.join(HERE, "plan", "screen_kcurve_20260914.json")
# Board-measured containment moves (sps points; CLAUDE.md "SPS factorizes"), the part of xfam6's
# sps that a single-family package does not have: the first ViT family (R21 T1 / T2R9) and the
# second ViT (R24, carried at the same value on Track 2, where T2R12 matched it).
LEDGER_CONTAINMENT = {1: {"vit1_R21": 0.184, "vit2_R24": 0.053},
                      2: {"vit1_T2R9": 0.163, "vit2_R24": 0.053}}
STANDALONE_BASIS = (
    "X x 4 fused at ONE invocation vs the anchor package (FNO x4 + ViT x2, 2 invocations). "
    "Errors: anchor board error x [X x1 proxy x kappa_FNO(4)] / base-twin proxy, kappa from the "
    "FNO cmb3sim twins' K = 1,2,3 curve fitted as e = a + b/K (plan/screen_kcurve_20260914.json). "
    "sps: anchor sps x per-sample factor at those errors, + 0.5 x the local X-vs-FNO containment "
    "at K = 1 (predict_board sps_1to1 - sps_pf, X alone vs one FNO), + the ledger containment "
    "REMOVED IN FULL: 1st ViT family (+0.184 R21 / +0.163 T2R9) and 2nd ViT (+0.053 R24). "
    "proj = 0.5 x (error + family-containment delta) + ledger + time. The K = 1 number is kept as "
    "standalone_k1, extrapolated, and never enters a verdict.")
LATENCY_TERMS = (
    "invocation term: +-1.151 board ms (T1) / +-0.866 (T2) per invocation (predict_board "
    "LAT_INV_*). Compute term: A100 fp16 ms x 1.850 (4060-eq) x 1.75 board ms per 4060-eq ms on "
    "T1 (b64 eager) / 0.60 on T2 (b1 CUDA-graph replay) (LAT_LOC_*). Every compared package is "
    "fp16 on both sides, so the precision-only slope PREC_SLOPE_T1 = 0.65 (R24 -> R25) does not "
    "enter. X x K fused compute = X x (FNO x4 fused / FNO x1) at the same batch.")
ROLE_OF_ARCH = {"ffno": "fno_member", "ffno_nff": "fno_member", "swin_unet": "vit_replacement",
                "fno": "control"}
FORMULA = ("proj = 0.5*final_v9.delta(anchor, [rel_p, tke_p, mvpe_p, time_anchor, sps_1to1]) "
           "+ final_v9.delta(anchor, [rel_a, tke_a, mvpe_a, time_p, sps_a]); subscores from "
           "predict_board.py --anchor (candidate and anchor err/sigma sets held out, bands refit "
           "by fit_band.py --band-p 0.45 at ALPHA 0.60, --anchor-local-ms 0, --local-ms = delta "
           "A100 ms x 1.850: b64 fp16 eager for T1, b1 fp16 CUDA-graph replay for T2 when "
           "the table has it, else b1 fp16 eager)")


# ------------------------------------------------------------------------------ ledger


@contextlib.contextmanager
def _locked(path: str):
    # The lock file lives on scratch, not beside the ledger: nothing generated goes in the repo tree.
    lock = os.path.join(SCREEN, "." + os.path.basename(path) + ".lock")
    os.makedirs(SCREEN, exist_ok=True)
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def upsert(row: dict, path: str = LEDGER) -> dict:
    """Merge `row` into the line with the same `arm` (new keys win) or append; atomic rewrite."""
    with _locked(path):
        rows = []
        if os.path.exists(path):
            with open(path) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
        for i, r in enumerate(rows):
            if r.get("arm") == row["arm"]:
                rows[i] = {**r, **row}
                row = rows[i]
                break
        else:
            rows.append(row)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    return row


# ------------------------------------------------------------------------------ instruments


def run_main(module: str, argv: list, log: str | None = None) -> str:
    """Run an instrument's own main() in-process with `argv`; return (and save) its stdout."""
    import importlib
    mod = importlib.import_module(module)
    old = sys.argv
    buf = io.StringIO()
    sys.argv = [module + ".py"] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"{module} exited {e.code}:\n{buf.getvalue()[-3000:]}") from e
    finally:
        sys.argv = old
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    out = buf.getvalue()
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(" ".join([module] + [str(a) for a in argv]) + "\n\n" + out)
    return out


def band_for(members: list[str], alpha: float) -> str:
    """fit_band.py's shipping band for this member set at this ALPHA, cached by content key."""
    key = hashlib.sha1(("|".join(sorted(os.path.realpath(m) for m in members))
                        + f"|a{alpha:.3f}|p{BAND_P}").encode()).hexdigest()[:16]
    out = os.path.join(SCREEN, "bands", f"band_{key}.npz")
    if not os.path.exists(out):
        run_main("fit_band", ["--ckpt", *members, "--err-ckpt", *members, "--alpha", alpha,
                              "--band-p", BAND_P, "--out", out], log=out[:-4] + ".txt")
    return out


class Data:
    def __init__(self):
        du = np.load(os.path.join(HERE, "val_unseen.npz"))
        self.X, self.Y = du["x"], du["y"]
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
        self.hi = np.isin(aoa, AOA_HI)


_MODEL_CACHE: dict = {}


def models_of(paths: list[str], dev: str):
    import _fno
    for p in paths:
        if p not in _MODEL_CACHE:
            _MODEL_CACHE[p] = _fno.load_any(p, dev)
    return [_MODEL_CACHE[p] for p in paths]


def infer(paths, X, dev, alpha):
    import _fno
    sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
    uv = _fno.predict_fno(models_of(paths, dev), X, dev, alpha=alpha, sigma_out=sig)
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), sig


def score_block(P, Y, sig, band_path, sel) -> dict:
    """rel / tke / mvpe / sps (1.39x board-like) / sps_local / wfinal on the rows `sel`."""
    import final_v9
    import sps9
    from sps9 import prep, sc, sps_score
    z = np.load(band_path)
    base, ref, p = z["base"][..., :2], z["sigma_ref"][..., :2], float(z["p"])
    Ps, Ys, ss = P[sel], Y[sel], sig[sel]
    half = (base * np.clip(ss / ref, 0.3, 3.0) ** np.float32(p)).astype(np.float32)
    rel = float(np.mean(sc.rel_l2_per_sample(Ps, Ys, 2)))
    tke = float(np.mean(sc.tke_rel_l2_per_sample(Ps, Ys, 2)))
    mvpe = float(sc.mvpe_rel_l2(Ps, Ys))
    A, a, scored, n = prep(Ps, Ys)
    sps_local = float(sps_score(A, a, scored, n, half))
    dm, tk, mv = sps9.sample_metrics(Ps, Ys)
    sps_b = float(sps_score(np.float32(F_BOARD) * A,
                            sps9.weights_from_metrics(F_BOARD * dm, F_BOARD * tk, F_BOARD * mv),
                            scored, n, half))
    wf = final_v9.predict([sc.score_error(F_BOARD * rel), sc.score_error(F_BOARD * tke),
                           sc.score_error(F_BOARD * mvpe), T_FIXED, sps_b])[0]
    return dict(rel=round(rel, 6), tke=round(tke, 6), mvpe=round(mvpe, 6), sps=round(sps_b, 4),
                sps_local=round(sps_local, 4), wfinal=round(float(wf), 4), n=int(len(Ps)))


def eval_set(paths, data, dev, alpha) -> dict:
    band = band_for(paths, alpha)
    P, sig = infer(paths, data.X, dev, alpha)
    return dict(band=band, full=score_block(P, data.Y, sig, band, slice(None)),
                aoa1520=score_block(P, data.Y, sig, band, data.hi))


def parse_ens_v9(text: str, n_sets: int) -> list[dict]:
    rows, seen_dash = [], False
    for line in text.splitlines():
        if line.startswith("-----"):
            seen_dash = True
            continue
        if seen_dash and line.count("|") == 2:
            left, mid, right = line.split("|")
            r, k, mv, s = (float(v) for v in mid.split())
            m4, fin = (float(v) for v in right.split())
            rows.append(dict(rel=r, tke=k, mvpe=mv, sps=s, mean4=m4, final=fin,
                             n=int(left.split()[-1])))
    if len(rows) != n_sets:
        raise RuntimeError(f"ens_v9: parsed {len(rows)} rows, expected {n_sets}:\n{text[-2000:]}")
    return rows


def anchors() -> tuple[str, str]:
    import predict_board as pb
    t1 = "r25" if "r25" in pb.ANCHORS else "r24"
    t2 = "t2r12" if "t2r12" in pb.ANCHORS else "t2r11"
    return t1, t2


_SUB = re.compile(r"^\s*(rel_l2|tke|mvpe|time|sps)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s*$")
_SPS = {"sps_1to1": re.compile(r"local sps transferred 1:1\s+sps\s+([-\d.]+)\s+final\s+([-\d.]+)"),
        "sps_pf": re.compile(r"per-sample factor ONLY\s+sps\s+([-\d.]+)\s+final\s+([-\d.]+)"),
        "sps_floor": re.compile(r"sps unchanged\s+sps\s+([-\d.]+)\s+final\s+([-\d.]+)")}
_MS = re.compile(r"-> anchored ([-\d.]+)")


def project(cand: list[str], cand_band: str, anchor_band: str, inv: int, dms_a100: float,
            anchor: str, log: str, anchor_set: list[str] | None = None,
            anchor_inv: int = ANCHOR_INV) -> dict:
    """predict_board.py --anchor for one candidate set; returns the halved projection and parts.

    `anchor_set` is the held-out set standing in for the anchor package (default: the base twins,
    FNO x3 + ViT x2, i.e. xfam6's twin set) and `anchor_band` must be that set's band."""
    import final_v9
    import predict_board as pb
    anchor_set = list(anchor_set) if anchor_set is not None else BASE_FNO + BASE_VIT
    local_ms = dms_a100 * A100_TO_4060
    argv = ["--ckpt", *cand, "--err-ckpt", *cand, "--band", cand_band, "--alpha", ALPHA,
            "--invocations", inv, f"--local-ms={local_ms:.6f}", "--anchor", anchor,
            "--anchor-err-ckpt", *anchor_set, "--anchor-ckpt", *anchor_set,
            "--anchor-band", anchor_band, "--anchor-alpha", ALPHA, "--anchor-local-ms=0",
            "--anchor-invocations", anchor_inv]
    text = run_main("predict_board", argv, log=log)
    subs = {}
    for line in text.splitlines():
        m = _SUB.match(line)
        if m:
            subs[m.group(1)] = float(m.group(3))
    if set(subs) != {"rel_l2", "tke", "mvpe", "time", "sps"}:
        raise RuntimeError(f"predict_board: could not parse subscores:\n{text[-2500:]}")
    spsv = {k: float(rx.search(text).group(1)) for k, rx in _SPS.items()}
    A = pb.ANCHORS[anchor]
    order = ("rel_l2", "tke", "mvpe", "time", "sps")
    a = [A[k] for k in order]

    def acc(sps_value):
        return final_v9.delta(a, [subs["rel_l2"], subs["tke"], subs["mvpe"], A["time"],
                                  sps_value])[0]

    lat = final_v9.delta(a, [A["rel_l2"], A["tke"], A["mvpe"], subs["time"], A["sps"]])[0]
    acc_1, acc_pf, acc_fl = acc(spsv["sps_1to1"]), acc(spsv["sps_pf"]), acc(spsv["sps_floor"])
    ms = _MS.search(text)
    return dict(anchor=anchor, anchor_set=[os.path.basename(p) for p in anchor_set],
                cand_set=[os.path.basename(p) for p in cand], invocations=inv,
                anchor_invocations=anchor_inv, d_a100_ms=round(dms_a100, 5),
                local_ms_4060eq=round(local_ms, 5),
                board_ms=float(ms.group(1)) if ms else None,
                subs_pred={k: subs[k] for k in order}, sps=spsv,
                acc_1to1=round(acc_1, 4), acc_pf=round(acc_pf, 4), acc_floor=round(acc_fl, 4),
                lat=round(lat, 4), proj=round(0.5 * acc_1 + lat, 4),
                proj_pf=round(0.5 * acc_pf + lat, 4), proj_floor=round(0.5 * acc_fl + lat, 4),
                unhalved_1to1=round(acc_1 + lat, 4))


# The lowest board latency this account has recorded at ONE invocation, per track: R20 (four FNOs
# fused, Track 1, 3.795 ms) and T2R8 (the same package on Track 2, 4.659 ms). predict_board's
# latency model is linear in invocations and local ms and was calibrated on small deltas; a
# standalone candidate that sheds the ViT group and most FLOPs extrapolates it below anything the
# board has returned (the harness-validation row read 0.61 board ms, worth +0.47 `final`).
FLOOR_ANCHOR = {1: "r20", 2: "t2r8"}


def conservative(p: dict, track: int) -> dict:
    """proj_cons = 0.5*max(acc, 0) + min(acc, 0) + latency at the floored board ms.

    The plan halves the local accuracy delta because the proxy over-reads error CUTS ~2x (R24).
    Nothing measures that it over-reads LOSSES, so this variant halves gains only -- which matters
    for every role whose accuracy delta can be negative (a ViT-pair replacement, a standalone
    package) -- and clamps the anchored board ms at the account's lowest one-invocation reading.
    For a partner whose accuracy delta is positive and whose latency grows it equals `proj`."""
    import final_v9
    import predict_board as pb
    A = pb.ANCHORS[p["anchor"]]
    a = [A[k] for k in ("rel_l2", "tke", "mvpe", "time", "sps")]
    floor_ms = 728.96 * (100.0 / pb.ANCHORS[FLOOR_ANCHOR[track]]["time"] - 1.0) ** 2
    ms = p.get("board_ms")
    if ms is None or ms >= floor_ms:
        lat_f = p["lat"]
    else:
        t = float(pb.score_time(floor_ms))
        lat_f = final_v9.delta(a, [A["rel_l2"], A["tke"], A["mvpe"], t, A["sps"]])[0]
    acc = p["acc_1to1"]
    return dict(board_ms_floor=round(floor_ms, 3), floored=bool(ms is not None and ms < floor_ms),
                lat_floored=round(lat_f, 4),
                proj_cons=round(0.5 * max(acc, 0.0) + min(acc, 0.0) + lat_f, 4))


def primary_parts(det: dict, role: str):
    """(standalone T1, partner T1, partner T2, primary T1) projections of a row, by role."""
    if role == "fno_member":
        return (det["t1_standalone"], det["t1_member_swap"], det["t2_member_swap"],
                det["t1_member_swap"])
    return (det["t1_standalone"], det["t1_partner"], det["t2_partner"],
            det["t1_replace"] if role == "vit_replacement" else det["t1_partner"])


def kcurve_data(dev, data, force: bool = False) -> dict:
    """The FNO member-count curve on the screen's own instruments, from the three cmb3sim twins:
    every subset of size K = 1, 2, 3 through eval_set (band refit per subset), averaged per K, then
    e(K) = a + b/K fitted per metric and evaluated at K = SHIP_FNO. Cached in KCURVE_JSON."""
    if os.path.exists(KCURVE_JSON) and not force:
        with open(KCURVE_JSON) as fh:
            return json.load(fh)
    import itertools
    metrics = ("rel", "tke", "mvpe", "sps", "sps_local", "wfinal")
    per_subset, per_k = {}, {}
    for k in (1, 2, 3):
        res = []
        for combo in itertools.combinations(BASE_FNO, k):
            r = eval_set(list(combo), data, dev, ALPHA)
            res.append(dict(members=[os.path.basename(c) for c in combo], full=r["full"],
                            aoa1520=r["aoa1520"]))
        per_subset[str(k)] = res
        per_k[str(k)] = {sp: {m: float(np.mean([r[sp][m] for r in res])) for m in metrics}
                         for sp in ("full", "aoa1520")}
    ks = np.array([1.0, 2.0, 3.0])
    design = np.stack([np.ones(3), 1.0 / ks], 1)
    fit = {}
    for sp in ("full", "aoa1520"):
        for m in metrics:
            y = np.array([per_k[str(int(k))][sp][m] for k in ks])
            a_, b_ = np.linalg.lstsq(design, y, rcond=None)[0]
            vk = a_ + b_ / SHIP_FNO
            fit[f"{sp}.{m}"] = dict(a=round(float(a_), 6), b=round(float(b_), 6),
                                    at_K=round(float(vk), 6),
                                    ratio_K_over_1=round(float(vk / y[0]), 6),
                                    delta_K_minus_1=round(float(vk - y[0]), 6))
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), K=SHIP_FNO, alpha=ALPHA,
               members=[os.path.basename(p) for p in BASE_FNO],
               model="e(K) = a + b/K, least squares on K = 1, 2, 3 (mean over all subsets of the "
                     "three FNO cmb3sim twins); kappa_m(K) = e_fit(K) / e(1)",
               per_k=per_k, per_subset=per_subset, fit=fit,
               reference="main 09-14: ens_v9 FNO K = 1/2/3 full sps 43.86 / 44.19 / 44.31",
               job=os.environ.get("SLURM_JOB_ID"))
    with open(KCURVE_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def standalone_xk(track: int, anchor: str, x_proxy: dict, base_proxy: dict, fam_cont: float,
                  ms_x: float, lat: dict, key: str, kc: dict) -> dict:
    """ONE family, K = SHIP_FNO members fused into one invocation, against the anchor package.
    See STANDALONE_BASIS and LATENCY_TERMS; every term is returned separately."""
    import final_v9
    import predict_board as pb
    import sps9
    from sps9 import sc
    A = pb.ANCHORS[anchor]
    a = [A[k] for k in ("rel_l2", "tke", "mvpe", "time", "sps")]
    ab = pb.errors_of(A)
    kap = {m: kc["fit"][f"aoa1520.{m}"]["ratio_K_over_1"] for m in ("rel", "tke", "mvpe")}
    pe = {m: ab[m] * x_proxy[m] * kap[m] / base_proxy[m] for m in ("rel", "tke", "mvpe")}
    subs = {"rel_l2": float(sc.score_error(pe["rel"])), "tke": float(sc.score_error(pe["tke"])),
            "mvpe": float(sc.score_error(pe["mvpe"]))}
    w_of = lambda e: float(sps9.weights_from_metrics(np.float64(e["rel"]), np.float64(e["tke"]),  # noqa: E731
                                                     np.float64(e["mvpe"])))
    sps_pf = A["sps"] * w_of(pe) / w_of(ab)
    ledger = -float(sum(LEDGER_CONTAINMENT[track].values()))
    f4, v2, f1 = lat["pkg_fno4_fused"][key], lat["pkg_vit2_fused"][key], lat["ref_fno_w96L8"][key]
    dms = (f4 / f1) * ms_x - f4 - v2
    anc_ms = 728.96 * (100.0 / A["time"] - 1.0) ** 2
    board_ms = anc_ms - pb.lat_inv_of(track) + pb.lat_loc_of(track) * A100_TO_4060 * dms
    floor_ms = 728.96 * (100.0 / pb.ANCHORS[FLOOR_ANCHOR[track]]["time"] - 1.0) ** 2
    t = float(pb.score_time(max(board_ms, floor_ms)))
    w_sps = float(np.asarray(final_v9.W0)[4])
    d_err = final_v9.delta(a, [subs["rel_l2"], subs["tke"], subs["mvpe"], A["time"], sps_pf])[0]
    d_fam, d_led = w_sps * fam_cont, w_sps * ledger
    d_time = final_v9.delta(a, [A["rel_l2"], A["tke"], A["mvpe"], t, A["sps"]])[0]
    local = d_err + d_fam
    return dict(K=SHIP_FNO, anchor=anchor, kappa=kap, pred_board_err=pe, subs_pred=subs,
                sps_per_sample=round(sps_pf, 4), fam_containment_sps=round(fam_cont, 4),
                ledger_containment_sps=round(ledger, 4), ledger_entries=LEDGER_CONTAINMENT[track],
                d_a100_ms=round(dms, 5), board_ms=round(board_ms, 3),
                board_ms_floor=round(floor_ms, 3), latency_floored=bool(board_ms < floor_ms),
                d_err=round(d_err, 4), d_fam=round(d_fam, 4), d_ledger=round(d_led, 4),
                d_time=round(d_time, 4),
                accuracy_only=round(0.5 * local + d_led, 4),
                accuracy_only_cons=round(0.5 * max(local, 0.0) + min(local, 0.0) + d_led, 4),
                proj=round(0.5 * local + d_led + d_time, 4),
                proj_cons=round(0.5 * max(local, 0.0) + min(local, 0.0) + d_led + d_time, 4))


# ------------------------------------------------------------------------------ latency


def load_lat(path: str) -> dict:
    with open(path) as fh:
        return {r["name"]: r for r in json.load(fh)["rows"]}


def lat_row_for(lat: dict, cfg: dict, explicit: str | None) -> dict:
    if explicit:
        return lat[explicit]
    arch, w, nl = cfg["arch"], cfg["width"], cfg["n_layers"]
    if arch == "fno" and w == 96 and nl == 8:
        return lat["ref_fno_w96L8"]
    if arch == "vit" and w == 384 and nl == 8:
        return lat["ref_vit_d384L8"]
    name = f"{arch}_w{w}L{nl}" + (f"_m{cfg['modes_y']}x{cfg['modes_x']}"
                                  if arch.startswith("ffno") else "")
    if name not in lat:
        raise SystemExit(f"latency table has no row {name}; pass --lat-row")
    return lat[name]


def t2_key(lat: dict) -> str:
    """Track 2 ships a CUDA-graph-captured step, so its latency proxy is the graphed batch-1
    forward when the table has one (lat2 on); eager batch 1 is launch-bound for every model."""
    return "ms_b1g_fp16" if lat["ref_fno_w96L8"].get("ms_b1g_fp16") else "ms_b1_fp16"


def plan_invocations(family: str, which: str) -> int:
    if which == "partner":
        return ANCHOR_INV + (1 if family == "new" else 0)
    if which == "replace":
        return 1 if family == "fno" else 2
    return 1


def plan_dms(family: str, which: str, lat: dict, xrow: dict, key: str) -> float:
    """A100 ms delta of the shipped package, per window (b64) or per step (b1)."""
    f4, v2 = lat["pkg_fno4_fused"][key], lat["pkg_vit2_fused"][key]
    x = xrow[key]
    member = {"fno": f4 / SHIP_FNO, "vit": v2 / 2.0}.get(family, x)
    if which == "partner":
        return member
    if which == "replace":
        return member - v2
    return x - f4 - v2


# ------------------------------------------------------------------------------ modes


def cfg_of(path: str) -> dict:
    import torch
    blob = torch.load(path, map_location="cpu")
    c = dict(blob.get("config", {}))
    for k in ("arch", "width", "n_layers", "modes_y", "modes_x"):
        if k in blob:
            c[k] = blob[k]
    c["_params_M"] = round(sum(v.numel() for v in blob["model_state_dict"].values()
                               if v.is_floating_point()) / 1e6, 4)
    return c


def arm_eval(args) -> None:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logdir = os.path.join(SCREEN, "eval", args.arm)
    os.makedirs(logdir, exist_ok=True)
    X = [os.path.abspath(c) for c in args.ckpt]
    cfg = cfg_of(X[0])
    arch = cfg["arch"]
    family = args.family if args.family != "auto" else (arch if arch in ("fno", "vit") else "new")
    role = args.role if args.role != "auto" else ROLE_OF_ARCH.get(arch, "partner")
    lat = load_lat(args.lat)
    xrow = lat_row_for(lat, cfg, args.lat_row)
    data = Data()
    t0 = time.time()
    print(f"[{args.arm}] {arch} w{cfg['width']} L{cfg['n_layers']} family={family} role={role} "
          f"ckpt={X} lat_row={xrow['name']}", flush=True)

    # 1. single model, ALPHA sweep
    sweep = {}
    for a in (0.4, 0.6, 0.8):
        sweep[f"{a:.1f}"] = s = eval_set(X[:1], data, dev, a)
        print(f"  single alpha {a}: full {s['full']}\n                  aoa  {s['aoa1520']}",
              flush=True)

    # 2. member sets
    base = BASE_FNO + BASE_VIT
    sets = {"base": base, "base_plus_x": base + X, "fno3_plus_x": BASE_FNO + X, "x_alone": X}
    if role == "fno_member":
        ctrl = [os.path.abspath(args.fno_ctrl)]
        if not os.path.exists(ctrl[0]):
            raise SystemExit(f"role fno_member needs the FNO control checkpoint {ctrl[0]}")
        sets.update(ctrl_alone=ctrl, vit2_plus_x=BASE_VIT + X, vit2_plus_ctrl=BASE_VIT + ctrl)
    ens_txt = run_main("ens_v9", ["--sets", *[",".join(v) for v in sets.values()],
                                  "--alpha", ALPHA], log=os.path.join(logdir, "ens_v9.txt"))
    ens_rows = parse_ens_v9(ens_txt, len(sets))
    set_res = {}
    for (name, paths), er in zip(sets.items(), ens_rows):
        r = eval_set(paths, data, dev, ALPHA)
        r["ens_v9_aoa1520_p0band"] = er
        r["members"] = [os.path.basename(p) for p in paths]
        set_res[name] = r
        print(f"  set {name:14s} ens_v9 {er}\n                 p0.45 aoa {r['aoa1520']}",
              flush=True)

    # 3. projections
    t1, t2 = anchors()
    k2 = t2_key(lat)
    proj = {}
    for track, anchor, key in ((1, t1, "ms_b64_fp16"), (2, t2, k2)):
        if role == "fno_member":
            dms = SHIP_FNO * xrow[key] - lat["pkg_fno4_fused"][key]
            jobs = (("standalone", "x_alone", "ctrl_alone", 1),
                    ("member_swap", "vit2_plus_x", "vit2_plus_ctrl", ANCHOR_INV),
                    ("member_swap_vs_base", "vit2_plus_x", "base", ANCHOR_INV))
            for which, cand, anc, inv in jobs:
                proj[f"t{track}_{which}"] = project(
                    sets[cand], set_res[cand]["band"], set_res[anc]["band"], inv, dms, anchor,
                    os.path.join(logdir, f"predict_board_t{track}_{which}.txt"),
                    anchor_set=sets[anc], anchor_inv=inv)
        else:
            for which, sname in (("partner", "base_plus_x"), ("replace", "fno3_plus_x"),
                                 ("standalone", "x_alone")):
                inv = plan_invocations(family, which)
                dms = plan_dms(family, which, lat, xrow, key)
                proj[f"t{track}_{which}"] = project(
                    sets[sname], set_res[sname]["band"], set_res["base"]["band"], inv, dms,
                    anchor, os.path.join(logdir, f"predict_board_t{track}_{which}.txt"))
        for k in [k for k in proj if k.startswith(f"t{track}_")]:
            p = proj[k]
            p.update(conservative(p, track))
            print(f"  proj {k:24s} inv {p['invocations']} vs {p['anchor_invocations']}  "
                  f"dA100ms {p['d_a100_ms']:+.4f} -> proj {p['proj']:+.4f} (acc1:1 "
                  f"{p['acc_1to1']:+.4f} pf {p['acc_pf']:+.4f} lat {p['lat']:+.4f}; cons "
                  f"{p['proj_cons']:+.4f}{' FLOORED' if p['floored'] else ''})", flush=True)

    # STANDALONE, fixed 09-14 after main's review of the validation row: no package is ONE model
    # (none has been on the board since R7). Project "X x K fused, one invocation" with K =
    # SHIP_FNO; predict_board's K = 1 standalone stays in proj_detail, marked extrapolated.
    x4 = None
    if role != "fno_member":
        kc = kcurve_data(dev, data)
        comp = os.path.abspath(args.fno_ctrl)
        if not os.path.exists(comp) or comp in X:
            comp = BASE_FNO[0]
        x4 = {}
        for track, anchor, key in ((1, t1, "ms_b64_fp16"), (2, t2, k2)):
            fc = project(X[:1], sweep["0.6"]["band"], band_for([comp], ALPHA), 1, 0.0, anchor,
                         os.path.join(logdir, f"predict_board_t{track}_family_vs_fno.txt"),
                         anchor_set=[comp], anchor_inv=1)
            fam = fc["sps"]["sps_1to1"] - fc["sps"]["sps_pf"]
            x4[track] = standalone_xk(track, anchor, sweep["0.6"]["aoa1520"],
                                      set_res["base"]["aoa1520"], fam, xrow[key], lat, key, kc)
            x4[track].update(fam_comparator=os.path.basename(comp), fam_detail=fc)
            s_ = x4[track]
            print(f"  standalone X x{SHIP_FNO} T{track}: proj {s_['proj']:+.4f} (cons "
                  f"{s_['proj_cons']:+.4f}; err {s_['d_err']:+.4f} fam {s_['d_fam']:+.4f} ledger "
                  f"{s_['d_ledger']:+.4f} time {s_['d_time']:+.4f}; board ms {s_['board_ms']}"
                  f"{' FLOORED' if s_['latency_floored'] else ''})", flush=True)
        for t in (1, 2):
            proj[f"t{t}_standalone"]["extrapolated"] = True

    sigma = None
    if os.path.exists(SIGMA_JSON):
        with open(SIGMA_JSON) as fh:
            sigma = json.load(fh).get("sigma_seed")
    if role == "fno_member":
        st1, pa1, pa2, st2 = (proj["t1_standalone"], proj["t1_member_swap"],
                              proj["t2_member_swap"], proj["t2_standalone"])
        primary = pa1
        semantics = ("fno_member: standalone = X alone vs the FNO control alone; partner = X "
                     "swapped for the control inside the ViT x2 twins (vit2+X vs vit2+ctrl); "
                     "latency as if xfam6's 4 fused FNOs became 4 X")
        extra = dict(proj_dfinal_t2_standalone=st2["proj"],
                     proj_dfinal_t1_member_swap_vs_base=proj["t1_member_swap_vs_base"]["proj"],
                     single_ctrl_full=set_res["ctrl_alone"]["full"],
                     single_ctrl_aoa1520=set_res["ctrl_alone"]["aoa1520"])
    else:
        st1, pa1, pa2, st2 = x4[1], proj["t1_partner"], proj["t2_partner"], x4[2]
        primary = proj["t1_replace"] if role == "vit_replacement" else pa1
        semantics = (f"{role}: partner = base + X; replace = FNO x3 + X in place of the ViT "
                     f"pair (family {family}: "
                     f"{'+1 invocation' if family == 'new' else 'fused into its group'}); "
                     f"standalone = X x{SHIP_FNO} fused at one invocation (see standalone_basis)")
        extra = dict(proj_dfinal_t1_replace_vits=proj["t1_replace"]["proj"],
                     proj_dfinal_t2_replace_vits=proj["t2_replace"]["proj"],
                     proj_dfinal_t2_standalone=st2["proj"], proj_cons_t2_standalone=st2["proj_cons"],
                     standalone_basis=STANDALONE_BASIS, standalone_x4_detail=x4,
                     standalone_k1={f"t{t}": dict(proj=proj[f"t{t}_standalone"]["proj"],
                                                  proj_cons=proj[f"t{t}_standalone"]["proj_cons"],
                                                  extrapolated=True) for t in (1, 2)})
    row = dict(
        date=time.strftime("%Y-%m-%d"), wave=args.wave, arm=args.arm,
        backbone=f"{arch} w{cfg['width']} L{cfg['n_layers']}"
                 + (f" modes {cfg.get('modes_y')}x{cfg.get('modes_x')}"
                    if arch.startswith("ffno") else ""),
        family=family, status=args.status, gpu_h=args.gpu_h, params_M=cfg["_params_M"],
        epochs=cfg.get("epochs"), dropout=cfg.get("dropout"),
        ms_b64_fp16=xrow["ms_b64_fp16"], ms_b64_fp32=xrow["ms_b64_fp32"],
        ms_b1_fp16=xrow["ms_b1_fp16"], ms_ratio_fno=xrow["ms_ratio_fno"],
        single_full={k: sweep["0.6"]["full"][k] for k in ("rel", "tke", "mvpe", "sps")},
        single_aoa1520={k: sweep["0.6"]["aoa1520"][k] for k in ("rel", "tke", "mvpe", "sps")},
        proj_dfinal_t1_standalone=st1["proj"], proj_dfinal_t1_partner=pa1["proj"],
        proj_dfinal_t2_partner=pa2["proj"], verdict="pending", note=args.note,
        # ---- extra keys ----
        role=role, proj_primary=primary["proj"], proj_semantics=semantics, **extra,
        proj_cons_t1_standalone=st1["proj_cons"], proj_cons_t1_partner=pa1["proj_cons"],
        proj_cons_t2_partner=pa2["proj_cons"], proj_cons_primary=primary["proj_cons"],
        anchor_t1=t1, anchor_t2=t2, proj_formula=FORMULA, latency_terms=LATENCY_TERMS,
        proj_detail=proj,
        single_detail={"alpha_sweep": sweep}, sets=set_res, sigma_seed=sigma,
        ckpt=X, seed=cfg.get("seed"), lat_row=xrow["name"], lat_table=os.path.abspath(args.lat),
        int8_MB=xrow.get("int8_MB"), ms_b1_fp32=xrow.get("ms_b1_fp32"),
        ms_b1g_fp16=xrow.get("ms_b1g_fp16"), ratio_b1g_fp16=xrow.get("ratio_b1g_fp16"),
        gflops_b1=xrow.get("gflops_b1"), flops_ratio_fno=xrow.get("flops_ratio_fno"),
        t2_latency_key=k2,
        eval_job=os.environ.get("SLURM_JOB_ID"), eval_s=round(time.time() - t0, 1))
    # READ FIRST (scout: every MSE-trained backbone under-predicts TKE, so a new family should move
    # rel and containment): the board-scale rel_l2 subscore delta and the CONTAINMENT part of the
    # sps delta (local 1:1 minus the per-sample-factor prediction) of every projection.
    row["read_first"] = {k: dict(d_rel_l2_subscore=round(p["subs_pred"]["rel_l2"]
                                                         - _anchor_sub(p["anchor"], "rel_l2"), 4),
                                 d_sps_containment=round(p["sps"]["sps_1to1"] - p["sps"]["sps_pf"], 4),
                                 d_tke_subscore=round(p["subs_pred"]["tke"]
                                                      - _anchor_sub(p["anchor"], "tke"), 4))
                         for k, p in proj.items()}
    for t in ((1, 2) if x4 else ()):
        s_ = x4[t]
        row["read_first"][f"t{t}_standalone_x{SHIP_FNO}"] = dict(
            d_rel_l2_subscore=round(s_["subs_pred"]["rel_l2"] - _anchor_sub(s_["anchor"], "rel_l2"), 4),
            d_sps_containment=round(0.5 * s_["fam_containment_sps"] + s_["ledger_containment_sps"], 4),
            d_tke_subscore=round(s_["subs_pred"]["tke"] - _anchor_sub(s_["anchor"], "tke"), 4))
    row["verdict_rule"] = verdict_rule(row, sigma)
    upsert(row)
    print(f"[{args.arm}] upserted into {LEDGER} ({row['eval_s']} s); primary {row['proj_primary']:+.4f}"
          f"; verdict_rule: {row['verdict_rule']}", flush=True)


def _anchor_sub(anchor: str, key: str) -> float:
    import predict_board as pb
    return pb.ANCHORS[anchor][key]


def verdict_rule(row: dict, sigma: dict | None) -> str:
    """Plan section 4's decision rules, applied mechanically. The main session decides."""
    if row.get("role") == "control":
        return "control (the no-sim recipe's FNO reference; no verdict)"
    if not sigma:
        return "no sigma_seed yet"
    # Plan formula on every term (main 09-14: partner and ViT-pair-replacement projections are
    # the calibrated case); the standalone entry is X x K, never the extrapolated K = 1 number.
    s_st = sigma.get("t1_standalone_x4") or sigma.get("t1_standalone")
    s_pa = sigma.get("t1_partner") or s_st
    cands = [(row["proj_dfinal_t1_standalone"], s_st), (row["proj_dfinal_t1_partner"], s_pa),
             (row["proj_primary"], s_pa)]
    if all(v < -s for v, s in cands):
        return "drop (every projection below 0 by more than 1 sigma_seed)"
    best, s_best = max(cands)
    if best >= 0.10:
        return "replicate, then advance if seed 2 holds (>= +0.10 needs a replicate)"
    if abs(best - 0.05) <= s_best:
        return "replicate (within 1 sigma_seed of +0.05)"
    return "hold (neither the drop nor the replicate rule fires)"


def sigma_seed(args) -> None:
    """Spread of the three FNO cmb3sim twins: single-model metrics and the projection's own noise."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logdir = os.path.join(SCREEN, "eval", "_sigma_seed")
    lat = load_lat(args.lat)
    data = Data()
    t1, t2 = anchors()
    k2 = t2_key(lat)
    base = BASE_FNO + BASE_VIT
    base_band = band_for(base, ALPHA)
    per = []
    for i, twin in enumerate(BASE_FNO):
        r = eval_set([twin], data, dev, ALPHA)
        others = [b for b in BASE_FNO if b != twin]
        st = {}
        for track, anchor, key in ((1, t1, "ms_b64_fp16"), (2, t2, k2)):
            dms = (lat["ref_fno_w96L8"][key] - lat["pkg_fno4_fused"][key]
                   - lat["pkg_vit2_fused"][key])
            st[track] = project([twin], r["band"], base_band, 1, dms, anchor,
                                os.path.join(logdir, f"pb_t{track}_standalone_{i}.txt"))
        # partner-style: the full base against the base WITHOUT this twin. The candidate is the
        # same for every twin; what varies is which member the anchor set lacks, so the spread is
        # the seed noise of ONE added member's accuracy delta (time is identical, excluded).
        sub_base = others + BASE_VIT
        pa = project(base, base_band, band_for(sub_base, ALPHA), ANCHOR_INV, 0.0, t1,
                     os.path.join(logdir, f"pb_t1_partner_{i}.txt"), anchor_set=sub_base)
        for tr in (1, 2):
            st[tr].update(conservative(st[tr], tr))
        per.append(dict(twin=os.path.basename(twin), single=r, t1_standalone=st[1]["proj"],
                        t2_standalone=st[2]["proj"], t1_standalone_cons=st[1]["proj_cons"],
                        t2_standalone_cons=st[2]["proj_cons"], standalone_detail=st,
                        t1_partner_detail=pa,
                        t1_partner=round(0.5 * pa["acc_1to1"], 4)))
        print(f"  twin {i}: full {r['full']}\n          aoa  {r['aoa1520']}\n          proj T1 "
              f"standalone {st[1]['proj']:+.4f}  T2 {st[2]['proj']:+.4f}  member-add "
              f"{0.5 * pa['acc_1to1']:+.4f}", flush=True)
    sd = lambda v: float(np.std(v, ddof=1))  # noqa: E731
    keys = ("rel", "tke", "mvpe", "sps", "wfinal")
    out = dict(
        date=time.strftime("%Y-%m-%d %H:%M"), twins=[p["twin"] for p in per],
        single_full_sd={k: sd([p["single"]["full"][k] for p in per]) for k in keys},
        single_aoa1520_sd={k: sd([p["single"]["aoa1520"][k] for p in per]) for k in keys},
        sigma_seed=dict(t1_standalone=round(sd([p["t1_standalone"] for p in per]), 4),
                        t2_standalone=round(sd([p["t2_standalone"] for p in per]), 4),
                        t1_partner=round(sd([p["t1_partner"] for p in per]), 4),
                        t1_standalone_cons=round(sd([p["t1_standalone_cons"] for p in per]), 4),
                        t2_standalone_cons=round(sd([p["t2_standalone_cons"] for p in per]), 4),
                        definition="sample sd (ddof 1) over the three FNO cmb3sim twins of the "
                                   "HALVED projection. standalone: the twin as the whole package "
                                   "vs base. partner: 0.5*acc_1to1 of base vs base minus that "
                                   "twin (one member's contribution; time identical)."),
        per_twin=per, anchors=[t1, t2], formula=FORMULA, job=os.environ.get("SLURM_JOB_ID"))
    with open(SIGMA_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("single_full_sd", "single_aoa1520_sd", "sigma_seed")},
                     indent=1))
    print(f"wrote {SIGMA_JSON}")


def kcurve_mode(args) -> None:
    """GPU: refit the FNO K-curve, then CALIBRATE the X x K standalone on the one case the board
    has scored -- the FNO x4 cmb3sim package alone at one invocation (R20 on Track 1, T2R8 on
    Track 2) against the anchor -- and write the seed spread of the X x K projection."""
    import final_v9
    import predict_board as pb
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lat = load_lat(args.lat)
    data = Data()
    kc = kcurve_data(dev, data, force=True)
    t1, t2 = anchors()
    k2 = t2_key(lat)
    base = eval_set(BASE_FNO + BASE_VIT, data, dev, ALPHA)
    calib = {}
    for track, anchor, key in ((1, t1, "ms_b64_fp16"), (2, t2, k2)):
        x4 = standalone_xk(track, anchor, kc["per_k"]["1"]["aoa1520"], base["aoa1520"], 0.0,
                           lat["ref_fno_w96L8"][key], lat, key, kc)
        truth = FLOOR_ANCHOR[track]
        R, A = pb.ANCHORS[truth], pb.ANCHORS[anchor]
        board_acc = (R["final"] - A["final"]) - float(np.asarray(final_v9.W0)[3]) * (R["time"] - A["time"])
        per = [standalone_xk(track, anchor, s_["aoa1520"], base["aoa1520"], 0.0,
                             lat["ref_fno_w96L8"][key], lat, key, kc)["proj"]
               for s_ in kc["per_subset"]["1"]]
        calib[f"t{track}"] = dict(
            truth=f"{truth} (FNO x4 cmb3sim alone, one invocation) minus {anchor}, time removed",
            board_accuracy_only=round(board_acc, 4), pred_accuracy_only=x4["accuracy_only"],
            pred_accuracy_only_cons=x4["accuracy_only_cons"], pred_proj=x4["proj"],
            sigma_x4_twins=round(float(np.std(per, ddof=1)), 4), detail=x4)
        print(f"T{track}: board accuracy-only {board_acc:+.4f}  predicted {x4['accuracy_only']:+.4f} "
              f"(halved) / {x4['accuracy_only_cons']:+.4f} (losses unhalved); proj {x4['proj']:+.4f}; "
              f"seed sd {calib[f't{track}']['sigma_x4_twins']:.4f}", flush=True)
    kc["calibration"] = calib
    with open(KCURVE_JSON, "w") as fh:
        json.dump(kc, fh, indent=1)
    if os.path.exists(SIGMA_JSON):
        with open(SIGMA_JSON) as fh:
            sj = json.load(fh)
        sj["sigma_seed"]["t1_standalone_x4"] = calib["t1"]["sigma_x4_twins"]
        sj["sigma_seed"]["t2_standalone_x4"] = calib["t2"]["sigma_x4_twins"]
        with open(SIGMA_JSON, "w") as fh:
            json.dump(sj, fh, indent=1)
    print(json.dumps({k: kc["fit"][k] for k in kc["fit"] if k.startswith("aoa1520")}, indent=1))
    print(f"wrote {KCURVE_JSON}")


def rescore(args) -> None:
    """Recompute the conservative projections and the verdict rule of stored rows (no GPU)."""
    sigma = None
    if os.path.exists(SIGMA_JSON):
        with open(SIGMA_JSON) as fh:
            sj = json.load(fh)
        sigma = sj.get("sigma_seed")
        if sigma is not None and "t1_standalone_cons" not in sigma:
            # Latency is identical across the twins and each twin's standalone accuracy delta
            # against the five-member base is negative, so the conservative spread (acc not
            # halved) is exactly twice the halved one. Recorded as such.
            sigma["t1_standalone_cons"] = round(2.0 * sigma["t1_standalone"], 4)
            sigma["t2_standalone_cons"] = round(2.0 * sigma["t2_standalone"], 4)
            sigma["cons_note"] = ("*_standalone_cons = 2 x the halved spread (identical latency "
                                  "across twins, negative standalone accuracy deltas)")
            with open(SIGMA_JSON, "w") as fh:
                json.dump(sj, fh, indent=1)
    with open(LEDGER) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    for r in rows:
        if (args.arm and r["arm"] != args.arm) or not r.get("proj_detail"):
            continue
        det = r["proj_detail"]
        for k, p in det.items():
            p.update(conservative(p, 1 if k.startswith("t1_") else 2))
        st1, pa1, pa2, prim = primary_parts(det, r.get("role", "partner"))
        upd = dict(arm=r["arm"], proj_detail=det, sigma_seed=sigma,
                   proj_cons_t1_partner=pa1["proj_cons"],
                   proj_cons_t2_partner=pa2["proj_cons"], proj_cons_primary=prim["proj_cons"])
        upd["verdict_rule"] = verdict_rule({**r, **upd}, sigma)
        upsert(upd)
        print(r["arm"], {k: v for k, v in upd.items() if k.startswith("proj_cons")},
              upd["verdict_rule"])


def register(args) -> None:
    row = dict(arm=args.arm)
    if args.status:
        row["status"] = args.status
    if args.note is not None:
        row["note"] = args.note
    if args.wave is not None:
        row["wave"] = args.wave
    if args.gpu_h is not None:
        row["gpu_h"] = args.gpu_h
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        row[k] = v
    row.setdefault("date", time.strftime("%Y-%m-%d"))
    print(json.dumps(upsert({"verdict": "pending", **row} if not _exists(args.arm) else row)))


def _exists(arm: str) -> bool:
    if not os.path.exists(LEDGER):
        return False
    with open(LEDGER) as fh:
        return any(json.loads(l).get("arm") == arm for l in fh if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--ckpt", nargs="+")
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--lat", default=os.path.join(SCREEN, "lat", "lat1.json"))
    ap.add_argument("--lat-row", default=None)
    ap.add_argument("--family", choices=["auto", "new", "fno", "vit"], default="auto")
    ap.add_argument("--role", choices=["auto", "partner", "vit_replacement", "fno_member",
                                       "control"], default="auto")
    ap.add_argument("--fno-ctrl", default=FNO_CTRL)
    ap.add_argument("--status", default=None)
    ap.add_argument("--gpu-h", type=float, default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--sigma-seed", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--kcurve", action="store_true",
                    help="GPU: FNO K-curve + X x K standalone calibration against R20 / T2R8")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute conservative projections + verdict of stored rows (--arm "
                         "to restrict); no GPU")
    ap.add_argument("--set", nargs="*", help="register: extra key=value (JSON values allowed)")
    args = ap.parse_args()
    if args.sigma_seed:
        return sigma_seed(args)
    if args.rescore:
        return rescore(args)
    if args.kcurve:
        return kcurve_mode(args)
    if not args.arm:
        raise SystemExit("--arm is required")
    if args.register:
        return register(args)
    if not args.ckpt:
        raise SystemExit("--ckpt is required to evaluate an arm")
    args.status = args.status or "done"
    args.wave = 1 if args.wave is None else args.wave
    args.note = args.note or ""
    arm_eval(args)


if __name__ == "__main__":
    main()
