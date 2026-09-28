"""Out-of-distribution stream instrument: regimes, member layouts, config checks, band fit split.

Everything here is DERIVED in code (rule 3): evaluation trajectories from the `train_real` file names, held-out sets
from `re_split.SETS`, the in-distribution fit split from `model.VAL_TRAJ`. Nothing is typed.

    python ood_lib.py regime  --regime A20                       # evaluation trajectories and groups
    python ood_lib.py members --regime A20 --layout mirror       # member paths (full-size layout), configs checked
    python ood_lib.py split   --regime A20 --write fit.npz       # band fit split; prints FIT_AOA=... for fit_band.py

Regimes
    A20 / A10   every train_real trajectory at that AoA            group = the regime itself
    RE          re_split.excluded()                                groups relo, rehi, rein (re_split.SETS)
    ID          model.VAL_TRAJ (the twins of record)               groups all8, a15_20

The band fit split of a regime ensemble is VAL_TRAJ minus any trajectory of the regime. The static band uses its
AoA 15-20 part (the shipping construction's --fit-aoa 15 20); the SIGBLEND refit uses all of it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

WT = os.environ.get("OOD_WT", "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/wt_ood")
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
CKPT = os.environ.get("OOD_CKPT", "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ckpt")
SCREEN = "/fs/scratch/PAS1187/qwu/tmp/realpde/screen/aoa"
UNET = os.path.join(WT, "Track_1", "unet")          # tracked code only; checkpoints of record come from the main tree
REPO_UNET = "/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet"
STATIC_AOA = (15, 20)                                # fit_band.py --fit-aoa default = the shipping construction

sys.path.insert(0, os.path.join(WT, "Track_1"))


def val_traj():
    """model.VAL_TRAJ, read from the worktree's source without importing torch."""
    import ast
    src = open(os.path.join(WT, "Track_1", "unet", "model.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VAL_TRAJ" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("VAL_TRAJ not found in model.py")


def all_real():
    return sorted(f[:-3] for f in os.listdir(DATA) if f.endswith(".h5"))


def aoa_of(n):
    return int(n.split("_")[1])


def re_of(n):
    return int(n.split("_")[0])


def regime(name):
    """-> (evaluation trajectories, {group: trajectories}, held-out-of-training trajectories)."""
    name = name.upper()
    if name.endswith("ID") and name != "ID":
        # the in-distribution read of a REGIME ensemble: VAL_TRAJ minus the regime (the ensemble trained on neither).
        # These trajectories are the regime band's own fit split, so any band-dependent number here is in sample.
        _, _, held = regime(name[:-2])
        ev = sorted(t for t in val_traj() if t not in set(held))
        return ev, {name: ev}, held
    if name in ("A20", "A10"):
        a = int(name[1:])
        ev = [n for n in all_real() if aoa_of(n) == a]
        return ev, {name: ev}, ev
    if name == "RE":
        import re_split
        groups = {k: re_split.names_for(v, DATA) for k, v in re_split.SETS.items()}
        ev = sorted(n for g in groups.values() for n in g)
        assert ev == re_split.excluded(DATA), "RE groups != re_split.excluded()"
        return ev, groups, ev
    if name == "ID":
        ev = sorted(val_traj())
        return ev, {"all8": ev, "a15_20": [n for n in ev if aoa_of(n) in STATIC_AOA]}, []
    raise SystemExit(f"unknown regime {name!r} (A20 | A10 | RE | ID)")


def layout(reg, lay):
    """Member paths for a named layout. F-FNO members first: the first fused group is family A (FAMW)."""
    reg = reg.upper(); r = reg.lower()
    d = os.path.join(CKPT, reg)
    ff = lambda ks: [os.path.join(d, f"{r}_ffnosim_s{k}.pth") for k in ks]          # noqa: E731
    vit = lambda ns: [os.path.join(d, f"{r}_{n}.pth") for n in ns]                  # noqa: E731
    if lay == "mirror":          # the package of record's shape: 4 F-FNO + sim, VITP s1, VITC s3
        return ff(range(1, 5)) + vit(["vitp_s1", "vitc_s3"])
    if lay == "six":             # arm T1: six F-FNO + sim members
        return ff(range(1, 7)) + vit(["vitp_s1", "vitc_s3"])
    if lay == "vitc2":           # arm T2: both ViTs co-trained with sim
        return ff(range(1, 5)) + vit(["vitc_s1", "vitc_s3"])
    if lay == "ffno4":           # the four F-FNO + sim members alone, no ViT (one invocation)
        return ff(range(1, 5))
    if lay == "ffno6":           # the six F-FNO + sim members alone
        return ff(range(1, 7))
    if lay == "six_vitc2":       # six F-FNO + sim, ViT-C s1 + ViT-C s3
        return ff(range(1, 7)) + vit(["vitc_s1", "vitc_s3"])
    if lay in ("t3", "t4"):      # arm T3 (sim dose 1.0) / T4 (Re-similarity augmentation): the arm's four F-FNO members
        root = os.path.join(os.path.dirname(CKPT), lay, "ckpt", reg)   # + the control's two ViTs
        stem = "ffnosim_lam10" if lay == "t3" else "ffnosim_resim"
        return [os.path.join(root, f"{r}_{stem}_s{k}.pth") for k in range(1, 5)] + vit(["vitp_s1", "vitc_s3"])
    if lay == "twins_ffno":      # the twins of record minus their ViTs (in distribution, F-FNO only)
        return [os.path.join(REPO_UNET, x + ".pth") for x in ["p2_ffno_l8_sim_s1", "p2_ffno_l8_sim_s2", "p2_ffno_l8_sim_s3"]]
    if lay == "standin":         # the sprint's stand-ins (screen/aoa), F-FNO family first, FNO w96 second
        if reg == "A20":
            n = ["aoa20_ffnosim_s1", "aoa20_ffnosim_s2", "aoa20_fno_s1"]
        elif reg == "A10":
            n = ["aoa10_ffnosim_s1", "aoa10_ffno_s1", "aoa10_fno_s1", "aoa10_nosim_s1"]
        else:
            raise SystemExit(f"no stand-in ensemble for {reg}")
        return [os.path.join(SCREEN, x + ".pth") for x in n]
    if lay == "twins":           # the five held-out twins of record (in distribution)
        return [os.path.join(REPO_UNET, x + ".pth") for x in
                ["p2_ffno_l8_sim_s1", "p2_ffno_l8_sim_s2", "p2_ffno_l8_sim_s3", "arch_vit_d384", "arch_vit_cmb3sim"]]
    raise SystemExit(f"unknown layout {lay!r} (mirror | six | vitc2 | ffno4 | ffno6 | six_vitc2 | t3 | t4 | standin | twins | twins_ffno)")


def check_members(reg, paths):
    """Refuse a member that saw any evaluation trajectory or the fit split, or is an all-data model."""
    import torch
    ev, _, held = regime(reg)
    must = set(held) | set(val_traj())
    rows, bad = [], []
    for p in paths:
        if not os.path.isfile(p):
            bad.append(f"MISSING {p}"); continue
        b = torch.load(p, map_location="cpu")
        c = b.get("config", {})
        # train2.py: excl = [] if all_data else (exclude if exclude is not None else VAL_TRAJ)
        ex = set() if c.get("all_data") else set(c["exclude"] if c.get("exclude") is not None else val_traj())
        miss = sorted(must - ex)
        sim = bool(c.get("sim"))
        row = (f"{os.path.basename(p):28s} arch {b.get('arch', 'fno'):6s} width {c.get('width')} L {c.get('n_layers')} "
               f"epochs {c.get('epochs')} seed {c.get('seed')} sim {'yes' if sim else 'no'} "
               f"lam_sim {c.get('lam_sim') if sim else '-'} sim_exclude_held {c.get('sim_exclude_held')} "
               f"all_data {c.get('all_data')} alpha_train {c.get('alpha_train')} excluded {len(ex)}")
        rows.append(row)
        if c.get("all_data"):
            bad.append(f"{p}: all_data model")
        if miss:
            bad.append(f"{p}: trained on {len(miss)} trajectories it must not have seen: {miss[:6]}")
        if sim and reg.upper() in ("A20", "A10", "RE") and not c.get("sim_exclude_held"):
            bad.append(f"{p}: sim co-training without --sim-exclude-held")
    return rows, bad


def seen_traj():
    """make_val.SEEN (the val_seen.npz trajectories, IN the training set of every model trained without --exclude)."""
    import ast
    src = open(os.path.join(WT, "Track_1", "make_val.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SEEN" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("SEEN not found in make_val.py")


def fit_split(reg, style="heldout"):
    """-> (fit trajectories [SIGBLEND], static-band trajectories, fit_band.py --fit-aoa list, --fit-on).

    style "heldout" (default, the planner's design): the static band's errors come from VAL_TRAJ minus the regime, AoA
    15-20 part -- OUT of sample for a regime ensemble. style "ship": the shipping construction's own split, val_seen AoA
    15-20 minus the regime -- trajectories the regime ensemble TRAINED on, as the twins trained on val_seen
    (fit_band.py --fit-on seen is its default). SIGBLEND uses VAL_TRAJ minus the regime in both styles, as shipped."""
    vt = val_traj()
    _, _, held = regime(reg)
    fit = [t for t in vt if t not in set(held)]
    pool, fit_on = (vt, "unseen") if style == "heldout" else (seen_traj(), "seen")
    if style not in ("heldout", "ship"):
        raise SystemExit(f"band style {style!r} (heldout | ship)")
    static = [t for t in pool if t not in set(held) and aoa_of(t) in STATIC_AOA]
    fit_aoa = sorted({aoa_of(t) for t in static})
    # fit_band.py --fit-on <split> --fit-aoa <fit_aoa> selects windows by AoA only: it must hit exactly `static`
    sel = sorted(t for t in pool if aoa_of(t) in fit_aoa)
    if sel != sorted(static):
        raise SystemExit(f"{reg}: --fit-on {fit_on} --fit-aoa {fit_aoa} would select {sel}, not the static fit split "
                         f"{static}; this regime needs a fit_band.py that filters by trajectory")
    if not static:
        raise SystemExit(f"{reg}: no AoA 15-20 trajectory left in the static fit split")
    return fit, static, fit_aoa, fit_on


def write_fit_npz(reg, out):
    fit = fit_split(reg)[0]
    d = np.load(os.path.join(WT, "Track_1", "val_unseen.npz"))
    tr = np.array([str(i).split(":")[0] for i in d["ids"]])
    if sorted(set(tr)) != sorted(val_traj()):
        raise SystemExit("val_unseen.npz does not hold exactly model.VAL_TRAJ")
    k = np.isin(tr, fit)
    np.savez(out, x=d["x"][k], y=d["y"][k], ids=d["ids"][k])
    return int(k.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regime", "members", "split"])
    ap.add_argument("--regime", required=True)
    ap.add_argument("--layout", default=None)
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--write", default=None)
    ap.add_argument("--style", default="heldout", choices=["heldout", "ship"])
    a = ap.parse_args()
    if a.cmd == "regime":
        ev, groups, held = regime(a.regime)
        print(f"# {a.regime}: {len(ev)} evaluation trajectories")
        for g, ts in groups.items():
            print(f"#   {g}: {len(ts)}  {' '.join(ts)}")
    elif a.cmd == "members":
        paths = a.members or layout(a.regime, a.layout)
        rows, bad = check_members(a.regime, paths)
        for r in rows:
            print("# " + r, file=sys.stderr)
        if bad:
            for b in bad:
                print("REFUSING: " + b, file=sys.stderr)
            sys.exit(1)
        print(" ".join(paths))
    elif a.cmd == "split":
        fit, static, fit_aoa, fit_on = fit_split(a.regime, a.style)
        print(f"# fit split (SIGBLEND, val_unseen): {len(fit)} trajectories {' '.join(fit)}", file=sys.stderr)
        print(f"# static band split (style {a.style}, val_{fit_on} AoA 15-20): {len(static)} {' '.join(static)}",
              file=sys.stderr)
        if a.write:
            n = write_fit_npz(a.regime, a.write)
            print(f"# wrote {a.write}: {n} windows", file=sys.stderr)
        print("FIT_AOA='" + " ".join(map(str, fit_aoa)) + "'; FIT_ON=" + fit_on)


if __name__ == "__main__":
    main()
