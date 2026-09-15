"""The internal leaderboard: what we tried, what we predicted, what the board actually said.

    python leaderboard/lb.py                 # everything, as text
    python leaderboard/lb.py --html          # also write ../leaderboard.html (OUTSIDE the git tree)

Two append-only ledgers, both JSONL, both edited by hand:

  `board.jsonl`   one record per SUBMISSION -- the package, the thesis, the PREDICTION made
                  before uploading, the actual subscores, and the local numbers that produced
                  the prediction (the shipped package's own errors, and the held-out-twin proxy
                  at the alpha that package deploys).
  `runs.jsonl`    one record per LOCAL experiment -- the arm, the instrument it was measured
                  on, the effect, whether it was tested AT ENSEMBLE SIZE, the verdict, and
                  which submission (if any) carried it to the board.

Why this exists.  Thirteen rounds of results lived in prose across eight reports, and the one
question that decides everything -- *does a local gain transfer?* -- could not be answered
without re-reading all of them.  It is answerable from these two files in one pass, and the
answer it gives (section 3 below) is not the comfortable one.

Appending after a board result:
  1. fill `actual` on that round's record in `board.jsonl`;
  2. add the arms it carried to `runs.jsonl` with `shipped` set to the round;
  3. rerun this; the calibration and the resolution limit refit themselves.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The rendered page lives OUTSIDE the git tree (user request, 2026-09-02): generated files are
# not source. Override with RPDE_LB_HTML if the workspace moves.
HTML_OUT = os.environ.get("RPDE_LB_HTML",
                          os.path.join(os.path.dirname(os.path.dirname(HERE)), "leaderboard.html"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Track_1"))

M = ["rel", "tke", "mvpe"]
SUB = ["rel", "tke", "mvpe", "time", "sps"]


# Every dated read of the best OTHER team's `final`: (date, track, final, note). 08-25 is np-user's
# Track-1 entry as read on 08-31; from 09-11 the two tracks have different leaders. Append a row
# whenever the board is read -- LEADER_T1 / LEADER_T2 and the curve's reference points follow it.
LEADER_READS = [
    ("2026-08-25", 1, 81.76136, "np-user, read 08-31"),
    ("2026-09-11", 1, 81.82371, ""),
    ("2026-09-11", 2, 81.99648, ""),
    ("2026-09-14", 1, 81.85874, ""),
    ("2026-09-14", 2, 81.99648, ""),
]


def leader_latest(track):
    """(date, final) of the latest leader read on a track."""
    r = max((x for x in LEADER_READS if x[1] == track), key=lambda x: x[0])
    return r[0], r[2]


LEADER_T1 = leader_latest(1)[1]
LEADER_T2 = leader_latest(2)[1]

# Backbone families in the FIXED slot order of the categorical palette (--f1 .. --f8 in the page
# CSS). A family keeps its slot on every chart and never takes a generated hue; an unrecognised
# family is drawn in --fx and named by its label. Slots 1-3 (FNO, ViT, U-Net) are the only three
# that stay distinguishable all-pairs in both themes (dataviz validator, 2026-09-14), so families
# from slot 4 on are also drawn as SQUARES, and every point carries a direct label.
FAMILIES = ["FNO", "ViT", "U-Net", "U-FNO", "F-FNO", "Swin U-Net", "SimVP", "ST-ViT"]
_FAMILY_ALIAS = {
    "fno": "FNO", "vit": "ViT", "vit2d": "ViT", "unet": "U-Net", "cnextunet": "U-Net",
    "convnextunet": "U-Net", "unetmod": "U-Net", "modernunet": "U-Net", "ufno": "U-FNO",
    "ffno": "F-FNO", "factorizedfno": "F-FNO", "swin": "Swin U-Net", "swinunet": "Swin U-Net",
    "swinv2": "Swin U-Net", "scot": "Swin U-Net", "simvp": "SimVP", "simvpv2": "SimVP",
    "gsta": "SimVP", "stvit": "ST-ViT", "spacetimevit": "ST-ViT", "dividedvit": "ST-ViT",
}


def family_key(name):
    """Canonical family of a free-text family or arch name, or None when it has no slot.

    Exact alias first; otherwise the longest alias the name STARTS with, so Phase-2 variants such as
    `swin_unet_b6`, `swin_unet_win16` or `ffno_w64` keep their family's colour.
    """
    k = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    if not k:
        return None
    if k in _FAMILY_ALIAS:
        return _FAMILY_ALIAS[k]
    for a in sorted(_FAMILY_ALIAS, key=len, reverse=True):
        if k.startswith(a):
            return _FAMILY_ALIAS[a]
    return None


def fam_var(name):
    """CSS colour of a family: its fixed slot, else --fx."""
    f = family_key(name)
    return f"var(--f{FAMILIES.index(f) + 1})" if f else "var(--fx)"


def fam_square(name):
    """True when a family is drawn as a square (slot 4 on, or unrecognised)."""
    f = family_key(name)
    return f is None or FAMILIES.index(f) >= 3


def families_of(backbone):
    """`"FNO w96 L8 ×4 + ViT ×2 fp16"` -> ["FNO", "ViT"]: the first word of each ` + ` part."""
    return [p.split()[0] for p in str(backbone or "").split(" + ") if p.split()]


def _num(v):
    """A finite float, or None -- screen records are partial while an arm is running."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def load_screen(name="screen.jsonl"):
    """The backbone-screen ledger, which forge rewrites in place (one record per arm).

    Returns (arms, state). Missing, empty and partly unreadable files all render: a line that does
    not parse is counted and skipped, and when an arm appears twice its LAST record wins.
    """
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return [], "missing"
    by, bad = {}, 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict) and r.get("arm"):
                by[str(r["arm"])] = r
            else:
                bad += 1
    if not by:
        return [], "empty" if not bad else f"{bad} unreadable"
    return list(by.values()), (f"{bad} unreadable" if bad else "")


ROLE_ZH = {"standalone": "单独", "partner": "作伙伴", "replace": "替换 ViT 对", "p2_l8": "P2-L8"}


def _extrapolated(r, role):
    """True when the ledger flags this role's projection as extrapolated (plan §4: K=1 figures)."""
    if r.get("extrapolated") is True or r.get(f"{role}_extrapolated") is True:
        return True
    return role in (r.get("extrapolated_roles") or [])


def best_proj(r):
    """(value, role) of an arm's best Track-1 projection -- standalone (X x4 fused, one invocation),
    partner (the package plus X), replacing the ViT pair, or for F-FNO the P2-L8 package -- never an
    extrapolated figure. An F-FNO row's `standalone` field is the K=1 single-model swap read (main 09-14),
    so for F-FNO it is shown relabelled and never plotted."""
    ffno = arm_family(r) == "F-FNO"
    c = []
    for k, role in (("proj_dfinal_t1_standalone", "standalone"), ("proj_dfinal_t1_partner", "partner"),
                    ("proj_dfinal_t1_replace_vits", "replace"), ("proj_dfinal_t1_p2_l8", "p2_l8")):
        if (role == "standalone" and ffno) or (role == "p2_l8" and not ffno):
            continue
        v = _num(r.get(k))
        if v is not None and not _extrapolated(r, role):
            c.append((v, role))
    return max(c) if c else (None, None)


def arm_family(r):
    """The ledger's `family`, or -- when forge writes a generic one such as "new" -- the family of
    `backbone`'s first word (cnext_unet, swin_unet, simvp, ffno, unet_mod, vit, fno)."""
    fam = r.get("family")
    if family_key(fam):
        return family_key(fam)
    for src in (r.get("arch"), r.get("backbone")):
        words = str(src or "").split()
        k = family_key(words[0]) if words else None
        if k:
            return k
    return str(fam) if fam else "?"


def screen_order(r):
    """Wave, then best projection (highest first), then arm name."""
    b, w = best_proj(r)[0], _num(r.get("wave"))
    return (w if w is not None else 99.0, -b if b is not None else float("inf"), str(r.get("arm")))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def err(score):
    """Invert score = 100/(1+0.5e)."""
    return 2.0 * (100.0 / score - 1.0)


# ---------------------------------------------------------------------------------------
def section_board(B, w):
    rows = []
    print("\n=== 1. Every submission: what we predicted, what the board said ===\n")
    print(f"  {'round':<6s} {'tr':>2s} {'date':<11s} {'thesis':<52s} {'pred':>8s} "
          f"{'actual':>9s} {'miss':>7s}")
    for r in B:
        a = r.get("actual")
        p = (r.get("pred") or {}).get("final")
        th = (r.get("thesis") or "")[:52]
        av = f"{a['final']:9.3f}" if a else f"{'pending':>9s}"
        pv = f"{p:8.3f}" if p else f"{'-':>8s}"
        ms = f"{a['final'] - p:+7.3f}" if (a and p) else f"{'':>7s}"
        print(f"  {r['round']:<6s} {r['track']:>2d} {r['date']:<11s} {th:<52s} {pv} {av} {ms}")
        if a and p:
            rows.append((r["round"], a["final"] - p))
    if rows:
        d = np.array([x[1] for x in rows])
        print(f"\n  prediction residual: mean {d.mean():+.3f}, rms {np.sqrt((d**2).mean()):.3f}, "
              f"n = {len(d)}")
        print("  The forward model (local errors -> board SUBSCORES) is in good shape; the")
        print("  PRICING model is what has been wrong. Round 13's subscores were predicted to")
        print("  rel_l2 0.02, and the whole +0.153 `final` miss was the weights.")

    print("\n--- per-subscore, where both are recorded ---")
    print(f"  {'round':<6s} " + "  ".join(f"{n:>16s}" for n in SUB))
    for r in B:
        a, p = r.get("actual"), r.get("pred") or {}
        if not a:
            continue
        cells = []
        for n in SUB:
            if n in p and n in a:
                cells.append(f"{p[n]:7.3f}/{a[n]:7.3f}")
            elif n in a:
                cells.append(f"{'':>7s} {a[n]:8.3f}")
            else:
                cells.append(" " * 16)
        print(f"  {r['round']:<6s} " + "  ".join(cells))
    print("  (predicted / actual)")


# ---------------------------------------------------------------------------------------
def section_progress(B, w):
    print("\n=== 2. Progress, with `time_score` held at the median ===\n")
    print("  Board `time_score` swings 2.8 points across the rounds and is only partly ours to")
    print("  control, so the raw ranking conflates model quality with eval-machine luck.")
    T = [r for r in B if r.get("actual") and r["track"] == 1]
    ref = float(np.median([r["actual"]["time"] for r in T]))
    print(f"  reference time_score {ref:.3f}\n")
    print(f"  {'round':<6s} {'date':<11s} {'final':>9s} {'@const t':>9s} {'rel-L2':>9s} "
          f"{'TKE':>9s} {'sps':>8s}")
    for r in T:
        a = r["actual"]
        norm = a["final"] + (ref - a["time"]) * w[3]
        print(f"  {r['round']:<6s} {r['date']:<11s} {a['final']:9.3f} {norm:9.3f} "
              f"{err(a['rel']):9.5f} {err(a['tke']):9.5f} {a['sps']:8.3f}")
    n = [r["actual"]["final"] + (ref - r["actual"]["time"]) * w[3] for r in T]
    print(f"\n  spread of the last four at constant latency: {max(n[-4:]) - min(n[-4:]):.3f}")


# ---------------------------------------------------------------------------------------
def calib(B):
    """Level ratios (all rounds, both tracks) and delta residuals (WITHIN each track).

    The two tracks share one test set and one scoring rule (ROADMAP 1.16), so a Track-2 round
    with a recorded proxy calibrates exactly as a Track-1 one does.  The DELTA chain is
    per-track, though: a round-to-round move only means something between two packages scored
    on the same board in sequence.  Transitions whose proxy is bit-identical to the previous
    round's are skipped -- R13 -> R14 changed nothing locally (the blend gate never fires on
    validation data), so it carries no information about transfer and would only pad the
    residual set with zeros.
    """
    R = [r for r in B if r.get("actual") and r.get("proxy")]
    rat = {m: [] for m in M}
    for r in R:
        for m in M:
            rat[m].append(err(r["actual"][m]) / r["proxy"][m])
    rows, res = [], []
    for tr in sorted({r["track"] for r in R}):
        Rt = [r for r in R if r["track"] == tr]
        for i in range(1, len(Rt)):
            a, b = Rt[i - 1], Rt[i]
            if all(b["proxy"][m] == a["proxy"][m] for m in M):
                continue
            for m in M:
                pd = (b["proxy"][m] / a["proxy"][m] - 1) * 100
                bd = (err(b["actual"][m]) / err(a["actual"][m]) - 1) * 100
                res.append(bd - pd)
                rows.append((a["round"], b["round"], m, pd, bd, bd - pd))
    rms = float(np.sqrt(np.mean(np.square(res)))) if res else float("nan")
    return R, rat, rows, res, rms


def section_calib(B):
    print("\n=== 3. Local -> board calibration, and the resolution limit ===\n")
    R, rat, rows, res, rms = calib(B)
    print("  The proxy is the held-out TWIN set on the AoA 15-20 slice of val_unseen, scored at")
    print("  the ALPHA the package ships. `r_x` is board error / proxy error.  BOTH TRACKS are")
    print("  listed: they share one test set and one scoring rule (ROADMAP 1.16).\n")
    print(f"  {'round':<6s} {'tr':>2s} " + "  ".join(f"{'proxy ' + m:>10s}" for m in M)
          + "  " + "  ".join(f"{'board ' + m:>10s}" for m in M)
          + "  " + "  ".join(f"{'r_' + m:>6s}" for m in M))
    for r in R:
        p, a = r["proxy"], r["actual"]
        b = {m: err(a[m]) for m in M}
        print(f"  {r['round']:<6s} {r['track']:>2d} " + "  ".join(f"{p[m]:10.5f}" for m in M)
              + "  " + "  ".join(f"{b[m]:10.5f}" for m in M)
              + "  " + "  ".join(f"{b[m]/p[m]:6.3f}" for m in M))
    print("\n  LEVELS are well calibrated:")
    for m in M:
        v = np.array(rat[m])
        print(f"    r_{m:<5s} mean {v.mean():.3f}   spread {v.min():.3f}-{v.max():.3f} "
              f"(+-{(v.max()-v.min())/2/v.mean()*100:.1f} %)")

    print("\n  DELTAS are not. Round-to-round WITHIN A TRACK, how much of the board's move did")
    print("  the proxy explain? (`residual` = board move - proxy move, in percent of the error.)\n")
    print(f"  {'transition':<16s} {'metric':<6s} {'proxy d%':>9s} {'board d%':>9s} {'residual':>9s}")
    for a, b, m, pd, bd, rr in rows:
        print(f"  {a + ' -> ' + b:<16s} {m:<6s} {pd:+9.2f} {bd:+9.2f} {rr:+9.2f}")
    res = np.array(res)
    print(f"\n  residual rms {rms:.2f} %, range {res.min():+.2f} .. {res.max():+.2f} %")
    print(f"\n  >>> THE RESOLUTION LIMIT. A local delta smaller than ~{rms:.0f} % of the error"
          " does not")
    print("  >>> survive the trip to the board, and MUST NOT be used to choose a package.")
    print("  >>> Every measured improvement since 2026-08-08 has been smaller than this, and so")
    print("  >>> was dropout's (-2.6 % on the interior Re bands, +0.66 % on the AoA proxy) --")
    print("  >>> which is why T2R2's +0.200 is 'the closest instrument was right about the")
    print("  >>> sign', not 'the instrument is confirmed'.")
    print("  >>> R14->R15 and T2R1->T2R2 are the SAME package change measured on the two")
    print("  >>> boards (blend8 -> dropout8), and their residuals agree to 0.7 pp on all three")
    print("  >>> metrics -- the residual is a property of the proxy, not of a board draw.")
    print("  >>> `mvpe` inverts sign between the proxy and the board in BOTH tracks (R12->R13:")
    print("  >>> proxy +2.34 %, board -3.17 %; T2R1->T2R2: proxy +1.25 %, board -7.79 %), which")
    print("  >>> is why the proxy cannot be trusted on it. NOTE mvpe is worth 0.094 under the")
    print("  >>> FREE weights -- 2.4x the constrained fit -- so it is no longer safe to IGNORE.")

    print("\n  Other calibrated constants, and where they came from:")
    print("    SPS element-tail stretch f   1.360 (T1) / 1.350 (T2)   calib_band13.py, round 13")
    print("      -- two boards returned two SPS values for one band, so this is measured twice")
    print("    w_time = 0.0956   R13 vs R14: identical error subscores, `time` alone moved")
    print("      -- the only weight in the family read off a single-subscore A/B, and it PICKS")
    print("         THE FIT: the free eighteen-round fit says 0.0967, the sum(w)=1 fit that")
    print("         rounds 9-16 used said 0.085.  Round 17 dropped the constraint.")
    print("    board latency, MEASURED as deltas (board_rounds.py): +1.151 / +0.866 board ms per")
    print("      extra invocation (R23); FLOPs 1.75 / 0.60 board ms per 4060-eq local ms (R24);")
    print("      a PRECISION cut ~0.65 on Track 1 (fp16, R25: 37 % of the local cut kept).")
    print("      The old level model (2.616 + 0.435*inv + 0.803*local) is withdrawn.")


# ---------------------------------------------------------------------------------------
def section_runs(Rn):
    print("\n=== 4. Local experiment ledger ===\n")
    order = {"SHIPPED": 0, "OPEN": 1, "MARGINAL": 2, "CLOSED": 3, "REJECTED": 4}
    by = {}
    for r in Rn:
        by.setdefault(r["verdict"], []).append(r)
    for v in sorted(by, key=lambda k: order.get(k, 9)):
        print(f"  --- {v} ({len(by[v])}) ---")
        for r in sorted(by[v], key=lambda x: x["date"]):
            ens = "ens" if r.get("ensemble_tested") else "  1"    # measured at ensemble size?
            sh = r.get("shipped") or "-"
            print(f"    {r['date']}  r{r['round']:<2d} [{ens}] {r['arm'][:34]:<34s} "
                  f"{r['direction'][:22]:<22s} -> {sh:<5s} {r['effect'][:62]}")
        print()
    n_ens = sum(1 for r in Rn if r.get("ensemble_tested"))
    print(f"  {n_ens}/{len(Rn)} arms were measured AT THE ENSEMBLE SIZE THAT SHIPS.")
    print("  That column is the single most predictive field here: `ozdo` was +3.84 sigma per")
    print("  model and -0.012 at four members, and `--var-mlp`'s 3.9 % fixed-field TKE cut")
    print("  became +0.08. An arm that has not been ensembled has not been measured.")


# ---------------------------------------------------------------------------------------
def section_screen(S, state):
    print("\n=== 5. Backbone screen (leaderboard/screen.jsonl, local proxy projections) ===\n")
    if not S:
        print(f"  no arms yet ({state or 'no records'})")
        return
    if state:
        print(f"  note: {state} line(s) skipped")
    print(f"  {'arm':<16s} {'family':<11s} {'status':<8s} {'gpu_h':>6s} {'xFNO':>6s} "
          f"{'T1 solo':>8s} {'T1 partner':>10s} {'T2 partner':>10s}  verdict")

    def f(r, k, fmt, w):
        v = _num(r.get(k))
        return (fmt % v if v is not None else "-").rjust(w)
    for r in sorted(S, key=screen_order):
        print(f"  {str(r.get('arm'))[:16]:<16s} {str(r.get('family') or '-')[:11]:<11s} "
              f"{str(r.get('status') or '-')[:8]:<8s} {f(r, 'gpu_h', '%.1f', 6)} "
              f"{f(r, 'ms_ratio_fno', '%.2f', 6)} {f(r, 'proj_dfinal_t1_standalone', '%+.3f', 8)} "
              f"{f(r, 'proj_dfinal_t1_partner', '%+.3f', 10)} "
              f"{f(r, 'proj_dfinal_t2_partner', '%+.3f', 10)}  {r.get('verdict') or '-'}")
    print("  Projections are local proxy numbers: a sign, not a size.")


# ---------------------------------------------------------------------------------------
# Bilingual rendering.  Every AUTHORED string below is declared once, both languages side by
# side, through `t(en, zh)`; every DATA string (the ledgers are written in English) goes
# through `tdata`, which falls back to the English when the record carries no `_zh` field.
# Both languages are emitted into the DOM and CSS hides one, so the page is complete at rest
# with no JavaScript -- the toggle only flips `data-lang` on <html>.
from html import escape as _esc


def t(en, zh):
    """An authored string, in both languages."""
    return f'<span class="en">{en}</span><span class="zh">{zh}</span>'


def tdata(en, zh=None):
    """A ledger string. Escaped; falls back to English when the `_zh` field is missing."""
    en = _esc(str(en if en is not None else ""))
    if not zh:
        return en
    return f'<span class="en">{en}</span><span class="zh">{_esc(str(zh))}</span>'


def tsvg(x, y, en, zh, attrs=""):
    """Two SVG <text> nodes at one anchor; CSS shows exactly one."""
    return (f'<text class="en" x="{x}" y="{y}" {attrs}>{en}</text>'
            f'<text class="zh" x="{x}" y="{y}" {attrs}>{zh}</text>')


NAV = [("top", "Read this first", "先读这里"),
       ("curve", "Optimization curve", "优化曲线"),
       ("screen", "Backbone screen", "骨干网络筛选"),
       ("submissions", "Submissions", "提交记录"),
       ("progress", "Progress at constant latency", "恒定延迟下的进展"),
       ("calibration", "Local &rarr; board correction", "本地 &rarr; board 校正"),
       ("ledger", "Experiment ledger", "实验台账")]

VERDICT_ZH = {"SHIPPED": "已上线", "OPEN": "未决",
              "MARGINAL": "边缘", "CLOSED": "已封闭",
              "REJECTED": "已否决"}

HTML_HEAD = """<title>RealPDE Calibration Log</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;600&display=swap">
<style>
/* Palette from the subject: PIV velocity fields are rendered on a blue-white-red diverging
   map, so cool/warm here are the domain's own colours, not decoration. Neutrals carry a blue
   bias so they read as chosen. */
:root{
  --ground:#f7f8fa; --panel:#ffffff; --sunk:#f1f3f7;
  --ink:#14181f; --dim:#6a7280; --faint:#9aa1ac; --line:#dfe3ea; --rule:#c8cfda;
  --probe:#2b5fa8; --cool:#1f6f8b; --warm:#b3402c; --hold:#8a6d1f;
  --coolw:#1f6f8b22; --warmw:#b3402c22;
  /* backbone families, fixed slot order (lb.FAMILIES); validated on these panels 2026-09-14 */
  --f1:#2a78d6; --f2:#eb6834; --f3:#1baf7a; --f4:#eda100;
  --f5:#e87ba4; --f6:#008300; --f7:#4a3aa7; --f8:#e34948; --fx:#9aa1ac;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#12151b; --panel:#191d24; --sunk:#141920;
  --ink:#e6e9ee; --dim:#98a1ae; --faint:#6b7481; --line:#262c35; --rule:#39414d;
  --probe:#7aa8e0; --cool:#5fb6c9; --warm:#ef8a72; --hold:#d4a94a;
  --coolw:#5fb6c92e; --warmw:#ef8a722e;
  --f1:#3987e5; --f2:#d95926; --f3:#199e70; --f4:#c98500;
  --f5:#d55181; --f6:#008300; --f7:#9085e9; --f8:#e66767; --fx:#6b7481;
}}
:root[data-theme="dark"]{
  --ground:#12151b; --panel:#191d24; --sunk:#141920;
  --ink:#e6e9ee; --dim:#98a1ae; --faint:#6b7481; --line:#262c35; --rule:#39414d;
  --probe:#7aa8e0; --cool:#5fb6c9; --warm:#ef8a72; --hold:#d4a94a;
  --coolw:#5fb6c92e; --warmw:#ef8a722e;
  --f1:#3987e5; --f2:#d95926; --f3:#199e70; --f4:#c98500;
  --f5:#d55181; --f6:#008300; --f7:#9085e9; --f8:#e66767; --fx:#6b7481;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;padding:0;
  font:400 14px/1.6 "IBM Plex Sans","Noto Sans SC","PingFang SC","Microsoft YaHei",
       "Helvetica Neue",Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
/* the language switch: both languages are in the DOM, CSS hides one. No JS -> English. */
html[data-lang="zh"] .en,html:not([data-lang="zh"]) .zh{display:none}
/* layout: fixed left rail, one content column */
.app{display:grid;grid-template-columns:220px 1fr;align-items:start}
.side{position:sticky;top:0;height:100vh;overflow-y:auto;z-index:5;
  background:var(--panel);border-right:1px solid var(--line);
  padding:26px 14px 18px;display:flex;flex-direction:column;gap:16px}
.brand{display:block}
.brand .nm{display:block;font-family:"IBM Plex Sans Condensed","IBM Plex Sans",Arial,sans-serif;
  font-weight:700;font-size:16px;line-height:1.2;letter-spacing:-.005em}
.brand .zhnm{display:block;color:var(--dim);font-size:12.5px;margin-top:2px}
.langsw{display:flex;border:1px solid var(--line);width:max-content}
.langsw button{appearance:none;background:var(--panel);color:var(--dim);border:0;cursor:pointer;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.06em;
  padding:4px 11px;line-height:1.5}
.langsw button+button{border-left:1px solid var(--line)}
.langsw button[aria-pressed="true"]{background:var(--sunk);color:var(--ink);font-weight:600}
.langsw button:hover{color:var(--ink)}
.langsw button:focus-visible,.side nav a:focus-visible,a:focus-visible{
  outline:2px solid var(--probe);outline-offset:2px}
.side nav{flex:1 1 auto}
.side nav ul{list-style:none;margin:0;padding:0}
.side nav a{display:block;padding:6px 9px;font-size:12.5px;line-height:1.35;
  color:var(--dim);text-decoration:none;border-left:2px solid transparent}
.side nav a:hover{color:var(--ink);background:var(--sunk)}
.side nav a.on{color:var(--ink);background:var(--sunk);border-left-color:var(--probe);
  font-weight:600}
.side .stamp{margin:0}
main{padding:34px 20px 80px;min-width:0}
.wrap{max-width:1200px;margin:0 auto}
h1{font-family:"IBM Plex Sans Condensed","IBM Plex Sans","Noto Sans SC",Arial,sans-serif;
  font-weight:700;font-size:30px;line-height:1.1;margin:0 0 6px;letter-spacing:-.005em;
  text-wrap:balance}
.lede{color:var(--dim);margin:0 0 22px;max-width:66ch;font-size:14.5px}
.stamp{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--faint);
  letter-spacing:.06em;text-transform:uppercase;margin:0 0 26px}
.stamp .zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace;
  letter-spacing:.02em;text-transform:none}
h2{font-family:"IBM Plex Sans Condensed","IBM Plex Sans","Noto Sans SC",Arial,sans-serif;
  font-weight:600;font-size:19px;margin:0;letter-spacing:.005em}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;font-weight:500;
  color:var(--faint);letter-spacing:.11em;text-transform:uppercase}
.eyebrow .zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace;
  letter-spacing:.04em;text-transform:none}
section{background:var(--panel);border:1px solid var(--line);border-top:2px solid var(--rule);
  padding:20px 22px 22px;margin:20px 0 0;scroll-margin-top:14px}
#top{scroll-margin-top:14px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  margin:0 0 14px;flex-wrap:wrap}
.head p{color:var(--dim);margin:6px 0 0;font-size:13px;max-width:70ch;flex-basis:100%}
/* readout strip */
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);margin:0 0 4px}
.cell{background:var(--panel);padding:13px 15px}
.cell .k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;color:var(--faint);
  letter-spacing:.1em;text-transform:uppercase}
.cell .k .zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace;
  letter-spacing:.03em;text-transform:none}
.cell .v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;
  font-size:25px;font-weight:500;line-height:1.15;margin-top:5px;letter-spacing:-.02em}
.cell .n{color:var(--dim);font-size:11.5px;margin-top:3px}
/* the thesis callout */
.thesis{border:1px solid var(--rule);background:var(--sunk);padding:15px 18px;margin:20px 0 0;
  font-size:14px;max-width:none}
.thesis .eyebrow{display:block;margin-bottom:6px}
.thesis b{font-weight:600}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
th{font-family:"IBM Plex Sans","Noto Sans SC","Helvetica Neue",Arial,sans-serif;
  color:var(--faint);font-weight:600;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;border-bottom:1px solid var(--rule)}
th .zh{letter-spacing:.02em;text-transform:none}
td.l,th.l{text-align:left}
td.t{font-family:"IBM Plex Sans","Noto Sans SC","Helvetica Neue",Arial,sans-serif;
  white-space:normal;min-width:16ch}
tbody tr:hover td{background:var(--sunk)}
.g{color:var(--cool)}.b{color:var(--warm)}.w{color:var(--hold)}.d{color:var(--faint)}
.chip{display:inline-block;padding:1px 8px;border:1px solid currentColor;border-radius:2px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;font-weight:500;
  letter-spacing:.05em}
.chip .zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace;letter-spacing:0}
.SHIPPED{color:var(--cool)}.OPEN{color:var(--probe)}
.MARGINAL,.CLOSED{color:var(--hold)}.REJECTED{color:var(--warm)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.92em;
  background:var(--sunk);padding:1px 5px;border:1px solid var(--line)}
svg{display:block;max-width:100%}
svg text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;fill:var(--dim)}
svg text.zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace}
svg text.lab{font-size:10.5px;fill:var(--faint);letter-spacing:.06em}
footer{color:var(--faint);font-size:11.5px;margin-top:34px;padding-top:14px;
  border-top:1px solid var(--line);font-family:"IBM Plex Mono",ui-monospace,monospace;
  line-height:1.8}
footer .zh{font-family:"IBM Plex Mono","Noto Sans SC",ui-monospace,monospace}
/* charts added 2026-09-14: legend, tooltip, empty state, screen table */
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:2px 0 12px;font-size:12px;color:var(--dim)}
.legend>span{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.legend svg{display:inline-block;flex:none}
svg.chart{min-width:900px}
.pt{outline:none}
.pt .fr{opacity:0}
.pt:hover .fr,.pt:focus .fr{opacity:1}
.tip{position:fixed;z-index:30;pointer-events:none;background:var(--panel);color:var(--ink);
  border:1px solid var(--rule);box-shadow:0 4px 14px rgba(20,24,31,.16);padding:7px 10px;
  font:12px/1.5 "IBM Plex Mono","Noto Sans SC",ui-monospace,monospace;max-width:340px}
.tip div:first-child{font-weight:600}
.tip div+div{color:var(--dim)}
.tip[hidden]{display:none}
.empty{border:1px dashed var(--rule);background:var(--sunk);padding:14px 16px;color:var(--dim);
  font-size:13px;margin:0 0 4px}
.cap{color:var(--dim);font-size:12.5px;margin:10px 0 0;max-width:86ch}
th.grp{text-align:center;border-bottom:1px solid var(--line);color:var(--dim)}
.sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
.sw.sq{border-radius:1px}
.v-advance{color:var(--cool)}.v-replicate{color:var(--probe)}.v-pending{color:var(--faint)}
.v-drop{color:var(--warm)}.st-failed{color:var(--warm)}.st-running{color:var(--probe)}
@media (prefers-reduced-motion:no-preference){
  tbody tr td{transition:background .12s ease}}
/* narrow: the rail becomes a top bar with a horizontally scrollable nav */
@media (max-width:880px){
  .app{grid-template-columns:1fr}
  .side{position:static;height:auto;overflow:visible;flex-direction:row;align-items:center;
    gap:12px;padding:10px 14px;border-right:0;border-bottom:1px solid var(--line)}
  .brand .zhnm{display:none}
  .side .stamp{display:none}
  .side nav{overflow-x:auto;min-width:0}
  .side nav ul{display:flex;gap:2px}
  .side nav a{white-space:nowrap;border-left:0;border-bottom:2px solid transparent;
    padding:5px 8px}
  .side nav a.on{border-left:0;border-bottom-color:var(--probe)}
}
</style>
<script>
/* Sets the language before the page paints: stored choice, else the browser's, else English.
   Both languages are always present in the DOM; this only flips an attribute. */
(function(){
  var d=document.documentElement,v=null;
  try{v=window.localStorage.getItem("rpde-lang")}catch(e){}
  if(v!=="zh"&&v!=="en"){
    v=((navigator.language||navigator.userLanguage||"").toLowerCase().indexOf("zh")===0)?"zh":"en";
  }
  d.setAttribute("data-lang",v);
  function mark(){
    var bs=document.querySelectorAll("[data-setlang]"),i;
    for(i=0;i<bs.length;i++){
      bs[i].setAttribute("aria-pressed",bs[i].getAttribute("data-setlang")===
        d.getAttribute("data-lang")?"true":"false");
    }
  }
  document.addEventListener("click",function(e){
    var n=e.target;
    while(n&&n!==document&&!(n.getAttribute&&n.getAttribute("data-setlang")))n=n.parentNode;
    if(!n||n===document)return;
    var s=n.getAttribute("data-setlang");
    d.setAttribute("data-lang",s);
    try{window.localStorage.setItem("rpde-lang",s)}catch(err){}
    mark();
  });
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",mark);
  else mark();
})();
</script>
"""

NAV_SCRIPT = """<script>
/* Active-section highlight. No smooth scrolling, no animation -- the anchors just jump. */
(function(){
  if(!("IntersectionObserver" in window))return;
  var ids=%s,links={},vis={},i;
  for(i=0;i<ids.length;i++){
    var a=document.querySelector('.side nav a[href="#'+ids[i]+'"]');
    if(a)links[ids[i]]=a;
  }
  var io=new IntersectionObserver(function(es){
    for(var k=0;k<es.length;k++)vis[es[k].target.id]=es[k].isIntersecting?es[k].intersectionRatio:0;
    var best=null,bv=0,j;
    for(j=0;j<ids.length;j++){if((vis[ids[j]]||0)>bv){bv=vis[ids[j]];best=ids[j];}}
    for(j=0;j<ids.length;j++){
      if(links[ids[j]])links[ids[j]].className=(ids[j]===best)?"on":"";
    }
  },{rootMargin:"-8%% 0px -55%% 0px",threshold:[0,0.05,0.2,0.5,1]});
  for(i=0;i<ids.length;i++){
    var el=document.getElementById(ids[i]);
    if(el)io.observe(el);
  }
})();
</script>"""

TIP_SCRIPT = """<script>
/* Chart tooltips. An element carrying data-tip-en holds its readout as newline-separated lines
   (data-tip-zh when the page is in Chinese); the lines go in through textContent. Every value a
   tooltip shows is also in a table on this page -- the tooltip only saves the lookup. */
(function(){
  var tip=document.createElement("div");tip.className="tip";tip.hidden=true;
  document.body.appendChild(tip);
  function target(n){
    while(n&&n!==document){if(n.getAttribute&&n.getAttribute("data-tip-en"))return n;n=n.parentNode;}
    return null;
  }
  function fill(el){
    var zh=document.documentElement.getAttribute("data-lang")==="zh";
    var s=(zh&&el.getAttribute("data-tip-zh"))||el.getAttribute("data-tip-en")||"";
    while(tip.firstChild)tip.removeChild(tip.firstChild);
    var ls=s.split("\\n"),i;
    for(i=0;i<ls.length;i++){var d=document.createElement("div");d.textContent=ls[i];tip.appendChild(d);}
    tip.hidden=false;
  }
  function place(x,y){
    var w=tip.offsetWidth,h=tip.offsetHeight,vw=window.innerWidth,vh=window.innerHeight;
    var l=x+14,t=y+14;
    if(l+w>vw-8)l=Math.max(8,x-w-14);
    if(t+h>vh-8)t=Math.max(8,y-h-14);
    tip.style.left=l+"px";tip.style.top=t+"px";
  }
  document.addEventListener("pointerover",function(e){var t=target(e.target);if(t){fill(t);place(e.clientX,e.clientY);}});
  document.addEventListener("pointermove",function(e){if(!tip.hidden&&target(e.target))place(e.clientX,e.clientY);});
  document.addEventListener("pointerout",function(e){var t=target(e.target);
    if(t&&!(e.relatedTarget&&t.contains(e.relatedTarget)))tip.hidden=true;});
  document.addEventListener("focusin",function(e){var t=target(e.target);
    if(t){fill(t);var r=t.getBoundingClientRect();place(r.right,r.bottom);}});
  document.addEventListener("focusout",function(){tip.hidden=true;});
})();
</script>"""


def _sidebar(B, Rn, S=()):
    # The fit pools this account's rounds AND the two collaborator rounds (X1, X2), which are
    # in board_rounds.py but not in board.jsonl -- they have subscores and no thesis of ours.
    # Count what the fit actually uses, not what this page can narrate.
    try:
        import sys as _sys
        _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Track_1"))
        from board_rounds import ROUNDS as _R1, ROUNDS_T2 as _R2
        n_scored = len(_R1) + len(_R2)
    except Exception:
        n_scored = len([r for r in B if r.get("actual")])
    o = ['<aside class="side">',
         '<div class="brand"><span class="nm">RealPDE Calibration Log</span>'
         '<span class="zhnm">RealPDE 校准日志</span></div>',
         '<div class="langsw" role="group" aria-label="Language / 语言">'
         '<button type="button" data-setlang="en" aria-pressed="true">EN</button>'
         '<button type="button" data-setlang="zh" aria-pressed="false">中文</button>'
         '</div>',
         '<nav aria-label="Sections"><ul>']
    for i, (sid, en, zh) in enumerate(NAV):
        cls = ' class="on"' if i == 0 else ""
        o.append(f'<li><a href="#{sid}"{cls}>{t(en, zh)}</a></li>')
    o.append("</ul></nav>")
    o.append('<p class="stamp">'
             + t(f"generated by leaderboard/lb.py<br>{len(B)} submissions &middot; {len(Rn)} "
                 f"local arms<br>{len(S)} screen arms<br>weights refit on {n_scored} scored rounds",
                 f"由 leaderboard/lb.py 生成<br>{len(B)} 次提交 "
                 f"&middot; {len(Rn)} 条本地实验<br>{len(S)} 个筛选实验臂<br>"
                 f"权重基于 {n_scored} 个已评分轮"
                 f"次拟合")
             + "</p></aside>")
    return "".join(o)


def _progress_svg(T, norm, ref):
    """Constant-latency board score over time, with the leader as a reference line."""
    W, H, L, Rp, Tp, Bp = 980, 210, 52, 118, 20, 34
    lo, hi = min(norm) - 0.12, 81.90
    x = lambda i: L + i * (W - L - Rp) / max(len(T) - 1, 1)
    y = lambda v: Tp + (hi - v) * (H - Tp - Bp) / (hi - lo)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="board score at constant latency">']
    for gv in (80.0, 80.5, 81.0, 81.5):
        o.append(f'<line x1="{L}" y1="{y(gv):.1f}" x2="{W-Rp}" y2="{y(gv):.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>'
                 f'<text x="{L-8}" y="{y(gv)+3.5:.1f}" text-anchor="end">{gv:.1f}</text>')
    o.append(f'<line x1="{L}" y1="{y(LEADER_T1):.1f}" x2="{W-Rp}" y2="{y(LEADER_T1):.1f}" '
             f'stroke="var(--warm)" stroke-width="1.2" stroke-dasharray="5 4"/>')
    o.append(tsvg(f"{W-Rp+8}", f"{y(LEADER_T1)+3.5:.1f}", f"leader {LEADER_T1:.3f}",
                  f"领先者 {LEADER_T1:.3f}", 'fill="var(--warm)"'))
    # the flat span
    i0 = len(T) - 4
    o.append(f'<rect x="{x(i0)-11:.1f}" y="{Tp}" width="{x(len(T)-1)-x(i0)+22:.1f}" '
             f'height="{H-Tp-Bp}" fill="var(--warmw)"/>')
    sp = max(norm[i0:]) - min(norm[i0:])
    o.append(tsvg(f"{(x(i0)+x(len(T)-1))/2:.1f}", f"{Tp+13}", f"span {sp:.3f}",
                  f"跨度 {sp:.3f}", 'text-anchor="middle" class="lab"'))
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(norm))
    o.append(f'<polyline points="{pts}" fill="none" stroke="var(--probe)" stroke-width="2" '
             f'stroke-linejoin="round"/>')
    for i, (r, v) in enumerate(zip(T, norm)):
        last = i == len(T) - 1
        o.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="{4.5 if last else 3}" '
                 f'fill="{"var(--probe)" if last else "var(--panel)"}" stroke="var(--probe)" '
                 f'stroke-width="2"/>')
        o.append(f'<text x="{x(i):.1f}" y="{H-14}" text-anchor="middle">{r["round"]}</text>')
        o.append(f'<text x="{x(i):.1f}" y="{H-3}" text-anchor="middle" '
                 f'fill="var(--faint)">{r["date"][5:]}</text>')
    o.append(f'<text x="{x(len(T)-1):.1f}" y="{y(norm[-1])-11:.1f}" text-anchor="end" '
             f'fill="var(--ink)" font-weight="600">{norm[-1]:.3f}</text>')
    o.append("</svg>")
    return "".join(o)


CURVE_X0, CURVE_X1 = "2026-08-03", "2026-09-15"     # a Monday, and the day after the last read
CURVE_Y0, CURVE_Y1 = 79.6, 82.1


def _date_num(d):
    """ISO date -> days since 2026-08-01."""
    import datetime as _dt
    return (_dt.date.fromisoformat(str(d)[:10]) - _dt.date(2026, 8, 1)).days


def _day_label(n):
    import datetime as _dt
    return (_dt.date(2026, 8, 1) + _dt.timedelta(days=n)).strftime("%m-%d")


def _attr(s):
    """An attribute value: escaped, with newlines kept as character references."""
    return _esc(str(s), quote=True).replace("\n", "&#10;")


def _collab_rounds():
    """The collaborator's scored rounds (X1, X2): in board_rounds.py, never staged here."""
    try:
        from board_rounds import ROUNDS as _R1
    except Exception:
        return []
    return [dict(round=r[0].split()[0], track=1, date=r[1], actual=dict(final=r[8]), collab=True)
            for r in _R1 if r[0].strip().startswith("X")]


def _marker(x, y, fams, rv=4.0, hollow=False, faded=False):
    """A family-coloured mark with a 2px surface ring: a disc, a split disc for a two-family
    package, or a square for a family outside the first three slots. `rv` is the visible radius."""
    op = ' opacity=".45"' if faded else ""
    if not fams:
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rv - 0.5:.1f}" fill="var(--panel)" '
                f'stroke="var(--fx)" stroke-width="1.5"{op}/>')
    c = fam_var(fams[0])
    if len(fams) == 1 or hollow:
        if fam_square(fams[0]):
            s = rv if hollow else rv + 1
            return (f'<rect x="{x - s:.1f}" y="{y - s:.1f}" width="{2 * s:.1f}" height="{2 * s:.1f}" '
                    f'fill="{"var(--panel)" if hollow else c}" '
                    f'stroke="{c if hollow else "var(--panel)"}" stroke-width="2"{op}/>')
        if hollow:
            return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rv:.1f}" fill="var(--panel)" '
                    f'stroke="{c}" stroke-width="2"{op}/>')
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rv + 1:.1f}" fill="{c}" '
                f'stroke="var(--panel)" stroke-width="2"{op}/>')
    c2 = fam_var(fams[1])
    return (f'<g{op}><path d="M{x:.1f},{y - rv:.1f}A{rv},{rv} 0 0 0 {x:.1f},{y + rv:.1f}Z" '
            f'fill="{c}"/><path d="M{x:.1f},{y - rv:.1f}A{rv},{rv} 0 0 1 {x:.1f},{y + rv:.1f}Z" '
            f'fill="{c2}"/><circle cx="{x:.1f}" cy="{y:.1f}" r="{rv + 1:.1f}" fill="none" '
            f'stroke="var(--panel)" stroke-width="2"/></g>')


def _hit(x, y, tip_en, tip_zh, body, r=12):
    """A focusable, hoverable point: a 24px transparent hit area, a focus ring, and the mark."""
    return (f'<g class="pt" tabindex="0" role="img" aria-label="{_attr(tip_en.replace(chr(10), ". "))}" '
            f'data-tip-en="{_attr(tip_en)}" data-tip-zh="{_attr(tip_zh)}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="transparent"/>'
            f'<circle class="fr" cx="{x:.1f}" cy="{y:.1f}" r="9" fill="none" stroke="var(--ink)" '
            f'stroke-width="1"/>{body}</g>')


def _swatch_svg(fam):
    f = (f'<rect x="2" y="2" width="8" height="8" fill="{fam_var(fam)}"/>' if fam_square(fam)
         else f'<circle cx="6" cy="6" r="4" fill="{fam_var(fam)}"/>')
    return f'<svg width="12" height="12" aria-hidden="true">{f}</svg>'


def _fam_legend(fams):
    """One legend entry per family present, in slot order; unrecognised names last, in --fx."""
    known = [f for f in FAMILIES if f in {family_key(x) for x in fams}]
    other = sorted({str(x) for x in fams if x and not family_key(x)})
    return ([f'<span>{_swatch_svg(f)}{_esc(f)}</span>' for f in known]
            + [f'<span>{_swatch_svg(x)}{_esc(x)}</span>' for x in other])


def _curve_svg(B, collab):
    """Both tracks as small multiples on one date axis and one score axis. Marks are families,
    the step line is our running best, diamonds are leader reads. Labels hang under each panel's
    axis and END at the point's tick with the round id, so the ids line up along the axis."""
    W, L, PW, PH = 1000, 58, 792, 230
    d0, d1 = _date_num(CURVE_X0), _date_num(CURVE_X1)

    def X(dn):
        return L + (dn - d0) * PW / (d1 - d0)

    o, y0 = [], 0
    for track, name in ((1, "Track 1 &middot; Sim2Real"), (2, "Track 2 &middot; LTTTA")):
        rows = [dict(r, _seq=i) for i, r in enumerate(B) if r.get("actual") and r["track"] == track]
        rows += [dict(c, _seq=-1) for c in collab if c["track"] == track]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["date"], r["_seq"]))
        for r in rows:
            r["_lab"] = r.get("backbone") or "collaborator"
        LH = 6.0 * max(len(r["_lab"]) + 1 + len(r["round"]) for r in rows) + 10
        top, bot = y0 + 26, y0 + 26 + PH

        def Y(v, top=top):
            return top + (CURVE_Y1 - v) * PH / (CURVE_Y1 - CURVE_Y0)

        o.append(f'<text x="{L}" y="{y0 + 13}" style="fill:var(--ink);font-weight:600;'
                 f'font-size:11px">{name}</text>')
        for gv in (80.0, 80.5, 81.0, 81.5, 82.0):
            o.append(f'<line x1="{L}" y1="{Y(gv):.1f}" x2="{L + PW}" y2="{Y(gv):.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/><text x="{L - 8}" '
                     f'y="{Y(gv) + 3.5:.1f}" text-anchor="end">{gv:.1f}</text>')
        o.append(f'<line x1="{L}" y1="{bot}" x2="{L + PW}" y2="{bot}" stroke="var(--rule)" '
                 f'stroke-width="1"/>')
        for dn in range(d0, d1 + 1, 7):
            o.append(f'<line x1="{X(dn):.1f}" y1="{bot}" x2="{X(dn):.1f}" y2="{bot + 6}" '
                     f'stroke="var(--rule)" stroke-width="1"/><text x="{X(dn):.1f}" y="{bot + 17}" '
                     f'text-anchor="middle" style="fill:var(--faint)">{_day_label(dn)}</text>')
        by = {}
        for r in rows:
            by.setdefault(r["date"], []).append(r)
        for d, rs in by.items():                        # same-day rounds sit side by side
            for i, r in enumerate(rs):
                r["_x"] = X(_date_num(d) + (i - (len(rs) - 1) / 2) * 0.66)
        best, path = None, []
        for r in rows:
            r["_prev"], r["_nb"] = best, False
            if r.get("collab"):
                continue
            v, x = r["actual"]["final"], r["_x"]
            if best is None:
                path.append(f"M{x:.1f},{Y(v):.1f}")
            else:
                path.append(f"H{x:.1f}" + (f"V{Y(v):.1f}" if v > best else ""))
            if best is None or v > best:
                best, r["_nb"] = v, True
        ours = [r for r in rows if not r.get("collab")]
        if path:
            path.append(f"H{L + PW:.1f}")
            o.append(f'<path d="{"".join(path)}" fill="none" stroke="var(--dim)" stroke-width="1.5" '
                     f'stroke-linejoin="round"/>')
        reads = sorted((x for x in LEADER_READS if x[1] == track), key=lambda x: x[0])
        if reads and ours:
            ld, lv = reads[-1][0], reads[-1][2]
            bestr = max(ours, key=lambda r: r["actual"]["final"])
            bv, xr = bestr["actual"]["final"], L + PW
            o.append(f'<line x1="{X(_date_num(ld)):.1f}" y1="{Y(lv):.1f}" x2="{xr}" y2="{Y(lv):.1f}" '
                     f'stroke="var(--ink)" stroke-width="1"/>')
            for k, (dt, _tr, v, note) in enumerate(reads):
                x, yy = X(_date_num(dt)), Y(v)
                then = [r["actual"]["final"] for r in ours if r["date"] <= dt]
                ten = (f"leader · Track {track} · read {dt}" + (f" ({note})" if note else "")
                       + f"\nfinal {v:.5f}"
                       + (f"\nour best then {max(then):.3f} · gap {v - max(then):.3f}" if then else ""))
                tzh = (f"领先者 · Track {track} · {dt} 读取" + (f"（{note}）" if note else "")
                       + f"\nfinal {v:.5f}"
                       + (f"\n当时我们的最佳 {max(then):.3f} · 差距 {v - max(then):.3f}" if then else ""))
                body = (f'<path d="M{x:.1f},{yy - 5.5:.1f}L{x + 5.5:.1f},{yy:.1f}L{x:.1f},{yy + 5.5:.1f}'
                        f'L{x - 5.5:.1f},{yy:.1f}Z" fill="var(--panel)" stroke="var(--ink)" '
                        f'stroke-width="1.5"/>')
                o.append(_hit(x, yy, ten, tzh, body))
                if k < len(reads) - 1:
                    o.append(f'<text x="{x:.1f}" y="{yy - 10:.1f}" text-anchor="middle">{v:.3f}</text>')
            o.append(tsvg(f"{xr + 10}", f"{Y(lv) + 3.5:.1f}", f"leader {lv:.3f}", f"领先者 {lv:.3f}",
                          'style="fill:var(--ink)"'))
            o.append(tsvg(f"{xr + 10}", f"{Y(bv) + 3.5:.1f}", f"best {bv:.3f} {bestr['round']}",
                          f"最佳 {bv:.3f} {bestr['round']}", 'style="fill:var(--ink);font-weight:600"'))
            o.append(f'<line x1="{xr + 4}" y1="{Y(lv) + 4:.1f}" x2="{xr + 4}" y2="{Y(bv) - 4:.1f}" '
                     f'stroke="var(--rule)" stroke-width="1"/>')
            o.append(tsvg(f"{xr + 10}", f"{(Y(lv) + Y(bv)) / 2 + 3.5:.1f}", f"gap {lv - bv:.3f}",
                          f"差距 {lv - bv:.3f}"))
        for r in rows:
            x, v = r["_x"], r["actual"]["final"]
            yy = Y(v)
            head = f"{r['round']} · Track {track} · {r['date']}\nfinal {v:.4f}"
            if r.get("collab"):
                ten = head + "\ncollaborator round, package not recorded"
                tzh = head + "\n合作者提交，未记录包"
                body = _marker(x, yy, [])
            else:
                prev, p = r["_prev"], (r.get("pred") or {}).get("final")
                if prev is None:
                    s_en, s_zh = "first round on this board", "本板第一轮"
                elif r["_nb"]:
                    s_en, s_zh = f"new best, +{v - prev:.4f}", f"新最佳，+{v - prev:.4f}"
                else:
                    s_en, s_zh = f"best then {prev:.4f}", f"此前最佳 {prev:.4f}"
                ten = (f"{head} ({s_en})\n{r['_lab']}"
                       + (f"\npredicted {p:.3f} · miss {v - p:+.3f}" if p else ""))
                tzh = (f"{head}（{s_zh}）\n{r['_lab']}"
                       + (f"\n预测 {p:.3f} · 偏差 {v - p:+.3f}" if p else ""))
                body = _marker(x, yy, families_of(r.get("backbone")))
            o.append(f'<line x1="{x:.1f}" y1="{bot}" x2="{x:.1f}" y2="{bot + 3}" stroke="var(--dim)" '
                     f'stroke-width="1"/>')
            o.append(_hit(x, yy, ten, tzh, body))
            lab_st = "fill:var(--faint)" if r.get("collab") else "fill:var(--dim)"
            id_st = ("fill:var(--ink);font-weight:600" if r["_nb"]
                     else ("fill:var(--faint)" if r.get("collab") else "fill:var(--ink)"))
            o.append(f'<text transform="translate({x:.1f},{bot + 26}) rotate(-90)" text-anchor="end" '
                     f'dy="3.5"><tspan style="{lab_st}">{_esc(r["_lab"])} </tspan>'
                     f'<tspan style="{id_st}">{_esc(r["round"])}</tspan></text>')
        y0 = bot + 26 + LH + 30
    return (f'<svg class="chart" viewBox="0 0 {W} {max(y0 - 20, 60):.0f}" role="img" '
            f'aria-label="board final of every scored round by upload date, Track 1 and Track 2">'
            + "".join(o) + "</svg>")


def _curve_section(B):
    collab = _collab_rounds()
    fams = [f for r in B if r.get("actual") for f in families_of(r.get("backbone"))]
    items = _fam_legend(fams) + [
        '<span><svg width="12" height="12" aria-hidden="true"><path d="M6,2A4,4 0 0 0 6,10Z" '
        'fill="var(--f1)"/><path d="M6,2A4,4 0 0 1 6,10Z" fill="var(--f2)"/></svg>'
        + t("two families in one package", "一个包里有两个家族") + "</span>",
        '<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="3.5" '
        'fill="none" stroke="var(--fx)" stroke-width="1.5"/></svg>'
        + t("collaborator round", "合作者提交") + "</span>",
        '<span><svg width="18" height="12" aria-hidden="true"><path d="M1,9H9V3H17" fill="none" '
        'stroke="var(--dim)" stroke-width="1.5"/></svg>' + t("running best", "历史最佳") + "</span>",
        '<span><svg width="12" height="12" aria-hidden="true"><path d="M6,1.5L10.5,6L6,10.5L1.5,6Z" '
        'fill="none" stroke="var(--ink)" stroke-width="1.5"/></svg>'
        + t("leader, dated read", "领先者，按读取日期") + "</span>"]
    return ('<section id="curve"><div class="head"><h2>' + t("Optimization curve", "优化曲线")
            + '</h2><span class="eyebrow">'
            + t("board final &middot; every scored round", "board final &middot; 每个已评分轮次")
            + "</span><p>"
            + t("Board <code>final</code> of every scored round against the day it was uploaded, "
                "Track&nbsp;1 above Track&nbsp;2 on one shared scale. Under each panel, every point "
                "is labelled with its backbone composition and round; a split marker is a package "
                "with two families. The step line is our running best, and the diamonds are dated "
                "reads of the best other team. Hover or focus a point for its prediction; every "
                "number is also in the Submissions table.",
                "每个已评分轮次的 board <code>final</code>，横轴为上传日期；Track&nbsp;1 在上、"
                "Track&nbsp;2 在下，共用同一纵轴。每个点在所在面板下方标注骨干组成与轮次；双色点表示"
                "一个包里有两个家族。阶梯线是我们的历史最佳，菱形是按日期读取的其他队伍最佳成绩。"
                "悬停或聚焦某个点可看其预测；所有数字也在「提交记录」表中。")
            + '</p></div><div class="legend">' + "".join(items) + '</div><div class="scroll">'
            + _curve_svg(B, collab) + "</div></section>")


SCREEN_VERDICTS = {"advance": ("advance", "晋级"), "replicate": ("replicate", "复现"),
                   "pending": ("pending", "待定"), "drop": ("drop", "淘汰")}
SCREEN_STATUS = {"queued": "排队", "running": "运行中", "training": "训练中", "trained": "已训练",
                 "done": "完成", "failed": "失败"}


def _screen_mark(x, y, r):
    """An arm's mark: the family's colour and shape; the FILL says the verdict -- hollow pending,
    filled replicate, filled and ringed advance, faded hollow drop."""
    fam = [arm_family(r)]
    v = str(r.get("verdict") or "pending").lower()
    if v == "drop":
        return _marker(x, y, fam, hollow=True, faded=True)
    if v not in ("replicate", "advance"):
        return _marker(x, y, fam, hollow=True)
    m = _marker(x, y, fam)
    if v == "advance":
        m = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="none" stroke="var(--ink)" '
             f'stroke-width="1.5"/>') + m
    return m


def _screen_svg(S):
    """Latency against projected accuracy, one point per arm. Returns (svg, plotted arm names), or
    (None, set()) when no arm has both numbers yet."""
    ratio = any(_num(r.get("ms_ratio_fno")) is not None for r in S)
    xk = "ms_ratio_fno" if ratio else "ms_b64_fp16"
    pts = [(r, _num(r.get(xk)), best_proj(r)) for r in S]
    pts = [(r, x, b) for r, x, b in pts if x is not None and x > 0 and b[0] is not None]
    if not pts:
        return None, set()
    W, H, L, Rm, T, Bm = 1000, 400, 64, 118, 18, 46
    PW, PH = W - L - Rm, H - T - Bm
    xs = [x for _, x, _ in pts] + ([1.0] if ratio else [])
    lo, hi = float(np.log2(min(xs))) - 0.35, float(np.log2(max(xs))) + 0.35
    if hi - lo < 1.5:
        c = (hi + lo) / 2
        lo, hi = c - 0.75, c + 0.75
    ys = [b[0] for _, _, b in pts] + [0.0, 0.10]
    ylo, yhi = min(ys) - 0.04, max(ys) + 0.04

    def X(v):
        return L + (float(np.log2(v)) - lo) * PW / (hi - lo)

    def Y(v):
        return T + (yhi - v) * PH / (yhi - ylo)

    o = []
    step = 0.05 if yhi - ylo <= 0.5 else (0.1 if yhi - ylo <= 1.0 else 0.25)
    g = float(np.ceil(ylo / step) * step)
    while g <= yhi + 1e-9:
        o.append(f'<line x1="{L}" y1="{Y(g):.1f}" x2="{L + PW}" y2="{Y(g):.1f}" stroke="var(--line)" '
                 f'stroke-width="1"/><text x="{L - 8}" y="{Y(g) + 3.5:.1f}" text-anchor="end">'
                 f'{g + 0.0:+.2f}</text>')
        g = round(g + step, 6)
    cands = ([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 16] if ratio
             else [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
    ticks = [c for c in cands if lo <= np.log2(c) <= hi]
    if len(ticks) > 7:
        ticks = [c for c in ticks if float(np.log2(c)).is_integer()]
    o.append(f'<line x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}" stroke="var(--rule)" '
             f'stroke-width="1"/>')
    for c in ticks:
        o.append(f'<line x1="{X(c):.1f}" y1="{T + PH}" x2="{X(c):.1f}" y2="{T + PH + 5}" '
                 f'stroke="var(--rule)" stroke-width="1"/><text x="{X(c):.1f}" y="{T + PH + 17}" '
                 f'text-anchor="middle">{("×%g" % c) if ratio else ("%g ms" % c)}</text>')
    if ratio and lo <= 0.0 <= hi:
        o.append(f'<line x1="{X(1):.1f}" y1="{T}" x2="{X(1):.1f}" y2="{T + PH}" stroke="var(--rule)" '
                 f'stroke-width="1"/>')
        o.append(tsvg(f"{X(1) + 5:.1f}", f"{T + 10}", "FNO w96 L8 latency", "FNO w96 L8 的延迟"))
    o.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{L + PW}" y2="{Y(0):.1f}" stroke="var(--dim)" '
             f'stroke-width="1"/>')
    o.append(f'<line x1="{L}" y1="{Y(0.10):.1f}" x2="{L + PW}" y2="{Y(0.10):.1f}" stroke="var(--cool)" '
             f'stroke-width="1.2" stroke-dasharray="5 4"/>')
    o.append(tsvg(f"{L + PW + 8}", f"{Y(0.10) + 3.5:.1f}", "advance +0.10", "晋级线 +0.10",
                  'style="fill:var(--ink)"'))
    o.append(tsvg(f"{L + PW + 8}", f"{Y(0) + 3.5:.1f}", "R25 = 0", "R25 = 0"))
    xt = (("A100 latency ÷ FNO w96 L8, batch 64 fp16 (log scale)", "A100 延迟 ÷ FNO w96 L8，batch 64 fp16（对数轴）")
          if ratio else ("A100 ms/window, batch 64 fp16 (log scale)", "A100 ms/window，batch 64 fp16（对数轴）"))
    o.append(tsvg(f"{L + PW / 2:.1f}", f"{H - 6}", xt[0], xt[1], 'text-anchor="middle" style="fill:var(--faint)"'))
    for cls, s in (("en", "projected Δ final, Track 1"), ("zh", "Track 1 预测 Δ final")):
        o.append(f'<text class="{cls}" transform="translate(15,{T + PH / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle" style="fill:var(--faint)">{s}</text>')
    placed, done = [], set()
    for r, xv, (bv, which) in sorted(pts, key=lambda p: (-p[2][0], str(p[0].get("arm")))):
        px, py, name = X(xv), Y(bv), str(r.get("arm"))
        w = 6.0 * len(name) + 2
        side = 1 if px + 10 + w < W - 4 else -1
        lx = px + 10 if side > 0 else px - 10 - w
        box, dy = None, 0
        for dy in (0, -12, 12, -24, 24, -36, 36, -48, 48):
            box = (lx, py + dy - 7, lx + w, py + dy + 5)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                break
        placed.append(box)
        lat = []
        if _num(r.get("ms_ratio_fno")) is not None:
            lat.append(f"×{_num(r.get('ms_ratio_fno')):.2f} of FNO")
        if _num(r.get("ms_b64_fp16")) is not None:
            lat.append(f"{_num(r.get('ms_b64_fp16')):.3f} ms b64 fp16")
        pr_en, pr_zh = [], []
        k1swap = arm_family(r) == "F-FNO"      # its `standalone` field is the K=1 single-model swap read
        for k, en, zh in (("proj_dfinal_t1_standalone", "T1 K=1 swap" if k1swap else "T1 standalone",
                           "T1 K=1 替换" if k1swap else "T1 单独"),
                          ("proj_dfinal_t1_partner", "T1 partner", "T1 作伙伴"),
                          ("proj_dfinal_t1_replace_vits", "T1 replace ViTs", "T1 替换 ViT 对"),
                          ("proj_dfinal_t1_p2_l8", "T1 P2-L8", "T1 P2-L8"),
                          ("proj_dfinal_t2_partner", "T2 partner", "T2 作伙伴")):
            v = _num(r.get(k))
            if v is not None:
                pr_en.append(f"{en} {v:+.3f}")
                pr_zh.append(f"{zh} {v:+.3f}")
        verdict = str(r.get("verdict") or "pending").lower()
        base = ([f"{name} · {arm_family(r)} · {r.get('status') or '?'}"]
                + ([str(r.get("backbone"))] if r.get("backbone") else []) + ([" · ".join(lat)] if lat else []))
        ten = "\n".join(base + [" · ".join(pr_en), f"plotted: {which} · verdict {verdict}"])
        tzh = "\n".join(base + [" · ".join(pr_zh),
                                f"作图取{ROLE_ZH.get(which, which)} · "
                                f"结论 {SCREEN_VERDICTS.get(verdict, (verdict, verdict))[1]}"])
        if dy:
            o.append(f'<line x1="{px + 5 * side:.1f}" y1="{py:.1f}" x2="{lx if side > 0 else lx + w:.1f}" '
                     f'y2="{py + dy - 1:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        o.append(_hit(px, py, ten, tzh, _screen_mark(px, py, r)))
        o.append(f'<text x="{lx if side > 0 else lx + w:.1f}" y="{py + dy + 3.5:.1f}" '
                 f'text-anchor="{"start" if side > 0 else "end"}" '
                 f'style="fill:var({"--faint" if verdict == "drop" else "--ink"})">{_esc(name)}</text>')
        done.add(name)
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="backbone screen: '
            f'latency against projected final">' + "".join(o) + "</svg>"), done


def _fmt(v, fmt):
    x = _num(v)
    return (fmt % x) if x is not None else '<span class="d">&mdash;</span>'


def k1cell(r):
    """The K=1 standalone figure: shown muted and tagged, never plotted, never in a verdict."""
    v = _num(((r.get("standalone_k1") or {}).get("t1") or {}).get("proj"))
    if v is None:
        return '<td><span class="d">&mdash;</span></td>'
    return f'<td><span class="d">{v:+.3f} ' + t("extrap.", "外推") + "</span></td>"


def _screen_table(S):
    """Every arm, whatever state its record is in."""
    th = lambda en, zh, cls="", extra="": f'<th{cls}{extra}>{t(en, zh)}</th>'  # noqa: E731
    o = ['<div class="scroll" style="margin-top:14px"><table><thead><tr>',
         th("Arm", "实验臂", ' class="l"', ' rowspan="2"'), th("Family", "家族", ' class="l"', ' rowspan="2"'),
         th("Status", "状态", ' class="l"', ' rowspan="2"'), th("Wave", "批次", "", ' rowspan="2"'),
         '<th rowspan="2">GPU-h</th>', th("Params M", "参数 M", "", ' rowspan="2"'),
         th("Epochs", "轮数", "", ' rowspan="2"'), '<th rowspan="2">dropout</th>',
         th("latency, A100 ms", "延迟，A100 ms", ' class="grp"', ' colspan="4"'),
         th("single model &middot; full split", "单模型 &middot; 全划分", ' class="grp"', ' colspan="4"'),
         th("single model &middot; AoA 15&ndash;20", "单模型 &middot; AoA 15&ndash;20", ' class="grp"', ' colspan="4"'),
         th("projected &Delta; final", "预测 &Delta; final", ' class="grp"', ' colspan="5"'),
         th("Verdict", "结论", ' class="l"', ' rowspan="2"'), th("Note", "备注", ' class="l t"', ' rowspan="2"'),
         "</tr><tr><th>b64 fp16</th><th>b64 fp32</th><th>b1 fp16</th><th>&times;FNO</th>",
         "<th>rel</th><th>tke</th><th>mvpe</th><th>sps</th>" * 2,
         th("T1 solo", "T1 单独"), th("T1 partner", "T1 作伙伴"), th("T1 replace", "T1 替换"),
         th("T2 partner", "T2 作伙伴"), th("K=1 solo, extrapolated", "K=1 单独（外推）"),
         "</tr></thead><tbody>"]
    for r in sorted(S, key=screen_order):
        fam, st = arm_family(r), str(r.get("status") or "")
        v = str(r.get("verdict") or "pending").lower()
        ven, vzh = SCREEN_VERDICTS.get(v, (v, v))

        def metric(block, k, fmt, r=r):
            d = r.get(block) if isinstance(r.get(block), dict) else {}
            return f"<td>{_fmt(d.get(k), fmt)}</td>"

        def proj(k, r=r, note=""):
            x = _num(r.get(k))
            if x is None:
                return '<td><span class="d">&mdash;</span></td>'
            cls = ' class="g"' if x >= 0.10 else (' class="b"' if x < 0 else "")
            return f"<td{cls}>{x:+.3f}{note}</td>"
        sq = " sq" if fam_square(fam) else ""
        stc = f' class="st-{_esc(st)}"' if st in ("failed", "running") else ""
        o.append(
            f'<tr><td class="l"><code>{_esc(str(r.get("arm")))}</code>'
            + (f'<br><span class="d">{_esc(str(r.get("backbone")))}</span>' if r.get("backbone") else "")
            + f'</td><td class="l"><span class="sw{sq}" style="background:{fam_var(fam)}"></span>'
            f'{_esc(str(fam or "?"))}</td>'
            f'<td class="l"><span{stc}>{t(_esc(st or "?"), SCREEN_STATUS.get(st, _esc(st or "?")))}</span></td>'
            f'<td>{_fmt(r.get("wave"), "%g")}</td><td>{_fmt(r.get("gpu_h"), "%.1f")}</td>'
            f'<td>{_fmt(r.get("params_M"), "%.1f")}</td><td>{_fmt(r.get("epochs"), "%g")}</td>'
            f'<td>{_fmt(r.get("dropout"), "%g")}</td><td>{_fmt(r.get("ms_b64_fp16"), "%.3f")}</td>'
            f'<td>{_fmt(r.get("ms_b64_fp32"), "%.3f")}</td><td>{_fmt(r.get("ms_b1_fp16"), "%.3f")}</td>'
            f'<td>{_fmt(r.get("ms_ratio_fno"), "%.2f")}</td>'
            + "".join(metric(b, k, f) for b in ("single_full", "single_aoa1520")
                      for k, f in (("rel", "%.4f"), ("tke", "%.4f"), ("mvpe", "%.4f"), ("sps", "%.2f")))
            + proj("proj_dfinal_t1_standalone",
                   note=(f'<br><span class="d">{t("K=1 swap", "K=1 替换")}</span>' if fam == "F-FNO" else ""))
            + proj("proj_dfinal_t1_partner")
            + proj("proj_dfinal_t1_replace_vits") + proj("proj_dfinal_t2_partner")
            + k1cell(r)
            + f'<td class="l"><span class="chip v-{_esc(v)}">{t(_esc(ven), _esc(vzh))}</span></td>'
            f'<td class="l t">{tdata(r.get("note") or "", r.get("note_zh"))}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def _screen_section(S, state):
    o = ['<section id="screen"><div class="head"><h2>' + t("Backbone screen", "骨干网络筛选")
         + '</h2><span class="eyebrow">' + t("phase 1 &middot; local proxy", "第一阶段 &middot; 本地代理")
         + "</span><p>"
         + t("Local proxy projections &mdash; <b>a sign, not a size</b>. Each arm is one held-out "
             "model trained with the shared screening recipe, projected against R25 (Track&nbsp;1) "
             "and T2R12 (Track&nbsp;2) with <code>predict_board.py</code>. <b>Accuracy gains are "
             "halved and losses are charged in full</b>; latency charges +1 invocation for a "
             "partner family, FLOPs through <code>LAT_LOC_*</code> and precision-only changes "
             "through <code>PREC_SLOPE_T1</code>. Three roles: <i>partner</i> = the package plus X; "
             "<i>replace</i> = FNO&times;3 + X in place of the ViT pair; <i>standalone</i> = "
             "X&times;4 fused in one invocation, accuracy along the FNO member-count curve, with "
             "the ViT pair&rsquo;s containment removed in full. The scatter plots each arm&rsquo;s "
             "best Track-1 role. A K=1 standalone figure is <i>extrapolated</i>: it appears in the "
             "table only, never in the scatter or a verdict. An arm advances at &ge;&nbsp;+0.10 on "
             "Track&nbsp;1, replicated. Source: <code>leaderboard/screen.jsonl</code>, written by "
             "forge (<code>Track_1/plan/plan_20260914.md</code> &sect;4, &sect;7).",
             "本地代理的预测——<b>只看方向，不看大小</b>。每个实验臂是用统一筛选配方训练的一个留出模型，"
             "以 R25（Track&nbsp;1）和 T2R12（Track&nbsp;2）为锚，用 <code>predict_board.py</code> "
             "计算。<b>精度收益减半，损失全额计入</b>；延迟方面，伙伴家族 +1 次调用，FLOPs 按 "
             "<code>LAT_LOC_*</code>，纯精度改动按 <code>PREC_SLOPE_T1</code>。三种角色："
             "<i>作伙伴</i> = 现有包加 X；<i>替换</i> = FNO&times;3 + X 取代 ViT 对；<i>单独</i> = "
             "X&times;4 融合为一次调用，精度沿 FNO 成员数曲线换算，并全额扣除 ViT 对的含盖率。"
             "散点取每个实验臂在 Track&nbsp;1 上最好的角色。K=1 的单独数字是<i>外推值</i>：只列在表中，"
             "不进散点，也不用于结论。Track&nbsp;1 上 &ge;&nbsp;+0.10 且复现后才晋级。数据来源："
             "<code>leaderboard/screen.jsonl</code>，由 forge 写入"
             "（<code>Track_1/plan/plan_20260914.md</code> &sect;4、&sect;7）。")
         + "</p></div>"]
    if not S:
        why = {"missing": ("<code>leaderboard/screen.jsonl</code> does not exist yet.",
                           "<code>leaderboard/screen.jsonl</code> 还不存在。"),
               "empty": ("<code>leaderboard/screen.jsonl</code> is empty.",
                         "<code>leaderboard/screen.jsonl</code> 是空的。")}.get(
            state, (f"<code>leaderboard/screen.jsonl</code> has no readable record ({_esc(state)}).",
                    f"<code>leaderboard/screen.jsonl</code> 没有可读的记录（{_esc(state)}）。"))
        o.append('<div class="empty">' + t("No screen results. " + why[0] + " forge writes one record "
                                            "per arm; rerun <code>lb.py --html</code> when it changes.",
                                            "暂无筛选结果。" + why[1] + "forge 每个实验臂写一条记录；"
                                            "文件变化后重新运行 <code>lb.py --html</code>。")
                 + "</div></section>")
        return "".join(o)
    by, vc = {}, {}
    for r in S:
        k = str(r.get("status") or "?")
        by[k] = by.get(k, 0) + 1
        k = str(r.get("verdict") or "pending").lower()
        vc[k] = vc.get(k, 0) + 1
    gh = sum(_num(r.get("gpu_h")) or 0.0 for r in S)
    o.append('<p class="cap" style="margin:0 0 10px">'
             + t(f"{len(S)} arms &middot; " + " &middot; ".join(f"{_esc(k)} {n}" for k, n in sorted(by.items()))
                 + f" &middot; {gh:.1f} GPU-h &middot; "
                 + " &middot; ".join(f"{_esc(k)} {n}" for k, n in sorted(vc.items())),
                 f"{len(S)} 个实验臂 &middot; "
                 + " &middot; ".join(f"{SCREEN_STATUS.get(k, _esc(k))} {n}" for k, n in sorted(by.items()))
                 + f" &middot; {gh:.1f} GPU 小时 &middot; "
                 + " &middot; ".join(f"{_esc(SCREEN_VERDICTS.get(k, (k, k))[1])} {n}"
                                     for k, n in sorted(vc.items())))
             + (" &middot; " + t(f"{_esc(state)} line(s) skipped", f"跳过 {_esc(state)} 行") if state else "")
             + "</p>")
    svg, done = _screen_svg(S)
    glyph = lambda body: f'<svg width="18" height="18" aria-hidden="true">{body}</svg>'  # noqa: E731
    legend = _fam_legend([arm_family(r) for r in S]) + [
        "<span>" + glyph('<circle cx="9" cy="9" r="4" fill="var(--panel)" stroke="var(--dim)" stroke-width="2"/>')
        + t("pending", "待定") + "</span>",
        "<span>" + glyph('<circle cx="9" cy="9" r="5" fill="var(--dim)"/>') + t("replicate", "复现") + "</span>",
        "<span>" + glyph('<circle cx="9" cy="9" r="8" fill="none" stroke="var(--ink)" stroke-width="1.5"/>'
                         '<circle cx="9" cy="9" r="5" fill="var(--dim)"/>') + t("advance", "晋级") + "</span>",
        "<span>" + glyph('<circle cx="9" cy="9" r="4" fill="var(--panel)" stroke="var(--dim)" stroke-width="2" '
                         'opacity=".45"/>') + t("drop", "淘汰") + "</span>",
        "<span>" + t("K=1 extrapolated figures: table only, never plotted",
                     "K=1 外推数字：只列在表中，不进散点") + "</span>"]
    o.append('<div class="legend">' + "".join(legend) + "</div>")
    if svg:
        o.append('<div class="scroll">' + svg + "</div>")
    else:
        o.append('<div class="empty">' + t("No arm has both a latency and a projection yet; the table "
                                            "lists what exists.",
                                            "还没有同时具备延迟与预测的实验臂；下表列出现有记录。") + "</div>")
    rest = sorted(str(r.get("arm")) for r in S if str(r.get("arm")) not in done)
    if svg and rest:
        o.append('<p class="cap">' + t("Not plotted yet (no latency or no projection): ",
                                       "尚未作图（缺延迟或缺预测）：")
                 + ", ".join(f"<code>{_esc(a)}</code>" for a in rest) + "</p>")
    o.append(_screen_table(S) + "</section>")
    return "".join(o)


def _residual_svg(rows, rms):
    """Diverging bars: how much of each board move the local proxy failed to explain."""
    n = len(rows)
    rowh, W, L, C = 21, 980, 210, 560
    H = n * rowh + 40
    sc = 46.0                                    # px per percentage point
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="proxy residual per transition">']
    o.append(f'<rect x="{C-rms*sc:.1f}" y="24" width="{2*rms*sc:.1f}" height="{n*rowh}" '
             f'fill="var(--coolw)"/>')
    o.append(tsvg(f"{C:.0f}", "16",
                  f"&#177;{rms:.1f}% rms &mdash; the resolution limit",
                  f"&#177;{rms:.1f}% rms &mdash; 分辨率极限",
                  'text-anchor="middle" class="lab"'))
    o.append(f'<line x1="{C}" y1="24" x2="{C}" y2="{24+n*rowh}" stroke="var(--rule)" '
             f'stroke-width="1"/>')
    for k, (tr, m, pd, bd, rr) in enumerate(rows):
        yy = 24 + k * rowh
        col = "var(--warm)" if rr > 0 else "var(--cool)"
        w = abs(rr) * sc
        o.append(f'<rect x="{C if rr>0 else C-w:.1f}" y="{yy+4.5:.1f}" width="{w:.1f}" '
                 f'height="{rowh-9}" fill="{col}" opacity=".85"/>')
        o.append(f'<text x="{L-10}" y="{yy+rowh/2+3.5:.1f}" text-anchor="end">{tr}</text>')
        o.append(f'<text x="{L}" y="{yy+rowh/2+3.5:.1f}" fill="var(--ink)">{m}</text>')
        o.append(f'<text x="{L+52}" y="{yy+rowh/2+3.5:.1f}">proxy {pd:+.2f}</text>')
        o.append(f'<text x="{L+150}" y="{yy+rowh/2+3.5:.1f}">board {bd:+.2f}</text>')
        tx = C + (w + 7) * (1 if rr > 0 else -1)
        o.append(f'<text x="{tx:.1f}" y="{yy+rowh/2+3.5:.1f}" fill="{col}" '
                 f'text-anchor="{"start" if rr>0 else "end"}" font-weight="600">{rr:+.2f}</text>')
    o.append(tsvg(f"{C-6}", f"{H-6}",
                  "board moved LESS than the proxy said",
                  "board 的变化小于 proxy 的预测",
                  'text-anchor="end" class="lab"'))
    o.append(tsvg(f"{C+6}", f"{H-6}", "board moved MORE",
                  "board 的变化更大", 'class="lab"'))
    o.append("</svg>")
    return "".join(o)


def html(B, Rn, w, path, S=(), state="missing"):
    T = [r for r in B if r.get("actual") and r["track"] == 1]
    ref = float(np.median([r["actual"]["time"] for r in T]))
    norm = [r["actual"]["final"] + (ref - r["actual"]["time"]) * w[3] for r in T]
    resid = [r["actual"]["final"] - r["pred"]["final"] for r in B
             if r.get("actual") and (r.get("pred") or {}).get("final")]
    R, rat0, rawrows, res, rms = calib(B)
    rows = [(f"{a}&#8594;{b}", m, pd, bd, rr) for a, b, m, pd, bd, rr in rawrows]
    best = max(T, key=lambda r: r["actual"]["final"])
    scored = [r for r in B if r.get("actual")]
    T2 = [r for r in scored if r["track"] == 2]
    best2 = max(T2, key=lambda r: r["actual"]["final"])
    (ld1, lv1), (ld2, lv2) = leader_latest(1), leader_latest(2)

    o = [HTML_HEAD, '<div class="app">', _sidebar(B, Rn, S), '<main><div class="wrap">',
         "<h1>" + t("RealPDE calibration log", "RealPDE 校准日志") + "</h1>",
         '<p class="lede">'
         + t("Every submission, the prediction recorded before it was uploaded, and what the "
             "board returned &mdash; then the local experiments behind them and the measured "
             "correction from our validation set to the hidden one.",
             "每一次提交、上传前记录的"
             "预测，以及 board 实际返回的结果"
             "——随后是它们背后的本地"
             "实验，和从我们的验证集到"
             "隐藏集的实测校正。")
         + "</p>", '<div id="top">']

    o.append('<div class="strip">')
    for k_en, k_zh, v, n_en, n_zh in [
            ("best, track 1", "最佳，track 1", f"{best['actual']['final']:.3f}",
             f"{best['round']} &middot; {_esc(best.get('backbone') or '')}",
             f"{best['round']} &middot; {_esc(best.get('backbone') or '')}"),
            ("best, track 2", "最佳，track 2", f"{best2['actual']['final']:.3f}",
             f"{best2['round']} &middot; {_esc(best2.get('backbone') or '')}",
             f"{best2['round']} &middot; {_esc(best2.get('backbone') or '')}"),
            ("gap to leader, track 1", "与领先者差距，track 1",
             f"{lv1 - best['actual']['final']:.3f}",
             f"leader {lv1:.3f} &middot; read {ld1[5:]}", f"领先者 {lv1:.3f} &middot; {ld1[5:]} 读取"),
            ("gap to leader, track 2", "与领先者差距，track 2",
             f"{lv2 - best2['actual']['final']:.3f}",
             f"leader {lv2:.3f} &middot; read {ld2[5:]}", f"领先者 {lv2:.3f} &middot; {ld2[5:]} 读取"),
            ("prediction rms", "预测残差 rms",
             f"{np.sqrt(np.mean(np.square(resid))):.3f}",
             f"{len(resid)} predicted rounds",
             f"{len(resid)} 个有预测的轮次"),
            ("resolution limit", "分辨率极限", f"{rms:.1f}%",
             "of error, to transfer",
             "占误差比，可迁移下限")]:
        o.append(f'<div class="cell"><div class="k">{t(k_en, k_zh)}</div>'
                 f'<div class="v">{v}</div><div class="n">{t(n_en, n_zh)}</div></div>')
    o.append("</div>")

    o.append('<div class="thesis"><span class="eyebrow">'
             + t("read this first", "先读这里") + "</span>"
             + t(f"<b>`sps` factorizes, and the residual names every lever.</b> The scoring rule is <code>sps = 100 &middot; mean(a_i) &middot; containment</code> with <code>a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe)</code>, so any round can be priced from its published subscores alone &mdash; no free parameter, no local measurement. THE RESIDUAL IS THEN THE CONTAINMENT TERM, and across all eighteen consecutive board steps it is nonzero for exactly four kinds of change: the <b>band</b> (+1.410, R9&#8594;R10), regularized <b>capacity</b> (+0.992 / +0.961, width 24&#8594;96 on both boards), <b>member count</b> (+0.668, 3&#8594;8), and the architecture <b>family</b> (+0.464 / &minus;0.440, U-FNO in and back out). Everything rounds 16&ndash;20 tried &mdash; sim co-training, depth at equal FLOPs, the coverage and temporal-gradient losses, <code>lam_v</code>, both latency rebuilds, the gated blend &mdash; lands inside &plusmn;0.05. It also re-prices the exchange rate, because <code>a_i</code> is computed from the other three subscores: per 10% of error removed, <code>rel_l2</code> is worth <b>+0.346</b>, <code>tke</code> <b>+0.290</b> and <code>mvpe</code> <b>+0.100</b>, so <code>tke</code> is 84% of <code>rel_l2</code>, not 74%. Two corrections it forced: round 16 closed every non-FNO family on U-FNO&rsquo;s &minus;0.45 miss, which was true of the <i>package</i> (batch-1 latency, worse <code>rel_l2</code>) and false of the <i>member</i>, whose +0.44 <code>sps</code> was filed as noise and was the third-largest containment move on record; and &ldquo;<code>sps</code> is unreachable&rdquo; was a statement about rms, not about <code>sps</code>. Section&nbsp;3 remains the reason every local reading is hedged: the proxy predicts the board&rsquo;s <i>level</i> to &plusmn;2&ndash;8% and almost none of its <i>movement</i> ({rms:.1f}% residual rms).",
                 f"<b>`sps` 可以分解，而残差把每一个杠杆都点了名。</b>评分规则是 <code>sps = 100 &middot; mean(a_i) &middot; 含盖率</code>，其中 <code>a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe)</code>，所以任何一轮都可以只用它已公布的子分数来定价——没有自由参数，也不需要任何本地测量。于是残差就是含盖率项：在全部十八对连续 board 步骤上，它只对四类改动非零——<b>band</b>（+1.410，R9&#8594;R10）、带正则的<b>容量</b>（+0.992 / +0.961，宽度 24&#8594;96，两块板都复现）、<b>成员数</b>（+0.668，3&#8594;8），以及架构<b>家族</b>（+0.464 / &minus;0.440，U-FNO 进再出）。第 16&ndash;20 轮试过的一切——sim 联合训练、等 FLOPs 的深度、coverage 与 temporal-gradient 损失、<code>lam_v</code>、两次延迟重构、门控混合——全在 &plusmn;0.05 之内。它同时重新给兑换率定了价，因为 <code>a_i</code> 由另外三个子分数算出：每削减 10% 误差，<code>rel_l2</code> 值 <b>+0.346</b>，<code>tke</code> 值 <b>+0.290</b>，<code>mvpe</code> 值 <b>+0.100</b>——<code>tke</code> 是 <code>rel_l2</code> 的 84%，不是 74%。它逼出两处更正：第 16 轮用 U-FNO 的 &minus;0.45 封了所有非 FNO 家族，那对<i>包</i>成立（batch 1 的延迟、更差的 <code>rel_l2</code>），对<i>成员</i>不成立——它那 +0.44 的 <code>sps</code> 被当噪声记下，实际是有记录以来第三大的含盖率移动；而「<code>sps</code> 拿不到」说的是 rms，不是 <code>sps</code>。第 3 节仍然是这里每个本地读数都要打折的理由：proxy 能把 board 的<i>水平</i>预测到 &plusmn;2&ndash;8%，却几乎解释不了它的<i>变化</i>（残差 rms {rms:.1f}%）。")
             + "</div></div>")
    o.append(_curve_section(B))
    o.append(_screen_section(S, state))

    # 1 -----------------------------------------------------------------------------------
    o.append('<section id="submissions"><div class="head"><h2>'
             + t("Submissions", "提交记录") + '</h2>'
             '<span class="eyebrow">'
             + t("board final", "board final 分数") + "</span><p>"
             + t("The prediction is recorded before upload; a prediction written afterwards is "
                 "worth nothing. Colour is the size of the miss.",
                 "预测必须在上传前记录；"
                 "事后补写的预测毫无价值。"
                 "颜色表示偏差的大小。")
             + "</p></div>"
             '<div class="scroll"><table><thead><tr>'
             '<th class="l">' + t("Round", "轮次") + "</th>"
             "<th>" + t("Tr", "赛道") + "</th>"
             '<th class="l">' + t("Date", "日期") + "</th>"
             '<th class="l">' + t("Backbone", "骨干组成") + "</th>"
             '<th class="l t">' + t("Thesis", "论点") + "</th>"
             "<th>" + t("Pred", "预测") + "</th>"
             "<th>" + t("Actual", "实际") + "</th>"
             "<th>" + t("Miss", "偏差") + "</th></tr></thead><tbody>")
    for r in B:
        a, p = r.get("actual"), (r.get("pred") or {}).get("final")
        miss, cls = "", ""
        act = '<span class="d">' + t("pending", "待定") + "</span>"
        pv = "&mdash;"
        if p is not None:
            pv = f"{p:.3f}"
        if a:
            act = f"{a['final']:.3f}"
        if a and p:
            miss = f"{a['final'] - p:+.3f}"
            d = abs(a["final"] - p)
            cls = ' class="g"' if d < 0.05 else (' class="b"' if d > 0.12 else ' class="w"')
        sws = "".join(f'<span class="sw{" sq" if fam_square(f) else ""}" style="background:{fam_var(f)}">'
                      f'</span>' for f in families_of(r.get("backbone")))
        o.append(f'<tr><td class="l"><b>{r["round"]}</b></td><td>{r["track"]}</td>'
                 f'<td class="l d">{r["date"]}</td>'
                 f'<td class="l">{sws}{_esc(r.get("backbone") or "")}</td>'
                 f'<td class="l t">{tdata(r.get("thesis"), r.get("thesis_zh"))}</td>'
                 f'<td class="d">{pv}</td><td><b>{act}</b></td>'
                 f"<td{cls}>{miss}</td></tr>")
    o.append("</tbody></table></div>")
    o.append('<div class="scroll" style="margin-top:14px"><table><thead><tr>'
             '<th class="l">' + t("Round", "轮次") + "</th>"
             + "".join(f"<th>{n}</th>" for n in SUB) + "</tr></thead><tbody>")
    for r in B:
        a, p = r.get("actual"), r.get("pred") or {}
        if not a:
            continue
        cells = []
        for n in SUB:
            if n in p and n in a:
                dd = a[n] - p[n]
                c = "g" if abs(dd) < 0.15 else ("b" if abs(dd) > 0.4 else "w")
                cells.append(f'<td><span class="d">{p[n]:.2f}</span> '
                             f'<span class="{c}">{a[n]:.3f}</span></td>')
            else:
                cells.append(f"<td>{a[n]:.3f}</td>")
        o.append(f'<tr><td class="l"><b>{r["round"]}</b></td>' + "".join(cells) + "</tr>")
    o.append('</tbody></table></div><p class="stamp" style="margin:10px 0 0">'
             + t("predicted &middot; actual",
                 "预测值 &middot; 实际值")
             + "</p></section>")

    # 2 -----------------------------------------------------------------------------------
    o.append('<section id="progress"><div class="head"><h2>'
             + t("Progress at constant latency",
                 "恒定延迟下的进展") + "</h2>"
             '<span class="eyebrow">'
             + t("time_score held at the median",
                 "time_score 固定在中位数") + "</span><p>"
             + t("Board <code>time_score</code> swings 2.8 points across the rounds and is "
                 "only partly ours to control, so the raw ranking conflates model quality "
                 "with eval-machine luck. Holding it fixed is the honest curve.",
                 "board 的 <code>time_score</code> 在各轮之间"
                 "摆动 2.8 分，且只有一部分由"
                 "我们控制，所以原始排名把"
                 "模型质量和评测机的运气"
                 "混在一起。把它固定住，"
                 "才是诚实的曲线。")
             + "</p></div>")
    o.append(_progress_svg(T, norm, ref) + "</section>")

    # 3 -----------------------------------------------------------------------------------
    o.append('<section id="calibration"><div class="head"><h2>'
             + t("Local &rarr; board correction", "本地 &rarr; board 校正")
             + "</h2><span class=\"eyebrow\">"
             + t("% of error", "误差百分比") + "</span><p>"
             + t("Proxy = the held-out <i>twin</i> members on the AoA&nbsp;15&ndash;20 slice "
                 "of <code>val_unseen</code>, scored at the ALPHA that package ships. "
                 "<code>r</code> is board error divided by proxy error. Both tracks are listed "
                 "&mdash; they share one test set and one scoring rule &mdash; but the delta "
                 "chain below is computed <i>within</i> a track, and a transition whose proxy "
                 "did not move (R13&#8594;R14) is skipped.",
                 "proxy = <code>val_unseen</code> 的 AoA&nbsp;15&ndash;20 切片"
                 "上的留出<i>孪生</i>成员，按"
                 "该包实际部署的 ALPHA 评分。"
                 "<code>r</code> 为 board 误差除以 proxy 误差"
                 "。两个赛道一并列出——"
                 "它们共用一份测试集和一套"
                 "评分规则——但下方的增量"
                 "链在<i>赛道内部</i>计算，proxy "
                 "未变的过渡（R13&#8594;R14）被跳"
                 "过。")
             + "</p></div>"
             '<div class="scroll"><table><thead><tr>'
             '<th class="l">' + t("Round", "轮次") + "</th>"
             "<th>" + t("Tr", "赛道") + "</th>"
             + "".join(f"<th>proxy {m}</th>" for m in M)
             + "".join(f"<th>board {m}</th>" for m in M)
             + "".join(f"<th>r_{m}</th>" for m in M) + "</tr></thead><tbody>")
    rat = {m: [] for m in M}
    for r in R:
        pp, aa = r["proxy"], {m: err(r["actual"][m]) for m in M}
        for m in M:
            rat[m].append(aa[m] / pp[m])
        o.append(f'<tr><td class="l"><b>{r["round"]}</b></td><td class="d">{r["track"]}</td>'
                 + "".join(f'<td class="d">{pp[m]:.5f}</td>' for m in M)
                 + "".join(f'<td class="d">{aa[m]:.5f}</td>' for m in M)
                 + "".join(f"<td><b>{aa[m]/pp[m]:.3f}</b></td>" for m in M) + "</tr>")
    o.append('<tr><td class="l d">'
             + t("mean &plusmn;&frac12;range", "均值 &plusmn;&frac12;极差")
             + "</td>" + "<td></td>" * 7
             + "".join(f'<td class="d">{np.mean(rat[m]):.3f}&#177;'
                       f"{(max(rat[m])-min(rat[m]))/2:.3f}</td>" for m in M)
             + "</tr></tbody></table></div>")
    o.append('<div class="thesis"><span class="eyebrow">'
             + t("the finding", "结论") + "</span>"
             + t(f"<b>The levels calibrate; the deltas do not.</b> Those ratios are stable to "
                 f"&plusmn;1&ndash;3%, so the proxy predicts the board&rsquo;s level well. "
                 f"Round to round it explains almost none of the board&rsquo;s movement "
                 f"&mdash; residual rms <b>{rms:.1f}%</b> of the error, range {min(res):+.1f} "
                 f"to {max(res):+.1f}%, and <code>mvpe</code> inverts sign in both tracks "
                 f"(R12&#8594;R13 proxy +2.34%, board &minus;3.17%; T2R1&#8594;T2R2 proxy "
                 f"+1.25%, board &minus;7.79%). <b>A local delta below ~{rms:.0f}% of the "
                 f"error does not survive the trip and must not be used to choose a "
                 f"package</b> &mdash; every improvement measured since 2026-08-08 has been "
                 f"smaller than it &mdash; dropout&rsquo;s (&minus;2.6% on the interior "
                 f"Re bands) and w96&rsquo;s (&minus;1.2% at 3-vs-3) included, and both "
                 f"transferred anyway. The instrument is not conservative, it is <i>blunt</i>: "
                 f"it gets the sign right and the size wrong. Corroboration: "
                 f"R14&#8594;R15 and T2R1&#8594;T2R2 are the SAME package change scored on the "
                 f"two boards, and their residuals agree to 0.7&nbsp;pp on all three metrics "
                 f"&mdash; the residual belongs to the proxy, not to a board draw.",
                 f"<b>水平可校准，增量不可。</b>"
                 f"这些比值稳定在 &plusmn;1&ndash;3%，"
                 f"所以 proxy 能很好地预测 board 的"
                 f"水平。但逐轮看，它几乎"
                 f"解释不了 board 的变化——残"
                 f"差 rms 为误差的 <b>{rms:.1f}%</b>，范围 "
                 f"{min(res):+.1f} 到 {max(res):+.1f}%，且 <code>mvpe</code> "
                 f"在两个赛道都反号（R12&#8594;R13 "
                 f"proxy +2.34%，board &minus;3.17%；T2R1&#8594;T2R2 proxy "
                 f"+1.25%，board &minus;7.79%）。<b>低于误差 "
                 f"~{rms:.0f}% 的本地增量无法完成到 "
                 f"board 的迁移，不得用于选包</b>"
                 f"——2026-08-08 以来测到的每一项"
                 f"改进都小于它——dropout"
                 f"（内部 Re 段 &minus;2.6%）和 w96"
                 f"（3-vs-3 下 &minus;1.2%）都在其中，"
                 f"而两者仍然迁移成功了。"
                 f"这台仪器不是保守，而是"
                 f"<i>粗</i>：符号判得对，幅度"
                 f"判不准。旁证：R14&#8594;R15 与 "
                 f"T2R1&#8594;T2R2 是同一处改动在两个 "
                 f"board 上的读数，三项指标的"
                 f"残差彼此相差不到 0.7&nbsp;pp"
                 f"——残差属于 proxy，而不"
                 f"属于某一次 board 抽签。")
             + "</div>")
    o.append(_residual_svg(rows, rms))
    o.append('<p class="stamp" style="margin:14px 0 0">'
             + t("also calibrated &mdash; sps element-tail f 1.360 (T1) / 1.350 (T2), "
                 "calib_band13.py &middot; board latency, measured as deltas: +1.151 / +0.866 "
                 "board ms per extra invocation (R23), FLOPs 1.75 / 0.60 board ms per 4060-eq "
                 "local ms (R24), a precision cut ~0.65 on Track 1 (fp16, R25) &middot; w_time "
                 "0.0956&ndash;0.0962 from five pairs whose error subscores were identical "
                 "(<code>time</code> alone moved)",
                 "其他已校准常数 &mdash; sps 元素尾部 f 1.360（T1）/ 1.350（T2），"
                 "calib_band13.py &middot; board 延迟按增量实测：每多一次调用 +1.151 / +0.866 "
                 "board ms（R23），FLOPs 每 4060 等效本地毫秒 1.75 / 0.60 board ms（R24），"
                 "精度削减在 Track 1 约 0.65（fp16，R25）&middot; w_time 0.0956&ndash;0.0962，"
                 "来自五对误差子分数完全相同、只有 <code>time</code> 变了的提交")
             + "</p></section>")

    # 4 -----------------------------------------------------------------------------------
    n_ens = sum(1 for r in Rn if r.get("ensemble_tested"))
    o.append('<section id="ledger"><div class="head"><h2>'
             + t("Experiment ledger", "实验台账") + "</h2>"
             '<span class="eyebrow">'
             + t(f"{len(Rn)} arms &middot; {n_ens} ensembled",
                 f"{len(Rn)} 条实验 &middot; {n_ens} 条做过"
                 f"集成验证") + "</span><p>"
             + t("<b>ens</b> marks an arm measured at the ensemble size that actually ships. "
                 "It is the most predictive column here: <code>ozdo</code> was +3.84&sigma; "
                 "per model and &minus;0.012 at four members, and <code>--var-mlp</code>&rsquo;s "
                 "3.9% fixed-field TKE cut became +0.08. An arm that has not been ensembled "
                 "has not been measured.",
                 "<b>ens</b> 标记那些在真正上线的"
                 "集成规模下测过的实验。"
                 "它是这里最有预测力的一"
                 "列：<code>ozdo</code> 单模型 +3.84&sigma;，"
                 "四成员时 &minus;0.012；<code>--var-mlp</code> "
                 "固定场 TKE 降 3.9%，最终只值 "
                 "+0.08。没有在集成规模下测"
                 "过的实验，等于没有测过。")
             + "</p></div>"
             '<div class="scroll"><table><thead><tr>'
             '<th class="l">' + t("Verdict", "结论") + "</th>"
             '<th class="l">' + t("Date", "日期") + "</th>"
             '<th class="l">' + t("Arm", "实验") + "</th>"
             '<th class="l">' + t("Direction", "方向") + "</th>"
             "<th>ens</th>"
             '<th class="l t">' + t("Effect", "效果") + "</th>"
             '<th class="l">' + t("Board", "上板") + "</th>"
             "</tr></thead><tbody>")
    order = {"SHIPPED": 0, "OPEN": 1, "MARGINAL": 2, "CLOSED": 3, "REJECTED": 4}
    for r in sorted(Rn, key=lambda x: (order.get(x["verdict"], 9), x["date"])):
        e = ('<span class="g">&#10003;</span>' if r.get("ensemble_tested")
             else '<span class="d">&mdash;</span>')
        v = r["verdict"]
        chip = t(v, VERDICT_ZH.get(v, v))
        sh = _esc(r.get("shipped") or "") or '<span class="d">&mdash;</span>'
        o.append(f'<tr><td class="l"><span class="chip {v}">{chip}</span>'
                 f'</td><td class="l d">{r["date"]}</td>'
                 f'<td class="l"><code>{_esc(r["arm"])}</code></td>'
                 f'<td class="l d">{tdata(r.get("direction"), r.get("direction_zh"))}</td>'
                 f'<td>{e}</td>'
                 f'<td class="l t">{tdata(r.get("effect"), r.get("effect_zh"))}</td>'
                 f'<td class="l">{sh}</td></tr>')
    o.append("</tbody></table></div></section>")

    import final_v9                       # the fit also carries two collaborator rounds that
    n_fit = len(final_v9.BOARD)           # were never staged here, so it is wider than board.jsonl
    o.append("<footer>"
             + t(f"subscore weights, FREE fit (no sum-to-one constraint), {n_fit} rounds pooled "
                 f"across both tracks, max residual "
                 f"{abs(final_v9.RESID).max():.3f}",
                 f"子分数权重，自由拟合（不强制权重和为 1），两个赛道合并 "
                 f"{n_fit} 轮拟合，最大残差 "
                 f"{abs(final_v9.RESID).max():.3f}")
             + " &nbsp;&mdash;&nbsp; "
             + " &nbsp; ".join(f"{n} {v:.3f}" for n, v in zip(SUB, w))
             + "<br>"
             + t("sources: leaderboard/board.jsonl &middot; leaderboard/runs.jsonl "
                 "&middot; Track_1/final_v9.py &middot; Track_1/proxy_calib.py",
                 "数据来源：leaderboard/board.jsonl &middot; "
                 "leaderboard/runs.jsonl &middot; Track_1/final_v9.py &middot; "
                 "Track_1/proxy_calib.py")
             + "</footer></div></main></div>")
    o.append(NAV_SCRIPT % json.dumps([n[0] for n in NAV]))
    o.append(TIP_SCRIPT)
    with open(path, "w") as fh:
        fh.write("\n".join(o))
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true")
    a = ap.parse_args()
    import final_v9
    w = final_v9.W0
    B, Rn = load("board.jsonl"), load("runs.jsonl")
    S, state = load_screen()
    section_board(B, w)
    section_progress(B, w)
    section_calib(B)
    section_runs(Rn)
    section_screen(S, state)
    if a.html:
        html(B, Rn, w, HTML_OUT, S, state)


if __name__ == "__main__":
    main()
