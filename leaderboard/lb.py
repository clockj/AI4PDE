"""The internal leaderboard: what we tried, what we predicted, what the board actually said.

    python leaderboard/lb.py                 # everything, as text
    python leaderboard/lb.py --html          # also write leaderboard/index.html

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
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Track_1"))

M = ["rel", "tke", "mvpe"]
SUB = ["rel", "tke", "mvpe", "time", "sps"]


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
def section_calib(B):
    print("\n=== 3. Local -> board calibration, and the resolution limit ===\n")
    R = [r for r in B if r.get("actual") and r.get("proxy") and r["track"] == 1]
    print("  The proxy is the held-out TWIN set on the AoA 15-20 slice of val_unseen, scored at")
    print("  the ALPHA the package ships. `r_x` is board error / proxy error.\n")
    print(f"  {'round':<6s} " + "  ".join(f"{'proxy ' + m:>10s}" for m in M)
          + "  " + "  ".join(f"{'board ' + m:>10s}" for m in M)
          + "  " + "  ".join(f"{'r_' + m:>6s}" for m in M))
    rat = {m: [] for m in M}
    for r in R:
        p, a = r["proxy"], r["actual"]
        b = {m: err(a[m]) for m in M}
        for m in M:
            rat[m].append(b[m] / p[m])
        print(f"  {r['round']:<6s} " + "  ".join(f"{p[m]:10.5f}" for m in M)
              + "  " + "  ".join(f"{b[m]:10.5f}" for m in M)
              + "  " + "  ".join(f"{b[m]/p[m]:6.3f}" for m in M))
    print("\n  LEVELS are well calibrated:")
    for m in M:
        v = np.array(rat[m])
        print(f"    r_{m:<5s} mean {v.mean():.3f}   spread {v.min():.3f}-{v.max():.3f} "
              f"(+-{(v.max()-v.min())/2/v.mean()*100:.1f} %)")

    print("\n  DELTAS are not. Round-to-round, how much of the board's move did the proxy")
    print("  explain? (`residual` = board move - proxy move, in percent of the error.)\n")
    print(f"  {'transition':<16s} {'metric':<6s} {'proxy d%':>9s} {'board d%':>9s} {'residual':>9s}")
    res = []
    for i in range(1, len(R)):
        a, b = R[i - 1], R[i]
        for m in M:
            pd = (b["proxy"][m] / a["proxy"][m] - 1) * 100
            bd = (err(b["actual"][m]) / err(a["actual"][m]) - 1) * 100
            res.append(bd - pd)
            print(f"  {a['round'] + ' -> ' + b['round']:<16s} {m:<6s} {pd:+9.2f} {bd:+9.2f} "
                  f"{bd - pd:+9.2f}")
    res = np.array(res)
    print(f"\n  residual rms {np.sqrt((res**2).mean()):.2f} %, range {res.min():+.2f} .. "
          f"{res.max():+.2f} %")
    print("\n  >>> THE RESOLUTION LIMIT. A local delta smaller than ~"
          f"{np.sqrt((res**2).mean()):.0f} % of the error does not")
    print("  >>> survive the trip to the board, and MUST NOT be used to choose a package.")
    print("  >>> Every measured improvement since 2026-08-08 has been smaller than this.")
    print("  >>> `mvpe` even inverts sign between the proxy and the board (R12->R13:")
    print("  >>> proxy +2.34 %, board -3.17 %), which is why it is worth 0.011 and ignored.")

    print("\n  Other calibrated constants, and where they came from:")
    print("    SPS element-tail stretch f   1.360 (T1) / 1.350 (T2)   calib_band13.py, round 13")
    print("      -- two boards returned two SPS values for one band, so this is measured twice")
    print("    board ms = 2.616 + 0.435*inv + 0.803*local_ms   rms 0.690 ms, 7 rounds")
    print("      -- residuals are ordered by DATE: the eval machine drifts, the intercept is not")
    print("         a constant. The one clean number is the same-day R12/R13 pair: an")
    print("         invocation costs 0.52-0.56 board ms = 0.036 `final`.")


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
HTML_HEAD = """<title>RealPDE Calibration Log</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Palette from the subject: PIV velocity fields are rendered on a blue-white-red diverging
   map, so cool/warm here are the domain's own colours, not decoration. Neutrals carry a blue
   bias so they read as chosen. */
:root{
  --ground:#f7f8fa; --panel:#ffffff; --sunk:#f1f3f7;
  --ink:#14181f; --dim:#6a7280; --faint:#9aa1ac; --line:#dfe3ea; --rule:#c8cfda;
  --probe:#2b5fa8; --cool:#1f6f8b; --warm:#b3402c; --hold:#8a6d1f;
  --coolw:#1f6f8b22; --warmw:#b3402c22;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#12151b; --panel:#191d24; --sunk:#141920;
  --ink:#e6e9ee; --dim:#98a1ae; --faint:#6b7481; --line:#262c35; --rule:#39414d;
  --probe:#7aa8e0; --cool:#5fb6c9; --warm:#ef8a72; --hold:#d4a94a;
  --coolw:#5fb6c92e; --warmw:#ef8a722e;
}}
:root[data-theme="dark"]{
  --ground:#12151b; --panel:#191d24; --sunk:#141920;
  --ink:#e6e9ee; --dim:#98a1ae; --faint:#6b7481; --line:#262c35; --rule:#39414d;
  --probe:#7aa8e0; --cool:#5fb6c9; --warm:#ef8a72; --hold:#d4a94a;
  --coolw:#5fb6c92e; --warmw:#ef8a722e;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;padding:34px 20px 80px;
  font:400 14px/1.6 "IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1200px;margin:0 auto}
h1{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",Arial,sans-serif;font-weight:700;
  font-size:30px;line-height:1.1;margin:0 0 6px;letter-spacing:-.005em;text-wrap:balance}
.lede{color:var(--dim);margin:0 0 6px;max-width:66ch;font-size:14.5px}
.stamp{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--faint);
  letter-spacing:.06em;text-transform:uppercase;margin:0 0 26px}
h2{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",Arial,sans-serif;font-weight:600;
  font-size:19px;margin:0;letter-spacing:.005em}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;font-weight:500;
  color:var(--faint);letter-spacing:.11em;text-transform:uppercase}
section{background:var(--panel);border:1px solid var(--line);border-top:2px solid var(--rule);
  padding:20px 22px 22px;margin:20px 0 0}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  margin:0 0 14px;flex-wrap:wrap}
.head p{color:var(--dim);margin:6px 0 0;font-size:13px;max-width:70ch;flex-basis:100%}
/* readout strip */
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);margin:0 0 4px}
.cell{background:var(--panel);padding:13px 15px}
.cell .k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;color:var(--faint);
  letter-spacing:.1em;text-transform:uppercase}
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
th{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;color:var(--faint);
  font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  border-bottom:1px solid var(--rule)}
td.l,th.l{text-align:left}
td.t{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;white-space:normal;
  min-width:16ch}
tbody tr:hover td{background:var(--sunk)}
.g{color:var(--cool)}.b{color:var(--warm)}.w{color:var(--hold)}.d{color:var(--faint)}
.chip{display:inline-block;padding:1px 8px;border:1px solid currentColor;border-radius:2px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;font-weight:500;
  letter-spacing:.05em}
.SHIPPED{color:var(--cool)}.OPEN{color:var(--probe)}
.MARGINAL,.CLOSED{color:var(--hold)}.REJECTED{color:var(--warm)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.92em;
  background:var(--sunk);padding:1px 5px;border:1px solid var(--line)}
svg{display:block;max-width:100%}
svg text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;fill:var(--dim)}
svg text.lab{font-size:10.5px;fill:var(--faint);letter-spacing:.06em}
footer{color:var(--faint);font-size:11.5px;margin-top:34px;padding-top:14px;
  border-top:1px solid var(--line);font-family:"IBM Plex Mono",ui-monospace,monospace;
  line-height:1.8}
@media (prefers-reduced-motion:no-preference){
  tbody tr td{transition:background .12s ease}}
</style>
"""


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
    o.append(f'<line x1="{L}" y1="{y(81.76136):.1f}" x2="{W-Rp}" y2="{y(81.76136):.1f}" '
             f'stroke="var(--warm)" stroke-width="1.2" stroke-dasharray="5 4"/>'
             f'<text x="{W-Rp+8}" y="{y(81.76136)+3.5:.1f}" fill="var(--warm)">leader 81.761</text>')
    # the flat span
    i0 = len(T) - 4
    o.append(f'<rect x="{x(i0)-11:.1f}" y="{Tp}" width="{x(len(T)-1)-x(i0)+22:.1f}" '
             f'height="{H-Tp-Bp}" fill="var(--warmw)"/>')
    o.append(f'<text x="{(x(i0)+x(len(T)-1))/2:.1f}" y="{Tp+13}" text-anchor="middle" '
             f'class="lab">span {max(norm[i0:])-min(norm[i0:]):.3f}</text>')
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


def _residual_svg(rows, rms):
    """Diverging bars: how much of each board move the local proxy failed to explain."""
    n = len(rows)
    rowh, W, L, C = 21, 980, 210, 560
    H = n * rowh + 40
    sc = 46.0                                    # px per percentage point
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="proxy residual per transition">']
    o.append(f'<rect x="{C-rms*sc:.1f}" y="24" width="{2*rms*sc:.1f}" height="{n*rowh}" '
             f'fill="var(--coolw)"/>')
    o.append(f'<text x="{C:.0f}" y="16" text-anchor="middle" class="lab">'
             f'&#177;{rms:.1f}% rms &mdash; the resolution limit</text>')
    o.append(f'<line x1="{C}" y1="24" x2="{C}" y2="{24+n*rowh}" stroke="var(--rule)" '
             f'stroke-width="1"/>')
    for k, (t, m, pd, bd, rr) in enumerate(rows):
        yy = 24 + k * rowh
        col = "var(--warm)" if rr > 0 else "var(--cool)"
        w = abs(rr) * sc
        o.append(f'<rect x="{C if rr>0 else C-w:.1f}" y="{yy+4.5:.1f}" width="{w:.1f}" '
                 f'height="{rowh-9}" fill="{col}" opacity=".85"/>')
        o.append(f'<text x="{L-10}" y="{yy+rowh/2+3.5:.1f}" text-anchor="end">{t}</text>')
        o.append(f'<text x="{L}" y="{yy+rowh/2+3.5:.1f}" fill="var(--ink)">{m}</text>')
        o.append(f'<text x="{L+52}" y="{yy+rowh/2+3.5:.1f}">proxy {pd:+.2f}</text>')
        o.append(f'<text x="{L+150}" y="{yy+rowh/2+3.5:.1f}">board {bd:+.2f}</text>')
        tx = C + (w + 7) * (1 if rr > 0 else -1)
        o.append(f'<text x="{tx:.1f}" y="{yy+rowh/2+3.5:.1f}" fill="{col}" '
                 f'text-anchor="{"start" if rr>0 else "end"}" font-weight="600">{rr:+.2f}</text>')
    o.append(f'<text x="{C-6}" y="{H-6}" text-anchor="end" class="lab">'
             f'board moved LESS than the proxy said</text>')
    o.append(f'<text x="{C+6}" y="{H-6}" class="lab">board moved MORE</text>')
    o.append("</svg>")
    return "".join(o)


def html(B, Rn, w, path):
    T = [r for r in B if r.get("actual") and r["track"] == 1]
    ref = float(np.median([r["actual"]["time"] for r in T]))
    norm = [r["actual"]["final"] + (ref - r["actual"]["time"]) * w[3] for r in T]
    resid = [r["actual"]["final"] - r["pred"]["final"] for r in B
             if r.get("actual") and (r.get("pred") or {}).get("final")]
    R = [r for r in B if r.get("actual") and r.get("proxy") and r["track"] == 1]
    rows, res = [], []
    for i in range(1, len(R)):
        a, b = R[i - 1], R[i]
        for m in M:
            pd = (b["proxy"][m] / a["proxy"][m] - 1) * 100
            bd = (err(b["actual"][m]) / err(a["actual"][m]) - 1) * 100
            res.append(bd - pd)
            rows.append((f'{a["round"]}&#8594;{b["round"]}', m, pd, bd, bd - pd))
    rms = float(np.sqrt(np.mean(np.square(res))))
    span = max(norm[-4:]) - min(norm[-4:])
    best = max(T, key=lambda r: r["actual"]["final"])

    o = [HTML_HEAD, '<div class="wrap">',
         "<h1>RealPDE calibration log</h1>",
         '<p class="lede">Every submission, the prediction recorded before it was uploaded, '
         'and what the board returned &mdash; then the local experiments behind them and the '
         'measured correction from our validation set to the hidden one.</p>',
         f'<p class="stamp">generated by leaderboard/lb.py &middot; {len(B)} submissions '
         f'&middot; {len(Rn)} local arms &middot; subscore weights refit on '
         f'{len([r for r in B if r.get("actual")])} scored rounds</p>']

    o.append('<div class="strip">')
    for k, v, n in [
            ("best, track 1", f"{best['actual']['final']:.3f}", f"round {best['round']}"),
            ("leader", "81.761", "np-user &middot; 08-25"),
            ("gap", f"{81.76136 - T[-1]['actual']['final']:.3f}",
             "&asymp;&thinsp;11.8% of error"),
            ("last 4 @ const t", f"{span:.3f}", "span, four rounds"),
            ("prediction rms", f"{np.sqrt(np.mean(np.square(resid))):.3f}",
             f"{len(resid)} predicted rounds"),
            ("resolution limit", f"{rms:.1f}%", "of error, to transfer")]:
        o.append(f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div>'
                 f'<div class="n">{n}</div></div>')
    o.append("</div>")

    o.append(f'<div class="thesis"><span class="eyebrow">read this first</span>'
             f'At constant board latency the last four rounds span <b>{span:.3f}</b> points. '
             f'The model has not measurably improved since 2026-08-08 &mdash; what improved is '
             f'latency, one composition constant, and our knowledge of the scoring rule. '
             f'Section&nbsp;3 measures why: the proxy predicts the board&rsquo;s <i>level</i> '
             f'to &plusmn;1&ndash;3%, but explains almost none of its <i>movement</i>, and every '
             f'gain we have chased has been smaller than that residual.</div>')

    # 1 -----------------------------------------------------------------------------------
    o.append('<section><div class="head"><h2>Submissions</h2>'
             '<span class="eyebrow">board final</span>'
             '<p>The prediction is recorded before upload; a prediction written afterwards is '
             'worth nothing. Colour is the size of the miss.</p></div>'
             '<div class="scroll"><table><thead><tr>'
             "<th class='l'>Round</th><th>Tr</th><th class='l'>Date</th><th class='l t'>Thesis"
             "</th><th>Pred</th><th>Actual</th><th>Miss</th></tr></thead><tbody>")
    for r in B:
        a, p = r.get("actual"), (r.get("pred") or {}).get("final")
        miss, cls = "", ""
        act = "<span class='d'>pending</span>"
        pv = "&mdash;"
        if p is not None:
            pv = f"{p:.3f}"
        if a:
            act = f"{a['final']:.3f}"
        if a and p:
            miss = f"{a['final'] - p:+.3f}"
            d = abs(a["final"] - p)
            cls = " class='g'" if d < 0.05 else (" class='b'" if d > 0.12 else " class='w'")
        o.append(f"<tr><td class='l'><b>{r['round']}</b></td><td>{r['track']}</td>"
                 f"<td class='l d'>{r['date']}</td><td class='l t'>{r.get('thesis') or ''}</td>"
                 f"<td class='d'>{pv}</td><td><b>{act}</b></td>"
                 f"<td{cls}>{miss}</td></tr>")
    o.append("</tbody></table></div>")
    o.append("<div class='scroll' style='margin-top:14px'><table><thead><tr><th class='l'>Round"
             "</th>" + "".join(f"<th>{n}</th>" for n in SUB) + "</tr></thead><tbody>")
    for r in B:
        a, p = r.get("actual"), r.get("pred") or {}
        if not a:
            continue
        cells = []
        for n in SUB:
            if n in p and n in a:
                dd = a[n] - p[n]
                c = "g" if abs(dd) < 0.15 else ("b" if abs(dd) > 0.4 else "w")
                cells.append(f"<td><span class='d'>{p[n]:.2f}</span> "
                             f"<span class='{c}'>{a[n]:.3f}</span></td>")
            else:
                cells.append(f"<td>{a[n]:.3f}</td>")
        o.append(f"<tr><td class='l'><b>{r['round']}</b></td>" + "".join(cells) + "</tr>")
    o.append("</tbody></table></div>"
             "<p class='stamp' style='margin:10px 0 0'>predicted &middot; actual</p></section>")

    # 2 -----------------------------------------------------------------------------------
    o.append('<section><div class="head"><h2>Progress at constant latency</h2>'
             '<span class="eyebrow">time_score held at the median</span>'
             '<p>Board <code>time_score</code> swings 2.8 points across the rounds and is only '
             'partly ours to control, so the raw ranking conflates model quality with '
             'eval-machine luck. Holding it fixed is the honest curve.</p></div>')
    o.append(_progress_svg(T, norm, ref) + "</section>")

    # 3 -----------------------------------------------------------------------------------
    o.append('<section><div class="head"><h2>Local &rarr; board correction</h2>'
             '<span class="eyebrow">% of error</span>'
             '<p>Proxy = the held-out <i>twin</i> members on the AoA&nbsp;15&ndash;20 slice of '
             '<code>val_unseen</code>, scored at the ALPHA that package ships. '
             '<code>r</code> is board error divided by proxy error.</p></div>'
             "<div class='scroll'><table><thead><tr><th class='l'>Round</th>"
             + "".join(f"<th>proxy {m}</th>" for m in M)
             + "".join(f"<th>board {m}</th>" for m in M)
             + "".join(f"<th>r_{m}</th>" for m in M) + "</tr></thead><tbody>")
    rat = {m: [] for m in M}
    for r in R:
        pp, aa = r["proxy"], {m: err(r["actual"][m]) for m in M}
        for m in M:
            rat[m].append(aa[m] / pp[m])
        o.append(f"<tr><td class='l'><b>{r['round']}</b></td>"
                 + "".join(f"<td class='d'>{pp[m]:.5f}</td>" for m in M)
                 + "".join(f"<td class='d'>{aa[m]:.5f}</td>" for m in M)
                 + "".join(f"<td><b>{aa[m]/pp[m]:.3f}</b></td>" for m in M) + "</tr>")
    o.append("<tr><td class='l d'>mean &plusmn;&frac12;range</td>"
             + "<td></td>" * 6
             + "".join(f"<td class='d'>{np.mean(rat[m]):.3f}&#177;"
                       f"{(max(rat[m])-min(rat[m]))/2:.3f}</td>" for m in M)
             + "</tr></tbody></table></div>")
    o.append(f'<div class="thesis"><span class="eyebrow">the finding</span>'
             f'<b>The levels calibrate; the deltas do not.</b> Those ratios are stable to '
             f'&plusmn;1&ndash;3%, so the proxy predicts the board&rsquo;s level well. Round to '
             f'round it explains almost none of the board&rsquo;s movement &mdash; residual rms '
             f'<b>{rms:.1f}%</b> of the error, range {min(res):+.1f} to {max(res):+.1f}%, and '
             f'<code>mvpe</code> inverts sign between R12 and R13. <b>A local delta below '
             f'~{rms:.0f}% of the error does not survive the trip and must not be used to choose '
             f'a package</b> &mdash; and every improvement measured since 2026-08-08 has been '
             f'smaller than it.</div>')
    o.append(_residual_svg(rows, rms))
    o.append("<p class='stamp' style='margin:14px 0 0'>also calibrated &mdash; sps element-tail "
             "f 1.360 (T1) / 1.350 (T2), calib_band13.py &middot; board ms = 2.616 + "
             "0.435&times;inv + 0.803&times;local, rms 0.690 over 7 rounds, residuals ordered "
             "by date &middot; one invocation = 0.52&ndash;0.56 board ms = 0.036 final</p>"
             "</section>")

    # 4 -----------------------------------------------------------------------------------
    n_ens = sum(1 for r in Rn if r.get("ensemble_tested"))
    o.append(f'<section><div class="head"><h2>Experiment ledger</h2>'
             f'<span class="eyebrow">{len(Rn)} arms &middot; {n_ens} ensembled</span>'
             f'<p><b>ens</b> marks an arm measured at the ensemble size that actually ships. It '
             f'is the most predictive column here: <code>ozdo</code> was +3.84&sigma; per model '
             f'and &minus;0.012 at four members, and <code>--var-mlp</code>&rsquo;s 3.9% '
             f'fixed-field TKE cut became +0.08. An arm that has not been ensembled has not been '
             f'measured.</p></div>'
             "<div class='scroll'><table><thead><tr><th class='l'>Verdict</th>"
             "<th class='l'>Date</th><th class='l'>Arm</th><th class='l'>Direction</th>"
             "<th>ens</th><th class='l t'>Effect</th><th class='l'>Board</th>"
             "</tr></thead><tbody>")
    order = {"SHIPPED": 0, "OPEN": 1, "MARGINAL": 2, "CLOSED": 3, "REJECTED": 4}
    for r in sorted(Rn, key=lambda x: (order.get(x["verdict"], 9), x["date"])):
        e = ("<span class='g'>&#10003;</span>" if r.get("ensemble_tested")
             else "<span class='d'>&mdash;</span>")
        o.append(f"<tr><td class='l'><span class='chip {r['verdict']}'>{r['verdict']}</span>"
                 f"</td><td class='l d'>{r['date']}</td>"
                 f"<td class='l'><code>{r['arm']}</code></td>"
                 f"<td class='l d'>{r['direction']}</td><td>{e}</td>"
                 f"<td class='l t'>{r['effect']}</td>"
                 f"<td class='l'>{r.get('shipped') or '<span class=d>&mdash;</span>'}</td></tr>")
    o.append("</tbody></table></div></section>")

    o.append("<footer>subscore weights, nine rounds, max residual 0.035 &nbsp;&mdash;&nbsp; "
             + " &nbsp; ".join(f"{n} {v:.3f}" for n, v in zip(SUB, w))
             + "<br>sources: leaderboard/board.jsonl &middot; leaderboard/runs.jsonl "
               "&middot; Track_1/final_v9.py &middot; Track_1/proxy_calib.py</footer></div>")
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
    section_board(B, w)
    section_progress(B, w)
    section_calib(B)
    section_runs(Rn)
    if a.html:
        html(B, Rn, w, os.path.join(HERE, "index.html"))


if __name__ == "__main__":
    main()
