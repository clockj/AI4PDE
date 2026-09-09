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


def _sidebar(B, Rn):
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
                 f"local arms<br>weights refit on {n_scored} scored rounds",
                 f"由 leaderboard/lb.py 生成<br>{len(B)} 次提交 "
                 f"&middot; {len(Rn)} 条本地实验<br>"
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
    o.append(f'<line x1="{L}" y1="{y(81.76136):.1f}" x2="{W-Rp}" y2="{y(81.76136):.1f}" '
             f'stroke="var(--warm)" stroke-width="1.2" stroke-dasharray="5 4"/>')
    o.append(tsvg(f"{W-Rp+8}", f"{y(81.76136)+3.5:.1f}", "leader 81.761",
                  "领先者 81.761", 'fill="var(--warm)"'))
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


def html(B, Rn, w, path):
    T = [r for r in B if r.get("actual") and r["track"] == 1]
    ref = float(np.median([r["actual"]["time"] for r in T]))
    norm = [r["actual"]["final"] + (ref - r["actual"]["time"]) * w[3] for r in T]
    resid = [r["actual"]["final"] - r["pred"]["final"] for r in B
             if r.get("actual") and (r.get("pred") or {}).get("final")]
    R, rat0, rawrows, res, rms = calib(B)
    rows = [(f"{a}&#8594;{b}", m, pd, bd, rr) for a, b, m, pd, bd, rr in rawrows]
    span = max(norm[-4:]) - min(norm[-4:])
    best = max(T, key=lambda r: r["actual"]["final"])
    # Best on either board, and the constant-latency series for Track 2 alongside Track 1's.
    scored = [r for r in B if r.get("actual")]
    best_any = max(scored, key=lambda r: r["actual"]["final"])
    T2 = [r for r in scored if r["track"] == 2]
    norm2 = [r["actual"]["final"] + (ref - r["actual"]["time"]) * w[3] for r in T2]
    gap = 81.76136 - best_any["actual"]["final"]
    # Exchange rate, re-priced with the FREE eighteen-round weights (round 17): 10 % of error
    # removed is worth ~0.95 `final` (Track_1/exchange_v9.py, round 9's instrument).  The
    # total barely moved when the constraint came off; its SPLIT across subscores did.
    gap_pct = gap / 0.0948
    best2 = max(T2, key=lambda r: r["actual"]["final"])

    o = [HTML_HEAD, '<div class="app">', _sidebar(B, Rn), '<main><div class="wrap">',
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
             f"round {best['round']}", f"轮次 {best['round']}"),
            ("best, track 2", "最佳，track 2", f"{best2['actual']['final']:.3f}",
             f"round {best2['round']}", f"轮次 {best2['round']}"),
            ("leader", "领先者", "81.761",
             "np-user &middot; 08-25", "np-user &middot; 08-25"),
            ("gap", "差距", f"{gap:.3f}",
             f"&asymp;&thinsp;{gap_pct:.1f}% of error",
             f"约为误差的 {gap_pct:.1f}%"),
            ("track 1, last 4 @ const t",
             "track 1，恒定延迟下最近四轮",
             f"{span:.3f}", "span, four rounds", "四轮跨度"),
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
        o.append(f'<tr><td class="l"><b>{r["round"]}</b></td><td>{r["track"]}</td>'
                 f'<td class="l d">{r["date"]}</td>'
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
                 "calib_band13.py &middot; board ms = 2.616 + 0.435&times;inv + "
                 "0.803&times;local, rms 0.690 over 7 rounds, residuals ordered by date "
                 "&middot; one invocation = 0.52&ndash;0.56 board ms = 0.036 final "
                 "&middot; w_time = 0.0956, measured directly from R13 vs R14 (identical "
                 "error subscores, <code>time</code> alone moved)",
                 "其他已校准常数 &mdash; sps 元素"
                 "尾部 f 1.360（T1）/ 1.350（T2），"
                 "calib_band13.py &middot; board ms = 2.616 + 0.435&times;inv + "
                 "0.803&times;local，7 轮 rms 0.690，残差按"
                 "日期排序 &middot; 一次调用 = "
                 "0.52&ndash;0.56 board ms = 0.036 final &middot; w_time = 0.0956，"
                 "由 R13 与 R14 直接测得（误差"
                 "子分数完全相同，只有 "
                 "<code>time</code> 变了）")
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
        html(B, Rn, w, HTML_OUT)


if __name__ == "__main__":
    main()
