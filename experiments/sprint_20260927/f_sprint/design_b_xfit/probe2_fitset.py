"""Probe 3: does the per-element base generalise ACROSS trajectories, and does a larger fit set help?
  insample : base fitted on all three AoA 15-20 trajectories, scored on them (the number every in-sample read reports)
  LOTO-2   : V0 (fit on the other two AoA 15-20 trajectories)
  LOTO-7   : fit on all seven other val_unseen trajectories (any AoA)
  LOTO-hi+10: fit on the other AoA>=10 trajectories
Only the AoA 15-20 columns are comparable across rows (the 'all' column of insample is partly in-sample)."""
import numpy as np
from bandlab import Lab
L = Lab()
ref = L.run(L.y0); L.table("LOTO-2 (V0)", ref)
hi = np.isin(L.traj, L.hi)
L.table("insample hi-fit", L.run(L.y0, fitmask={t: hi for t in L.names}), ref)
L.table("LOTO-7 all-AoA", L.run(L.y0, fitmask={t: L.traj != t for t in L.names}), ref)
L.table("LOTO AoA>=10", L.run(L.y0, fitmask={t: (L.traj != t) & (L.aoa >= 10) for t in L.names}), ref)
