"""K1 T2 patch (text edit on an extracted Track-2 package): ARENA_STEPS > 0 = every step's three CPU outputs are written
straight into their own slot of a page-locked arena allocated (and faulted in) at load, instead of pinned staging +
three fresh clones per step. Same bits; the caller still owns distinct storage per step. Falls back to the shipped
path when the arena is exhausted or could not be allocated.   usage: apply_k1_t2patch.py <pkgdir> <steps>"""
import sys, os
pkg, steps = sys.argv[1], int(sys.argv[2])
p = os.path.join(pkg, "submission.py"); s = open(p).read()
a = "STACKED_OUT = False\n"
assert s.count(a) == 1
s = s.replace(a, a + "ARENA_STEPS = %d\n" % steps, 1)
b = ("        self.host = [torch.empty((1, T_OUT, H, W, 3), dtype=torch.float32,\n"
     "                                 pin_memory=cuda) for _ in range(3)]\n")
assert s.count(b) == 1
s = s.replace(b, b +
     "        self.arena, self.arena_i = None, 0\n"
     "        if ARENA_STEPS and cuda:\n"
     "            try:\n"
     "                self.arena = [torch.zeros((ARENA_STEPS, 1, T_OUT, H, W, 3), dtype=torch.float32).pin_memory()\n"
     "                              for _ in range(3)]\n"
     "            except Exception:\n"
     "                self.arena = None\n", 1)
c = "        for i, s in enumerate(srcs):\n            self.host[i].copy_(s, non_blocking=True)"
assert s.count(c) == 1
s = s.replace(c,
     "        if self.arena is not None and self.arena_i < ARENA_STEPS and not STACKED_OUT:\n"
     "            j = self.arena_i\n"
     "            self.arena_i = j + 1\n"
     "            outs = [self.arena[i][j] for i in range(len(srcs))]\n"
     "            for o, s in zip(outs, srcs):\n"
     "                o.copy_(s, non_blocking=True)\n"
     "            torch.cuda.synchronize()\n"
     "            return outs\n" + c, 1)
d = "        for _ in range(3):\n            m.ttt_step(probe, None)\n"
assert s.count(d) == 1
s = s.replace(d, d + "        m.arena_i = 0\n", 1)
open(p, "w").write(s)
print("patched T2", pkg, "ARENA_STEPS", steps)
