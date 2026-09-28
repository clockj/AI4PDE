"""Apply the K1 Track-1 knobs to an EXTRACTED package (same text edits as the wt_1 clone; for measurement only --
ship through build_round13.sh). usage: apply_k1_full.py <pkgdir> [--pin N] [--cap N] [--ln] [--warm]"""
import sys, os
pkg = sys.argv[1]; a = sys.argv[2:]
def opt(k, d=0):
    return int(a[a.index(k) + 1]) if k in a else d
pin, cap, ln, warm = opt("--pin"), opt("--cap"), "--ln" in a, "--warm" in a
p = os.path.join(pkg, "submission.py"); s = open(p).read()
b0 = ("        pred = np.zeros((len(xin), 20, 32, 64, 3), dtype=np.float32)\n"
      "        sig = np.empty((len(xin), 1, 32, 64, 2), dtype=np.float32)\n"
      "        lower = np.empty_like(pred)\n        upper = np.empty_like(pred)\n")
assert s.count(b0) == 1
s = s.replace(b0, "        _pb = _pool_take(len(xin))\n        if _pb is not None:\n            pred, lower, upper = _pb\n        else:\n"
      "            pred = np.zeros((len(xin), 20, 32, 64, 3), dtype=np.float32)\n            lower = np.empty_like(pred)\n"
      "            upper = np.empty_like(pred)\n        sig = np.empty((len(xin), 1, 32, 64, 2), dtype=np.float32)\n", 1)
b1 = "\ndef predict(input_array, metadata=None):\n"
assert s.count(b1) == 1
s = s.replace(b1, '''
PIN_OUT = %d
_POOL = None
_POOL_I = 0


def _pool_init():
    global _POOL
    if not (PIN_OUT and torch.cuda.is_available()):
        return
    try:
        _POOL = [torch.zeros((int(PIN_OUT), 20, 32, 64, 3), dtype=torch.float32).pin_memory() for _ in range(3)]
    except Exception:
        _POOL = None


def _pool_take(n):
    global _POOL_I
    if _POOL is None or _WARMING or _POOL_I + n > int(PIN_OUT):
        return None
    i = _POOL_I
    _POOL_I = i + n
    return tuple(t.numpy()[i:i + n] for t in _POOL)


_pool_init()


def predict(input_array, metadata=None):
''' % pin, 1)
c0 = "    batch = _pick_batch(models, dev)\n"
assert s.count(c0) == 1
if cap:
    s = s.replace(c0, c0 + "    batch = min(int(batch), %d)\n" % cap, 1)
d0 = "WARM_TAILS = False\n"
assert s.count(d0) == 1
s = s.replace(d0, ("WARM_TAILS = True\n" if warm else d0) + "_fno.LN_MANUAL = %s\n" % ("True" if ln else "False"), 1)
open(p, "w").write(s)
p = os.path.join(pkg, "_fno.py"); s = open(p).read()
a1 = ("    def forward(self, x):\n        return torch.addcmul(self.b, _F.layer_norm(x, x.shape[-1:], eps=self.eps), self.w)\n")
assert s.count(a1) == 1
s = s.replace(a1, "    def forward(self, x):\n        if LN_MANUAL and x.is_cuda:\n            x32 = x.float()\n"
      "            var, mu = torch.var_mean(x32, dim=-1, keepdim=True, unbiased=False)\n"
      "            return torch.addcmul(self.b, (x32 - mu).mul_(torch.rsqrt(var + self.eps)), self.w)\n"
      "        return torch.addcmul(self.b, _F.layer_norm(x, x.shape[-1:], eps=self.eps), self.w)\n", 1)
b = "class _StkLN(nn.Module):\n"
s = s.replace(b, "LN_MANUAL = False\n\n\n" + b, 1)
open(p, "w").write(s)
print("patched", pkg, dict(pin=pin, cap=cap, ln=ln, warm=warm))
