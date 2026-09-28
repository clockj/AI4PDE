import sys, re
p = sys.argv[1]; s = open(p).read()
i = s.index("SIGPOW = "); j = s.index("SIGEMA_DDOF = 0 ")
s = s[:i] + s[j:]
i = s.index("        self.sigpow = None\n"); j = s.index("        self.Ad = torch.from_numpy(_A).to(device)\n")
s = s[:i] + s[j:]
i = s.index("            if self.sigpow is not None:\n"); j = s.index("            if self.err is not None and er is not None and self.eref")
s = s[:i] + s[j:]
s = s.replace(" sigpow {SIGPOW}", "")
assert "SIGPOW" not in s and "sigpow" not in s
open(p, "w").write(s)
