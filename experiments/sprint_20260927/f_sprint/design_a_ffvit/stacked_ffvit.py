class StackedFFViT(nn.Module):
    """K FFViT2d hybrids (same structure) in ONE invocation (final-day design A, executor EA).

    The F-FNO streams run as a StackedFFNO, the ViT streams as a StackedViT (its per-block math is
    reproduced inline, unchanged), and the cross-stitch links fold the member axis the same way:
    LayerNorm once + per-member affine, the 4x4/4 patch conv and the 1x1 conv as grouped convs over
    member-major channels, the Linears as stacked einsums. Every member's field and sigma are formed
    exactly as FFViT2d forms them; the group returns the member mean of the field and
    log(mean_k exp(clamp(logstd_k, -8, 2))) -- the StackedFFNO contract -- with n_members = K."""

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.n_members = k = len(members)
        sd0 = first.state_dict()
        for m in members[1:]:
            sd = m.state_dict()
            assert sd.keys() == sd0.keys(), "state_dict keys differ"
            for key, v in sd.items():
                assert v.shape == sd0[key].shape, f"{key} shape mismatch"
        self.sf = StackedFFNO([m.f for m in members])
        self.sv = StackedViT([m.v for m in members]).to(first.mix.device)
        self.width, self.pp = first.width, first.v.patch
        self.gh, self.gw = first.v.gh, first.v.gw
        self.fpad = tuple(first.f.PAD)
        self.register_buffer("grid", first.f.grid.detach().clone(), persistent=False)
        self.links = [int(i) for i in first.f2v.keys()]
        with torch.no_grad():
            self.register_buffer("mix", torch.stack([m.mix.detach() for m in members]), persistent=False)
            self.lk = nn.ModuleDict()
            for i in self.links:
                key = str(i)
                f2v = [m.f2v[key] for m in members]
                v2f = [m.v2f[key] for m in members]
                mod = nn.Module()
                mod.fln = _StkLN([s[0] for s in f2v])                    # (K,B,H,W,40)
                mod.c1 = _StkConv([s[1] for s in f2v])                   # grouped 4x4/4
                mod.c2 = _StkConv([s[3] for s in f2v])                   # grouped 1x1
                mod.vln_eps = float(v2f[0][0].eps)
                mod.register_buffer("vln_w", torch.stack([s[0].weight.detach() for s in v2f])[:, None, None]
                                    .clone(), persistent=False)
                mod.register_buffer("vln_b", torch.stack([s[0].bias.detach() for s in v2f])[:, None, None]
                                    .clone(), persistent=False)
                mod.register_buffer("l1_w", torch.stack([s[1].weight.detach() for s in v2f]).clone(),
                                    persistent=False)
                mod.register_buffer("l1_b", torch.stack([s[1].bias.detach() for s in v2f])[:, None, None]
                                    .clone(), persistent=False)
                mod.register_buffer("l2_w", torch.stack([s[3].weight.detach() for s in v2f]).clone(),
                                    persistent=False)
                mod.register_buffer("l2_b", torch.stack([s[3].bias.detach() for s in v2f])[:, None, None]
                                    .clone(), persistent=False)
                self.lk[key] = mod
        self.self_norm = bool(getattr(first, "self_norm", False))

    def set_spectral_path(self, path: str = "fft"):
        self.sf.set_spectral_path(path)
        return self

    def _vblock(self, i, t):
        sv, dim = self.sv, self.sv.dim
        k, b, n = t.shape[0], t.shape[1], t.shape[2]
        hd = dim // sv.heads
        lnorm = torch.nn.functional.layer_norm
        q = lnorm(t, (dim,)) * sv.n1_w[i] + sv.n1_b[i]
        qkv = sv._lin(q, sv.qkv_w[i], sv.qkv_b[i])
        qkv = qkv.reshape(k * b, n, 3, sv.heads, hd).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        a = a.transpose(1, 2).reshape(k, b, n, dim)
        t = t + sv._lin(a, sv.op_w[i], sv.op_b[i])
        h = lnorm(t, (dim,)) * sv.n2_w[i] + sv.n2_b[i]
        h = torch.nn.functional.gelu(sv._lin(h, sv.f1_w[i], sv.f1_b[i]))
        return t + sv._lin(h, sv.f2_w[i], sv.f2_b[i])

    def forward(self, prev, detach_var: bool = False):
        sf, sv = self.sf, self.sv
        b, k, wd, pp = len(prev), self.n_members, self.width, self.pp
        dim, gh, gw = sv.dim, self.gh, self.gw
        x = torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1)
        h = sf.lift2(_F.gelu(sf.lift0(x)))
        h = _F.pad(h, (0, self.fpad[1], 0, self.fpad[0]))
        t = sv.embed(x).reshape(b, k, dim, gh * gw).permute(1, 0, 3, 2)
        t = (t + sv.pos).contiguous()                                        # (k, b, N, dim)
        for i in range(sv.depth):
            h = sf.layers[i](h)
            t = self._vblock(i, t)
            key = str(i)
            if key in self.lk:
                L = self.lk[key]
                hc = h[..., :H, :W].reshape(b, k, wd, H, W).permute(1, 0, 3, 4, 2)   # (k,b,H,W,wd)
                z = L.fln(hc).permute(1, 0, 4, 2, 3).reshape(b, k * wd, H, W)
                dt = L.c2(_F.gelu(L.c1(z)))                                          # (b, k*dim, gh, gw)
                dt = dt.reshape(b, k, dim, gh * gw).permute(1, 0, 3, 2)              # (k, b, N, dim)
                q = torch.nn.functional.layer_norm(t, (dim,), eps=L.vln_eps) * L.vln_w + L.vln_b
                q = _F.gelu(sv._lin(q, L.l1_w, L.l1_b))
                q = sv._lin(q, L.l2_w, L.l2_b)                                       # (k, b, N, wd*pp*pp)
                dh = (q.reshape(k, b, gh, gw, wd, pp, pp)
                      .permute(1, 0, 4, 2, 5, 3, 6).reshape(b, k * wd, H, W))
                t = t + dt
                h = h + _F.pad(dh, (0, self.fpad[1], 0, self.fpad[0]))
        # F-FNO heads, per member: (k, b, H, W, c)
        hc = h[..., :H, :W]
        tf = sf.norm_out(hc.reshape(b, k, wd, H, W).permute(1, 0, 3, 4, 2))
        ff = sf.hf[1](_F.gelu(sf.hf[0](tf))).float().permute(0, 1, 4, 2, 3)        # (k, b, CH, H, W)
        # ViT heads, per member
        tv = torch.nn.functional.layer_norm(t, (dim,)) * sv.no_w + sv.no_b
        tv = sv._lin(tv, sv.pj_w, sv.pj_b)
        wid = sv.wid
        hv = (tv.reshape(k, b, gh, gw, wid, pp, pp)
              .permute(1, 0, 4, 2, 5, 3, 6).reshape(b, k * wid, H, W))
        fv = sv.head_field(hv).float().reshape(b, k, CH, H, W).transpose(0, 1)    # (k, b, CH, H, W)
        s = torch.sigmoid(self.mix).reshape(k, 1, 1, 1, 1)
        field = (s * ff + (1.0 - s) * fv).mean(0)
        logstd = None
        if sf.hl is not None and sv.head_logstd is not None:
            lf = sf.hl[1](_F.gelu(sf.hl[0](tf))).float().permute(0, 1, 4, 2, 3)    # (k, b, C, H, W)
            lv = sv.head_logstd(hv).float().reshape(b, k, C, H, W).transpose(0, 1)
            mx = self.mix.reshape(k, 1, 1, 1, 1)
            ls = torch.logaddexp(lf + _F.logsigmoid(mx), lv + _F.logsigmoid(-mx))
            logstd = ls.clamp(-8.0, 2.0).exp().mean(0).log()
        return field, logstd
