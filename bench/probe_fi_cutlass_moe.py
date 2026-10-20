#!/usr/bin/env python3
"""Probe: does flashinfer's CUTLASS fused MoE (sm100 module) JIT-compile for
sm_110a and execute correctly on a Jetson Thor?

Mirrors the nvfp4 path of vllm's FlashInferExperts.apply():
  - bf16 activations, nvfp4 weights (uint8 packed), fp8 block scales swizzled
    with vllm's swizzle_blockscale, [w1,w3]->[w3,w1] reorder for gated act
  - cutlass_fused_moe(quant_scales=[a1g, w1s, g1, a2g, w2s, g2])
Compares against a bf16 reference MoE with the same routing.
"""
import time
import torch
import flashinfer
from flashinfer.fused_moe.core import cutlass_fused_moe

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)

torch.manual_seed(0)
dev = "cuda"
M, K, N, E, TOPK = 128, 512, 256, 8, 4

print(f"capability: {torch.cuda.get_device_capability()}")
print(f"flashinfer: {flashinfer.__version__}")

# ---- reference weights (bf16) ----
w13_ref = (torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16) * 0.05)
w2_ref = (torch.randn(E, K, N, device=dev, dtype=torch.bfloat16) * 0.05)
x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5)

# ---- nvfp4-quantize weights (blocks of 16 along last dim, e4m3 scales) ----
def q_block(t):
    shp = t.shape
    tb = t.float().view(*shp[:-1], shp[-1] // 16, 16)
    amax = tb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = scale.float()
    # quantize-dequantize to e2m1 grid via round-trip through float8 math
    q = (tb / sf).clamp(-6.0, 6.0)
    # e2m1 grid: 0,0.5,1,1.5,2,3,4,6
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    sign = q.sign()
    mag = q.abs().unsqueeze(-1)
    idx = (mag - grid).abs().argmin(dim=-1)
    deq = sign * grid[idx]  # dequantized value
    packed_val = deq
    # pack two e2m1 nibbles per byte (low nibble = even element)
    def to_nib(v):
        m = v.abs()
        s = (v < 0).to(torch.uint8)
        g = torch.stack([torch.zeros_like(m), torch.full_like(m, 0.5),
                         torch.ones_like(m), torch.full_like(m, 1.5),
                         torch.full_like(m, 2.0), torch.full_like(m, 3.0),
                         torch.full_like(m, 4.0), torch.full_like(m, 6.0)], -1)
        idx2 = (m.unsqueeze(-1) - g).abs().argmin(dim=-1).to(torch.uint8)
        return (idx2 & 0x7) | (s << 3)
    nib = to_nib(packed_val)  # [..., 16]
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    packed = (lo | (hi << 4)).reshape(*shp[:-1], shp[-1] // 2)
    scale_out = scale.squeeze(-1).reshape(*shp[:-1], shp[-1] // 16)
    return packed.contiguous(), scale_out.contiguous(), (deq * sf).reshape(shp)

w13_q, w13_s, w13_dq = q_block(w13_ref)
w2_q, w2_s, w2_dq = q_block(w2_ref)
print(f"w13_q {tuple(w13_q.shape)} {w13_q.dtype}, w13_s {tuple(w13_s.shape)} {w13_s.dtype}")

# ---- vLLM FI-CUTLASS prep: reorder [w1,w3]->[w3,w1] + swizzle scales ----
def reorder(t):
    e, m, k = t.shape
    return torch.cat([t[:, N:, :], t[:, :N, :]], dim=1).contiguous()

w13_q_r = reorder(w13_q)
w13_s_r = reorder(w13_s)
w13_dq_r = reorder(w13_dq)
w13_s_sw = swizzle_blockscale(w13_s_r)
w2_s_sw = swizzle_blockscale(w2_s)
pad = w13_s_sw.shape[1] - w13_q_r.shape[1]
if pad > 0:
    raise SystemExit(f"unexpected pad {pad}")
print(f"swizzled: {tuple(w13_s_sw.shape)} {w13_s_sw.dtype}, {tuple(w2_s_sw.shape)}")

# ---- routing ----
logits = torch.randn(M, E, device=dev)
topv, topi = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
topw = (topv / topv.sum(-1, keepdim=True)).float().contiguous()
topi = topi.to(torch.int).contiguous()

# ---- reference MoE in bf16 (dequantized weights, same routing) ----
ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
for m in range(M):
    for j in range(TOPK):
        e = int(topi[m, j])
        g = x[m].float() @ w13_dq_r[e, :N].T.float()
        u = x[m].float() @ w13_dq_r[e, N:].T.float()
        h = torch.nn.functional.silu(g) * u
        o = h @ w2_dq[e].T.float()
        ref[m] += float(topw[m, j]) * o

# ---- flashinfer CUTLASS fused MoE ----
gsc = torch.full((E,), 1.0, device=dev, dtype=torch.float32)
t0 = time.time()
out = cutlass_fused_moe(
    input=x,
    token_selected_experts=topi,
    token_final_scales=topw,
    fc1_expert_weights=w13_q_r.view(torch.long),
    fc2_expert_weights=w2_q.view(torch.long),
    fc1_expert_biases=None,
    fc2_expert_biases=None,
    output_dtype=torch.bfloat16,
    quant_scales=[
        gsc,                             # gemm1 activation global scale
        w13_s_sw.view(torch.int32),      # gemm1 weight block scales (fp8 bit-view)
        gsc.clone(),                    # gemm1 dequant scale
        gsc.clone(),                    # gemm2 activation global scale
        w2_s_sw.view(torch.int32),      # gemm2 weight block scales
        gsc.clone(),                    # gemm2 dequant scale
    ],
)
torch.cuda.synchronize()
print(f"kernel call ok in {time.time()-t0:.1f}s; out type={type(out).__name__}")

out = out[0] if isinstance(out, list) else out
print(f"out {tuple(out.shape)} {out.dtype}")
o = out.float()
finite = torch.isfinite(o).all().item()
err = (o - ref).abs().max().item()
rel = err / ref.abs().max().item()
cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
print(f"finite={finite} max_abs_err={err:.4f} rel={rel:.4f} cos={cos:.6f}")
print("out sample:", o[0, :4].tolist())
print("ref sample:", ref[0, :4].tolist())
print("PROBE PASS" if finite and cos > 0.98 else "PROBE FAIL")
