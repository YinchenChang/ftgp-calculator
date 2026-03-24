"""
FTGP Token Generation Calculator
Computation chain: Config → WP_Param → TL_Param → Revenue_Model
Faithfully reproduced from 20260319_FTGP_FIXED.xlsx
"""
import streamlit as st
import streamlit.components.v1 as components
import math
import json
import pandas as pd

st.set_page_config(page_title="FTGP Token Generation Calculator", layout="wide")
st.title("Tokenomics")
st.subheader("From Token Generation to Profitability (FTGP)")
st.caption("Computation chain: Config → WP_Param → TL_Param → Revenue_Model")

# ─── RACK PRESETS ───────────────────────────────────────────────────────────
RACK_PRESETS = {
    "Vera Rubin NVL72": dict(
        cpu_type="Vera CPU", n_cpu=36, gpu_type="Rubin GPU", n_gpu=72,
        nvlink_bw=260, nvlink_c2c_bw=65, gpu_mem=20.7, mem_bw=1580,
        fp4_inf=3600, fp4_train=2520, fp8=1260, fp16=288,
        power_per_rack=190, rack_price=6_000_000,
    ),
    "GB200 NVL72": dict(
        cpu_type="Grace CPU", n_cpu=36, gpu_type="Blackwell B200", n_gpu=72,
        nvlink_bw=130, nvlink_c2c_bw=32.4, gpu_mem=13.5, mem_bw=576,
        fp4_inf=648, fp4_train=648, fp8=324, fp16=162,
        power_per_rack=139, rack_price=3_000_000,
    ),
}

# ─── PRICING TIERS ──────────────────────────────────────────────────────────
PRICING_TIERS = [
    ("Nano (<50B)", 0, 0.10, 0.40),
    ("Small (50B–200B)", 50e9, 0.25, 2.00),
    ("Mid (200B–500B)", 200e9, 1.25, 10.00),
    ("Frontier (500B–1.5T)", 500e9, 2.50, 10.00),
    ("Premium (>1.5T)", 1500e9, 5.00, 25.00),
]


def match_pricing_tier(params):
    matched_idx = 0
    for i, (_, threshold, _, _) in enumerate(PRICING_TIERS):
        if params >= threshold:
            matched_idx = i
    tier = PRICING_TIERS[matched_idx]
    return tier[0], tier[2], tier[3]


# ─── SIDEBAR: CONFIG INPUTS ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Config")

    st.subheader("AI DC Spec")
    total_power_gw = st.number_input("Total Power (GW)", value=1.0, step=0.1, format="%.1f")
    total_power = total_power_gw * 1e9
    pue = st.number_input("PUE", value=1.2, step=0.05, format="%.2f")
    rack_type = st.selectbox("Rack Type", list(RACK_PRESETS.keys()))
    rp = RACK_PRESETS[rack_type]

    with st.expander("▶ KEY INPUTS — Model & Tokens", expanded=True):
        input_tokens = st.number_input("Input Tokens", value=4000, step=100)
        output_tokens = st.number_input("Output Tokens", value=1000, step=100)

    st.subheader("Model Spec")
    param_presets = {"1.5T": 1.5e12, "1T": 1e12, "500B": 500e9, "200B": 200e9, "70B": 70e9}
    param_preset = st.selectbox("Parameters (preset)", list(param_presets.keys()), index=1)
    parameters = st.number_input("Parameters (exact)", value=int(param_presets[param_preset]),
                                 step=int(1e9), format="%d")
    vocab_size = st.number_input("Vocab Size (V)", value=128000, step=1000)
    if parameters < 100_000_000:
        d_model = parameters ** 0.25
    else:
        d_model = 2**11 * 10 ** math.log10(parameters / 1e11)
    d_model = round(d_model)
    n_layers = st.number_input("n_layers", value=128, step=1)
    n_heads = st.number_input("n_heads", value=160, step=1)
    d_head = d_model / n_heads if n_heads > 0 else 0
    d_ff = (parameters / n_layers - 4 * d_model**2) / (3 * d_model) if n_layers > 0 and d_model > 0 else 0
    mfu = st.number_input("MFU", value=0.50, step=0.05, format="%.2f")
    precision = st.selectbox("Precision", ["FP4", "FP8", "FP16"], index=1)
    bytes_per_param: float = {"FP4": 0.5, "FP8": 1, "FP16": 2}[precision]
    inference_pflops = {"FP4": rp["fp4_inf"], "FP8": rp["fp8"], "FP16": rp["fp16"]}[precision]

    st.subheader("Parallelism & Optimization")
    tp = rp["n_gpu"]

    use_gqa = st.checkbox("GQA (Grouped Query Attention)", value=True)
    if use_gqa:
        gqa_kv_heads = st.number_input("GQA KV heads", value=8, step=1)
    else:
        gqa_kv_heads = n_heads  # baseline: full attention (no KV head reduction)

    use_batching = st.checkbox("Batch Size Optimization", value=True)
    if use_batching:
        batch_size = st.number_input("Batch Size", value=64, step=1)
    else:
        batch_size = 1  # baseline: single request

    use_spec_decode = st.checkbox("Speculative Decoding", value=True)
    if use_spec_decode:
        spec_window = st.number_input("Speculation window N", value=8, step=1)
        acceptance_rate = st.number_input("Acceptance rate a", value=0.70, step=0.05, format="%.2f")
        expected_tokens_accepted = (1 - acceptance_rate**(spec_window + 1)) / (1 - acceptance_rate) if acceptance_rate != 1 else spec_window + 1
    else:
        spec_window = 0
        acceptance_rate = 1.0
        expected_tokens_accepted = 1  # baseline: one token per step

    use_nvlink_sharp = st.checkbox("NVLink SHARP", value=True)
    if use_nvlink_sharp:
        nvlink_sharp_reduction = st.number_input("NVLink SHARP reduction", value=0.5, step=0.1, format="%.1f")
        nvlink_time_override = st.number_input("NVLink time/tok with SHARP (s)", value=0.0005, step=0.0001, format="%.4f")
    else:
        nvlink_sharp_reduction = 1.0  # baseline: no SHARP reduction
        nvlink_time_override = None   # will use raw NVLink time

    use_overlap = st.checkbox("Compute-Comm Overlap", value=True)
    if use_overlap:
        overlap_efficiency = st.number_input("Compute-Comm Overlap efficiency", value=0.5, step=0.1, format="%.1f")
    else:
        overlap_efficiency = 0.0  # baseline: no overlap

    use_pp = st.checkbox("Pipeline Parallelism (PP)", value=True)
    if use_pp:
        new_tp_degree = st.number_input("New TP degree (for PP)", value=36, step=1)
    else:
        new_tp_degree: int = int(rp["n_gpu"])  # baseline: full TP, no PP
    pp_stages = rp["n_gpu"] / new_tp_degree if new_tp_degree > 0 else 1

    use_nvlink_latency = st.checkbox("NVLink Hop Latency", value=True)
    if use_nvlink_latency:
        nvlink_latency_us = st.number_input("NVLink latency per hop (μs)", value=1.0, step=0.5, format="%.1f")
        nvlink_hops = st.number_input("NVLink hops (NVL72 fabric)", value=2, step=1)
    else:
        nvlink_latency_us = 0.0
        nvlink_hops = 0

    st.subheader("Revenue Parameters")
    gpu_utilization = st.number_input("GPU Utilization Rate", value=0.70, step=0.05, format="%.2f")
    uptime = st.number_input("Uptime (scheduled)", value=0.995, step=0.001, format="%.3f")

# ─── Effective decode steps (speculative decoding speedup) ──────────────────
eff_decode_steps = output_tokens / expected_tokens_accepted if expected_tokens_accepted > 0 else output_tokens

# ─── DERIVED: Rack counts per type ──────────────────────────────────────────
n_gpu: int = int(rp["n_gpu"])
n_racks = math.floor(total_power / 1000 / rp["power_per_rack"])
nvlink_bw: float = float(rp["nvlink_bw"])
mem_bw: float = float(rp["mem_bw"])

# Also compute for the *other* rack type (Revenue_Model shows both)
other_type = "GB200 NVL72" if rack_type == "Vera Rubin NVL72" else "Vera Rubin NVL72"
orp = RACK_PRESETS[other_type]
other_n_racks = math.floor(total_power / 1000 / orp["power_per_rack"])
other_inference_pflops = {"FP4": orp["fp4_inf"], "FP8": orp["fp8"], "FP16": orp["fp16"]}[precision]

# ─── PRICING ─────────────────────────────────────────────────────────────────
tier_name, input_price, output_price = match_pricing_tier(parameters)

# ─── WP_PARAM CALCULATIONS ──────────────────────────────────────────────────

# --- Prefill Phase ---
params_per_layer_att = 4 * d_model**2
params_per_layer_ffn = 3 * d_model * d_ff
total_params_per_layer = params_per_layer_att + params_per_layer_ffn
weight_memory = parameters * bytes_per_param

# Prefill FLOP per layer
pf_qkv = 2 * input_tokens * d_model**2 * 3
pf_qkt = 2 * input_tokens**2 * d_model
pf_atv = 2 * input_tokens**2 * d_model
pf_out = 2 * input_tokens * d_model**2
pf_ffn = 2 * input_tokens * d_model * d_ff * 3
pf_norms = 5 * input_tokens * d_model
pf_flop_per_layer = pf_qkv + pf_qkt + pf_atv + pf_out + pf_ffn + pf_norms

# All layers
pf_flop_total = pf_flop_per_layer * n_layers

# Prefill compute time
pf_compute_theoretical = pf_flop_total / (inference_pflops * 1e15) if inference_pflops > 0 else 0
pf_compute_mfu = pf_compute_theoretical / mfu if mfu > 0 else 0

# Prefill HBM traffic per layer
pf_hbm_weights = bytes_per_param * (4 * d_model**2 + 3 * d_model * d_ff)
pf_hbm_activation = bytes_per_param * 4 * input_tokens * d_model
pf_hbm_kv_write = bytes_per_param * 2 * input_tokens * n_heads * d_head
pf_hbm_flash = bytes_per_param * 2 * input_tokens * d_model
pf_hbm_per_layer = pf_hbm_weights + pf_hbm_activation + pf_hbm_kv_write + pf_hbm_flash
pf_hbm_total = pf_hbm_per_layer * n_layers
pf_hbm_time = pf_hbm_total / (mem_bw * 1e12) if mem_bw > 0 else 0

# Prefill NVLink communication
pf_nvl_per_allreduce = bytes_per_param * 2 * (n_gpu - 1) / n_gpu * input_tokens * d_model
pf_nvl_per_layer = pf_nvl_per_allreduce * 2  # 2 all-reduces per layer
pf_nvl_total = pf_nvl_per_layer * n_layers
per_gpu_nvlink_bw = nvlink_bw / n_gpu  # TB/s per GPU
pf_nvl_time = pf_nvl_total / (per_gpu_nvlink_bw * 1e12) if per_gpu_nvlink_bw > 0 else 0
pf_nvl_time_sharp = pf_nvl_time * nvlink_sharp_reduction if use_nvlink_sharp else pf_nvl_time

total_prefill_time = pf_compute_mfu + pf_hbm_time + pf_nvl_time_sharp

# --- Decode Phase ---
avg_context = input_tokens + 0.5 * output_tokens

# Decode FLOP per layer per token
dc_qkv = 2 * 1 * d_model**2 * 3
dc_qkt = 2 * d_model * avg_context
dc_atv = 2 * d_model * avg_context
dc_out = 2 * 1 * d_model**2
dc_ffn = 2 * d_model * d_ff * 3
dc_flop_per_layer = dc_qkv + dc_qkt + dc_atv + dc_out + dc_ffn
dc_flop_all_layers = dc_flop_per_layer * n_layers
dc_flop_all_tokens = dc_flop_all_layers * eff_decode_steps

# Decode compute time
dc_compute_per_tok = dc_flop_all_layers / (inference_pflops * 1e15) if inference_pflops > 0 else 0
dc_compute_all = dc_compute_per_tok * eff_decode_steps

# Decode HBM per layer per token
dc_hbm_weights = total_params_per_layer * bytes_per_param
dc_hbm_kv_read = bytes_per_param * 2 * avg_context * n_heads * d_head
dc_hbm_kv_write = bytes_per_param * 2 * 1 * n_heads * d_head
dc_hbm_per_layer = dc_hbm_weights + dc_hbm_kv_read + dc_hbm_kv_write
dc_hbm_all = dc_hbm_per_layer * n_layers
dc_hbm_time = dc_hbm_all / (mem_bw * 1e12) if mem_bw > 0 else 0

# Decode NVLink per token
dc_nvl_msg = bytes_per_param * d_model * batch_size
dc_nvl_allreduces_per_layer = 2
dc_nvl_allreduces_all = dc_nvl_allreduces_per_layer * n_layers
dc_nvl_per_allreduce = 2 * dc_nvl_msg * (n_gpu - 1) / n_gpu
dc_nvl_traffic = dc_nvl_allreduces_all * dc_nvl_per_allreduce
dc_nvl_time_raw = dc_nvl_traffic / (per_gpu_nvlink_bw * 1e12) if per_gpu_nvlink_bw > 0 else 0
dc_nvl_time_sharp = float(nvlink_time_override) if (use_nvlink_sharp and nvlink_time_override is not None) else dc_nvl_time_raw  # Config override

dc_data_time_per_tok = dc_hbm_time + dc_nvl_time_sharp
dc_total_time_per_tok = dc_compute_per_tok + dc_hbm_time + dc_nvl_time_sharp

# --- GQA Optimization ---
gqa_kv_per_tok_per_layer = bytes_per_param * 2 * d_head * gqa_kv_heads
gqa_kv_all_layers_per_tok = gqa_kv_per_tok_per_layer * avg_context * n_layers
gqa_hbm_traffic = gqa_kv_all_layers_per_tok + weight_memory
gqa_hbm_time = gqa_hbm_traffic / (mem_bw * 1e12) if mem_bw > 0 else 0

# --- Batch Size Optimization ---
batch_gqa_kv_total = gqa_kv_all_layers_per_tok * batch_size
batch_hbm_traffic = weight_memory + batch_gqa_kv_total
batch_hbm_time_per_tok = batch_hbm_traffic / (mem_bw * 1e12) / batch_size if mem_bw > 0 and batch_size > 0 else 0

# --- NVLink Optimization ---
# Step 1: Compute-Communication Overlap
eff_nvlink_time_overlap = dc_nvl_time_sharp * (1 - overlap_efficiency)

# Step 2: Reduced TP via PP
orig_ring_factor = (n_gpu - 1) / n_gpu
new_ring_factor = (new_tp_degree - 1) / new_tp_degree if new_tp_degree > 0 else 0
nvl_data_new_tp = 2 * dc_nvl_msg * new_ring_factor
nvl_traffic_new_tp = nvl_data_new_tp * dc_nvl_allreduces_all
nvl_time_raw_new = nvl_traffic_new_tp / (per_gpu_nvlink_bw * 1e12) if per_gpu_nvlink_bw > 0 else 0
nvl_time_sharp_new = nvl_time_raw_new * nvlink_sharp_reduction if use_nvlink_sharp else nvl_time_raw_new
nvl_time_optimized = nvl_time_sharp_new * (1 - overlap_efficiency)
nvl_bw_all_tokens_opt = nvl_time_optimized * eff_decode_steps  # bandwidth component only
# Hop latency: per-token cost = latency_per_hop × hops × all-reduces; overlap hides same fraction
nvl_latency_per_tok = nvlink_latency_us * 1e-6 * nvlink_hops * dc_nvl_allreduces_all * (1 - overlap_efficiency)
nvl_latency_all_tokens = nvl_latency_per_tok * eff_decode_steps
nvl_time_all_tokens_opt = nvl_bw_all_tokens_opt + nvl_latency_all_tokens

# ─── TL_PARAM: TIMELINE ─────────────────────────────────────────────────────

# Helper: compute for a specific rack type
def compute_timeline_for_rack(rack_name):
    rr = RACK_PRESETS[rack_name]
    r_mem_bw: float = float(rr["mem_bw"])
    r_nvlink_bw: float = float(rr["nvlink_bw"])
    r_n_gpu: int = int(rr["n_gpu"])
    r_inference_pflops: float = float({"FP4": rr["fp4_inf"], "FP8": rr["fp8"], "FP16": rr["fp16"]}[precision])
    r_per_gpu_nvl = r_nvlink_bw / r_n_gpu

    # Prefill compute
    r_pf_compute_th = pf_flop_total / (r_inference_pflops * 1e15) if r_inference_pflops > 0 else 0
    r_pf_compute = r_pf_compute_th / mfu if mfu > 0 else 0
    # Prefill HBM
    r_pf_hbm_time = pf_hbm_total / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0
    # Prefill NVLink
    r_pf_nvl_time = pf_nvl_total / (r_per_gpu_nvl * 1e12) if r_per_gpu_nvl > 0 else 0
    r_pf_nvl_sharp = r_pf_nvl_time * nvlink_sharp_reduction if use_nvlink_sharp else r_pf_nvl_time

    # Decode HBM (GQA)
    r_gqa_hbm_traffic = gqa_kv_all_layers_per_tok + weight_memory
    r_gqa_hbm_time = r_gqa_hbm_traffic / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0

    # Decode HBM (batch)
    r_batch_hbm_traffic = weight_memory + batch_gqa_kv_total
    r_batch_hbm_per_tok = r_batch_hbm_traffic / (r_mem_bw * 1e12) / batch_size if r_mem_bw > 0 and batch_size > 0 else 0

    # Decode NVLink optimized
    r_nvl_time_raw_new = nvl_traffic_new_tp / (r_per_gpu_nvl * 1e12) if r_per_gpu_nvl > 0 else 0
    r_nvl_time_sharp_new = r_nvl_time_raw_new * nvlink_sharp_reduction if use_nvlink_sharp else r_nvl_time_raw_new
    r_nvl_time_optimized = r_nvl_time_sharp_new * (1 - overlap_efficiency)
    r_nvl_latency_per_tok = nvlink_latency_us * 1e-6 * nvlink_hops * dc_nvl_allreduces_all * (1 - overlap_efficiency)

    # Timeline steps (fully optimized)
    d_tok = 0.05  # tokenization
    d_embed_hbm = bytes_per_param * vocab_size * d_model / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0
    d_embed_nvl = bytes_per_param * 2 * input_tokens * d_model * 2 * (71/72) / (r_nvlink_bw * 1e12) if r_nvlink_bw > 0 else 0
    d_embed = d_embed_hbm + d_embed_nvl

    # Prefill (batched): multiply by batch_size
    d_prefill_f = r_pf_compute * batch_size
    d_prefill_h = r_pf_nvl_sharp * batch_size
    d_prefill = d_prefill_f + d_prefill_h

    # Decode first token (GQA)
    d_decode1 = r_gqa_hbm_time

    # Decode all tokens (batch + NVLink opt)
    d_decode_all_hbm = r_batch_hbm_per_tok * eff_decode_steps * batch_size
    d_decode_all_nvl = (r_nvl_time_optimized + r_nvl_latency_per_tok) * eff_decode_steps
    d_decode_all = d_decode_all_hbm + d_decode_all_nvl

    d_detok = 0.05

    e2e = d_tok + d_embed + d_prefill + d_decode1 + d_decode_all + d_detok
    avg_time_per_tok = e2e / (batch_size * output_tokens) if batch_size > 0 and output_tokens > 0 else 0
    tok_per_sec = 1 / avg_time_per_tok if avg_time_per_tok > 0 else 0

    # E2E through decode-all (excluding detokenization)
    e2e_through_decode = d_tok + d_embed + d_prefill + d_decode1 + d_decode_all

    return dict(
        d_tok=d_tok, d_embed=d_embed, d_prefill=d_prefill,
        d_decode1=d_decode1, d_decode_all=d_decode_all, d_detok=d_detok,
        e2e=e2e, e2e_through_decode=e2e_through_decode,
        avg_time_per_tok=avg_time_per_tok, tok_per_sec=tok_per_sec,
        # Sub-components for display
        pf_compute=d_prefill_f, pf_hbm=r_pf_hbm_time * batch_size,
        pf_nvl=d_prefill_h,
        dc_hbm_per_tok=r_batch_hbm_per_tok,
        dc_nvl_opt=r_nvl_time_optimized,
    )


# The Excel uses a specific formula structure for the "fully optimized" timeline.
# Let me reproduce it exactly as in TL_Param rows 25-35:
def compute_tl_exact(rack_name):
    """Exact reproduction of TL_Param optimized timeline (rows 25-35)"""
    rr = RACK_PRESETS[rack_name]
    r_mem_bw: float = float(rr["mem_bw"])
    r_nvlink_bw: float = float(rr["nvlink_bw"])
    r_n_gpu: int = int(rr["n_gpu"])
    r_inference_pflops: float = float({"FP4": rr["fp4_inf"], "FP8": rr["fp8"], "FP16": rr["fp16"]}[precision])
    r_per_gpu_nvl = r_nvlink_bw / r_n_gpu

    # Step 1: Tokenization
    d1 = 0.05

    # Step 2: Embedding Lookup + Broadcast
    g2 = bytes_per_param * vocab_size * d_model / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0
    h2 = bytes_per_param * 2 * input_tokens * d_model * 2 * (71/72) / (r_nvlink_bw * 1e12) if r_nvlink_bw > 0 else 0
    d2 = g2 + h2

    # Step 3: Prefill + KV Cache Write (batched)
    r_pf_compute_th = pf_flop_total / (r_inference_pflops * 1e15) if r_inference_pflops > 0 else 0
    r_pf_compute = r_pf_compute_th / mfu if mfu > 0 else 0
    f3 = r_pf_compute * batch_size
    r_pf_hbm_time = pf_hbm_total / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0
    g3 = r_pf_hbm_time * batch_size
    r_pf_nvl_time = pf_nvl_total / (r_per_gpu_nvl * 1e12) if r_per_gpu_nvl > 0 else 0
    r_pf_nvl_sharp = r_pf_nvl_time * nvlink_sharp_reduction if use_nvlink_sharp else r_pf_nvl_time
    h3 = r_pf_nvl_sharp * batch_size
    d3 = f3 + h3

    # Step 4: Decode-First Token (GQA)
    r_gqa_hbm_traffic = gqa_kv_all_layers_per_tok + weight_memory
    r_gqa_hbm_time = r_gqa_hbm_traffic / (r_mem_bw * 1e12) if r_mem_bw > 0 else 0
    d4 = r_gqa_hbm_time

    # Step 5: Decode-All Tokens
    r_batch_hbm_traffic = weight_memory + batch_gqa_kv_total
    r_batch_hbm_per_tok = r_batch_hbm_traffic / (r_mem_bw * 1e12) / batch_size if r_mem_bw > 0 and batch_size > 0 else 0
    g5 = r_batch_hbm_per_tok * eff_decode_steps * batch_size

    # NVLink optimized
    r_nvl_data_new_tp = 2 * dc_nvl_msg * (new_tp_degree - 1) / new_tp_degree if new_tp_degree > 0 else 0
    r_nvl_traffic_new_tp = r_nvl_data_new_tp * dc_nvl_allreduces_all
    r_nvl_time_raw_new = r_nvl_traffic_new_tp / (r_per_gpu_nvl * 1e12) if r_per_gpu_nvl > 0 else 0
    r_nvl_time_sharp_new = r_nvl_time_raw_new * nvlink_sharp_reduction if use_nvlink_sharp else r_nvl_time_raw_new
    r_nvl_time_optimized = r_nvl_time_sharp_new * (1 - overlap_efficiency)
    r_nvl_latency_per_tok = nvlink_latency_us * 1e-6 * nvlink_hops * dc_nvl_allreduces_all * (1 - overlap_efficiency)
    h5 = (r_nvl_time_optimized + r_nvl_latency_per_tok) * eff_decode_steps
    d5 = g5 + h5

    # Step 6: De-tokenization
    d6 = 0.05

    e2e = d1 + d2 + d3 + d4 + d5 + d6
    e2e_through_decode = d1 + d2 + d3 + d4 + d5
    avg_tok = e2e / (batch_size * output_tokens) if batch_size > 0 and output_tokens > 0 else 0
    tps = 1 / avg_tok if avg_tok > 0 else 0

    return dict(
        steps=[d1, d2, d3, d4, d5, d6],
        e2e=e2e, e2e_through_decode=e2e_through_decode,
        avg_time_per_tok=avg_tok, tok_per_sec=tps,
        f3=f3, g3=g3, h3=h3, g5=g5, h5=h5, d4=d4,
    )


# However the Excel TL_Param uses cross-rack scaling for the N/O columns.
# Let me reproduce the exact Excel formulas for the "Rack-Independent" outputs.
# The Excel computes everything using the *selected* rack, then scales to the other rack
# using hardware ratios. But looking more closely, N27:N32 compute VR-specific values
# and O27:O32 compute GB200-specific values independently of the dropdown.

def compute_tl_for_rack_excel(target_rack):
    """
    Reproduces TL_Param N/O columns exactly.
    These compute rack-specific E2E latency independent of the Config dropdown.
    """
    # The Excel formulas in N/O columns use WP_Param values (which depend on the selected rack)
    # and then apply scaling ratios when the target differs from selected.
    # But the "selected" rack in WP_Param uses Config!B7 values.
    #
    # For perfect fidelity, we compute using the selected rack's WP_Param values,
    # then apply the Excel's IF/scaling logic.

    selected = rack_type
    sel_rp = RACK_PRESETS[selected]
    tgt_rp = RACK_PRESETS[target_rack]

    # WP_Param values are always computed from the selected rack
    # (already done above in the global scope)

    # Step 1: Tokenization - always 0.05
    n1 = 0.05

    # Step 2: Embedding
    # N28/O28 formulas:
    # If selected == VR: D28 is used for VR; for GB200 scale HBM and NVLink
    # If selected != VR: scale for VR, use D28 for GB200
    g28 = bytes_per_param * vocab_size * d_model / (mem_bw * 1e12) if mem_bw > 0 else 0
    h28 = bytes_per_param * 2 * input_tokens * d_model * 2 * (71/72) / (nvlink_bw * 1e12) if nvlink_bw > 0 else 0
    d28 = g28 + h28

    if target_rack == selected:
        n2 = d28
    elif target_rack == "Vera Rubin NVL72":
        # Scale from selected (GB200) to VR
        n2 = g28 * (RACK_PRESETS["Vera Rubin NVL72"]["mem_bw"] / sel_rp["mem_bw"]) + \
             h28 * (RACK_PRESETS["Vera Rubin NVL72"]["nvlink_bw"] / sel_rp["nvlink_bw"])
        # Wait, the Excel formula is: G28*(Config!G18/Config!F18) + H28*(Config!G15/Config!F15)
        # When selected is VR (F columns), for GB200 (O column):
        # O28 = G28*(F18/G18) + H28*(F15/G15) - this scales FROM VR TO GB200
        # Actually let me re-read:
        # N28: IF(Config!B7="Vera Rubin NVL72", D28, G28*(Config!G18/Config!F18) + H28*(Config!G15/Config!F15))
        # O28: IF(Config!B7="Vera Rubin NVL72", G28*(Config!F18/Config!G18) + H28*(Config!F15/Config!G15), D28)
        #
        # Config!F18 = VR mem_bw = 1580, Config!G18 = GB200 mem_bw = 576
        # Config!F15 = VR nvlink_bw = 260, Config!G15 = GB200 nvlink_bw = 130
        #
        # So when selected=VR: N28 = D28 (VR), O28 = G28*(1580/576) + H28*(260/130)
        # G28 uses selected rack's mem_bw, H28 uses selected rack's nvlink_bw
        # For O28 (GB200): scale HBM time by VR_bw/GB_bw ratio (higher time for GB200 since lower BW)
        # Wait: G28*(F18/G18) = G28 * (VR_memBW / GB_memBW) -- this would INCREASE time, which is wrong
        # Actually G28 = time, and if VR has higher BW, the time with VR BW is shorter.
        # The ratio F18/G18 = 1580/576 ≈ 2.74 which increases time -- meaning GB200 is slower.
        # That's correct: GB200 has lower mem_bw so things take longer.
        #
        # But wait - G28 was computed with selected rack's mem_bw. If selected=VR, G28 uses VR BW.
        # To get GB200 time: time_GB = data / BW_GB = (data / BW_VR) * (BW_VR / BW_GB) = G28 * (BW_VR / BW_GB)
        # Yes that's correct.
        pass

    vr = RACK_PRESETS["Vera Rubin NVL72"]
    gb = RACK_PRESETS["GB200 NVL72"]

    if selected == "Vera Rubin NVL72":
        if target_rack == "Vera Rubin NVL72":
            n2 = d28
        else:  # GB200
            n2 = g28 * (vr["mem_bw"] / gb["mem_bw"]) + h28 * (vr["nvlink_bw"] / gb["nvlink_bw"])
    else:  # selected == GB200
        if target_rack == "GB200 NVL72":
            n2 = d28
        else:  # VR
            n2 = g28 * (gb["mem_bw"] / vr["mem_bw"]) + h28 * (gb["nvlink_bw"] / vr["nvlink_bw"])

    # Step 3: Prefill
    # F29 = WP_Param!B70 * WP_Param!B169 = pf_compute_mfu * batch_size
    f29 = pf_compute_mfu * batch_size
    g29 = pf_hbm_time * batch_size
    h29 = pf_nvl_time_sharp * batch_size
    d29 = f29 + h29

    # N29: scales compute by PFLOPS ratio and NVLink by BW ratio
    if selected == "Vera Rubin NVL72":
        if target_rack == "Vera Rubin NVL72":
            n3 = d29
            n3_compute, n3_hbm, n3_nvlink = f29, g29, h29
        else:
            sel_pflops = {"FP4": vr["fp4_inf"], "FP8": vr["fp8"], "FP16": vr["fp16"]}[precision]
            tgt_pflops = {"FP4": gb["fp4_inf"], "FP8": gb["fp8"], "FP16": gb["fp16"]}[precision]
            n3_compute = f29 * (sel_pflops / tgt_pflops)
            n3_hbm = g29 * (vr["mem_bw"] / gb["mem_bw"])
            n3_nvlink = h29 * (vr["nvlink_bw"] / gb["nvlink_bw"])
            n3 = n3_compute + n3_nvlink
    else:
        if target_rack == "GB200 NVL72":
            n3 = d29
            n3_compute, n3_hbm, n3_nvlink = f29, g29, h29
        else:
            sel_pflops = {"FP4": gb["fp4_inf"], "FP8": gb["fp8"], "FP16": gb["fp16"]}[precision]
            tgt_pflops = {"FP4": vr["fp4_inf"], "FP8": vr["fp8"], "FP16": vr["fp16"]}[precision]
            n3_compute = f29 * (sel_pflops / tgt_pflops)
            n3_hbm = g29 * (gb["mem_bw"] / vr["mem_bw"])
            n3_nvlink = h29 * (gb["nvlink_bw"] / vr["nvlink_bw"])
            n3 = n3_compute + n3_nvlink

    # Step 4: Decode-First Token (GQA HBM time)
    # G30 = WP_Param!B164 = gqa_hbm_time (uses selected rack's mem_bw)
    d30 = gqa_hbm_time

    if selected == "Vera Rubin NVL72":
        if target_rack == "Vera Rubin NVL72":
            n4 = d30
        else:
            n4 = d30 * (vr["mem_bw"] / gb["mem_bw"])
    else:
        if target_rack == "GB200 NVL72":
            n4 = d30
        else:
            n4 = d30 * (gb["mem_bw"] / vr["mem_bw"])
    n4_hbm = n4  # step 4 is purely HBM-bound

    # Step 5: Decode-All Tokens
    # G31 = WP_Param!B172 * WP_Param!B23 * batch = batch_hbm_time_per_tok * output_tokens * batch_size
    # Wall-clock HBM time: weights read once per output token step for entire batch
    g31 = batch_hbm_time_per_tok * eff_decode_steps * batch_size
    # H31 = BW component + hop latency component (latency does not scale with BW ratio)
    h31_bw = nvl_bw_all_tokens_opt   # scales with NVLink BW when cross-rack
    h31_lat = nvl_latency_all_tokens  # same for both racks (same NVL72 fabric topology)
    h31 = h31_bw + h31_lat
    d31 = g31 + h31

    if selected == "Vera Rubin NVL72":
        if target_rack == "Vera Rubin NVL72":
            n5 = d31
            n5_hbm, n5_nvlink = g31, h31
        else:
            n5_hbm = g31 * (vr["mem_bw"] / gb["mem_bw"])
            n5_nvlink = h31_bw * (vr["nvlink_bw"] / gb["nvlink_bw"]) + h31_lat
            n5 = n5_hbm + n5_nvlink
    else:
        if target_rack == "GB200 NVL72":
            n5 = d31
            n5_hbm, n5_nvlink = g31, h31
        else:
            n5_hbm = g31 * (gb["mem_bw"] / vr["mem_bw"])
            n5_nvlink = h31_bw * (gb["nvlink_bw"] / vr["nvlink_bw"]) + h31_lat
            n5 = n5_hbm + n5_nvlink

    # Step 6: De-tokenization
    n6 = 0.05

    e2e_through_decode = n1 + n2 + n3 + n4 + n5
    e2e = e2e_through_decode + n6
    avg_tok = e2e / (batch_size * output_tokens) if batch_size > 0 and output_tokens > 0 else 0
    tps = 1 / avg_tok if avg_tok > 0 else 0

    return dict(
        steps=[n1, n2, n3, n4, n5, n6],
        step_names=["Tokenization", "Embedding+Broadcast", "Prefill+KV Write",
                     "Decode-First Token", "Decode-All Tokens", "De-tokenization"],
        e2e=e2e, e2e_through_decode=e2e_through_decode,
        avg_time_per_tok=avg_tok, tok_per_sec=tps,
        # Sub-components for steps 3–5
        n3_compute=n3_compute, n3_hbm=n3_hbm, n3_nvlink=n3_nvlink,
        n4_hbm=n4_hbm,
        n5_hbm=n5_hbm, n5_nvlink=n5_nvlink,
    )


vr_tl = compute_tl_for_rack_excel("Vera Rubin NVL72")
gb_tl = compute_tl_for_rack_excel("GB200 NVL72")


# ─── DC_COST_MODEL ──────────────────────────────────────────────────────────
def compute_dc_cost(rack_name):
    rr = RACK_PRESETS[rack_name]
    r_n_racks = math.floor(total_power / 1000 / rr["power_per_rack"])
    it_power_mw = total_power / 1e6
    facility_power_mw = it_power_mw * pue
    total_gpus = r_n_racks * rr["n_gpu"]

    # CapEx
    it_racks = rr["rack_price"] * r_n_racks / 1e6
    it_networking = 150_000 * r_n_racks / 1e6
    it_switches = 35_000 * r_n_racks / 1e6
    it_storage = 25_000 * r_n_racks / 1e6
    it_hw = it_racks + it_networking + it_switches + it_storage

    land = 200_000 * 100 / 1e6
    building = 3_000_000 * facility_power_mw / 1e6
    sitework = 500_000 * facility_power_mw / 1e6
    land_building = land + building + sitework

    substation = 1_500_000 * facility_power_mw / 1e6
    mv_dist = 800_000 * facility_power_mw / 1e6
    generators = 600_000 * facility_power_mw / 1e6
    ups = 400_000 * it_power_mw / 1e6
    electrical = substation + mv_dist + generators + ups

    liquid_cool = 56_000 * r_n_racks / 1e6
    cooling_towers = 300_000 * facility_power_mw / 1e6
    fire_hvac = 100_000 * facility_power_mw / 1e6
    cooling = liquid_cool + cooling_towers + fire_hvac

    fiber = 200_000 * facility_power_mw / 1e6
    security = 50_000 * facility_power_mw / 1e6
    fiber_security = fiber + security

    non_it = land_building + electrical + cooling + fiber_security
    design = 0.03 * non_it
    pm = 0.02 * non_it
    contingency = 0.10 * non_it
    soft_costs = design + pm + contingency

    total_capex = it_hw + non_it + soft_costs

    # OpEx
    depr_it = it_hw / 5
    depr_bldg = land_building / 30
    depr_elec = electrical / 20
    depr_cool = cooling / 15
    depr_fiber = fiber_security / 10
    depr_total = depr_it + depr_bldg + depr_elec + depr_cool + depr_fiber

    elec_cost = facility_power_mw * 8760 * 1000 * 0.06 / 1e6

    maint_it = 0.03 * it_hw
    maint_fac = 0.02 * non_it
    staff = 500 * 150_000 / 1e6
    prop_tax = 0.005 * total_capex
    water = 50_000 * facility_power_mw / 1e6
    sw_licenses = 200
    maint_ops = maint_it + maint_fac + staff + prop_tax + water + sw_licenses

    total_opex = depr_total + elec_cost + maint_ops

    return dict(
        n_racks=r_n_racks, total_gpus=total_gpus,
        it_power_mw=it_power_mw, facility_power_mw=facility_power_mw,
        total_capex=total_capex, total_opex=total_opex,
        it_hw=it_hw, depr_total=depr_total, elec_cost=elec_cost, maint_ops=maint_ops,
    )


vr_dc = compute_dc_cost("Vera Rubin NVL72")
gb_dc = compute_dc_cost("GB200 NVL72")


# ─── REVENUE MODEL ──────────────────────────────────────────────────────────
def compute_revenue(tl, dc, rack_name):
    rr = RACK_PRESETS[rack_name]
    r_n_racks = dc["n_racks"]

    eff_annual_sec = 8760 * 3600 * gpu_utilization * uptime
    requests_per_rack_yr = math.floor(eff_annual_sec / tl["e2e"]) * batch_size if tl["e2e"] > 0 else 0
    total_requests_yr = requests_per_rack_yr * r_n_racks

    annual_input_tokens = total_requests_yr * input_tokens
    annual_output_tokens = total_requests_yr * output_tokens
    annual_total_tokens = annual_input_tokens + annual_output_tokens
    annual_total_m = annual_total_tokens / 1e6
    annual_input_m = annual_input_tokens / 1e6
    annual_output_m = annual_output_tokens / 1e6

    tokens_per_sec = annual_total_tokens / (8760 * 3600) if True else 0
    requests_per_sec = total_requests_yr / (8760 * 3600) if True else 0

    input_revenue = annual_input_m * input_price / 1e6
    output_revenue = annual_output_m * output_price / 1e6
    total_revenue = input_revenue + output_revenue
    blended_rate = (input_price * annual_input_m + output_price * annual_output_m) / annual_total_m if annual_total_m > 0 else 0

    rev_per_rack = total_revenue / r_n_racks * 1e6 if r_n_racks > 0 else 0
    rev_to_opex = total_revenue / dc["total_opex"] if dc["total_opex"] > 0 else 0
    cost_per_mtok = blended_rate / rev_to_opex if rev_to_opex > 0 else 0

    return dict(
        eff_annual_sec=eff_annual_sec,
        requests_per_rack_yr=requests_per_rack_yr,
        total_requests_yr=total_requests_yr,
        annual_input_m=annual_input_m, annual_output_m=annual_output_m,
        annual_total_m=annual_total_m,
        tokens_per_sec=tokens_per_sec, requests_per_sec=requests_per_sec,
        input_revenue=input_revenue, output_revenue=output_revenue,
        total_revenue=total_revenue, blended_rate=blended_rate,
        rev_per_rack=rev_per_rack, rev_to_opex=rev_to_opex,
        cost_per_mtok=cost_per_mtok,
    )


vr_rev = compute_revenue(vr_tl, vr_dc, "Vera Rubin NVL72")
gb_rev = compute_revenue(gb_tl, gb_dc, "GB200 NVL72")


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

# ─── Config Summary ──────────────────────────────────────────────────────────
with st.expander("📋 Config Summary", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**AI DC**")
        st.write(f"Power: {total_power/1e9:.1f} GW, PUE: {pue}")
        st.write(f"Rack: {rack_type}")
        st.write(f"Racks: {n_racks:,}")
        st.write(f"GPUs/rack: {n_gpu}, Total: {n_racks * n_gpu:,}")
    with c2:
        st.markdown("**Model**")
        st.write(f"Params: {parameters/1e9:.0f}B ({parameters/1e12:.1f}T)")
        st.write(f"d_model={d_model:,}, n_layers={n_layers}, n_heads={n_heads}")
        st.write(f"d_head={d_head:.0f}, d_ff={d_ff:,.0f}")
        st.write(f"Precision: {precision}, MFU: {mfu}")
    with c3:
        st.markdown("**Optimization**")
        st.write(f"GQA KV heads: {gqa_kv_heads}, Batch: {batch_size}")
        st.write(f"Spec decode: N={spec_window}, a={acceptance_rate}, E[tok]={expected_tokens_accepted:.2f}")
        st.write(f"TP={new_tp_degree}, PP={pp_stages:.0f}")
        st.write(f"Pricing: {tier_name} (${input_price}/M in, ${output_price}/M out)")

# ─── Key Metrics ─────────────────────────────────────────────────────────────
st.header("Key Metrics")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("VR E2E Latency (s)", f"{vr_tl['e2e']:.6f}")
with m2:
    st.metric("VR Output tok/s", f"{vr_tl['tok_per_sec']:,.1f}")
with m3:
    st.metric("VR Annual Revenue ($M)", f"{vr_rev['total_revenue']:,.1f}")
with m4:
    st.metric("VR Rev/OpEx Ratio", f"{vr_rev['rev_to_opex']:.1f}x")

m5, m6, m7, m8 = st.columns(4)
with m5:
    st.metric("GB200 E2E Latency (s)", f"{gb_tl['e2e']:.6f}")
with m6:
    st.metric("GB200 Output tok/s", f"{gb_tl['tok_per_sec']:,.1f}")
with m7:
    st.metric("GB200 Annual Revenue ($M)", f"{gb_rev['total_revenue']:,.1f}")
with m8:
    st.metric("GB200 Rev/OpEx Ratio", f"{gb_rev['rev_to_opex']:.1f}x")

# ─── ANIMATED FLOWCHART ──────────────────────────────────────────────────────
st.header("Token Generation Pipeline — Animated Flowchart")

# Helper: format large numbers
def _fmt_bytes(b):
    if b == 0: return "0"
    if b >= 1e15: return f"{b/1e15:.2f} PB"
    if b >= 1e12: return f"{b/1e12:.2f} TB"
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.2f} MB"
    if b >= 1e3: return f"{b/1e3:.2f} KB"
    return f"{b:.0f} B"

def _fmt_flops(f):
    if f == 0: return "0"
    if f >= 1e18: return f"{f/1e18:.2f} EFLOP"
    if f >= 1e15: return f"{f/1e15:.2f} PFLOP"
    if f >= 1e12: return f"{f/1e12:.2f} TFLOP"
    if f >= 1e9: return f"{f/1e9:.2f} GFLOP"
    return f"{f:.2e}"

def _fmt_time(t):
    if t == 0: return "0"
    if t >= 1: return f"{t:.4f}s"
    if t >= 1e-3: return f"{t*1e3:.4f}ms"
    if t >= 1e-6: return f"{t*1e6:.4f}us"
    return f"{t:.3e}s"

# Build per-step data for the selected rack
# Step 2 sub-components
embed_hbm_bytes = bytes_per_param * (vocab_size * d_model + input_tokens * d_model)  # embedding table + broadcast
embed_nvl_bytes = bytes_per_param * 2 * input_tokens * d_model * 2 * (71/72)
embed_hbm_time_val = bytes_per_param * vocab_size * d_model / (mem_bw * 1e12) if mem_bw > 0 else 0
embed_nvl_time_val = embed_nvl_bytes / (nvlink_bw * 1e12) if nvlink_bw > 0 else 0

# Step 3 sub-components (per-request, then ×batch)
pf_hbm_bytes_total = pf_hbm_total  # all layers
pf_nvl_bytes_total = pf_nvl_total  # all layers

# Step 4: decode first token
dc1_hbm_bytes = gqa_hbm_traffic  # weight_memory + GQA KV cache

# Step 5: decode all tokens
dc_all_hbm_bytes_per_tok = batch_hbm_traffic  # weight + batch GQA KV
dc_all_nvl_bytes_per_tok = nvl_traffic_new_tp  # optimized NVLink per tok (before SHARP/overlap)
dc_all_hbm_bytes_total = dc_all_hbm_bytes_per_tok * eff_decode_steps
dc_all_nvl_bytes_total = nvl_bw_all_tokens_opt * per_gpu_nvlink_bw * 1e12  # back-derive bytes from BW component only

# Use selected rack timeline
sel_tl = vr_tl if rack_type == "Vera Rubin NVL72" else gb_tl

flowchart_steps = [
    {
        "id": 1, "name": "Tokenization",
        "subtitle": "CPU",
        "duration": sel_tl["steps"][0],
        "compute_flops": 0,
        "hbm_bytes": 0, "hbm_time": 0,
        "nvl_bytes": 0, "nvl_time": 0,
        "bottleneck": "CPU",
        "color": "#6366f1",
        "per_tok_time": 0,
        "all_tok_time": sel_tl["steps"][0],
    },
    {
        "id": 2, "name": "Embedding Lookup",
        "subtitle": "+ Broadcast",
        "duration": sel_tl["steps"][1],
        "compute_flops": 0,
        "hbm_bytes": embed_hbm_bytes,
        "hbm_time": embed_hbm_time_val,
        "nvl_bytes": embed_nvl_bytes,
        "nvl_time": embed_nvl_time_val,
        "bottleneck": "NVLink",
        "color": "#8b5cf6",
        "per_tok_time": 0,
        "all_tok_time": sel_tl["steps"][1],
    },
    {
        "id": 3, "name": "Prefill",
        "subtitle": f"+ KV Cache Write (x{batch_size} batch)",
        "duration": sel_tl["steps"][2],
        "compute_flops": pf_flop_total * batch_size,
        "hbm_bytes": pf_hbm_bytes_total * batch_size,
        "hbm_time": pf_hbm_time * batch_size,
        "nvl_bytes": pf_nvl_bytes_total * batch_size,
        "nvl_time": pf_nvl_time_sharp * batch_size,
        "bottleneck": "Compute",
        "color": "#ec4899",
        "per_tok_time": pf_compute_mfu,
        "all_tok_time": sel_tl["steps"][2],
    },
    {
        "id": 4, "name": "Decode",
        "subtitle": "First Token (GQA)",
        "duration": sel_tl["steps"][3],
        "compute_flops": dc_flop_all_layers,
        "hbm_bytes": dc1_hbm_bytes,
        "hbm_time": gqa_hbm_time,
        "nvl_bytes": 0,
        "nvl_time": 0,
        "bottleneck": "HBM BW",
        "color": "#f59e0b",
        "per_tok_time": gqa_hbm_time,
        "all_tok_time": sel_tl["steps"][3],
    },
    {
        "id": 5, "name": "Decode",
        "subtitle": f"All {output_tokens} Tokens (x{batch_size} batch)",
        "duration": sel_tl["steps"][4],
        "compute_flops": dc_flop_all_tokens * batch_size,
        "hbm_bytes": dc_all_hbm_bytes_total,
        "hbm_time": batch_hbm_time_per_tok * eff_decode_steps * batch_size,
        "nvl_bytes": dc_all_nvl_bytes_total,
        "nvl_time": nvl_time_all_tokens_opt,
        "bottleneck": "HBM BW",
        "color": "#ef4444",
        "per_tok_time": batch_hbm_time_per_tok + nvl_time_optimized,
        "all_tok_time": sel_tl["steps"][4],
    },
    {
        "id": 6, "name": "De-tokenization",
        "subtitle": "CPU",
        "duration": sel_tl["steps"][5],
        "compute_flops": 0,
        "hbm_bytes": 0, "hbm_time": 0,
        "nvl_bytes": 0, "nvl_time": 0,
        "bottleneck": "CPU",
        "color": "#10b981",
        "per_tok_time": 0,
        "all_tok_time": sel_tl["steps"][5],
    },
]

# Format for display
for s in flowchart_steps:
    s["duration_fmt"] = _fmt_time(s["duration"])
    s["compute_fmt"] = _fmt_flops(s["compute_flops"])
    s["hbm_bytes_fmt"] = _fmt_bytes(s["hbm_bytes"])
    s["hbm_time_fmt"] = _fmt_time(s["hbm_time"])
    s["nvl_bytes_fmt"] = _fmt_bytes(s["nvl_bytes"])
    s["nvl_time_fmt"] = _fmt_time(s["nvl_time"])
    s["per_tok_fmt"] = _fmt_time(s["per_tok_time"])
    s["all_tok_fmt"] = _fmt_time(s["all_tok_time"])

e2e_total = sel_tl["e2e"]

flowchart_html = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',system-ui,sans-serif; background:#0f172a; color:#e2e8f0; }}
  .fc-container {{ padding:20px; max-width:1400px; margin:auto; }}
  .fc-title {{ font-size:14px; color:#94a3b8; margin-bottom:16px; text-align:center; }}

  /* Pipeline row */
  .pipeline {{ display:flex; align-items:stretch; gap:6px; margin-bottom:24px; position:relative; }}
  .step-box {{
    flex:1; border-radius:12px; padding:12px 10px; position:relative; overflow:hidden;
    border:1px solid rgba(255,255,255,0.08); cursor:pointer; transition:all 0.3s;
    min-width:0;
  }}
  .step-box:hover {{ transform:translateY(-3px); box-shadow:0 8px 25px rgba(0,0,0,0.4); }}
  .step-box.active {{ border-color:rgba(255,255,255,0.4); transform:translateY(-3px);
    box-shadow:0 8px 25px rgba(0,0,0,0.4); }}
  .step-num {{ font-size:11px; font-weight:700; opacity:0.5; }}
  .step-name {{ font-size:13px; font-weight:700; line-height:1.2; margin:4px 0 2px; }}
  .step-sub {{ font-size:10px; opacity:0.6; line-height:1.2; }}
  .step-dur {{ font-size:18px; font-weight:700; margin:8px 0 4px; }}
  .step-pct {{ font-size:11px; opacity:0.7; }}
  .step-bneck {{ font-size:10px; padding:2px 6px; border-radius:4px; display:inline-block;
    margin-top:6px; font-weight:600; }}
  .bneck-cpu {{ background:#334155; color:#94a3b8; }}
  .bneck-compute {{ background:#831843; color:#f9a8d4; }}
  .bneck-hbm {{ background:#78350f; color:#fcd34d; }}
  .bneck-nvlink {{ background:#1e3a5f; color:#7dd3fc; }}

  /* Animated progress bar inside each step */
  .step-progress {{
    position:absolute; bottom:0; left:0; height:3px; border-radius:0 0 12px 12px;
    transition:width 0.05s linear;
  }}

  /* Arrow connectors */
  .arrow {{ display:flex; align-items:center; color:#475569; font-size:18px; flex-shrink:0; }}

  /* Detail panel */
  .detail-panel {{
    background:#1e293b; border-radius:12px; padding:20px; margin-bottom:24px;
    border:1px solid rgba(255,255,255,0.08); min-height:180px;
    display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;
  }}
  .detail-section h4 {{ font-size:12px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.5px; margin-bottom:10px; }}
  .detail-row {{ display:flex; justify-content:space-between; padding:3px 0; font-size:12px; }}
  .detail-row .label {{ color:#94a3b8; }}
  .detail-row .value {{ font-weight:600; font-variant-numeric:tabular-nums; }}

  /* Resource bars */
  .resource-bars {{ margin-top:10px; }}
  .rbar {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
  .rbar-label {{ font-size:10px; width:55px; text-align:right; color:#94a3b8; flex-shrink:0; }}
  .rbar-track {{ flex:1; height:14px; background:#0f172a; border-radius:7px; overflow:hidden; position:relative; }}
  .rbar-fill {{ height:100%; border-radius:7px; transition:width 0.6s ease; position:relative; }}
  .rbar-fill span {{ position:absolute; right:4px; top:0; font-size:9px; line-height:14px; font-weight:600; }}
  .rbar-compute {{ background:linear-gradient(90deg,#ec4899,#f472b6); }}
  .rbar-hbm {{ background:linear-gradient(90deg,#f59e0b,#fbbf24); }}
  .rbar-nvlink {{ background:linear-gradient(90deg,#3b82f6,#60a5fa); }}

  /* Timeline bar */
  .timeline {{ position:relative; height:40px; background:#1e293b; border-radius:8px;
    overflow:hidden; margin-bottom:8px; }}
  .tl-seg {{ position:absolute; top:0; height:100%; display:flex; align-items:center;
    justify-content:center; font-size:10px; font-weight:600; border-right:1px solid #0f172a;
    transition:opacity 0.3s; overflow:hidden; white-space:nowrap; }}
  .tl-cursor {{ position:absolute; top:0; width:2px; height:100%; background:#fff;
    box-shadow:0 0 8px rgba(255,255,255,0.8); transition:left 0.05s linear; z-index:10; }}
  .tl-labels {{ display:flex; justify-content:space-between; font-size:11px; color:#64748b; }}

  /* Controls */
  .controls {{ display:flex; align-items:center; gap:12px; margin-bottom:20px; justify-content:center; }}
  .btn {{
    padding:8px 20px; border-radius:8px; border:1px solid #334155; background:#1e293b;
    color:#e2e8f0; font-size:13px; font-weight:600; cursor:pointer; transition:all 0.2s;
  }}
  .btn:hover {{ background:#334155; }}
  .btn.playing {{ background:#3b82f6; border-color:#3b82f6; }}
  .speed-label {{ font-size:12px; color:#94a3b8; }}

  /* Token particles */
  .particles {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:5; }}
  .particle {{
    position:absolute; width:6px; height:6px; border-radius:50%;
    background:#fbbf24; box-shadow:0 0 6px rgba(251,191,36,0.6);
    animation:particleFade 0.8s ease-out forwards;
  }}
  @keyframes particleFade {{
    0% {{ opacity:1; transform:scale(1); }}
    100% {{ opacity:0; transform:scale(0.3) translateY(-10px); }}
  }}

  /* E2E summary */
  .e2e-summary {{
    text-align:center; margin:16px 0 8px; font-size:14px; color:#94a3b8;
  }}
  .e2e-summary strong {{ color:#fbbf24; font-size:20px; }}
</style>

<div class="fc-container" id="fcApp">
  <div class="fc-title">Selected Rack: <strong>{rack_type}</strong> &nbsp;|&nbsp;
    Model: {parameters/1e12:.1f}T &nbsp;|&nbsp; I/O: {input_tokens}/{output_tokens} tokens
    &nbsp;|&nbsp; Batch: {batch_size}</div>

  <div class="controls">
    <button class="btn" id="playBtn" onclick="togglePlay()">Play</button>
    <span class="speed-label">Speed:</span>
    <button class="btn" onclick="setSpeed(0.5)">0.5x</button>
    <button class="btn" id="sp1" onclick="setSpeed(1)">1x</button>
    <button class="btn" onclick="setSpeed(3)">3x</button>
    <button class="btn" onclick="setSpeed(10)">10x</button>
    <button class="btn" onclick="resetAnim()">Reset</button>
  </div>

  <div class="pipeline" id="pipeline"></div>

  <div class="detail-panel" id="detailPanel"></div>

  <div style="margin-bottom:4px;font-size:12px;color:#64748b;font-weight:600;">
    Cumulative Timeline</div>
  <div class="timeline" id="timeline"></div>
  <div class="tl-labels">
    <span>0s</span>
    <span id="tlEnd">{_fmt_time(e2e_total)}</span>
  </div>

  <div class="e2e-summary">
    E2E Latency: <strong>{_fmt_time(e2e_total)}</strong>
    &nbsp;&nbsp;|&nbsp;&nbsp; Output tok/s: <strong>{sel_tl['tok_per_sec']:,.1f}</strong>
  </div>
</div>

<script>
const STEPS = {json.dumps(flowchart_steps)};
const E2E = {e2e_total};
let speed = 1, playing = false, animTime = 0, animFrame = null, lastTs = null;
let activeStep = 0;

// Build pipeline boxes
const pipeline = document.getElementById('pipeline');
STEPS.forEach((s, i) => {{
  if (i > 0) {{
    const arrow = document.createElement('div');
    arrow.className = 'arrow';
    arrow.innerHTML = '&#9654;';
    pipeline.appendChild(arrow);
  }}
  const box = document.createElement('div');
  box.className = 'step-box';
  box.style.background = `linear-gradient(135deg, ${{s.color}}22, ${{s.color}}11)`;
  box.id = 'step-' + s.id;
  box.onclick = () => showDetail(i);

  const bneckClass = s.bottleneck === 'CPU' ? 'bneck-cpu' :
    s.bottleneck === 'Compute' ? 'bneck-compute' :
    s.bottleneck.includes('HBM') ? 'bneck-hbm' : 'bneck-nvlink';

  const pct = E2E > 0 ? (s.duration / E2E * 100) : 0;

  box.innerHTML = `
    <div class="step-num">STEP ${{s.id}}</div>
    <div class="step-name">${{s.name}}</div>
    <div class="step-sub">${{s.subtitle}}</div>
    <div class="step-dur">${{s.duration_fmt}}</div>
    <div class="step-pct">${{pct.toFixed(1)}}% of E2E</div>
    <div class="step-bneck ${{bneckClass}}">${{s.bottleneck}}</div>
    <div class="step-progress" style="background:${{s.color}};width:0%"></div>
    <div class="particles"></div>
  `;
  pipeline.appendChild(box);
}});

// Build timeline segments
const tl = document.getElementById('timeline');
let cumPct = 0;
STEPS.forEach((s, i) => {{
  const pct = E2E > 0 ? (s.duration / E2E * 100) : 0;
  const seg = document.createElement('div');
  seg.className = 'tl-seg';
  seg.style.left = cumPct + '%';
  seg.style.width = pct + '%';
  seg.style.background = s.color + '88';
  seg.innerHTML = pct > 5 ? s.name : '';
  seg.id = 'tl-seg-' + i;
  tl.appendChild(seg);
  cumPct += pct;
}});
// cursor
const cursor = document.createElement('div');
cursor.className = 'tl-cursor';
cursor.style.left = '0%';
cursor.id = 'tlCursor';
tl.appendChild(cursor);

function showDetail(idx) {{
  activeStep = idx;
  const s = STEPS[idx];
  const dp = document.getElementById('detailPanel');

  // Find max values for bar scaling
  const maxFlops = Math.max(...STEPS.map(x => x.compute_flops || 0)) || 1;
  const maxHBM = Math.max(...STEPS.map(x => x.hbm_bytes || 0)) || 1;
  const maxNVL = Math.max(...STEPS.map(x => x.nvl_bytes || 0)) || 1;
  const maxTime = Math.max(...STEPS.map(x => x.duration || 0)) || 1;

  const computePct = s.compute_flops / maxFlops * 100;
  const hbmPct = s.hbm_bytes / maxHBM * 100;
  const nvlPct = s.nvl_bytes / maxNVL * 100;

  // Time breakdown bars (within this step)
  const stepMax = Math.max(s.hbm_time, s.nvl_time, s.duration) || 1;
  const hbmTimePct = s.hbm_time / stepMax * 100;
  const nvlTimePct = s.nvl_time / stepMax * 100;

  dp.innerHTML = `
    <div class="detail-section">
      <h4 style="color:${{s.color}}">Step ${{s.id}}: ${{s.name}} ${{s.subtitle}}</h4>
      <div class="detail-row"><span class="label">Total Duration</span><span class="value">${{s.duration_fmt}}</span></div>
      <div class="detail-row"><span class="label">Per Output Token</span><span class="value">${{s.per_tok_fmt}}</span></div>
      <div class="detail-row"><span class="label">All Output Tokens</span><span class="value">${{s.all_tok_fmt}}</span></div>
      <div class="detail-row"><span class="label">Bottleneck</span><span class="value">${{s.bottleneck}}</span></div>
      <div class="detail-row"><span class="label">% of E2E</span><span class="value">${{(s.duration/E2E*100).toFixed(2)}}%</span></div>
      <div class="detail-row"><span class="label">Compute FLOPs</span><span class="value">${{s.compute_fmt}}</span></div>
    </div>
    <div class="detail-section">
      <h4 style="color:#fbbf24">Data Transit</h4>
      <div class="detail-row"><span class="label">HBM Traffic</span><span class="value">${{s.hbm_bytes_fmt}}</span></div>
      <div class="detail-row"><span class="label">HBM Time</span><span class="value">${{s.hbm_time_fmt}}</span></div>
      <div class="detail-row"><span class="label">NVLink Traffic</span><span class="value">${{s.nvl_bytes_fmt}}</span></div>
      <div class="detail-row"><span class="label">NVLink Time</span><span class="value">${{s.nvl_time_fmt}}</span></div>
      <div class="resource-bars">
        <div class="rbar">
          <span class="rbar-label">HBM</span>
          <div class="rbar-track"><div class="rbar-fill rbar-hbm" style="width:${{hbmTimePct.toFixed(1)}}%"><span>${{s.hbm_time_fmt}}</span></div></div>
        </div>
        <div class="rbar">
          <span class="rbar-label">NVLink</span>
          <div class="rbar-track"><div class="rbar-fill rbar-nvlink" style="width:${{nvlTimePct.toFixed(1)}}%"><span>${{s.nvl_time_fmt}}</span></div></div>
        </div>
      </div>
    </div>
    <div class="detail-section">
      <h4 style="color:#a78bfa">Resource Scale (vs. max step)</h4>
      <div class="resource-bars">
        <div class="rbar">
          <span class="rbar-label">Compute</span>
          <div class="rbar-track"><div class="rbar-fill rbar-compute" style="width:${{computePct.toFixed(1)}}%"><span>${{s.compute_fmt}}</span></div></div>
        </div>
        <div class="rbar">
          <span class="rbar-label">HBM</span>
          <div class="rbar-track"><div class="rbar-fill rbar-hbm" style="width:${{hbmPct.toFixed(1)}}%"><span>${{s.hbm_bytes_fmt}}</span></div></div>
        </div>
        <div class="rbar">
          <span class="rbar-label">NVLink</span>
          <div class="rbar-track"><div class="rbar-fill rbar-nvlink" style="width:${{nvlPct.toFixed(1)}}%"><span>${{s.nvl_bytes_fmt}}</span></div></div>
        </div>
      </div>
    </div>
  `;

  // Highlight active step box
  document.querySelectorAll('.step-box').forEach((b, bi) => {{
    b.classList.toggle('active', bi === idx);
  }});
}}

// Animation
function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  document.getElementById('playBtn').classList.toggle('playing', playing);
  if (playing) {{
    lastTs = null;
    animFrame = requestAnimationFrame(tick);
  }} else if (animFrame) {{
    cancelAnimationFrame(animFrame);
  }}
}}

function setSpeed(s) {{
  speed = s;
  document.querySelectorAll('.controls .btn').forEach(b => {{
    if (b.textContent === s + 'x') b.classList.add('playing');
    else if (b.id !== 'playBtn') b.classList.remove('playing');
  }});
}}

function resetAnim() {{
  playing = false;
  animTime = 0;
  lastTs = null;
  document.getElementById('playBtn').textContent = 'Play';
  document.getElementById('playBtn').classList.remove('playing');
  if (animFrame) cancelAnimationFrame(animFrame);
  updateVisuals();
}}

function tick(ts) {{
  if (!playing) return;
  if (lastTs === null) lastTs = ts;
  const dt = (ts - lastTs) / 1000 * speed;
  lastTs = ts;
  animTime = Math.min(animTime + dt, E2E);
  updateVisuals();
  if (animTime < E2E) {{
    animFrame = requestAnimationFrame(tick);
  }} else {{
    // Auto-reset after a brief pause
    setTimeout(() => {{
      playing = false;
      animTime = 0;
      lastTs = null;
      document.getElementById('playBtn').textContent = 'Play';
      document.getElementById('playBtn').classList.remove('playing');
      updateVisuals();
      showDetail(0);
    }}, 800);
  }}
}}

let particleCounter = 0;
function updateVisuals() {{
  let cumTime = 0;
  STEPS.forEach((s, i) => {{
    const box = document.getElementById('step-' + s.id);
    const prog = box.querySelector('.step-progress');
    const start = cumTime;
    const end = cumTime + s.duration;
    let pct = 0;
    if (animTime >= end) pct = 100;
    else if (animTime > start) pct = (animTime - start) / s.duration * 100;
    prog.style.width = pct + '%';

    // Auto-select detail for current step
    if (animTime >= start && animTime < end && activeStep !== i) {{
      showDetail(i);
    }}

    // Spawn particles during active step
    if (pct > 0 && pct < 100 && playing) {{
      particleCounter++;
      if (particleCounter % 3 === 0) {{
        const pDiv = box.querySelector('.particles');
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = pct + '%';
        p.style.top = (20 + Math.random() * 60) + '%';
        p.style.background = s.color;
        p.style.boxShadow = '0 0 6px ' + s.color + '99';
        pDiv.appendChild(p);
        setTimeout(() => p.remove(), 800);
      }}
    }}

    cumTime = end;
  }});

  // Timeline cursor
  const cursorPct = E2E > 0 ? (animTime / E2E * 100) : 0;
  document.getElementById('tlCursor').style.left = cursorPct + '%';
}}

// Initialize
showDetail(0);
setSpeed(1);
</script>
"""

components.html(flowchart_html, height=720, scrolling=False)

# ─── TL_Param Timeline ──────────────────────────────────────────────────────
_active_opts = [
    label for label, flag in [
        ("GQA", use_gqa),
        ("Batching", use_batching),
        ("SpecDecode", use_spec_decode),
        ("SHARP", use_nvlink_sharp),
        ("Overlap", use_overlap),
        ("PP", use_pp),
    ] if flag
]
_opt_label = ", ".join(_active_opts) if _active_opts else "No Optimizations"
st.header(f"TL_Param — Latency Timeline ({_opt_label})")

tl_data = []
for i, name in enumerate(vr_tl["step_names"]):
    tl_data.append({
        "Step": f"{i+1}. {name}",
        "VR Duration (s)": vr_tl["steps"][i],
        "GB200 Duration (s)": gb_tl["steps"][i],
    })
    if i == 2:  # Step 3: Prefill+KV Write
        tl_data.append({"Step": "   ↳ Compute",
                         "VR Duration (s)": vr_tl["n3_compute"],
                         "GB200 Duration (s)": gb_tl["n3_compute"]})
        tl_data.append({"Step": "   ↳ HBM",
                         "VR Duration (s)": vr_tl["n3_hbm"],
                         "GB200 Duration (s)": gb_tl["n3_hbm"]})
        tl_data.append({"Step": "   ↳ NVLink",
                         "VR Duration (s)": vr_tl["n3_nvlink"],
                         "GB200 Duration (s)": gb_tl["n3_nvlink"]})
    elif i == 3:  # Step 4: Decode-First Token
        tl_data.append({"Step": "   ↳ HBM",
                         "VR Duration (s)": vr_tl["n4_hbm"],
                         "GB200 Duration (s)": gb_tl["n4_hbm"]})
    elif i == 4:  # Step 5: Decode-All Tokens
        tl_data.append({"Step": "   ↳ HBM",
                         "VR Duration (s)": vr_tl["n5_hbm"],
                         "GB200 Duration (s)": gb_tl["n5_hbm"]})
        tl_data.append({"Step": "   ↳ NVLink",
                         "VR Duration (s)": vr_tl["n5_nvlink"],
                         "GB200 Duration (s)": gb_tl["n5_nvlink"]})
tl_data.append({"Step": "E2E",
                 "VR Duration (s)": vr_tl["e2e"],
                 "GB200 Duration (s)": gb_tl["e2e"]})
tl_data.append({"Step": "Avg Time/Output Token (s)",
                 "VR Duration (s)": vr_tl["avg_time_per_tok"],
                 "GB200 Duration (s)": gb_tl["avg_time_per_tok"]})
tl_data.append({"Step": "Output tok/s",
                 "VR Duration (s)": vr_tl["tok_per_sec"],
                 "GB200 Duration (s)": gb_tl["tok_per_sec"]})

st.dataframe(pd.DataFrame(tl_data), use_container_width=True, hide_index=True)

# ─── Revenue Model ───────────────────────────────────────────────────────────
st.header("Revenue Model")

rev_col1, rev_col2 = st.columns(2)

def show_revenue(col, title, rev, dc, tl_data_dict):
    with col:
        st.subheader(title)
        st.markdown("**Section 1: Inference Configuration**")
        st.write(f"E2E Latency: {tl_data_dict['e2e']:.6f}s")
        st.write(f"Racks: {dc['n_racks']:,} | GPUs: {dc['total_gpus']:,}")
        st.write(f"Utilization: {gpu_utilization:.0%} | Uptime: {uptime:.1%}")
        st.write(f"Eff. Annual Seconds: {rev['eff_annual_sec']:,.0f}")
        st.write(f"Requests/rack/yr: {rev['requests_per_rack_yr']:,.0f}")

        st.markdown("**Section 2: Annual Throughput**")
        st.write(f"Total Requests/yr: {rev['total_requests_yr']:,.0f}")
        st.write(f"Input Tokens (M): {rev['annual_input_m']:,.0f}")
        st.write(f"Output Tokens (M): {rev['annual_output_m']:,.0f}")
        st.write(f"Tokens/sec (DC-wide): {rev['tokens_per_sec']:,.0f}")
        st.write(f"Requests/sec: {rev['requests_per_sec']:,.0f}")

        st.markdown("**Section 3: Revenue**")
        st.write(f"Input Price: \${input_price}/M | Output Price: \${output_price}/M")
        st.write(f"Blended Rate: ${rev['blended_rate']:.2f}/M tokens")
        st.write(f"Input Revenue: **${rev['input_revenue']:,.1f}M**")
        st.write(f"Output Revenue: **${rev['output_revenue']:,.1f}M**")
        st.write(f"**TOTAL ANNUAL REVENUE: ${rev['total_revenue']:,.1f}M**")
        st.write(f"Revenue/rack/yr: ${rev['rev_per_rack']:,.0f}")
        st.write(f"Revenue/OpEx: {rev['rev_to_opex']:.1f}x")
        st.write(f"Cost/Mtok: ${rev['cost_per_mtok']:.2f}")


show_revenue(rev_col1, "Vera Rubin NVL72", vr_rev, vr_dc, vr_tl)
show_revenue(rev_col2, "GB200 NVL72", gb_rev, gb_dc, gb_tl)

# ─── DC Cost Model ───────────────────────────────────────────────────────────
with st.expander("🏗️ DC Cost Model", expanded=False):
    dc1, dc2 = st.columns(2)
    for col, title, dc in [(dc1, "Vera Rubin NVL72", vr_dc), (dc2, "GB200 NVL72", gb_dc)]:
        with col:
            st.subheader(title)
            st.write(f"IT Power: {dc['it_power_mw']:,.0f} MW | Facility: {dc['facility_power_mw']:,.0f} MW")
            st.write(f"Racks: {dc['n_racks']:,} | GPUs: {dc['total_gpus']:,}")
            st.write(f"**Total CapEx: ${dc['total_capex']:,.0f}M**")
            st.write(f"  IT Hardware: ${dc['it_hw']:,.0f}M")
            st.write(f"**Total Annual OpEx: ${dc['total_opex']:,.0f}M**")
            st.write(f"  Depreciation: ${dc['depr_total']:,.0f}M")
            st.write(f"  Electricity: ${dc['elec_cost']:,.0f}M")
            st.write(f"  Maint & Ops: ${dc['maint_ops']:,.0f}M")

# ─── WP_Param Details ────────────────────────────────────────────────────────
with st.expander("🔧 WP_Param — Workload Parameters (selected rack)", expanded=False):
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Prefill Phase**")
        st.write(f"Prefill FLOPs: {pf_flop_total:.3e}")
        st.write(f"Prefill Compute Time (with MFU): {pf_compute_mfu:.6f}s")
        st.write(f"Prefill HBM Traffic: {pf_hbm_total:.3e} bytes")
        st.write(f"Prefill HBM Time: {pf_hbm_time:.6f}s")
        st.write(f"Prefill NVLink Time (SHARP): {pf_nvl_time_sharp:.6f}s")
        st.write(f"**Total Prefill Time: {total_prefill_time:.6f}s**")
    with w2:
        st.markdown("**Decode Phase**")
        st.write(f"Decode FLOPs/tok (all layers): {dc_flop_all_layers:.3e}")
        st.write(f"Decode Compute/tok: {dc_compute_per_tok:.3e}s")
        st.write(f"Decode HBM Time/tok (baseline): {dc_hbm_time:.6f}s")
        st.write(f"Decode HBM Time/tok (GQA): {gqa_hbm_time:.6f}s")
        st.write(f"Decode HBM Time/tok (GQA+Batch): {batch_hbm_time_per_tok:.6e}s")
        st.write(f"NVLink Time/tok (optimized): {nvl_time_optimized:.3e}s")
        st.write(f"Weight Memory: {weight_memory/1e12:.2f} TB")
        st.write(f"GPU Memory Headroom: {(float(rp['gpu_mem'])/n_gpu*1e12 - (parameters*bytes_per_param/n_gpu + 2*d_model*(input_tokens+output_tokens)*n_layers/n_gpu))/1e9:.1f} GB")

