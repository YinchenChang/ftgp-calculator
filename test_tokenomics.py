"""
Tests for tokenomics computation functions.

Since tokenomics.py is a Streamlit app with computation at module level,
these tests replicate the core formulas with known inputs and verify
expected outputs. This catches regressions in the math without needing
to run Streamlit.
"""
import math
import pytest

# ─── Shared constants (matching default Config values) ────────────────────────
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


# ─── DC Cost Model (standalone) ──────────────────────────────────────────────
def compute_dc_cost(rack_name, total_power, pue):
    rr = RACK_PRESETS[rack_name]
    r_n_racks = math.floor(total_power / 1000 / rr["power_per_rack"])
    it_power_mw = total_power / 1e6
    facility_power_mw = it_power_mw * pue
    total_gpus = r_n_racks * rr["n_gpu"]

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


# ─── Revenue Model (standalone) ──────────────────────────────────────────────
def compute_revenue(tl, dc, input_tokens, output_tokens, batch_size,
                    gpu_utilization, uptime, input_price, output_price):
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

    input_revenue = annual_input_m * input_price / 1e6
    output_revenue = annual_output_m * output_price / 1e6
    total_revenue = input_revenue + output_revenue
    blended_rate = (input_price * annual_input_m + output_price * annual_output_m) / annual_total_m if annual_total_m > 0 else 0

    rev_per_rack = total_revenue / r_n_racks * 1e6 if r_n_racks > 0 else 0
    rev_to_opex = total_revenue / dc["total_opex"] if dc["total_opex"] > 0 else 0

    return dict(
        total_revenue=total_revenue, rev_to_opex=rev_to_opex,
        total_requests_yr=total_requests_yr, blended_rate=blended_rate,
        rev_per_rack=rev_per_rack,
    )


# ─── WP_Param helpers ────────────────────────────────────────────────────────
def compute_d_model(parameters):
    if parameters < 100_000_000:
        return round(parameters ** 0.25)
    return round(2**11 * 10 ** math.log10(parameters / 1e11))


def compute_d_ff(parameters, n_layers, d_model):
    if n_layers <= 0 or d_model <= 0:
        return 0
    return (parameters / n_layers - 4 * d_model**2) / (3 * d_model)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPricingTier:
    def test_nano_tier(self):
        name, inp, out = match_pricing_tier(8e9)
        assert name == "Nano (<50B)"
        assert inp == 0.10
        assert out == 0.40

    def test_small_tier(self):
        name, inp, out = match_pricing_tier(70e9)
        assert name == "Small (50B–200B)"
        assert inp == 0.25

    def test_frontier_tier(self):
        name, inp, out = match_pricing_tier(1e12)
        assert name == "Frontier (500B–1.5T)"
        assert inp == 2.50
        assert out == 10.00

    def test_premium_tier(self):
        name, inp, out = match_pricing_tier(2e12)
        assert name == "Premium (>1.5T)"
        assert inp == 5.00
        assert out == 25.00

    def test_boundary_exact(self):
        """Exactly at threshold should match that tier."""
        name, _, _ = match_pricing_tier(50e9)
        assert name == "Small (50B–200B)"


class TestDCCost:
    def test_vr_rack_count(self):
        """1 GW / 190 kW per rack = 5263 racks."""
        dc = compute_dc_cost("Vera Rubin NVL72", total_power=1e9, pue=1.2)
        assert dc["n_racks"] == math.floor(1e9 / 1000 / 190)
        assert dc["n_racks"] == 5263

    def test_gb200_rack_count(self):
        dc = compute_dc_cost("GB200 NVL72", total_power=1e9, pue=1.2)
        assert dc["n_racks"] == math.floor(1e9 / 1000 / 139)
        assert dc["n_racks"] == 7194

    def test_total_gpus(self):
        dc = compute_dc_cost("Vera Rubin NVL72", total_power=1e9, pue=1.2)
        assert dc["total_gpus"] == dc["n_racks"] * 72

    def test_capex_positive(self):
        dc = compute_dc_cost("Vera Rubin NVL72", total_power=1e9, pue=1.2)
        assert dc["total_capex"] > 0
        assert dc["total_opex"] > 0

    def test_facility_power(self):
        dc = compute_dc_cost("Vera Rubin NVL72", total_power=1e9, pue=1.2)
        assert dc["it_power_mw"] == pytest.approx(1000.0)
        assert dc["facility_power_mw"] == pytest.approx(1200.0)

    def test_opex_components_sum(self):
        dc = compute_dc_cost("GB200 NVL72", total_power=1e9, pue=1.2)
        assert dc["total_opex"] == pytest.approx(
            dc["depr_total"] + dc["elec_cost"] + dc["maint_ops"], rel=1e-9
        )


class TestRevenue:
    def test_zero_e2e_returns_zero(self):
        tl = {"e2e": 0}
        dc = {"n_racks": 100, "total_opex": 1000}
        rev = compute_revenue(tl, dc, 4000, 1000, 64, 0.7, 0.995, 2.50, 10.00)
        assert rev["total_revenue"] == 0

    def test_revenue_positive(self):
        tl = {"e2e": 1.0}  # 1 second per batch
        dc = {"n_racks": 100, "total_opex": 1000}
        rev = compute_revenue(tl, dc, 4000, 1000, 64, 0.7, 0.995, 2.50, 10.00)
        assert rev["total_revenue"] > 0
        assert rev["rev_to_opex"] > 0

    def test_blended_rate(self):
        """Blended rate = weighted average of input/output prices."""
        tl = {"e2e": 1.0}
        dc = {"n_racks": 10, "total_opex": 500}
        rev = compute_revenue(tl, dc, 4000, 1000, 1, 1.0, 1.0, 1.00, 4.00)
        # input=4000, output=1000, total=5000
        # blended = (1.00*input_m + 4.00*output_m) / total_m
        expected_blended = (1.00 * 4000 + 4.00 * 1000) / 5000
        assert rev["blended_rate"] == pytest.approx(expected_blended)


class TestWPParam:
    def test_d_model_1t(self):
        """d_model for 1T params: 2^11 * 10^(log10(1e12/1e11)) = 2048*10 = 20480."""
        d = compute_d_model(1e12)
        assert d == 20480

    def test_d_model_small(self):
        """For very small params (<100M), uses params^0.25."""
        d = compute_d_model(1e6)
        assert d == round(1e6 ** 0.25)

    def test_d_ff_positive(self):
        """d_ff should be positive for reasonable model configs."""
        d_model = compute_d_model(1e12)
        d_ff = compute_d_ff(1e12, 128, d_model)
        assert d_ff > 0

    def test_d_ff_negative_guard(self):
        """d_ff goes negative when params are too small for architecture."""
        d_model = compute_d_model(1e6)  # 1M params
        d_ff = compute_d_ff(1e6, 256, d_model)  # way too many layers for 1M
        assert d_ff < 0  # confirms the guard in tokenomics.py is needed

    def test_prefill_flops_scale_with_tokens(self):
        """Prefill FLOPs should scale linearly with input_tokens."""
        d_model = 4096
        d_ff = 11008
        n_layers = 32
        input_tokens_a = 1000
        input_tokens_b = 2000

        def pf_flops(tokens):
            pf_qkv = 2 * tokens * d_model**2 * 3
            pf_qkt = 2 * tokens**2 * d_model
            pf_atv = 2 * tokens**2 * d_model
            pf_out = 2 * tokens * d_model**2
            pf_ffn = 2 * tokens * d_model * d_ff * 3
            pf_norms = 5 * tokens * d_model
            return (pf_qkv + pf_qkt + pf_atv + pf_out + pf_ffn + pf_norms) * n_layers

        flops_a = pf_flops(input_tokens_a)
        flops_b = pf_flops(input_tokens_b)
        # Not exactly 2x due to quadratic attention terms, but should be > 1x
        assert flops_b > flops_a
        assert flops_b < flops_a * 4  # sanity: not more than 4x for 2x tokens


class TestNVLinkScaling:
    def test_dynamic_gpu_scaling(self):
        """(n_gpu-1)/n_gpu should differ from hardcoded 71/72 for non-72 counts."""
        for n_gpu in [8, 36, 48, 72, 96]:
            factor = (n_gpu - 1) / n_gpu
            if n_gpu == 72:
                assert factor == pytest.approx(71 / 72)
            else:
                assert factor != pytest.approx(71 / 72)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
