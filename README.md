# FTGP Token Generation Calculator

An interactive calculator for estimating **AI inference throughput and revenue** for large-scale AI data centers, built with [Streamlit](https://streamlit.io).

## Overview

This tool models the full computation chain for LLM inference:

**Config → WP_Param → TL_Param → Revenue_Model**

Given a hardware configuration and model spec, it calculates:

- **WP_Param**: Per-layer FLOP counts, HBM traffic, and NVLink communication for both prefill and decode phases
- **TL_Param**: End-to-end latency timeline (tokenization → embedding → prefill → decode → detokenization) with GQA, batching, and NVLink SHARP optimizations
- **Revenue Model**: Token throughput, pricing tier, and estimated revenue across rack configurations

## Supported Hardware

| Rack Type | GPU | GPUs/Rack | Power |
|---|---|---|---|
| GB200 NVL72 | Blackwell B200 | 72 | 139 kW |
| Vera Rubin NVL72 | Rubin GPU | 72 | 190 kW |

## Key Inputs

| Category | Parameters |
|---|---|
| Data Center | Total power (GW), PUE, rack type |
| Model | Parameters, vocab size, layers, heads, precision (FP4/FP8/FP16) |
| Tokens | Input tokens, output tokens |
| Optimization | GQA KV heads, batch size, speculative decoding, NVLink SHARP, TP/PP |
| Revenue | GPU utilization rate, uptime |

## Getting Started

### Run locally

```bash
pip install -r requirements.txt
streamlit run ftgp_calculator.py
```

### Deploy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Deploy via [Streamlit Community Cloud](https://share.streamlit.io) by connecting this repository.

## Requirements

- Python 3.9+
- streamlit
- pandas
- openpyxl
