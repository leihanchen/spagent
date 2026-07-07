# Depth Anything V3 Tool Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Depth Anything V3 (`DA3NESTED-GIANT-LARGE-1.1`) as a new `backend="v3"` option for the existing `DepthEstimationTool` while keeping V2 as the default and completely untouched.

**Architecture:** New V3 model code, server, client, and mock under `external_experts/Depth_AnythingV2/` alongside V2. Tool (`depth_tool.py`) gains a `backend` param that dispatches to either V2 or V3 client. Catalog entry gains `backend="v2"` default — users opt into V3 via overrides.

**Tech Stack:** Python 3.11+, torch >= 2.0, transformers >= 4.45, huggingface_hub, Flask, opencv-python, numpy, Pillow, matplotlib, DA3NESTED-GIANT-LARGE-1.1

## Global Constraints

- V2 backend remains the default — existing users see zero change
- Same tool name (`depth_estimation_tool`) and catalog key (`"depth"`)
- Same `use_mock` semantics for both backends
- V2 files (`depth_server.py`, `depth_client.py`, `depth_anything_v2/`) are completely untouched
- V3 server runs on port 20039
- Model auto-downloads from HuggingFace on first server launch
- Error handling: image not found → `{"success": False, "error": "..."}`; invalid mode → same; server unreachable → same
- Mock mode returns synthetic data, never fails unless image missing
- Outputs saved under `outputs/` directory

---
