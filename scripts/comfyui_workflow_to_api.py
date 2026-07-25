#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow (nodes+links, the kind you get from
"Save (non-API)" or by downloading an example from a repo) to ComfyUI's API
format (node_id -> {class_type, inputs}, the shape POST /prompt expects),
using the target ComfyUI's own live /object_info schemas to get exact
required+optional input ordering.

Written because Claude for Chrome wasn't reachable from this session (no
browser automation available) to drive the ComfyUI UI directly -- this
reproduces what the frontend does internally when you click "Queue Prompt",
so a render can be triggered over SSH/curl alone. Algorithm, verified
against the actual InfiniteTalk example workflow's own widgets_values
arrays for several nodes before trusting it generally:

  1. /object_info/<class_type> gives required+optional input names in order.
  2. The UI workflow's node["inputs"] lists every SOCKET-capable input
     (whether actually connected or not) -- these are excluded from the
     widget name list.
  3. What's left, in order, are the pure-widget names; zip them with
     node["widgets_values"] (unless it's already a dict, as some nodes'
     widgets_values are, e.g. VHS_VideoCombine).
  4. For each socket input with a non-null link, resolve via the links
     table to [str(source_node_id), source_output_slot]. Unconnected
     optional sockets are omitted (let the node use its own default).
  5. One extra wrinkle: the frontend auto-injects a "control_after_generate"
     string (fixed/randomize/...) immediately after a seed-control INT
     widget's value -- it's a UI-only concept, absent from /object_info, so
     it's dropped before zipping if detected.

Usage:
    python3 scripts/comfyui_workflow_to_api.py <workflow_ui.json> <out_api.json> \\
        [--comfy-url http://localhost:8188]

The PROJECT_OVERRIDES block below is specific to this project's InfiniteTalk
single-speaker example workflow (see docker/infinitetalk/, the bundled
/opt/workflows/infinitetalk_single_example.json) -- editing it to target a
different workflow means checking which node ids actually exist in *that*
graph first (open the raw JSON, or just skip overrides and inspect the
converted output).
"""
import argparse
import json
import urllib.request


def get_schema(comfy_url, class_type, cache):
    if class_type not in cache:
        with urllib.request.urlopen(f"{comfy_url}/object_info/{class_type}", timeout=15) as r:
            d = json.load(r)
        info = list(d.values())[0]
        req = list(info["input"].get("required", {}).keys())
        opt = list(info["input"].get("optional", {}).keys())
        cache[class_type] = req + opt
    return cache[class_type]


def convert(workflow_ui, comfy_url, skip_node_ids=frozenset()):
    nodes = workflow_ui["nodes"]
    links = {l[0]: (l[1], l[2]) for l in workflow_ui["links"]}  # link_id -> (src_node_id, src_slot)
    schema_cache = {}
    api = {}

    for n in nodes:
        nid = n["id"]
        if nid in skip_node_ids:
            continue
        ctype = n["type"]
        socket_inputs = n.get("inputs", []) or []
        socket_names = {i["name"] for i in socket_inputs}

        ordered_names = get_schema(comfy_url, ctype, schema_cache)
        widget_names = [nm for nm in ordered_names if nm not in socket_names]

        node_inputs = {}

        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            for k, v in wv.items():
                if k == "videopreview":  # UI-only preview state, not a real input
                    continue
                node_inputs[k] = v
        elif isinstance(wv, list):
            if len(wv) == len(widget_names) + 1 and "seed" in widget_names:
                drop_at = widget_names.index("seed") + 1
                wv = wv[:drop_at] + wv[drop_at + 1:]
            if len(wv) != len(widget_names):
                print(f"WARN node {nid} ({ctype}): {len(wv)} values vs {len(widget_names)} "
                      f"widget names {widget_names} -- values {wv}")
            for name, val in zip(widget_names, wv):
                node_inputs[name] = val

        for i in socket_inputs:
            link_id = i.get("link")
            if link_id is None:
                continue
            src_node, src_slot = links[link_id]
            if src_node in skip_node_ids:
                continue
            node_inputs[i["name"]] = [str(src_node), src_slot]

        api[str(nid)] = {"class_type": ctype, "inputs": node_inputs}

    return api


def apply_infinitetalk_single_overrides(api):
    """Overrides specific to this project's bundled InfiniteTalk single-
    speaker example workflow (node ids match that exact graph -- see
    docker/infinitetalk/Dockerfile's bundled /opt/workflows/
    infinitetalk_single_example.json). Points weight loaders at the
    filenames scripts/download_weights_infinitetalk.sh actually produces,
    swaps in this project's test image/audio, and reverts the sampler to
    non-distilled defaults (the demo's steps=4/cfg=1.0/flowmatch_distill
    are tuned for the step-distillation LoRA, node 138, which this skips
    entirely -- keeping those settings without the LoRA starves the
    diffusion process of steps to converge, confirmed live on the pod
    2026-07-25: first render was blurry/unstable with the demo's distilled
    settings and no LoRA)."""
    api["122"]["inputs"]["model"] = "wan2.1_i2v_720p_14B_fp16.safetensors"
    api["122"]["inputs"]["quantization"] = "disabled"
    api["122"]["inputs"].pop("lora", None)  # LoRA node (138) skipped

    api["120"]["inputs"]["model"] = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"
    api["129"]["inputs"]["model_name"] = "Wan2_1_VAE_bf16.safetensors"

    api["202"]["inputs"]["input file"] = "face_8.jpg"
    api["125"]["inputs"]["audio"] = "celyj.mp3"

    api["135"]["inputs"]["positive_prompt"] = (
        "A woman speaks calmly to the camera in a steady, quiet tone, with a "
        "neutral, composed expression."
    )

    api["131"]["inputs"]["save_output"] = True

    api["128"]["inputs"]["steps"] = 30
    api["128"]["inputs"]["cfg"] = 6.0
    api["128"]["inputs"]["scheduler"] = "unipc"
    api["128"]["inputs"]["shift"] = 5.0

    return api


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workflow_ui_json")
    p.add_argument("out_api_json")
    p.add_argument("--comfy-url", default="http://localhost:8188")
    p.add_argument("--skip-node", type=int, action="append", default=[138],
                    help="Node id(s) to drop entirely (default: 138, the "
                         "InfiniteTalk demo's step-distill LoRA node).")
    p.add_argument("--no-overrides", action="store_true",
                    help="Skip apply_infinitetalk_single_overrides -- use "
                         "for a different workflow than this project's "
                         "bundled InfiniteTalk single-speaker example.")
    args = p.parse_args()

    wf = json.load(open(args.workflow_ui_json))
    api = convert(wf, args.comfy_url, skip_node_ids=frozenset(args.skip_node))
    if not args.no_overrides:
        api = apply_infinitetalk_single_overrides(api)

    json.dump(api, open(args.out_api_json, "w"), indent=2)
    print(f"Wrote {args.out_api_json}, {len(api)} nodes")
