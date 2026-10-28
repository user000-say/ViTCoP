"""Read-only model-config compatibility probe. NO weights are loaded."""
import argparse, json
from pathlib import Path


def inspect_config(data: dict) -> dict:
    mt = data.get("model_type","unknown")
    txt = data.get("text_config") or data
    vis = data.get("vision_config") or {}
    types = txt.get("layer_types") or []
    l = txt.get("num_hidden_layers") or data.get("num_hidden_layers")
    if l is None:
        raise ValueError("decoder layers not found in local config.json")
    return {
        "model_type":mt,
        "decoder_layers":int(l),
        "vision_depth":vis.get("depth",vis.get("num_hidden_layers")),
        "spatial_merge_size":vis.get("spatial_merge_size"),
        "vision_patch_size":vis.get("patch_size"),
        "deepstack_indexes":vis.get("deepstack_visual_indexes",[]),
        "hybrid_layer_types":sorted(set(types)),
        "full_attention_layers_1based":[i+1 for i,t in enumerate(types) if t=="full_attention"],
        "linear_attention_layers":sum(t=="linear_attention" for t in types),
        "declared_precision":txt.get("dtype", data.get("dtype")),
        "guardrails":[
            "No CUDA model run performed",
            "Use observed post-merge image-token N, not raw pre-merge patch count",
            "Preserve original visual coordinates / M-RoPE and later cache state",
            "Validate image placeholder count and DeepStack injection before real pruning",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config-json",required=True)
    args=p.parse_args()
    inp=json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    print(json.dumps(inspect_config(inp),indent=2,ensure_ascii=False))


if __name__=="__main__":main()
