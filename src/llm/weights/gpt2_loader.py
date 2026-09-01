import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt_download3 import download_and_load_gpt2

from llm.gpt.model import GPTModel
from llm.gpt.config import GPT_CONFIG_124M




MODEL_CONFIGS = {
    "124M": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    },
    "355M": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
    },
    "774M": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20,
    },
    "1558M": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25,
    },
}


def assign(left, right):
    """
    Convert a NumPy array into a PyTorch Parameter
    after checking that both tensors have the same shape.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. "
            f"Left: {left.shape}, "
            f"Right: {right.shape}"
        )

    return torch.nn.Parameter(
        torch.tensor(right)
    )


def load_weights_into_gpt(model, params):
    """
    Load OpenAI GPT-2 pretrained weights into our GPTModel.
    """

    # Token and positional embeddings
    model.pos_emb.weight = assign(
        model.pos_emb.weight,
        params["wpe"],
    )

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["wte"],
    )

    # Transformer blocks
    for b in range(len(params["blocks"])):

        # Attention QKV weights

        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"],
            3,
            axis=-1,
        )

        model.trf_blocks[b].att.W_query.weight = assign(
            model.trf_blocks[b].att.W_query.weight,
            q_w.T,
        )

        model.trf_blocks[b].att.W_key.weight = assign(
            model.trf_blocks[b].att.W_key.weight,
            k_w.T,
        )

        model.trf_blocks[b].att.W_value.weight = assign(
            model.trf_blocks[b].att.W_value.weight,
            v_w.T,
        )

        # Attention QKV biases

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"],
            3,
            axis=-1,
        )

        model.trf_blocks[b].att.W_query.bias = assign(
            model.trf_blocks[b].att.W_query.bias,
            q_b,
        )

        model.trf_blocks[b].att.W_key.bias = assign(
            model.trf_blocks[b].att.W_key.bias,
            k_b,
        )

        model.trf_blocks[b].att.W_value.bias = assign(
            model.trf_blocks[b].att.W_value.bias,
            v_b,
        )

        # Attention output projection

        model.trf_blocks[b].att.out_proj.weight = assign(
            model.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )

        model.trf_blocks[b].att.out_proj.bias = assign(
            model.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # Feed-forward network

        model.trf_blocks[b].ff.layers[0].weight = assign(
            model.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )

        model.trf_blocks[b].ff.layers[0].bias = assign(
            model.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )

        model.trf_blocks[b].ff.layers[2].weight = assign(
            model.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )

        model.trf_blocks[b].ff.layers[2].bias = assign(
            model.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # Layer normalization

        model.trf_blocks[b].norm1.scale = assign(
            model.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )

        model.trf_blocks[b].norm1.shift = assign(
            model.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )

        model.trf_blocks[b].norm2.scale = assign(
            model.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"],
        )

        model.trf_blocks[b].norm2.shift = assign(
            model.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"],
        )

    # Final layer normalization

    model.final_norm.scale = assign(
        model.final_norm.scale,
        params["g"],
    )

    model.final_norm.shift = assign(
        model.final_norm.shift,
        params["b"],
    )

    # GPT-2 uses tied input/output embeddings
    model.out_head.weight = assign(
        model.out_head.weight,
        params["wte"],
    )

    return model


def load_pretrained_gpt2(
    model_size="124M",
    models_dir="gpt2",
    device=None,
):
    """
    Download/load pretrained GPT-2 weights and
    load them into our GPTModel implementation.
    """

    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model size: {model_size}"
        )

    # Download/load OpenAI weights
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=models_dir,
    )

    # Build model configuration
    config = GPT_CONFIG_124M.copy()

    config.update(
        MODEL_CONFIGS[model_size]
    )

    config.update(
        {
            "context_length": settings["n_ctx"],
            "qkv_bias": True,
        }
    )

    # Create model
    model = GPTModel(config)

    # Load pretrained weights
    model = load_weights_into_gpt(
        model,
        params,
    )

    # Device
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model.to(device)
    model.eval()

    return model