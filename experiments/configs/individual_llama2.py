import os
import torch
from pathlib import Path

os.sys.path.append("..")
from configs.template import get_config as default_config

# ---- 仓库根推导：configs 位于 <repo>/experiments/configs/ ----
# 模型/数据目录可通过环境变量覆盖（默认 <repo>/models、<repo>/data）
_REPO = Path(__file__).resolve().parents[2]
_MODELS = Path(os.environ.get("GCG_MODELS_DIR", str(_REPO / "models")))
_DATA = Path(os.environ.get("GCG_DATA_DIR", str(_REPO / "data")))


def get_config():

    config = default_config()

    # 结果写到 <repo>/results，与分析脚本读取位置一致
    config.result_prefix = str(_REPO / "results" / "llama-2")

    config.tokenizer_paths = [str(_MODELS / "llama-2-7b-chat-hf")]
    config.model_paths = [str(_MODELS / "llama-2-7b-chat-hf")]
    config.conversation_templates = ['llama-2']
    config.attack = 'gcg'

    config.train_data = str(_DATA / "advbench" / "harmful_behaviors.csv")
    config.n_train_data = 10
    config.data_offset = 0

    # Llama-2 需 fast tokenizer（slow 模式在新版 transformers 报错）
    config.tokenizer_kwargs = [{"use_fast": True}]

    # 8GB 显存限制：7B fp16 权重约 13.5GB 必然 OOM，改用 4bit 量化（权重约 3.5GB）
    config.model_kwargs = [{
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "device_map": "cuda:0",
    }]

    # 7B 激活内存大于 3B，batch 32 起步，OOM 时再降
    config.batch_size = 32

    return config
