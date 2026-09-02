import os
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
    config.result_prefix = str(_REPO / "results" / "llama-3.2")

    config.tokenizer_paths = [str(_MODELS / "Llama-3.2-3B-Instruct")]
    config.model_paths = [str(_MODELS / "Llama-3.2-3B-Instruct")]
    config.conversation_templates = ['llama-3']
    config.attack = 'gcg'

    config.train_data = str(_DATA / "advbench" / "harmful_behaviors.csv")
    config.n_train_data = 10
    config.data_offset = 0

    # llama-3.2 slow tokenizer 加载异常（返回 bool），必须用 fast 模式
    config.tokenizer_kwargs = [{"use_fast": True}]

    # 8GB 显存限制：batch 512 的 logits 张量会 OOM（模型权重 6.4GB），降为 32
    config.batch_size = 32

    return config
