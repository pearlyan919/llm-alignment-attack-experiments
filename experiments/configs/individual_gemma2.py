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
    config.result_prefix = str(_REPO / "results" / "gemma-2")

    config.tokenizer_paths = [str(_MODELS / "gemma-2-2b-it")]
    config.model_paths = [str(_MODELS / "gemma-2-2b-it")]
    config.conversation_templates = ['gemma']
    config.attack = 'gcg'

    config.train_data = str(_DATA / "advbench" / "harmful_behaviors.csv")
    # 与权威结果 JSON（gemma2_advbench_final2）一致：
    # 8GB 显存下 gemma-2-2b 全量 10 目标难以容纳，权威运行取 offset=3 的 7 条
    config.n_train_data = 7
    config.data_offset = 3

    # 8GB 显存限制：2B 模型 logits 显存仍大，batch 512 会 OOM，权威运行降为 16
    config.batch_size = 16

    # topk 256 时 top-k 采样扩展过慢，权威运行降为 64
    config.topk = 64

    return config
