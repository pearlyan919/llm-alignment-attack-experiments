"""
仓库统一路径与字体加载。

各脚本通过本模块从仓库根推导 results / outputs / data 的位置，
模型目录可用环境变量 GCG_MODELS_DIR 覆盖（默认 <repo>/models）。
绘图时自动探测本机中文字体；找不到时降级为默认字体，图中中文可能乱码。
"""
import os
import sys
from pathlib import Path

# 仓库根 = 本文件的上两级目录（scripts/paths.py -> repo root）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 三个实验的权威原始结果（experiments/main.py 输出后归档至此）
RESULTS_DIR = REPO_ROOT / "results"

# 输出目录：三实验产物
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUT_EXP1 = OUTPUTS_DIR / "exp1"
OUT_EXP2 = OUTPUTS_DIR / "exp2"
OUT_EXP3 = OUTPUTS_DIR / "exp3"
for _d in (OUTPUTS_DIR, OUT_EXP1, OUT_EXP2, OUT_EXP3):
    _d.mkdir(parents=True, exist_ok=True)

# 数据目录（官方 advbench + 迁移行为集）
DATA_DIR = REPO_ROOT / "data"
ADVBENCH_DIR = DATA_DIR / "advbench"

# 模型目录：环境变量 GCG_MODELS_DIR 优先，否则 <repo>/models
MODELS_DIR = Path(os.environ.get("GCG_MODELS_DIR", str(REPO_ROOT / "models")))

# 三模型在 MODELS_DIR 下的目录名
MODEL_SUBDIRS = {
    "llama-2": "llama-2-7b-chat-hf",
    "llama-3.2": "Llama-3.2-3B-Instruct",
    "gemma-2": "gemma-2-2b-it",
}

# 兼容前缀：experiments/configs/*.py 的 result_prefix 与历史归档名可能不同，
# 例如 gemma-2 历史上以 gemma2_advbench_final2_*.json 归档，
# 现在 config 的 result_prefix 为 results/gemma-2，输出 gemma-2_*.json。
PREFIX_ALIASES = {
    "llama-2": ["llama-2_"],
    "llama-3.2": ["llama-3.2_"],
    "gemma-2": ["gemma2_advbench_final2", "gemma-2_"],
}


def result_json(tag: str):
    """返回某模型的 GCG 结果 JSON 路径：跨全部兼容前缀按修改时间取最新，
    兼容官方 main.py 输出的时间戳命名（Windows 下冒号会被替换为 '-'）。"""
    import glob
    files = []
    for pref in PREFIX_ALIASES[tag]:
        files += list(glob.glob(str(RESULTS_DIR / (pref + "*.json"))))
    if not files:
        raise FileNotFoundError("results/ 下找不到 %s 的 GCG 结果 JSON（前缀 %s*）" % (
            tag, " / ".join(PREFIX_ALIASES[tag])))
    return Path(max(files, key=os.path.getmtime))


def model_path(tag: str) -> Path:
    """返回某模型目录；若以环境变量显式指定了完整路径则优先。"""
    env_key = {"llama-2": "GCG_MODEL_LLAMA2", "llama-3.2": "GCG_MODEL_LLAMA32",
               "gemma-2": "GCG_MODEL_GEMMA2"}[tag]
    if os.environ.get(env_key):
        return Path(os.environ[env_key])
    return MODELS_DIR / MODEL_SUBDIRS[tag]


def setup_matplotlib_font(plt=None):
    """
    配置 matplotlib 中文字体：
    依次尝试 Windows 字体 / 常见 Linux 中文字体；找到即注册并设置。
    无中文字体时返回 False，脚本可据此提醒用户，但不中断。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager

    candidates = [
        # Windows（本实验原开发环境）
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        # Linux 常见中文字体
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                prop = font_manager.FontProperties(fname=fp)
                if plt is not None:
                    plt.rcParams["font.family"] = prop.get_name()
                    plt.rcParams["axes.unicode_minus"] = False
                else:
                    import matplotlib
                    matplotlib.rcParams["font.family"] = prop.get_name()
                    matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False


def ensure_repo_on_syspath():
    """确保仓库根在 sys.path，便于 `import llm_attacks`。"""
    r = str(REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)

