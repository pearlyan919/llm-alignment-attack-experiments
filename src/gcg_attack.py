"""
GCG 攻击的统一调用入口。
基于 llm_attacks 官方包封装，提供单模型/单行为的 GCG 攻击快捷接口。
"""
import sys, os
from pathlib import Path

# 确保仓库根在路径中，以便 import llm_attacks
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import numpy as np
from llm_attacks import (
    AttackPrompt,
    PromptManager,
    IndividualPromptAttack,
    get_goals_and_targets,
    get_workers,
)


def run_gcg_attack(
    model_path: str,
    tokenizer_path: str,
    conversation_template: str,
    goals: list,
    targets: list,
    n_steps: int = 500,
    test_steps: int = 50,
    batch_size: int = 32,
    topk: int = 256,
    lr: float = 0.01,
    control_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    device: str = "cuda:0",
    model_kwargs: dict = None,
    tokenizer_kwargs: dict = None,
    logfile: str = None,
):
    """
    对单个模型运行 GCG 对抗后缀优化。

    参数
    ----
    model_path : str
        HuggingFace 模型目录或模型 ID。
    tokenizer_path : str
        分词器目录或 ID（通常与 model_path 相同）。
    conversation_template : str
        对话模板名称，如 'llama-2', 'llama-3', 'gemma', 'vicuna' 等。
    goals : list[str]
        攻击目标（有害行为指令）。
    targets : list[str]
        期望模型输出的目标前缀。
    n_steps, test_steps, batch_size, topk, lr :
        GCG 优化超参。
    control_init : str
        对抗后缀初始字符串。
    device : str
        运行设备，如 'cuda:0' 或 'cpu'。
    model_kwargs : dict | None
        额外模型加载参数，如 {"load_in_4bit": True, ...}。
    tokenizer_kwargs : dict | None
        额外分词器参数，如 {"use_fast": True}。
    logfile : str | None
        结果保存路径（JSON）。默认不保存。

    返回
    ----
    controls : list[str]
        每个目标对应的最终对抗后缀。
    tests : list[dict]
        每一步的测试记录。
    """
    if model_kwargs is None:
        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {"use_fast": False}

    # 构建 mock config（兼容 llm_attacks 的 get_workers/get_goals_and_targets）
    class _Config:
        pass

    cfg = _Config()
    cfg.model_paths = [model_path]
    cfg.tokenizer_paths = [tokenizer_path]
    cfg.model_kwargs = [model_kwargs]
    cfg.tokenizer_kwargs = [tokenizer_kwargs]
    cfg.conversation_templates = [conversation_template]
    cfg.devices = [device]
    cfg.train_data = ""
    cfg.test_data = ""
    cfg.n_train_data = len(goals)
    cfg.n_test_data = 0
    cfg.data_offset = 0
    cfg.control_init = control_init
    cfg.attack = "gcg"
    cfg.target_weight = 1.0
    cfg.control_weight = 0.0
    cfg.progressive_goals = False
    cfg.progressive_models = False
    cfg.anneal = False
    cfg.incr_control = False
    cfg.stop_on_success = False
    cfg.verbose = True
    cfg.allow_non_ascii = False
    cfg.num_train_models = 1
    cfg.transfer = False
    cfg.result_prefix = "results/gcg"
    cfg.lr = lr
    cfg.temp = 1.0
    cfg.filter_cand = True
    cfg.gbda_deterministic = True

    workers, test_workers = get_workers(cfg)

    # 目标处理：官方代码中的随机替换策略
    process_fn = lambda s: s.replace("Sure, h", "H")
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    targets = [
        process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in targets
    ]

    managers = {
        "AP": AttackPrompt,
        "PM": PromptManager,
        "MPA": IndividualPromptAttack,
    }

    attack = IndividualPromptAttack(
        goals,
        targets,
        workers,
        control_init=control_init,
        logfile=logfile,
        managers=managers,
        test_goals=[],
        test_targets=[],
        test_workers=test_workers,
        mpa_deterministic=True,
        mpa_lr=lr,
        mpa_batch_size=batch_size,
        mpa_n_steps=n_steps,
    )

    attack.run(
        n_steps=n_steps,
        batch_size=batch_size,
        topk=topk,
        temp=1.0,
        target_weight=1.0,
        control_weight=0.0,
        test_steps=test_steps,
        anneal=False,
        incr_control=False,
        stop_on_success=False,
        verbose=True,
        filter_cand=True,
        allow_non_ascii=False,
    )

    for w in workers + test_workers:
        w.stop()

    return attack.controls, attack.tests


if __name__ == "__main__":
    # 最小演示：加载 yaml 配置并运行（需 GPU）
    import yaml, argparse

    parser = argparse.ArgumentParser(description="GCG Attack Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 读取目标数据（CSV）
    import csv
    train_file = cfg["data"]["train"]
    n = cfg["data"]["n_train"]
    offset = cfg["data"]["offset"]
    goals, targets = [], []
    with open(train_file, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < offset:
                continue
            if len(goals) >= n:
                break
            goals.append(row["goal"])
            targets.append(row["target"])

    mcfg = cfg["model"]
    acfg = cfg["attack"]
    outcfg = cfg["output"]

    controls, tests = run_gcg_attack(
        model_path=mcfg["path"],
        tokenizer_path=cfg["tokenizer"]["path"],
        conversation_template=mcfg["conversation_template"],
        goals=goals,
        targets=targets,
        n_steps=acfg["n_steps"],
        test_steps=acfg["test_steps"],
        batch_size=acfg["batch_size"],
        topk=acfg["topk"],
        lr=acfg["lr"],
        control_init=acfg["control_init"],
        model_kwargs=mcfg.get("kwargs", {}),
        tokenizer_kwargs=cfg["tokenizer"].get("kwargs", {}),
        logfile=f"{outcfg['result_prefix']}.json" if outcfg.get("result_prefix") else None,
    )
    print(f"攻击完成，生成了 {len(controls)} 个对抗后缀。")

