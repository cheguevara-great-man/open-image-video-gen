# 调研报告

2026-11-08 的两份深度调研原文（判定口径：fits / fits-with-quantization-or-offload / does-not-fit，逐条带来源）。本仓库的图像部分代码就是从这两份调研里落地的。

| 文件 | 内容 |
|---|---|
| `kaggle-gpu-tpu-what-can-run.md` | Kaggle 免费 GPU（2× T4）与 TPU（v5e-8）上到底能跑什么：图像、视频、LLM、训练的逐项判定（英文工作稿，是中文成品报告的底稿） |
| `kaggle-dual-t4-what-can-run.md` | 双卡专题：第二张 T4 到底解锁什么 — 三种双卡模式（吞吐并行 / 组件拆分 / 单图并行）的收益与边界，本项目"双开"路线即出自这里 |

这两份报告在 Muse 里另有排版成页的版本：《Colab 上能跑什么》（Colab 口径）和《Kaggle 双卡 T4 能跑什么》。
