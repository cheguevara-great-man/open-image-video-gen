# 图像生成

## FLUX.1-dev Q8 双卡并行（主成果）

在 Kaggle 免费 GPU（双 Tesla T4）上跑 FLUX.1-dev 的 Q8 GGUF 量化版，**双卡并行出图**：一张卡一个独立管线，一趟出 2 张 1024×1024。全部数字均为实测（2026-11-08 / 09）。

### 文件

| 文件 | 说明 |
|---|---|
| `flux-q8-dual-hf.ipynb` | **下载路线（推荐）**：权重开跑后从 Hugging Face 现下，编码后双开 |
| `flux-q8-dual.ipynb` | **数据集路线**：挂载预置数据集，整包拷到本地盘后双开 |
| `archive/flux-q8-ab-test.ipynb` | 同会话 A/B 测试：单卡 vs 双开（实验档案，不再日常使用） |

### 实测数据（T4 x2，1024×1024 / 28 步 / guidance 3.5）

- 双开出图墙钟 **234–264 秒 / 对 ≈ 117–132 秒 / 张**；同会话单卡全程 243.5 秒 / 张 → 双开产能约 **2.08×**
- 下载路线搬运：全部权重 **43.7 秒**（热缓存）/ 约 **90 秒**（冷启动）
- 数据集路线：整包拷本地 55.8 秒；但把 16.8 GiB 数据集搬运进环境约 12 分钟（交互式会话在开机前发生、不扣额度；API 批处理里计入运行窗口）
- API 批处理总窗口：下载版 **21.4 分钟** vs 数据集版 **31.5 分钟**（批处理自带约 15 分钟固定开销）；浏览器干净会话额度约 14–15 分钟
- 对照：全精度 FP16 纯生成 138.3 秒（最快），但要先下 53.9 GB + 加载约 2 分钟；Q8 免下载、加载约 14 秒，端到端更划算，故定为默认
- 其他实测：SDXL base 1.0 fp16 双卡 1024×1024 为 34.7 秒；FLUX 全精度 BF16 生成 754.1 秒（T4 不支持 BF16 张量核）；GGUF Q4_K_S 生成 190.5 秒、Q8_0 为 172.8 秒（均为单卡纯生成口径）

### 怎么跑

**浏览器**：把 `.ipynb` 导入 Kaggle → 数据集路线先 Add Data 挂数据集；下载路线先在笔记本设置里挂名为 `HF_TOKEN` 的 Secret → Run All。

**API**：`kaggle kernels push`（每次推送 = 新版本 = 一次运行）；取产物用 `kaggle kernels output --file-pattern ".*\.(png|log)$"`，不要整包拉。

前置条件：

- 下载路线：HF 账号需已获 FLUX.1-dev 访问授权，令牌存 Kaggle Secrets（**本仓库版本只走 Secrets，不含任何令牌**）。注意：经 API/CLI 推送的版本读不到网页端挂载的 Secrets（两次实测确认），API 场景需自行解决令牌注入
- 数据集路线：需要一个同构数据集（Q8_0 GGUF + fp8 T5 + CLIP/VAE/tokenizer/配置），本项目用的是私有数据集 `lei1237/flux1-dev-q8-gguf`，他人需自备

### 关键坑（全部实测踩过）

- GGUF 只能单卡加载（diffusers 的 GGUF 加载无法双卡拆分）；"双开" = 两卡各跑一个完整单卡管线，Q8 单卡峰值约 13.7 GB，T4 塞得下
- GGUF 加载必须传 `GGUFQuantizationConfig(compute_dtype=torch.float16)`，且 `import gguf` 要先于 `diffusers`，否则形状报错
- 量化模型只能 `.to(device)`，不能 `.to(dtype)` 强转精度
- 编码阶段必须关梯度（`torch.set_grad_enabled(False)`），否则 OOM
- 挂载盘随机读极慢（fp8 T5 直接从挂载盘加载 967 秒）：永远先整包顺序拷本地再加载
- `/kaggle/working` 只有 19.5 GiB：权重本地副本放 `/tmp`，别放 working，否则取产物时会把 16.8 GB 副本一起拉下来
- T4（图灵）张量核只认 FP16 不认 BF16：全精度 BF16 生成 754.1 秒 vs FP16 138.3 秒
- IPython 展示图片用 `Image(filename=...)`

### 环境

Kaggle Notebook，GPU T4 x2，diffusers 0.42.0，gguf 0.19.0。FLUX.1-dev 为非商业许可，仅自用 / 研究。
