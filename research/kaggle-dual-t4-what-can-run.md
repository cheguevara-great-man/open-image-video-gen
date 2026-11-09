# Kaggle 免费双卡 T4（2×16 GB）可行性重估：第二张卡到底解锁了什么

> **任务前提（与单卡版的唯一差别）**：Kaggle 会话提供 **2× Tesla T4**（每卡 16 GB GDDR6、Turing、计算能力 CC 7.5、**无 NVLink**、PCIe 3.0 x16），每周约 30 h 配额**按会话小时计**——Kaggle 员工在 T4 x2 上线时明确答复：用 T4 x2 跑 1 小时只扣 1 小时配额，与单卡 P100 同池同速。因此**第二张卡在配额意义上是免费的**，问题只剩：技术上怎么用、能解锁什么。本文只讨论 GPU 双卡，不涉及 TPU。
>
> 证据标注：【实测/官方】= 官方文档、官方模型卡或论文实测；【二手】= 可信二手整理；【外推/薄弱】= 没有 2×T4 直接证据、由其他硬件类比推得，文中明确标出。检索日期均为 2026-11-08。

## 摘要（先给结论）

1. **双卡 ≠ 一张 32 GB 卡。** 两卡可用显存约 **15.0–15.4 GiB/卡、合计约 30 GiB**（`nvidia-smi` 口径），且没有任何框架能把两卡自动当一块显存用。必须主动选择双卡模式，三种模式的收益完全不同：
   - **模式 C「吞吐并行」（两条独立管线，一卡一条）——收益最大、最可靠、实现最简单。** 凡是单卡能跑的工作负载，吞吐近乎翻倍（≈2×），配额成本为零。这是双卡的默认用法。
   - **模式 B「组件/层拆分」（device_map / DisTorch 把一个管线的部件或层摊到两卡）——唯一能抬高"单任务显存天花板"的模式。** 它让 **FLUX.1 BF16 全精度（31.4 GB）从"单卡装不下"变成"双卡勉强装下"**，也让 SDXL 重型栈、14B 级 LLM FP16 等跨过门槛。但它**不加速**（甚至略慢），且对 FLUX 有一个关键细节：只把编码器搬到第二卡**不够**，23.8 GB 的 transformer 本身必须再按层切开。
   - **模式 A「单图并行」（PipeFusion / DistriFusion / 序列并行把一张图的去噪拆到两卡）——技术上成立、工程上最贵、T4 上无实测。** 论文证据来自 8×A100/L40；在 2 卡、1024px、PCIe 的条件下预期收益最小（外推约 1.2–1.8×），且只支持 DiT 类模型。除非专门研究并行推理，不推荐在 Kaggle 上首选。
2. **视频**：双卡的主要红利同样是"同时出两条 480p 片子"。14B 级（Wan2.1-14B）靠 GGUF 量化 + 拆分从"完全不可行"变成"480p 勉强可行、很慢"；FP16 全精度 14B 即使双卡合并（约 30 GiB 可用）**仍装不下**。
3. **训练**：DDP 数据并行**不降低每卡显存**（每卡一份完整副本），它买的是吞吐：有效批大小翻倍或累积步数减半。计算密集的 SDXL/FLUX LoRA 预期 1.6–1.9×（外推，无 2×T4 直接基准）；小模型 DDP 在 2×T4 上实测反而更慢。对 LoRA 这种小梯度负载，另一个零复杂度的选择是**两卡各训一个 LoRA**（不同超参/种子），吞吐同样翻倍。
4. **LLM 推理**：双卡把可用池抬到约 30 GiB，新增两个实打实的解锁：**14B FP16**（28 GB 权重，单卡装不下 → TP=2 每卡约 14 GB）和 **32B AWQ/GPTQ 4-bit**（约 17–20 GB → 每卡约 9–10 GB 且 KV cache 有余量）。32B FP16（64 GB）、32B INT8（约 32–35 GB）、70B 4-bit（约 38–42 GB）**双卡仍装不下**。T4 特别限制：vLLM 必须 `--dtype half`（无原生 BF16）、FP8 W8A8 在 Turing 上**不受支持**、注意力只能用 xformers 类后端（FlashAttention-2 不支持 CC 7.5）。

### 全局硬件约束（贯穿所有判定）

| 约束 | 数值/事实 | 来源 |
|---|---|---|
| 每卡可用显存 | 约 15.0–15.4 GiB（标称 16 GB），双卡池约 30 GiB | 【二手】Kaggle 双卡指南的 `nvidia-smi`/诊断输出 |
| 互联 | PCIe 3.0 x16（理论单向约 16 GB/s，经主机桥实际 NCCL 带宽远低于此），**T4 本身不支持 NVLink** | 【二手】T4 规格页（PCIe 3.0 x16、NVLink: Not supported、显存带宽 320 GB/s） |
| BF16 | Turing 张量核**只有 FP16/INT8/INT4，没有 BF16**；PyTorch 在 T4 上跑 BF16 是软件回退，有约 4× 的静默慢速惩罚——实际应 cast 到 FP16（注意溢出风险） | 【二手】T4 BF16 实测博客 |
| FP8 | T4 无 FP8 张量核；扩散侧 FP8 权重只能当"省显存的存储格式"（计算时反量化回 FP16），vLLM 侧 FP8 W8A8 官方矩阵对 Turing 直接标 ✗ | 【实测/官方】vLLM 量化支持表 |
| 会话资源 | 4 vCPU、约 29 GB 系统 RAM、单会话最长 12 h——**两条并行管线要抢这同一份 CPU/RAM**；凡是靠 CPU offload 的方案，双开时 RAM 是第一瓶颈 | 【二手】Kaggle 官方文档转述（2026-10-31 读取） |

---

## 1. 双卡如何工作：扩散/图像生成的三种模式

### 1a 单图并行：一张图的去噪拆到两卡

**机制与证据**

- **DistriFusion（displaced patch parallelism）**：把一张图的 latent 切成 patch 分给各卡，用上一步的特征图作上下文以异步通信掩盖开销。【实测/官方】CVPR 2024 论文在 SDXL 上报告 UNet 延迟加速 **2 卡 1.8×、4 卡 3.4×、8 卡 6.1×（A100）**，无质量下降。注意：这是 NVLink 级互联的 A100 数据；2 卡时加速比本来就只有 1.8×，换到 PCIe 的 T4 只会更低。
- **PipeFusion / xDiT（xfuser）**：把 DiT 按层切成流水线级、图像切成 patch，利用相邻去噪步的相似性复用特征，通信量比张量/序列并行低，**专门为 PCIe 环境设计**。【实测/官方】PipeFusion 论文在 **8×L40（PCIe）** 上对 PixArt、SD3、**FLUX.1** 取得同类方法中最低延迟；xDiT 论文同样在 8×L40 PCIe 与 8×A100 NVLink 上验证了 FLUX.1、CogVideoX 等五个 DiT。Google Cloud 官方文档也收录 xDiT 部署（FLUX.1 schnell、CogVideoX-2b、Wan2.1），含 PipeFusion 并行度参数。
- **diffusers 原生上下文并行（Context Parallelism）**：新版 diffusers 已内置 Ring Attention / Ulysses 序列并行，`transformer.enable_parallelism(ContextParallelConfig(ring_degree=2 / ulysses_degree=2))` + `torchrun --nproc-per-node 2` 即可对 FLUX 类 transformer 启用。【实测/官方】diffusers 分布式推理文档给出完整示例，但其基准只在 **4×H100** 上测过。
- **ComfyUI 侧**：原版 ComfyUI **不能**把单个 KSampler 拆到两卡——`--cuda-device` 只是选卡。社区节点 ComfyUI-MultiGPU 做的是组件/层放置（见 1b），不是单图并行。个别新节点（如针对特定视频模型的序列并行节点）是单模型特例，不通用。

**对 2×T4 的判定：可行但性价比最低【外推/薄弱】**

- 没有找到任何 **2×T4（或任何 2×Turing 卡）** 上 PipeFusion/DistriFusion 的公开实测。按 2×A100 的 1.8×（DistriFusion）与 PipeFusion 在 1024px 任务上增益最小的规律外推，2×T4 在 1024×1024 的现实预期约 **1.2–1.6×**，且要付出：`torchrun` 多进程启动、xfuser 安装与模型适配（仅支持名单内的 DiT；SDXL UNet 走 DistriFusion 是另一套代码）、PipeFusion 需要 warmup 步对齐流水线、schnell 这类 1–4 步模型流水线填充开销占比极大、几乎吃不到收益。
- 无 NVLink 不是阻断项——PipeFusion 本就是为 PCIe 设计的，DistriFusion 的异步通信也是为掩盖通信而设计——但 PCIe 3.0（T4）比论文用的 L40（PCIe 4.0 级）更窄，收益只会进一步打折。

### 1b 组件/层拆分：把一个大管线摊到两卡

**diffusers：官方支持，且官方示例就是冲着 2×16 GB 写的**

- 【实测/官方】diffusers「分布式推理」文档的 `device_map` 示例**明确假设两张 GPU 可用**，`max_memory={0:"16GB", 1:"16GB"}`，分三段走完 FLUX.1-dev：① 只加载两个文本编码器、`device_map="balanced"` 摊到两卡，先算完 prompt embeddings；② 释放编码器，把 **transformer 用 `device_map="auto"` 按层切分到两卡**做去噪（输出 latent）；③ 最后单独加载 VAE 解码。文档结论："通过按阶段选择性加载/卸载、并把最大的模型切分到多卡，可以在消费级 GPU 上跑大模型推理。"
- 【实测/官方】diffusers「显存优化」文档补充了管线级拆分：`device_map="balanced"`（目前唯一支持的策略，文档标注为 experimental）对 SDXL 的实际映射是 `{'unet': 1, 'vae': 1, 'text_encoder': 0, ...}`——即 UNet+VAE 一卡、文本编码器另一卡；同一文档还给出**模型级**手工映射示例，把 FLUX transformer 的 `single_transformer_blocks.10–20` 指定到第二卡。也明确写了 `enable_model_cpu_offload()` **只适用于单卡**，与 device_map 互斥（需先 `reset_device_map()`）。
- 【实测/官方】ComfyUI-MultiGPU（pollockjj）提供 `UNETLoader/CLIPLoader/VAELoader` 的 MultiGPU 版本与 **DisTorch/DisTorch2** 层切分（可按 bytes/ratio 指定每卡分多少 GB，支持 safetensors 与 GGUF，含 WanVideoWrapper 节点）。其 README 自述定位非常诚实（经二手整理引用）："**这增强的是显存管理，不是并行处理；工作流步骤仍顺序执行**"。⚠️ 该仓库 README 显示**已于 2026-09-30 归档、不再维护**——能用，但要防版本漂移。

**关键显存账：FLUX.1 为什么"搬编码器"不够、必须切 transformer**

FLUX.1-dev/schnell 全精度权重（BF16，【实测/官方】HF 量化博客，H100 实测）：

| 组件 | 显存 |
|---|---|
| transformer（12B） | 23.8 GB |
| T5-XXL 编码器 | 9.52 GB |
| CLIP-L 编码器 | 0.25 GB |
| VAE | 0.17 GB |
| **合计（加载）** | **31.45 GB** |
| **单卡峰值（含激活/工作区）** | **36.17 GB** |

- ❌ 朴素组件拆分（transformer 独占一卡、编码器+VAE 另一卡）：transformer 单件 **23.8 GB > 单卡可用约 15 GiB**，直接 OOM。**"UNet 一卡、编码器一卡"对 SDXL 成立、对 FLUX 不成立**——这是单卡思维最容易误判的一点。
- ✅ 可行拆法（即官方三段式）：编码阶段两编码器合计 9.77 GB，可整放一卡或摊两卡；去噪阶段 transformer 按层对切，**每卡约 11.9 GB 权重**，再加各自那半层的激活/工作区（单卡总激活开销约 4.7 GB，切分后每卡粗估 2–4 GB），**每卡合计约 14–16 GB，对着约 15 GiB 的可用线是贴边通过**——1024×1024 大概率可行，VAE 阶段单独加载 + tiling 更稳；更高分辨率或大 batch 会先爆。判定：**fits-with-caveats（见第 2 节）**。
- 速度代价：层切分后每步的前向要跨 PCIe 搬运层间激活（每个 transformer block 边界一次点对点传输，量不大），主要代价不是通信，而是 T4 本身算得慢 + BF16 在 T4 上无张量核加速（须 FP16 化，FLUX 对 FP16 有溢出/黑图风险需实测）。官方 12 秒/张是 H100 数据，**T4 双卡上 dev（28–50 步）应按"分钟级/张"预期【外推】**；schnell 只有 1–4 步，绝对时间仍可接受。

**SDXL 的组件拆分**则宽松得多：UNet 5.2 GB 一卡、双编码器 1.65 GB + VAE 另一卡，还能把 refiner 的 UNet（另约 5.2 GB）常驻第二卡，实现 base→refiner 全程不换模型（见第 2 节）。

### 1c 吞吐并行：两条独立管线，一卡一条（默认推荐）

**已确认的成熟模式**

- 【实测/官方】diffusers 官方分布式推理文档的第一种用法就是它：`accelerate launch --num_processes=2` + `PartialState`，每个进程把完整管线放到自己的卡上，`split_between_processes` 自动把 prompt 列表分给各进程——官方定位即"处理多个 prompt 以获得更高吞吐"。
- 【实测/官方】ComfyUI 的标准做法是**两个实例**：`python main.py --cuda-device 0 --port 8188` 与 `--cuda-device 1 --port 8189`，社区文档称之为"最可靠的多 GPU 模式：一卡一个队列"；也有启动器工具把"一卡一实例、多端口"做成了一键管理。
- 【二手】Kaggle 场景的直接验证：一篇 Kaggle 双卡指南实测（2026-10-28 至 11-07，2×T4）用 `CUDA_VISIBLE_DEVICES` 分别钉住两张卡跑两个独立进程，**两个任务的总墙钟约为串行的一半**（作者自注：一次测试，非正式基准）。另一个 Kaggle 项目（vLLM 批量生成）也用了同一模式并写明"一小时 T4 x2 只扣一小时配额，第二个模型并行跑、墙钟减半且不额外花配额"。

**收益与打折因素**

- 纯 GPU-bound 的去噪阶段（SDXL、量化 FLUX、Wan 1.3B 等单卡可跑的负载）：吞吐 **≈1.9–2.0×**【二手+外推】。
- 打折项：两进程共享 4 vCPU 与 29 GB RAM——文本编码若走 CPU、VAE 解码、图片 IO、模型加载阶段会互相抢；**若两条管线都开 CPU offload，29 GB RAM 装两份卸载权重很容易先爆**，此时实际增益明显低于 2×；磁盘上两份模型权重各加载一份（Kaggle 持久盘有限，权重宜放同一缓存目录共享文件、各自读入显存）。

---

## 2. 修订图像判定（2×16 GB）

判定口径：单图 1024×1024（SD 1.5 为 512×512）、batch=1、FP16（T4 口径；权重按 BF16/FP16 同为 2 字节/参数计）。

### 2.1 FLUX.1 schnell / dev（权重账见 1b：合计 31.4 GB 加载 / 36.2 GB 单卡峰值）

| 双卡模式 | 判定 | 说明 |
|---|---|---|
| 单卡基准（对照） | ❌ BF16 不可行；NF4+offload ~12.2 GB 可行但慢 | 与既有单卡结论一致。补充：HF 博客的 BnB 4-bit 口径是加载 12.58 GB / **峰值 17.28 GB**（H100、无 offload），高于单卡可用线——这正是单卡 NF4 仍需 offload 的原因，两种 NF4 构建口径不同，勿混用 |
| (a) 单图并行（xDiT PipeFusion） | ⚠️ 可行但不推荐首选 | 权重随层切分天然装下，延迟或有 1.2–1.6× 改善【外推，无 T4 实测】；schnell 步数太少（1–4 步），流水线 warmup 占比大，收益最差；dev 步数多相对划算。工程成本最高 |
| (b) 组件+层拆分（diffusers device_map 三段式 / ComfyUI-MultiGPU DisTorch） | ⚠️ **fits-with-caveats——双卡最实在的解锁** | 每卡 transformer 权重约 11.9 GB + 激活 2–4 GB ≈ 贴边 15 GiB 可用线；编码器分阶段加载/释放是必需动作，不是可选项。1024px 可行性高；更高分辨率需降步数/开 VAE tiling 并实测。速度慢（T4 算力 + 无 BF16 核），dev 按分钟级预期 |
| (c) 两条并行管线 | ❌ BF16 仍不可行（每条管线各自要 31 GB）；✅ 量化版吞吐翻倍 | NF4/FP8/GGUF 版单卡已可跑，双卡的正确用法是两条量化管线并行出图，吞吐 ≈2×。注意 FP8 在 T4 上只是存储压缩、计算反量化回 FP16，无 FP8 加速 |

> 一句话：**双卡把 FLUX BF16 从"不可能"抬到"贴边可能"，但没有把"量化版 FLUX"变得更快——量化版的收益仍在模式 (c) 的翻倍吞吐。**

### 2.2 SDXL（+ refiner / ControlNet / IP-Adapter）

权重账（FP16，【二手】组件级拆分表）：UNet 5.2 + CLIP-L 0.25 + OpenCLIP-G 1.4 + VAE 约 0.2 ≈ **7.1 GB 权重、约 8.5 GB 实测峰值**（与单卡结论一致）；refiner 另约 5.2 GB UNet（整 checkpoint 文件约 6–6.9 GB）；单个 ControlNet 约 1.3–2.5 GB；IP-Adapter 本体很小但其 CLIP Vision 编码器约 1.3 GB 量级【二手，薄弱】。

| 双卡模式 | 判定 | 说明 |
|---|---|---|
| 单卡基准（对照） | ✅ base 约 8.5 GB；重型栈贴边 | base+1×ControlNet+IP-Adapter 约 12 GB 可行；再叠 refiner 常驻约 16 GB+，单卡须靠卸载/换入换出 |
| (a) 单图并行（DistriFusion） | ⚠️ 可行、收益中 | SDXL 是 DistriFusion 的原生支持模型，2×A100 实测 UNet 1.8×；T4/PCIe 外推 1.3–1.6×【薄弱】。为 SDXL 单独上这套工程，性价比不如 (c) |
| (b) 组件拆分 | ✅ **重型栈全常驻** | 例：GPU0 = base UNet 5.2 + ControlNet 2.5 + IP-Adapter；GPU1 = refiner UNet 5.2 + 双编码器 1.65 + VAE。全栈约 17–19 GB 摊到两卡后每卡 ≤10 GB，base→refiner 不再换模型、ControlNet 多挂 1–2 个也不爆。延迟改善有限（步骤仍顺序执行），赢在"不卸载、不等待" |
| (c) 两条并行管线 | ✅✅ **最佳模式** | 每条管线甚至能各带一个 ControlNet（约 12 GB/卡），吞吐 ≈2×；或一卡跑 base 批量、一卡专门跑 refiner/放大流水线 |

### 2.3 SD 1.5

权重账：UNet 860M，FP16 整管线峰值约 **3.5–4 GB**（512×512）【二手】。

| 双卡模式 | 判定 | 说明 |
|---|---|---|
| (a) 单图并行 | — 无意义 | 单卡延迟已低，拆分通信开销占比过高，无任何框架值得为它做 |
| (b) 组件拆分 | — 无意义 | 整管线不到 4 GB，无天花板可抬 |
| (c) 两条并行管线 | ✅✅ 吞吐翻倍，且可异构并行 | 例：一卡批量出图、一卡跑 AnimateDiff/ControlNet 批处理；或单卡加大 batch（单卡装 batch 4–8 都绰绰有余）本身已能吃满，双卡纯粹再 ×2 份工作 |

---

## 3. 修订视频判定（2 卡）

通用前提：视频的激活显存随分辨率×帧数暴涨，各家公布的"最低显存"几乎都绑定特定分辨率/帧数/offload 组合，跨来源数字冲突时以官方模型卡为准并标注。Kaggle 的 **29 GB RAM** 对一切 offload 方案都是硬约束（不少消费级 offload 教程默认 32 GB+ RAM）。

### 3.1 逐模型判定

**Wan2.1 T2V-1.3B**（官方：仅需 **8.19 GB**，480p，`--offload_model True --t5_cpu`；【实测/官方】官方仓库/HF 卡）
- (a) 双并行：✅✅ 两条 480p 管线各占约 8.2 GB，同时出两条 5 秒片——视频区最干净的翻倍。
- (b) 拆分/加速单条：⚠️ 官方仓库原生支持多卡（FSDP + xDiT USP，`--dit_fsdp --t5_fsdp --ulysses_size/--ring_size`，`torchrun` 启动；注意 1.3B 的注意力头数 12 不能被 8 整除、官方建议 Ring——2 卡时 Ulysses 12÷2 可整除，无此问题）。但官方示例与基准都以 8 卡为场景，**2×T4 的加速比无实测【薄弱】**；单条 480p 本来就只要 8 GB，拆分的主要意义是冲 720p（官方明言 1.3B 的 720p 训练不足、质量不稳，意义有限）。
- (c) 不适用（非 14B）。

**Wan2.1 14B**（DiT 本体 FP16 约 28 GB；含 umT5 编码器等全量 FP16 口径二手整理为 54–65 GB 显存 @720p 级设置【二手，来源间冲突：另一云厂商口径 65–80 GB】）
- (a) 双并行：❌ 每条都装不下，谈不上并行（量化后见 (c)，那时每卡一条 GGUF 反而又回到模式 (a)：两条 Q4/Q5 480p 并行 ✅⚠️）。
- (b) FP16 拆分常驻：❌ **仍不可行**——仅 DiT+编码器权重已超过约 30 GiB 的双卡可用池，更不用说 81 帧的激活。官方多卡路径（FSDP+USP）的设计场景是 8 卡把权重+激活一起摊开，2 卡摊不开 FP16 14B。【实测/官方（多卡代码存在）+ 显存账推算】
- (c) GGUF/量化 + 拆分：⚠️ **从"完全不可行"升为"480p 勉强可行"**。二手口径：GGUF Q4 transformer + T5 CPU offload 时单卡约 7–9 GB @480p 本已可跑（慢）；双卡的增量价值是把 T5 编码器（FP8 版约 6.7 GB 量级）或 GGUF 层片放到第二卡（ComfyUI-MultiGPU 有 WanVideoWrapper 的 block-swap/多设备节点），减少 CPU offload 的搬运、并给 VAE 解码留余量。720p 在量化+拆分下属实验性质【薄弱】，激活与时长都会先撞墙。速度预期：T4 算力跑 14B DiT × 81 帧，**单条按十分钟级预期**【外推】。

**CogVideoX-2B / 5B**（【实测/官方】官方模型卡：diffusers 全优化下单卡 2B 自 4 GB、5B 自 5 GB 起——该数字含 `enable_model_cpu_offload` 等全部优化；SAT 原生口径则为 18/26 GB）
- (a) 双并行：✅ 两条管线同时跑（各自带 offload 时慢但稳；RAM 29 GB 两份 offload 需留意）。这是双卡对 CogVideoX 最实在的收益。
- (b) 拆分单条：⚠️ 官方卡专门列了"多 GPU 推理显存"行：**2B FP16 10 GB、5B BF16 15 GB（diffusers）**，且注明多卡时**必须关闭** `enable_model_cpu_offload()`。解读：双卡的价值是让 5B **不用 CPU offload 也能跑**（15 GB 对 T4 约 15 GiB 可用线是贴边），速度应明显好于单卡 offload 版；但官方注明该套数字**只在 A100/H100（Ampere+）上测过**、FP8 需 H100+，T4 上 BF16 还要面对无 BF16 核的问题（5B 官方推荐精度正是 BF16）。判定 fits-with-caveats，T4 实测缺失【薄弱】。xDiT 也支持 CogVideoX-5B 的序列并行（L40 级证据），属模式 (a) 的单图并行变体，同 1a 的保留意见。

**HunyuanVideo-1.5**（8.3B；【实测/官方】腾讯官方：**最低 14 GB（开模型 offload 时）**；ComfyUI 口径 FP8 transformer 文件 8.33 GB、Qwen2.5-VL 编码器 FP8 9.38 GB，两者顺序加载）
- 单卡状态先说清：14 GB 的官方下限对着 T4 约 15 GiB 可用线，单卡已是"贴边可行"而非"不可行"。
- (a) 双并行：⚠️ 两条都走官方 offload 路线会争抢 29 GB RAM（编码器+transformer 的卸载副本都在 RAM 里），风险高；改走 FP8 文件路线（transformer 8.3 GB 常驻 + 编码器用完即走）时每卡自包含，双并行才稳。判定 fits-with-caveats，瓶颈在 RAM 不在显存。
- (b) 拆分单条：⚠️/【薄弱】腾讯官方仓库未见多卡推理支持；理论上可把 Qwen2.5-VL 编码器放第二卡、FP8 transformer 放第一卡省掉顺序换入，或用通用 device_map 切 FP16 transformer（单文件 16.65 GB，刚好超单卡、切两卡每半约 8.3 GB）——均为拼凑路线，无公开实测，不应当作承诺。

**LTX-Video**（按任务语境指 0.9.x 线：2B distilled 与 13B；⚠️ 注意与 2025 年底的 LTX-2/2.3 不是同一模型，后者官方口径 32 GB+ 起步，不在本节结论内）
- 2B distilled：BF16 文件约 5.91 GiB【二手】，单卡宽松。(a) 双并行 ✅✅ 两条同时出片；(b) 无必要。
- 13B：transformer BF16 约 26.6 GiB + T5-XXL 约 9.5 GB，单卡 ❌。(b) 拆分 ⚠️ 算术上成立（transformer 对切约 13.3 GiB/卡已贴边，再叠激活风险高；更现实是 FP8 文件约 14.6 GiB 单卡贴边 + 编码器放第二卡/分阶段），diffusers 官方 LTX 示例本身也是 FP8 层铸 + group offload 的约 10 GB 单卡路线——**双卡对 13B 的增量主要是"少 offload 一点"，不是质变**【薄弱】。(a) 两卡各跑一条 FP8/offload 版 13B ⚠️ 可行但两条都很慢。

**AnimateDiff**（SD1.5 motion module v3 约 1.7 GB 文件；SD1.5 基座 + 运动模块 512×512×16 帧峰值约 6–8 GB【二手，社区口径 8–13 GB 随帧数/ControlNet 浮动】；SDXL 版官方口径约 13 GB）
- (a) 双并行：✅✅ 两条 16 帧短片同时生成，是 AnimateDiff 在双卡上的全部意义（批量出片、双种子对比）。
- (b) 拆分：— 无必要（单条本就装得下）；SDXL-AnimateDiff 约 13 GB 单卡贴边时，第二卡的用法仍是再开一条，而不是拆分。
- 长帧数/高分辨率时激活增长很快，双卡对此无帮助（除非走 (a) 分段渲染再拼接，属工作流技巧而非显存解锁）。

### 3.2 视频小结表

| 模型 | (a) 双并行两条 | (b) 拆分/offload 跑单条 | (c) 量化+拆分冲大模型 |
|---|---|---|---|
| Wan2.1-1.3B | ✅✅ 480p ×2 | ⚠️ 官方 FSDP+USP 支持，2 卡收益无实测 | — |
| Wan2.1-14B | ❌（FP16）；⚠️ 量化后可双开 480p | ❌ FP16 权重+激活 > 双卡池 | ⚠️ GGUF Q4/Q5+编码器分卡，480p 勉强，720p 实验性 |
| CogVideoX-2B | ✅ | ✅ 10 GB 多卡口径、可关 offload | — |
| CogVideoX-5B | ✅（带 offload） | ⚠️ 15 GB 贴边、官方仅测 Ampere+ | INT8 单卡已可跑，双卡并行即可 |
| HunyuanVideo-1.5 | ⚠️ RAM 是瓶颈，FP8 路线才稳 | ⚠️ 无官方多卡，拼凑路线无实测 | — |
| LTX-Video 2B | ✅✅ | — | — |
| LTX-Video 13B | ⚠️ 两条量化版，慢 | ⚠️ 增量有限（少 offload） | ⚠️ 同 (b) |
| AnimateDiff (SD1.5) | ✅✅ | — | — |
| AnimateDiff (SDXL) | ✅ 两条贴边并行 | — | — |

---

## 4. 训练：双卡 LoRA / QLoRA（DDP 数据并行）

**机制先行——DDP 不省显存**

- 【实测/官方口径】DDP 在每张卡上放**模型的完整副本**（权重+梯度+优化器状态都在每卡），每步 all-reduce 平均梯度。因此：**每卡显存下限与单卡训练完全相同**，SDXL LoRA 单卡要约 12 GB（16 GB 舒适，kohya 口径【二手】）、FLUX LoRA 单卡要 16 GB（量化/优化后；传统口径 24 GB），双卡 DDP 后每卡还是这么多。想要"两卡摊一份训练状态"是 FSDP/ZeRO 的事（把参数/梯度/优化器分片），那是另一套配置，且 QLoRA+FSDP 有已知的兼容坑（见下）。
- **DDP 买的是批与步**：有效批 = 每卡批 × 2 × 累积步数。两种用法——① 保持有效批不变、累积步数减半 → 每 epoch 的优化步墙钟近乎减半；② 累积不变、有效批翻倍 → 步数不变但每 epoch 处理两倍样本（需相应调学习率，不是免费的等价交换）。

**速度：分模型大小，结论相反**

- 【实测/官方】直接相关的 Kaggle 证据：一个 4M 参数 nanoGPT 在 **Kaggle 2×T4 上 DDP 实测反而更慢**（单步 26.4 ms vs 单卡 22 ms；通信占 42.6%，缩放效率 41.8%）——小模型的计算/通信比太差，PCIe all-reduce 吃掉一切。
- 【实测/官方】PCIe-only 机器（4×A6000、无 NVLink、经主机桥）跑 124M/355M GPT 的 DDP：2 卡仅 1.25×/1.18×，实测 NCCL 总线带宽仅约 3.9–4.3 GB/s；靠梯度压缩+累积才把效率拉起来。说明 PCIe 互联下 DDP 效率对"每步计算量"极度敏感。
- 【实测/官方】反例（大计算量时 DDP 很好）：Axolotl QLoRA 在 2×H100 **PCIe**（同样无 P2P、被迫 `NCCL_P2P_DISABLE=1`）实测吞吐 15.92 vs 8.04 samples/s ≈ **2×**（注意其口径是"每步样本翻倍"，固定步数的墙钟相近——正对应上面①②的区别）。
- **对 SDXL/FLUX LoRA 的推论【外推/薄弱，无 2×T4 直接基准】**：扩散 LoRA 每步是整网前向+反向（1024px、秒级/步），且**冻结基座时 all-reduce 的只有 LoRA 梯度（几十 MB 级）**，计算/通信比远好于全量训练——预期 DDP 效率 **1.6–1.9×** 是合理区间，但这是推理而非实测，实际数字取决于 kohya/diffusers 训练脚本的多卡实现（kohya sd-scripts 经 accelerate 配置支持多卡；diffusers 训练示例同样走 accelerate）。
- **LLM QLoRA**：7–8B 的 4-bit 基座约 4 GB/卡、加 LoRA/优化器/激活后约 6–8 GB/卡（2 卡 FSDP2+QLoRA 实测拆解【二手】），单卡本就宽松。⚠️ HF Alignment Handbook 明确提示：**多卡 DDP + QLoRA 不推荐**（量化与某些分布式策略不兼容），多卡量化训练应走 FSDP+QLoRA；也有实测记录 bitsandbytes 4-bit + DDP 在某些版本组合下报错、需回退。Kaggle 上的稳妥路线：单卡 QLoRA 照旧，第二卡**并行跑另一组超参/另一个模型**——训练版的模式 (c)，零分布式风险、吞吐同样翻倍。

**训练判定表**

| 任务 | 单卡 | 双卡 DDP | 双卡其他用法 |
|---|---|---|---|
| SDXL LoRA | ✅ 约 12 GB | ⚠️→✅ 每卡显存不变；有效批 ×2 或累积减半；速度预期 1.6–1.9×【外推】 | 两组 LoRA 并行训 ✅✅ |
| FLUX LoRA | ⚠️ 16 GB 量化/优化版贴边 | ⚠️ 同上；DDP 不解决"每卡装不下"——装不下时 DDP 直接 OOM，需 FSDP/ZeRO 分片（无 T4 实测【薄弱】） | 两组并行（各一份量化版）✅ |
| LLM QLoRA 7–8B | ✅ 约 6–8 GB | ⚠️ DDP+4bit 有兼容风险，官方不推荐；FSDP+QLoRA 可行但配置复杂 | 两组并行 ✅✅（Kaggle 上最稳） |
| 全量微调（任何大模型） | 多数 ❌ | ❌ DDP 每卡全副本，不解锁 | FSDP/ZeRO-3 才是解锁路径，超出本文双卡红利的常规用法、且 T4 证据薄弱 |

---

## 5. LLM 推理：2×16 GB 的"约 32 GB 池"

先纠正口径：池子是**约 30 GiB 可用**，且只有两种框架用法能真正" pooled "——vLLM 张量并行（TP=2，每层切两半、每卡约一半权重+一半 KV）和 transformers `device_map="auto"`（按层切分、整层放一卡，流水线式）。两者性质不同：

- **vLLM TP=2**：每生成一个 token、每一层都要做 all-reduce，通信频率极高——【实测/官方】vLLM 社区/官方口径一致建议"**用能装下的最小 TP**"，模型本可单卡装下时 TP 只会更慢，PCIe 机器尤其如此；能单卡装下的模型应改用**两个独立实例（数据并行）**提吞吐。TP=2 的正当用途只有一个：**模型单卡装不下**。
- **transformers device_map**：层间只在边界传一次激活，通信便宜得多，但层是顺序执行的（第二卡干活时第一卡在等），延迟不降、吞吐低；胜在任何 HF 模型零改动可用、且 Kaggle 上已有项目实测用它在 2×T4 上自动分层跑通。适合"能跑就行"的离线批量生成，不适合交互速度。

**逐档判定（权重 = 参数量 × 字节数，另需 KV cache 与运行时开销）**

| 模型档 | 单卡（约 15 GiB） | 双卡 TP=2 / device_map（约 30 GiB 池） | 备注 |
|---|---|---|---|
| 7–8B FP16（约 15–16 GB） | ⚠️ 贴边（权重就占满，KV 几乎无余量，实用上多走 AWQ） | ✅ 装下且 KV 有余量，但**不推荐 TP**——两卡各开一个 AWQ 实例吞吐更高 | TP 在此纯属浪费互联 |
| 7–8B AWQ/GPTQ（约 4–5 GB） | ✅ 宽松 | — 用双实例并行 ✅✅ | |
| **14B FP16（约 28 GB）** | ❌ | ✅ **新增解锁**：每卡约 14 GB 权重，余量只够短-中上下文 KV，`gpu_memory_utilization` 与 `max_model_len` 需收着设 | 这是 TP=2 在双 T4 上最干净的用例 |
| 14B AWQ/GPTQ（约 8–10 GB） | ✅ 已可跑 | TP 无意义；双实例并行 ✅✅ | |
| **32B AWQ/GPTQ 4-bit（约 17–20 GB）** | ❌（超可用线，或只剩零 KV） | ✅ **新增解锁**：每卡约 9–10 GB 权重，KV 余量可观（4k–8k 上下文现实） | 权重口径来源间有 17 vs 20 GB 的出入【二手】，两种口径下结论不变 |
| 32B INT8（约 32–35 GB） | ❌ | ❌ 仍超池（权重 alone 已 > 30 GiB 可用） | |
| 32B FP16（约 64 GB） | ❌ | ❌ | |
| 70B AWQ/GPTQ（约 38–42 GB） | ❌ | ❌ | 别被"双卡 = 32 GB"误导，70B 4-bit 也进不来 |

**T4 专属的速度与兼容 caveat**

- 【实测/官方】vLLM 要求 CC ≥ 7.0，T4（7.5）可跑，但必须 `--dtype half`：BF16 权重模型不转 FP16 会直接报错（vLLM 论坛的 T4 案例即如此）；注意力后端走 xformers 路径而非 FlashAttention-2。
- 【实测/官方】vLLM 官方量化支持表（Turing 列）：AWQ ✅、GPTQ ✅、Marlin ✅*（脚注：Turing 不支持 Marlin MXFP4）、INT8 W8A8 ✅、**FP8 W8A8 ✗**。选模型文件时只认 AWQ/GPTQ 版。（二手资料对 Marlin 在 T4 上是否可用口径不一，以官方表为准、按版本复核。）
- **速度预期【薄弱/外推】**：未找到 2×T4 上 vLLM TP=2 的可信 tok/s 实测。可定性判断：解码是显存带宽-bound，TP=2 理论上聚合 2×320 GB/s 带宽，但每层 all-reduce 走 PCIe 3.0 会吃掉相当一部分；14B FP16 / 32B AWQ 的单流解码应按"**可用但慢、明显低于同模型在单张现代 24–32 GB 卡上的表现**"预期。若负载是批量离线生成，两个独立实例（各跑一个量化模型）的总吞吐大概率高于一个 TP=2 实例——与 vLLM 官方"能单卡就用数据并行"的建议一致。
- transformers `device_map` 路线同理：32B AWQ 甚至可以不切分…（17–20 GB 仍超单卡，必须切）；切分后每 token 两次跨卡边界传输，批量离线场景可接受。

---

## 6. 核心对比表：单卡判定 vs 双卡判定

图例：✅ 可行 / ⚠️ 有条件可行（贴边、需特定路线或慢）/ ❌ 不可行。双卡判定注明所用模式：(a) 单图并行 (b) 组件/层拆分 (c) 双管线并行。

| 工作负载 | 单卡判定（1×16 GB） | 双卡判定（2×16 GB）及模式 | 双卡带来的本质变化 |
|---|---|---|---|
| SD 1.5 出图 | ✅ 约 4 GB | ✅✅ (c) 吞吐 ×2 | 纯翻倍 |
| SDXL base 出图 | ✅ 约 8.5 GB | ✅✅ (c) 吞吐 ×2；(b) 亦可 | 纯翻倍 |
| SDXL + refiner + ControlNet + IP-Adapter 重型栈 | ⚠️ 约 16 GB+，靠卸载/换入 | ✅ (b) 全栈常驻两卡、免卸载；或 (c) 两条带 ControlNet 的管线并行 | 从"贴边折腾"变"宽松常驻" |
| FLUX.1 schnell/dev **BF16 全精度** | ❌ 31.4 GB 加载 / 36.2 GB 峰值 | ⚠️ (b) transformer 按层对切 + 编码器分阶段，贴边可行（慢）；(a) 亦可但工程贵 | **从不可能 → 勉强可能（本报告最大解锁）** |
| FLUX.1 NF4/FP8/GGUF 量化版 | ⚠️ NF4+offload 约 12.2 GB，慢 | ✅✅ (c) 两条量化管线并行，吞吐 ×2 | 纯翻倍（精度路线不变） |
| Wan2.1-1.3B 480p | ✅ 8.19 GB（官方） | ✅✅ (a/c) 同时出两条 480p | 纯翻倍 |
| Wan2.1-14B | ❌ FP16 远超单卡 | ⚠️ (c-量化) GGUF Q4/Q5 + 编码器分卡，480p 勉强；❌ FP16 双卡合并仍装不下 | 从"不可行"→"量化慢速可行" |
| CogVideoX-2B | ✅ diffusers 优化下自 4 GB（慢） | ✅ (c) 双开；(b) 官方多卡口径 10 GB 可关 offload | 翻倍 + 单条可提速 |
| CogVideoX-5B | ⚠️ diffusers 优化下自 5 GB（重 offload、慢） | ⚠️ (b) 官方多卡口径 15 GB 贴边、免 offload（仅测过 Ampere+）；(c) 双开稳妥 | 单条从"靠 offload"→"贴边常驻" |
| HunyuanVideo-1.5 | ⚠️ 官方下限 14 GB（开 offload），贴边 | ⚠️ (c) 双开受 29 GB RAM 制约，FP8 路线才稳；(b) 无官方支持 | 边际改善，非质变 |
| LTX-Video 2B (0.9.x) | ✅ 约 6 GiB | ✅✅ (c) 双开 | 纯翻倍 |
| LTX-Video 13B (0.9.x) | ❌ BF16 约 26.6 GiB + T5 | ⚠️ (b) 切分/FP8+分卡，少 offload 一点；或 (c) 两条量化版慢速并行 | 边际改善 |
| AnimateDiff SD1.5 | ✅ 约 6–8 GB | ✅✅ (c) 两条短片并行 | 纯翻倍 |
| AnimateDiff SDXL | ⚠️ 约 13 GB 贴边 | ✅ (c) 两条并行、各占一卡 | 贴边变宽松 + 翻倍 |
| SDXL LoRA 训练 | ✅ 约 12 GB/卡 | ⚠️→✅ DDP：每卡显存不变，有效批 ×2/累积减半，速度预期 1.6–1.9×【外推】；或两组 LoRA 并行训 ✅✅ | 吞吐，非显存 |
| FLUX LoRA 训练 | ⚠️ 16 GB 量化版贴边 | ⚠️ DDP 不降每卡门槛；FSDP 分片理论可降、无 T4 实测；两组并行 ✅ | 同上 |
| LLM QLoRA 7–8B 训练 | ✅ 约 6–8 GB | ⚠️ DDP+4bit 官方不推荐；两组并行 ✅✅ | 吞吐，非显存 |
| LLM 14B FP16 推理 | ❌ 约 28 GB | ✅ (b/TP=2) 每卡约 14 GB，KV 需收着设 | **新增解锁** |
| LLM 32B AWQ/GPTQ 推理 | ❌ 约 17–20 GB | ✅ (b/TP=2) 每卡约 9–10 GB，KV 有余量 | **新增解锁** |
| LLM 32B FP16 / INT8 推理 | ❌ 64 / 约 33 GB | ❌ 双卡池仍不够 | 无变化 |
| LLM 70B 4-bit 推理 | ❌ 约 38–42 GB | ❌ | 无变化 |
| LLM 7–14B AWQ 推理 | ✅ | ✅✅ 双实例并行，吞吐 ×2（勿用 TP） | 纯翻倍 |

---

## 7. 实践建议（按性价比排序）

1. **默认模式 (c)**：任何单卡能跑的负载，先想"能不能开两份"。ComfyUI 双实例（8188/8189、`--cuda-device` 分卡）或两个 `CUDA_VISIBLE_DEVICES` 进程，是零风险、近 2× 的配额套利。
2. **为 FLUX BF16 专门用模式 (b)**：照 diffusers 官方三段式（编码器先行并释放 → transformer `device_map` 切两卡 → VAE 单独解码），或 ComfyUI-MultiGPU 的 DisTorch 切分；1024px 先行验证峰值，不要一上来冲高分辨率。
3. **视频优先排 (a/c) 的"两条 480p"**，而不是把两卡押在一条大片上；Wan-14B 量化版当实验项目，不当生产路线。
4. **训练优先"两组并行"而非 DDP**（LoRA/QLoRA 场景）；真要 DDP，先用小步数实测缩放效率再投入配额——小模型 DDP 在 2×T4 上有实测变慢的先例。
5. **LLM 只在"装不下"时用 TP=2**（14B FP16、32B AWQ 两档）；能单卡装下的，一律双实例。

## 8. 未能证实 / 薄弱证据清单

- **无任何 2×T4 直接实测**：PipeFusion/DistriFusion/diffusers 上下文并行在双 T4 上的加速比（本文 1.2–1.6× 为由 A100/L40 数据外推）；SDXL/FLUX LoRA 的 DDP 缩放效率（1.6–1.9× 为外推）；vLLM TP=2 在双 T4 上的 tok/s；CogVideoX 官方多卡数字在 Turing 上的复现；HunyuanVideo-1.5 与 LTX-13B 的拆分路线。
- **来源冲突已按官方优先处理**：Wan-14B FP16 显存（54–65 GB vs 65–80 GB，均为含激活的二手口径，结论方向一致）；32B AWQ 权重（约 17 vs 20 GB）；vLLM Marlin 在 Turing 上的支持（官方表标 ✅*，部分二手称不可用）；FLUX NF4 显存（HF 博客 BnB 口径峰值 17.3 GB vs 单卡版采用的约 12.2 GB+offload 口径，系不同构建）。
- **Kaggle 侧未核实项**：两卡间的实际 PCIe 拓扑（是否经主机桥、P2P 是否可用）未见官方说明，本文按最保守的 PCIe 经主机桥假设；配额"按会话计"依据 Kaggle 员工上线答复的二手转述，官方文档只写"30 h/周"总量。
- **版本口径**：LTX-Video 按 0.9.x 线判定；若指 LTX-2/2.3（32 GB+ 起步），双卡结论需另行重估，大概率为"量化+offload 贴边、官方多卡支持缺失"。

## Sources（主要来源，2026-11-08 读取；【开】= 全文打开核读，其余为检索结果核读）

**平台与硬件**
- Kaggle 笔记本规格/配额（转述 Kaggle 官方文档，2026-10-31 读取）：https://fastero.com/blog/kaggle-notebooks
- Kaggle T4 x2 配额按会话计（员工答复转述）：https://medium.com/@hari12.nhk/kaggles-free-gpus-for-ai-experiments-what-you-actually-get-abff5684a53c
- Kaggle 双卡两进程实测与用法：https://dev.undress.workers.dev/read-https-dev.to/amrendra_singh_c700783152/kaggle-gpu-free-limit-what-you-get-and-how-to-use-it-f8i
- T4 规格（PCIe 3.0 x16、NVLink 不支持、320 GB/s）：https://www.sell-server.com/product/nvidia-tesla-t4-16gb-inference-gpu/
- T4 无 BF16 张量核实测：https://github.com/earino/applied-deep-learning/blob/HEAD/site/blog/05_bf16_on_t4.md

**扩散双卡机制**
- 【开】diffusers 分布式推理（device_map 三段式 FLUX、PartialState 吞吐并行、上下文并行）：https://huggingface.co/docs/diffusers/main/en/training/distributed_inference
- 【开】diffusers 显存优化（device_map=balanced、模型级层映射、offload 单卡限制）：https://huggingface.co/docs/diffusers/main/optimization/memory
- 【开】HF 博客：diffusers 量化后端（FLUX BF16 31.447/36.166 GB 与各量化口径）：https://github.com/huggingface/blog/blob/HEAD/diffusers-quantization.md
- PipeFusion 论文（8×L40 PCIe）：https://storage.googleapis.com/lacuna-pdfs-lunar-airport-385023/pdfs/a5/6d/a56d4913319d1b18b288b33b263e8acb6856de903cc2b92ecdde701e9a38eca5.pdf
- xDiT 论文：https://arxiv.org/html/2411.01738v1/
- DistriFusion（CVPR 2024，SDXL 2 卡 1.8×）：https://cvpr.thecvf.com/virtual/2024/poster/30271
- xDiT 官方部署文档（Google Cloud）：https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/xdit
- ComfyUI-MultiGPU / DisTorch README：https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/refs/heads/main/README.md
- ComfyUI 双实例模式：https://wonderfullauncher.com/docs/comfyui-multi-gpu
- "什么能跨两卡 pooled"对照（含 ComfyUI-MultiGPU 自述引用）：https://localaimaster.com/blog/two-gpus-vs-one-local-llm

**图像/视频显存**
- SDXL 组件级显存表：https://gigagpu.com/sdxl-vram-requirements-2/
- SD 1.5 / SDXL 显存总表：https://gigagpu.com/stable-diffusion-vram-requirements/
- Wan2.1 官方卡（1.3B 8.19 GB、多卡 FSDP+USP）：https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- Wan 14B 量化显存整理（二手）：https://willitrunai.com/blog/wan-2-2-vram-requirements
- 【开】CogVideoX-5B 官方卡（单卡/多卡显存行、offload 与多卡互斥、仅测 Ampere+）：https://huggingface.co/THUDM/CogVideoX-5b
- HunyuanVideo-1.5 官方 14 GB 下限（官方仓库转述）：https://localaimaster.com/blog/hunyuan-video-guide
- HunyuanVideo-1.5 文件级拆解（FP8 8.33 GB、编码器 FP8 9.38 GB）：https://www.genvidkit.com/blog/hunyuanvideo-1-5-vram-requirements
- LTX-Video 0.9.8 文件/显存：https://nexgpu.net/en/models/ltx-video/
- AnimateDiff 版本/显存：https://nexgpu.net/en/models/animatediff/

**训练**
- PCIe-only DDP 实测（NCCL 带宽、2 卡效率）：https://discuss.pytorch.org/t/ddp-on-a-pcie-only-4x-rtx-a6000-no-nvlink-plain-ddp-gave-2x-bf16-gradient-compression-plus-accumulation-gave-3-6x/225650
- Kaggle 2×T4 DDP 变慢实测：https://dev.to/thokozani_buthelezi_2cd41/ddp-is-not-always-faster-23mh
- Axolotl 双卡 QLoRA 实测（PCIe、samples/s 翻倍口径）：https://massedcompute.com/fine-tune-llm-axolotl-multi-gpu/
- FSDP2+QLoRA 双卡显存拆解：https://numb3r33.github.io/experiments/deeplearning/unsloth/backpropagation/2025/12/15/unsloth-fsdp2-qlora.html
- kohya LoRA 显存口径（SDXL/FLUX）：https://localaimaster.com/blog/image-lora-training-local-guide

**LLM 推理**
- vLLM 官方量化支持表（Turing：AWQ/GPTQ ✅、FP8 W8A8 ✗）：https://docs.vllm.ai/en/latest/features/quantization/index.html
- vLLM TP 使用原则（最小 TP、PCIe 开销、数据并行优先）：https://discuss.vllm.ai/t/is-there-recommended-max-upper-limit-for-tensor-parallel/1184/1
- vLLM 在 T4 上运行条件（CC 7.5、dtype half）：https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615
- Qwen2.5 14B/32B 各精度显存：https://www.botserv.de/en/local-ai/local-ai-models/qwen-models-alibaba-multilingual-series/
- 14B/32B AWQ 权重口径：https://blog.none.at/pdf/2026/2026-06-02-llm-inference-on-ovh-serving.pdf
