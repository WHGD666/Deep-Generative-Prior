# 实验日志（EXP_LOG）

> 规则：只追加、不删除；失败的正式 run 也要记录。详细 manifest 见 `experiments/<run_id>.json`。
> run_id 模式：`<日期>_<主题>_<短哈希>`，如 `20260908_diffbir_baseline_a3f2`。

## 正式实验一览

| run_id | 日期 | role | 主题/假设 | 相对控制组改动 | 关键指标（val） | 结论 | git commit |
|---|---|---|---|---|---|---|---|
| 20260907_val_probe | 2026-09-07 | diagnostic | DiffBIR v2.1 同分辨率管线首图探路（仅 val case1，512 瓦片/50 步/fp16/upscale=1） | 无（首个 run） | PSNR 29.04 · SSIM 0.741 · LPIPS 0.592 · DISTS 0.314 · MUSIQ 25.7 · NIQE 4.01 · 综合(equal) 0.570 | 端到端跑通（402.6s/张，4K 输出尺寸一致）；分数待 LQ 不增强基线对照后才有意义 | dc719e0 |
| lq_baseline | 2026-09-07 | control | LQ 不增强对照组：val 5 张原始输入直接评测 | 无（控制组，量管线增益用） | PSNR 28.03 · SSIM 0.788 · MS-SSIM 0.805 · LPIPS 0.357 · DISTS 0.279 · MUSIQ 21.3 · CLIPIQA 0.333 · NIQE 9.53 · MANIQA 0.268 · 综合 0.538/0.631/0.440 | NR 侧空间巨大（NIQE 9.5 极差）；FR 侧因 lq-gt 像素对齐天然偏高——扩散增强预计"NR 大赚、FR 有让"，靠 λ/低频回填找平衡 | 91f920f |
| 20260907_diffbir_val | 2026-09-07 | competition | DiffBIR v2.1 同分辨率 val 5 张正式基线（512 瓦片/64 overlap/50 步/fp16/upscale=1） | 相对 lq_baseline：去噪 + DiffBIR + 色彩对齐 + 低频回填 + λ=0.7 | PSNR 24.9021 · SSIM 0.6414 · MS-SSIM 0.6854 · LPIPS 0.6367 · DISTS 0.2999 · MUSIQ 39.9418 · CLIPIQA 0.5742 · NIQE 4.6868 · MANIQA 0.3413 · 综合 0.5427/0.5529/0.5115 | NR 显著改善，FR 明显下降；equal/NR-heavy 高于控制组，但 FR-heavy 低于控制组，未通过完整调优闸门；下一步优先降低 λ 或增强低频保真，再做同口径复评 | 307be4d |
| 20260908_val_lam05 | 2026-09-08 | competition | 仅降低 λ 的保真度对照实验 | 相对 20260907_diffbir_val：仅 λ `0.7 → 0.5`，其余配置、数据和评测口径不变 | PSNR 26.2581 · SSIM 0.6965 · MS-SSIM 0.7311 · LPIPS 0.5591 · DISTS 0.2922 · MUSIQ 37.6864 · CLIPIQA 0.5121 · NIQE 4.8324 · MANIQA 0.3308 · 综合 0.5550/0.5846/0.5069 | 相比 λ=0.7，FR 全面回升且 equal 提升；NR-heavy 基本持平，但 FR-heavy 仍低于 lq_baseline 的 0.6306，未通过完整闸门；原计划继续测试 λ=0.3，后续决策调整为直接用 λ=0.5 做 test-100 平台校准 | bd41f5f |

<!-- 示例行（复制使用）：
| 20260908_diffbir_baseline_a3f2 | 2026-09-08 | diagnostic | DiffBIR 原生管线 baseline | 无（首测） | PSNR:xx SSIM:xx LPIPS:xx MUSIQ:xx | 建立基准 | abc1234 |
-->

## 当前闸门状态

- 当前阶段：阶段 4 的 val 控制变量扫描已暂停，转入 competition candidate 的 test-100 渲染与平台校准准备；尚未启动 test-100 或官方提交。
- 本轮 `20260907_diffbir_val` 与 `lq_baseline` 使用相同的 val 5 对、相同指标实现和相同本地启发式综合分，具备可比性。
- 相对控制组：equal 综合分 `0.5382 → 0.5427`，NR-heavy `0.4397 → 0.5115`；但 FR-heavy `0.6306 → 0.5529`，说明当前生成强度过高。
- 决策：`lambda=0.5` 已验证为比 `0.7` 更好的折中，当前作为 test-100 的临时 competition candidate；停止继续扫描 `lambda=0.3`，改用官方平台得分校准本地启发式综合分的可靠性。
- 下一步计划：用 `configs/experiments/20260908_val_lam05.yaml` 渲染 `20260909_diffbir_test` 的 100 张 test，做 NR-only 本地检查，通过提交校验后作为首次校准包上传平台。
- 选择说明：本次 test-100/首次提交是平台校准实验，不是最终模型冻结，也不代表已获得官方成绩；官方分数和榜单排名待平台返回后单独登记到 `SUBMISSIONS.md`。
- 闸门：test-100 输出必须 100 张齐全、命名正确、分辨率一致且无失败图片；官方提交前仍需经过 `src/submit/validate.py` 和打包结构检查。
- 保密状态：test 的 GT 仍未公开，当前没有内部 holdout 分数，也没有消耗官方提交额度。

## 指标口径说明（比较前必读）

- 评测范围：val 5 组（case1~5），lq 输入 → 与 gt 计算 FR 指标。
- FR：PSNR / SSIM / MS-SSIM / LPIPS / DISTS；NR：MUSIQ / CLIPIQA / NIQE / MANIQA。
- 综合分暂用三套候选权重：等权、FR 偏重、NR 偏重（官方公式公布后切换为官方口径）。
- 两个结果只有共享同一评测口径与同一数据指纹才可比较。
