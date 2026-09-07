# 实验日志（EXP_LOG）

> 规则：只追加、不删除；失败的正式 run 也要记录。详细 manifest 见 `experiments/<run_id>.json`。
> run_id 模式：`<日期>_<主题>_<短哈希>`，如 `20260908_diffbir_baseline_a3f2`。

## 正式实验一览

| run_id | 日期 | role | 主题/假设 | 相对控制组改动 | 关键指标（val） | 结论 | git commit |
|---|---|---|---|---|---|---|---|
| 20260907_val_probe | 2026-09-07 | diagnostic | DiffBIR v2.1 同分辨率管线首图探路（仅 val case1，512 瓦片/50 步/fp16/upscale=1） | 无（首个 run） | PSNR 29.04 · SSIM 0.741 · LPIPS 0.592 · DISTS 0.314 · MUSIQ 25.7 · NIQE 4.01 · 综合(equal) 0.570 | 端到端跑通（402.6s/张，4K 输出尺寸一致）；分数待 LQ 不增强基线对照后才有意义 | dc719e0 |
| lq_baseline | 2026-09-07 | control | LQ 不增强对照组：val 5 张原始输入直接评测 | 无（控制组，量管线增益用） | PSNR 28.03 · SSIM 0.788 · MS-SSIM 0.805 · LPIPS 0.357 · DISTS 0.279 · MUSIQ 21.3 · CLIPIQA 0.333 · NIQE 9.53 · MANIQA 0.268 · 综合 0.538/0.631/0.440 | NR 侧空间巨大（NIQE 9.5 极差）；FR 侧因 lq-gt 像素对齐天然偏高——扩散增强预计"NR 大赚、FR 有让"，靠 λ/低频回填找平衡 | 91f920f |

<!-- 示例行（复制使用）：
| 20260908_diffbir_baseline_a3f2 | 2026-09-08 | diagnostic | DiffBIR 原生管线 baseline | 无（首测） | PSNR:xx SSIM:xx LPIPS:xx MUSIQ:xx | 建立基准 | abc1234 |
-->

## 指标口径说明（比较前必读）

- 评测范围：val 5 组（case1~5），lq 输入 → 与 gt 计算 FR 指标。
- FR：PSNR / SSIM / MS-SSIM / LPIPS / DISTS；NR：MUSIQ / CLIPIQA / NIQE / MANIQA。
- 综合分暂用三套候选权重：等权、FR 偏重、NR 偏重（官方公式公布后切换为官方口径）。
- 两个结果只有共享同一评测口径与同一数据指纹才可比较。
