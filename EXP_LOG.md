# 实验日志（EXP_LOG）

> 规则：只追加、不删除；失败的正式 run 也要记录。详细 manifest 见 `experiments/<run_id>.json`。
> run_id 模式：`<日期>_<主题>_<短哈希>`，如 `20260908_diffbir_baseline_a3f2`。

## 正式实验一览

| run_id | 日期 | role | 主题/假设 | 相对控制组改动 | 关键指标（val） | 结论 | git commit |
|---|---|---|---|---|---|---|---|

<!-- 示例行（复制使用）：
| 20260908_diffbir_baseline_a3f2 | 2026-09-08 | diagnostic | DiffBIR 原生管线 baseline | 无（首测） | PSNR:xx SSIM:xx LPIPS:xx MUSIQ:xx | 建立基准 | abc1234 |
-->

## 指标口径说明（比较前必读）

- 评测范围：val 5 组（case1~5），lq 输入 → 与 gt 计算 FR 指标。
- FR：PSNR / SSIM / MS-SSIM / LPIPS / DISTS；NR：MUSIQ / CLIPIQA / NIQE / MANIQA。
- 综合分暂用三套候选权重：等权、FR 偏重、NR 偏重（官方公式公布后切换为官方口径）。
- 两个结果只有共享同一评测口径与同一数据指纹才可比较。
