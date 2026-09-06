# src/evaluate — 评测体系

模拟官方综合评分，把"感觉变好了"变成**可比较、可复现的数字**。

## 规划模块

- `metrics_fr.py` — 有参考指标：PSNR / SSIM / MS-SSIM / LPIPS / DISTS
- `metrics_nr.py` — 无参考指标：MUSIQ / CLIPIQA / NIQE / MANIQA
- `composite.py` — 综合分：多套候选权重（等权 / FR 偏重 / NR 偏重）；官方公式公布后切换官方口径
- `run_eval.py` — 评测入口：一组输出图 + GT → 指标 JSON（落盘 `experiments/<run_id>/`）
- `report.py` — 从指标 JSON 汇总对照表（供 EXP_LOG 与 docs/reports 引用）

## 纪律

- 指标实现尽量用成熟库（pyiqa 等）并对齐公认实现；自实现必须配 `tests/` 校验用例
- **本地验证分永远不得表述为官方分数**（对外材料红线）
- 每次评测记录环境与库版本，指标数值随库版本可能漂移
