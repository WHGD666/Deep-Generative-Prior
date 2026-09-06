# experiments — 正式实验记录

每个正式 run 一个子目录 + 一个 manifest，是**项目的事实数据库**（所有报告数字从这里来，不允许手工另造）。

## 目录约定

```text
experiments/
├── README.md                  # 本文件
├── <run_id>/                  # 每个 run 的产物
│   ├── outputs/               # 【不入库】输出图片
│   └── eval.json              # 【入库】该 run 在验证集上的指标
└── <run_id>.json              # 【入库】run manifest（见根 README 7.3）
```

## 纪律（源自实验管理规范）

- run_id 唯一且不复用：`<日期>_<主题>_<短哈希>`
- 失败的 run **保留**：status=fail + 失败类型/日志摘要 + 下一步决策
- 一次正式对比只改一个有意义的变量；不同口径的结果不可直接比较
- 汇总表：根目录 `EXP_LOG.md`（人读），JSON（机器读），两者一致
