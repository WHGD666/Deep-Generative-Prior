# submissions — 提交快照记录

每次官方提交在这里留一份**轻量快照**（zip 本体留在本地磁盘，不入库）。

## 目录约定

```text
submissions/
└── sub01-20260909/            # 第 N 次提交 + 日期
    ├── manifest.md            # 来源 run_id、zip 文件名、文件清单摘要、打包校验结果
    └── local_scores.json      # 该包在验证集上的本地指标
```

## 纪律

- 上传前：`src/submit/validate.py` 必须全绿（命名/数量/分辨率/格式）
- 上传后：官方分数与榜单反馈**当天**回填根 `SUBMISSIONS.md` 台账
- 台账与快照共同构成 10 次额度的完整审计链
