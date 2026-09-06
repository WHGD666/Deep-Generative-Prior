# configs/experiments — 实验配置快照

- 文件名 = run_id，与 `experiments/<run_id>.json` 一一对应
- 快照入 git 后**不可修改**；调参 = 复制出新文件 + 新 run_id
- 建议继承 defaults 全量展开（不写"引用默认值"），保证单文件自足可读
