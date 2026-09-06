# src — 源代码

三层管线的全部实现代码。**运行环境为远程 RTX 5090**，本地仅做代码编辑与小规模调试。

## 模块划分（自上游到下游）

| 模块 | 职责 | 依赖上游 |
|---|---|---|
| [`data_prep/`](data_prep/) | 数据解压、指纹、统计核查 | 官方 zip |
| [`preprocess/`](preprocess/) | 去噪/去模糊等输入净化 | data_prep |
| [`enhance/`](enhance/) | 扩散增强核心：模型封装、4K 分块推理、融合 | preprocess |
| [`postprocess/`](postprocess/) | 低频回填、λ 混合、色彩对齐、人脸分支 | enhance |
| [`evaluate/`](evaluate/) | FR/NR 指标、综合分、评测报告 | 任意输出图 |
| [`submit/`](submit/) | 提交包组装、命名/尺寸校验、清单 | postprocess |
| [`utils/`](utils/) | IO、日志、随机种子等通用工具 | 无 |

## 编码纪律

- 每个模块职责单一，跨模块只通过显式函数/配置传参，不隐式共享状态
- 随机性（种子、采样步数）必须进入配置文件，禁止散落硬编码
- 公共逻辑先查 `utils/`，避免复制粘贴分叉
- 新增依赖必须在 `environment/` 记录版本并说明用途
