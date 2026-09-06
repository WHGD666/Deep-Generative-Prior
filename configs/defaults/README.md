# configs/defaults — 管线默认配置

按管线层级拆分的默认参数（骨架阶段为空，随模块实现逐个落盘）：

- `preprocess.yaml` — 去噪/去模糊强度
- `enhance.yaml` — 模型选择、分块尺寸与重叠、扩散步数/CFG/种子
- `postprocess.yaml` — 低频回填开关与强度、λ、色彩对齐方式
- `evaluate.yaml` — 指标集与候选权重组合
- `submit.yaml` — 提交包命名信息（作品名/队名/联系方式占位，提交前填写）
