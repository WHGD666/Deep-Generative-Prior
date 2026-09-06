"""后处理：扩散输出 -> 可提交结果之间的提分关键层。

管线内的固定执行顺序（见 run_pipeline.run_image）：
  1. color_align        色彩对齐（对输入的全局统计）
  2. lowfreq_backfill   低频回填（对输入的低频分量）
  3. lambda_blend       保真-生成混合（fidelity=预处理后的输入）
  4. face_branch        人脸分支（默认关闭）
每一层都可独立开关（enabled: false），全关时输出 = 纯扩散增强结果。
"""
