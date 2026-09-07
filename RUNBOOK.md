# RUNBOOK — 全流程作战手册（从开机连接到官方提交）

> 适用场景：拿到（或重启）一台租用的 RTX 5090 实例，从连接开始，跑验证、跑 100 张、
> 打包、拉回本地、完成官方提交的**完整命令指导**。
> 配套文档：根 `README.md`（方案与快速开始）、`EXP_LOG.md`（实验记录）、
> `SUBMISSIONS.md`（提交台账）、`environment/TROUBLESHOOTING.md`（18+ 条排错）。
>
> ⚠️ 铁律：**密码等凭证只存在于平台控制台，绝不写进本仓库**（本仓库在 GitHub 公开）。

---

## 0. 连接信息（每次租用/重启后都可能变化）

| 项 | 去哪拿 |
|---|---|
| SSH 端口 / 主机 / 密码 | 租用平台实例控制台（重启后端口和主机名会变，密码可能重置） |
| 网页版 VS Code | 控制台提供的 `https://<主机>:<端口>` 链接（和 SSH 二选一，效果相同） |

本仓库所有命令的**项目根目录**固定为：

```bash
cd /data/coding/Deep-Generative-Prior
```

---

## 1. 每次开机的标准流程（重启恢复，约 5 分钟）

### 1.1 本地连接远程

```bash
# 在 Windows 本机的 CMD 或 Git Bash 里执行（端口/主机按平台控制台替换）
ssh -p <端口> root@<主机>
# 粘贴控制台显示的密码（输入时不回显，粘完直接回车）
```

### 1.2 进项目 + 激活环境（⚠️ 每开一个新 shell/tmux 窗口都要做这两步）

```bash
cd /data/coding/Deep-Generative-Prior
source /data/miniconda/etc/profile.d/conda.sh && conda activate camera310
```

确认提示符变为 `(camera310) root@...:/data/coding/Deep-Generative-Prior#` 再继续。
（`(torch)` 开头 = 还在平台基础环境，跑我们脚本必出错）

### 1.3 重启后健康检查（容器重启 = 只有 /data 存活）

```bash
nvidia-smi                                                   # GPU 可见
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
                                                             # 应打印 2.7.1+cu128 (12, 0)
git log --oneline -1                                         # 记下当前 commit
ls third_party/DiffBIR/weights/                              # 3 个权重文件应在（/data 存活）
ls data/val data/test | head -6                              # 数据应在
```

### 1.4 补装系统工具 + 同步代码

```bash
# tmux 随系统盘丢失，重启后必重装（setup 第 0 步也会自动装）
apt-get update -qq && apt-get install -y tmux

# 同步最新代码（必须在仓库目录里！在 /data/coding 执行会报 not a git repository）
git pull
# 若卡住/超时，先恢复代理配置再 pull：
git config url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
```

### 1.5 权重缓存自愈（/root/.cache 随系统盘丢失，按需触发）

```bash
# 冒烟或评测报"下载/Connection reset"错误时才需要跑，两者都幂等（已有的自动跳过）：
bash environment/download_diffbir_weights.sh   # DiffBIR 权重（/data 上，重启后一般还在）
bash environment/download_iqa_weights.sh       # pyiqa 评测权重（在 /root/.cache，重启后必重下）
```

### 1.6 一键回归验证（可选但推荐，约 2 分钟）

```bash
pytest tests/ -v -m "not remote"                       # CPU 用例须全绿
pytest tests/test_smoke_diffbir.py -v -m remote -s     # 应输出 [冒烟] ... k=1
```

---

## 2. 全新机器（/data 也空了）才需要的部分

```bash
# ① 环境一键搭建（幂等，20-40 分钟）
bash environment/setup_5090.sh
source /data/miniconda/etc/profile.d/conda.sh && conda activate camera310

# ② 数据上传（本机执行；数据 zip 在本地，改名 data.zip 规避中文编码问题）
#    scp -P <端口> D:\path\to\data.zip root@<主机>:/data/coding/Deep-Generative-Prior/
bash scripts/prepare_data.sh data.zip        # 解压+指纹+统计核查

# ③ 权重下载（tmux 里，DiffBIR 6.3GB + IQA 12GB，断点续传）
bash environment/download_diffbir_weights.sh
bash environment/download_iqa_weights.sh

# ④ 验证
pytest tests/ -v -m "not remote"
pytest tests/test_smoke_diffbir.py -v -m remote -s
```

---

## 3. 实验主线（闸门协议：val 调优 → 过闸 → test 100 → 提交）

> 核心纪律：**官方提交只有 10 次**。一切配置先在 val 5 对上调到最优，
> 本地综合分 ≥ 历史最优才允许消耗 test/提交额度。

### 3.1 阶段一 · LQ 对照组（约 5 分钟；每个数据集指纹只需做一次）

```bash
bash scripts/eval_lq_baseline.sh
```

记录 `[均值]` `[综合]` —— 这是"什么都不做"的基准线（历史值见 `EXP_LOG.md`）。

### 3.2 阶段二 · val 全量增强（约 35 分钟，tmux 里跑）

```bash
tmux new -s val                 # 已存在会报 duplicate session，改用 tmux attach -t val
# （tmux 内新 shell 记得重新做 1.2 的两步：cd + conda activate）
bash scripts/run_enhance.sh 20260907_diffbir_val data/val "{case}_lq.jpg" \
    experiments/20260907_diffbir_val/outputs
```

**tmux 操作**：`Ctrl+B` 松开再按 `D` = 脱离（任务继续跑，可安全断网关机）；
`tmux attach -t val` = 回来查看；进度日志在 `experiments/<run_id>` 下的 `.log`。

### 3.3 阶段三 · val 评测（约 2 分钟）

```bash
bash scripts/run_eval.sh 20260907_diffbir_val experiments/20260907_diffbir_val/outputs data/val
```

★ **检查点**：把 `[均值]` `[综合]` 与 `EXP_LOG.md` 里的 lq_baseline 和历史最优对比，
差值方向决定调参动作（分析结论发给 AI 或自行对照 README §6）。

### 3.4 阶段四 · 调优循环（每轮只改一个变量）

改一个参数 → 新 run_id 重跑 3.2/3.3 → 记 `EXP_LOG.md`：

```bash
nano configs/defaults/postprocess.yaml     # 例：lambda 0.7 -> 0.5
bash scripts/run_enhance.sh 20260908_val_lam05 data/val "{case}_lq.jpg" \
    experiments/20260908_val_lam05/outputs
bash scripts/run_eval.sh 20260908_val_lam05 experiments/20260908_val_lam05/outputs data/val
```

可调旋钮速查：`postprocess.yaml` 的 `lambda`（保真↔生成）、`lowfreq_backfill.sigma/strength`、
`color_align.strength`；`enhance.yaml` 的 `steps`（50→30 提速 40%）、`cfg_scale`、`tile_size`；
`preprocess.yaml` 的 `denoise.gaussian_sigma`。
候选 run_id 命名：`<日期>_val_<改动>_<值>`，如 `20260908_val_lam05`。

### 3.5 ★闸门★

新配置的**综合分（equal 与 fr_heavy 为主）≥ 历史最优** → 允许进阶段五；
否则回 3.4 继续调。历史最优永远以 `EXP_LOG.md` 最新记录为准。

### 3.6 阶段五 · test 100 张渲染（约 11 小时，过闸才跑）

```bash
tmux new -s test100
# （tmux 内：cd + conda activate）
bash scripts/run_enhance.sh 20260909_diffbir_test data/test "{case}.jpg" \
    experiments/20260909_diffbir_test/outputs
```

中断/失败续跑（**已完成的不重算**）：

```bash
bash scripts/run_enhance.sh 20260909_diffbir_test data/test "{case}.jpg" \
    experiments/20260909_diffbir_test/outputs --resume
```

### 3.7 阶段六 · test 评测（无 GT，仅无参考指标，约 10 分钟）

```bash
bash scripts/run_eval.sh 20260909_diffbir_test experiments/20260909_diffbir_test/outputs none
```

### 3.8 阶段七 · 打包（内置强制校验，不过即拒绝）

```bash
nano configs/defaults/submit.yaml       # 三个占位字段必须填：work_name / team_name / phone
bash scripts/pack_submission.sh 20260909_diffbir_test sub01
# 产出：赛题一_<作品名>_<队名>_<手机号>.zip（仓库根）+ submissions/sub01-<日期>/manifest.md
# 复制一份 ASCII 名便于传输（官方要求 zip 用中文名，传输完本地再改回）：
cp 赛题一_*.zip sub01.zip
```

### 3.9 阶段八 · 拉回本地并提交（⚠️ 这一步在**本机**执行，不是远程）

```bash
# Windows 本机 CMD / Git Bash（端口主机用当前实例的）：
mkdir D:\submit
scp -P <端口> root@<主机>:/data/coding/Deep-Generative-Prior/sub01.zip D:\submit\
```

下载完成后，在本地把 `sub01.zip` **改回官方命名**：

```text
赛题一_<作品名>_<队名>_<手机号>.zip
```

上传到官方提交入口。**提交当天**把官方分数填进根目录 `SUBMISSIONS.md` 台账
（本地分数→官方分数的校准从 sub01 开始建立）。

---

## 4. 常见事故速查（详版见 environment/TROUBLESHOOTING.md）

| 症状 | 原因 | 处置 |
|---|---|---|
| `tmux: command not found` | 容器重启清了系统盘 | `apt-get update -qq && apt-get install -y tmux` |
| `not a git repository` | 站在 /data/coding 没进仓库 | `cd /data/coding/Deep-Generative-Prior` |
| `duplicate session: val` | 会话已存在 | `tmux attach -t val` |
| 提示符是 `(torch)` 不是 `(camera310)` | 新 shell 未激活环境 | 1.2 的两条命令 |
| `git pull` 卡死/超时 | 容器重启丢了代理配置 | 1.4 的 `git config ... insteadOf` 后重试 |
| 评测/冒烟报 Connection reset（下载权重） | /root/.cache 被重启清空 | 1.5 两个下载脚本（幂等续传） |
| 脚本报 `ModuleNotFoundError` | 用了平台 (torch) 基础环境 | `conda activate camera310` |
| 管线 OOM | 瓦片过大或显存被占 | 确认 `upscale: 1`；降 tile_size 512→384；`nvidia-smi` 查占卡 |
| 跑到一半断了（test 100） | SSH 断线且没用 tmux | 原命令末尾加 `--resume` 续跑 |

---

## 5. 每次收工的登记义务

1. 实验结果 → 根 `EXP_LOG.md` 追加一行（run_id / 指标 / 结论 / commit）；
2. 提交结果 → 根 `SUBMISSIONS.md`（序号 / 日期 / 本地分 / 官方分 / 下一步）；
3. 代码与文档改动 → 里程碑式 commit（`feat:/fix:/docs:/exp:` 前缀）后 `git push`；
4. 数据、权重、输出图、zip **永不入库**（.gitignore 已兜底）。
