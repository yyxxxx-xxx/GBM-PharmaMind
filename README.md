# GBM-PharmaMind

基于多模态大语言模型的胶质母细胞瘤（GBM）新药候选分子生成与优化系统。

## 项目结构

```
gbm_project/
├── src/
│   ├── gbm_knowledge_base.py          # GBM知识库（靶点、临床、分子数据）
│   ├── gbm_prompt_generator.py        # Prompt生成器（CoT/ToT模板）
│   ├── gbm_generator.py               # CoT模式分子生成器
│   ├── gbm_evaluator.py               # 多维评估器（BBB、活性、毒性等）
│   ├── gbm_physical_evaluator.py      # 物理评估（Vina对接、ADMET）
│   ├── gbm_admet_predictor.py         # ADMET预测
│   ├── gbm_lora_finetuner.py          # LoRA微调模块
│   ├── gbm_lora_dataset.py            # LoRA数据集构建
│   ├── structural_knowledge_injector.py # 结构化学知识注入
│   └── sascorer.py                    # 合成难度评分
├── scripts/
│   ├── generate_tot_molecules.py       # ToT模式分子生成器（BFS树搜索）
│   └── train_gbm_lora_with_graph_models.py  # LoRA微调主脚本
├── experiments/
│   ├── generate_gbm_candidates.py      # CoT/ToT候选分子生成入口
│   └── evaluate_gbm_candidates.py       # 生成结果评估分析入口
├── configs/
│   ├── gbm_generation.yaml            # 生成主配置
│   ├── english_gbm_prompts.yaml        # 英文Prompt模板
└── data/
    ├── gbm_targets/                   # GBM靶点信息
    ├── gbm_clinical/                  # GBM临床数据
    ├── gbm_molecules/                 # GBM相关分子数据
    └── lora_datasets/                 # LoRA微调数据集
```

## 依赖说明

### 预训练模型准备

本项目使用 Qwen2-7B-Instruct 作为基座模型（需自行下载），推荐从 Hugging Face 获取：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载 Qwen2-7B-Instruct（需要约 14GB 磁盘空间）
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir models/Qwen2-7B-Instruct
```

> **注意**：LoRA 适配器和图模型由 Llamole 项目提供，请参考 [Llamole 官方仓库](https://github.com/your-llamole-repo) 获取对应版本的 `saves/` 目录。

### 推荐目录结构

```
GBM-PharmaMind/
├── models/
│   └── Qwen2-7B-Instruct/             # Qwen2 基座模型（需从 Hugging Face 下载）
├── saves/
│   ├── graph_decoder/                 # 图模型decoder（Llamole 提供）
│   ├── graph_encoder/                 # 图模型encoder（Llamole 提供）
│   ├── graph_predictor/               # 图模型predictor（Llamole 提供）
│   ├── Llamole-Qwen2-7B-Instruct-Adapter/           # Llamole 分子生成适配器
│   └── Llamole-Qwen2-7B-Instruct-Adapter/connector/  # 图-LM连接器
└── gbm_project/                      # 本项目代码
```

## 运行步骤

### 步骤一：环境配置

**方式一：一键自动配置（推荐）**

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

**方式二：手动安装依赖**

```bash
# 1. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 2. 安装其余依赖
pip install -r requirements.txt
```

- `setup_environment.sh`：自动创建 conda 环境、安装所有依赖
- `requirements.txt`：Python 包依赖列表（不含 PyTorch，请先按上述命令安装适合的 CUDA 版本）

### 步骤二：LoRA微调（可选，已微调的模型可直接使用）

```bash
cd gbm_project/scripts
python train_gbm_lora_with_graph_models.py
```

输出保存至 `saves/Llamole-Qwen2-7B-Instruct-Adapter-gbm-with-graph-models/`。

### 步骤三：生成GBM候选分子

**CoT模式**（链式思维推理）：

```bash
cd gbm_project/experiments
python generate_gbm_candidates.py \
    --config ../configs/gbm_generation.yaml \
    --num_candidates 20 \
    --target EGFR \
    --use_cot \
    --output_dir ./outputs
```

**ToT模式**（树状思维搜索，BFS策略）：

```bash
cd gbm_project/experiments
python generate_gbm_candidates.py \
    --num_candidates 20 \
    --target EGFR \
    --use_tot \
    --tot_k 3 \
    --tot_b 2 \
    --tot_depth 3 \
    --output_dir ./outputs
```

ToT三层搜索流程：骨架设计（Scaffold）→ 分子组装（Assembly）→ SMILES生成。每层生成 k 个候选，保留 b 个最优分支，并通过物理评估反馈（Vina对接、ADMET）进行硬剪枝。

**ToT模式直接运行脚本**（独立使用）：

```bash
cd gbm_project/scripts
python generate_tot_molecules.py --target EGFR --k 3 --b 2 --gpu_id 0
```

### 步骤四：评估分析生成结果（需要导入外部工具测试的结果后使用）

```bash
cd gbm_project/experiments
python evaluate_gbm_candidates.py \
    --results_file ./outputs/generated_molecules.json \
    --output_dir ./analysis
```

输出包含：综合得分统计、靶点分布、BBB穿透性散点图、得分相关性热力图、分子性质分布、Top候选雷达图等。

## 支持的靶点

`EGFR`、`VEGF_VEGFR`、`IDH1_IDH2`、`MGMT`、`PD1_PDL1`、`PI3K_AKT_mTOR`、`p53_MDM2`

## 核心评估指标

| 指标 | 说明 |
|---|---|
| BBB穿透性 | 血脑屏障通透性预测 |
| GBM细胞活性 | 对GBM细胞的生长抑制活性 |
| 正常细胞毒性 | 对正常细胞的安全性 |
| 选择性指数 | GBM活性与正常毒性的比值 |
| 合成可行性 | 基于分子复杂度的合成难度 |
| 临床相似度 | 与已知GBM药物的结构相似性 |
| Vina对接评分 | AutoDock Vina分子对接结合能 |
| ADMET | 吸收、分布、代谢、排泄、毒性预测 |
| MPO综合奖励 | 多参数加权几何平均综合评分 |

## 配置说明

主要配置文件 `configs/gbm_generation.yaml`：

- `model_name_or_path`：基座模型路径
- `adapter_name_or_path`：GBM LoRA适配器路径
- `molecular_constraints`：分子性质约束（MW、LogP、TPSA等）
- `gbm_target_weights`：各靶点生成权重（用户输入未规定靶点时自动使用）
