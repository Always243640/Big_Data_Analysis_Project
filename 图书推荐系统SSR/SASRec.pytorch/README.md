# SASRec 图书推荐模型使用指南

本说明基于仓库中 `图书推荐系统SSR/SASRec.pytorch` 目录下的代码，指导你如何利用本项目在你自己的图书借阅数据上训练 [SASRec](https://arxiv.org/abs/1808.09781) 模型，并导出 Top-K（包括 Top-1、Top-5）推荐结果。

## 1. 环境准备

1. 安装 Python (建议 3.8 及以上)。
2. 创建并激活虚拟环境（可选）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell 中使用 .venv\Scripts\Activate.ps1
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   若仓库未提供 `requirements.txt`，可以直接安装关键依赖：
   ```bash
   pip install torch numpy pandas
   ```

> **提示**：如果需要 GPU 训练，请参照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与显卡兼容的 `torch` 版本，并在运行脚本时将 `--device` 参数设为 `cuda`。

## 2. 数据准备

- 将你的交互数据放在 `mydata/inter_reevaluation.csv`（或任意 CSV 路径）中，确保至少包含以下列：
  - `user_id`：用户唯一标识（数字）。
  - `book_id`：物品/图书唯一标识（数字）。
  - 可选时间字段（如 `borrow_time`、`timestamp` 等），用于保证交互按照时间顺序排序。
- 如果提供的是一个目录（例如 `mydata/`），脚本会自动在目录下寻找 `inter_reevaluation.csv`。

## 3. 训练模型

进入 `图书推荐系统SSR/SASRec.pytorch` 目录后，执行类似如下命令：

```bash
python main.py \
  --dataset ../../mydata \
  --batch_size 128 \
  --num_epochs 200 \
  --device cuda \
  --topk 1,5
```

关键参数说明：

- `--dataset`：数据集的目录或文件路径。可以是 `../../mydata`（目录），也可以直接写 `../../mydata/inter_reevaluation.csv`（文件）。
- `--train_dir`：训练日志与模型权重的保存子目录名。**如果不写该参数，程序会自动使用默认值 `run1`。** 最终结果会保存在 `数据集名称_train_dir` 的文件夹中。例如默认设置会生成 `mydata_run1/` 目录；如果你想把这次实验命名为 `baseline`, 则可以传入 `--train_dir baseline`，输出目录会变成 `mydata_baseline/`。
- `--topk`：需要导出的 Top-K 列表，多个值用英文逗号分隔，默认已经包含 `1,5`。
- `--device`：设为 `cpu` 或 `cuda`。
- 其他可选参数可根据需要调整，如 `--maxlen`（序列最大长度）、`--hidden_units`、`--num_blocks` 等。

训练过程中程序会定期在验证集/测试集上打印 NDCG@10、HR@10 等指标。当训练完成且达到最后一个 epoch 时，会在输出目录下保存一份模型权重：

```
mydata_run1/
├── args.txt                 # 运行参数记录
├── log.txt                  # 指标日志
├── recommendations.csv      # Top-K 推荐结果
└── SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth
```

## 4. 仅导出推荐结果（推理模式）

若已经有训练好的权重（例如位于 `mydata_run1/SASRec.epoch=200...pth`），可以在不重新训练的情况下直接生成 Top-K 推荐：

```bash
python main.py \
  --dataset ../../mydata \
  --inference_only true \
  --state_dict_path mydata_run1/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth \
  --generate_topk true \
  --topk 1,5 \
  --recommend_output recommendations_eval.csv
```

常用参数：
- `--inference_only true`：跳过训练，仅加载模型并执行评估/推荐。
- `--state_dict_path`：训练完成后保存的 `.pth` 权重文件路径。
- `--train_dir`：在推理模式下依然会用来确定输出目录。可不指定使用默认 `run1`，或根据需要另起名（如 `--train_dir eval_only`）。
- `--generate_topk`：是否在推理结束后生成推荐 CSV，默认 `true`。
- `--recommend_output`：Top-K 结果保存文件名，默认 `recommendations.csv`。

## 5. 推荐结果格式

生成的 `recommendations.csv` 文件包含每个用户的推荐列表，结构如下：

```csv
user_id,top1,top5
10001,301,301 22 18 45 90
10002,17,17 50 42 96 12
...
```

- `user_id` 是原始数据中的用户标识。
- `top1`、`top5` 等列以空格分隔的形式列出了对应的推荐物品（原始 `book_id`）。

## 6. 常见问题

1. **报错：`main.py: error: the following arguments are required: --dataset, --train_dir`**
   现在只需提供 `--dataset`，`--train_dir` 不写时会默认用 `run1`。如果你想更清晰地区分多次实验，可以主动传入不同的 `--train_dir` 名称。

2. **环境缺失 `numpy` / `torch`**  
   请确保按照“环境准备”部分安装了必要依赖。

3. **Top-K 推荐为空**  
   请确认用户在训练集/验证集/测试集中至少有一次交互，并且 `maxlen` 设置能够覆盖最近的历史记录。

如有更多问题，可检查输出目录下的 `log.txt` 或 `args.txt` 了解运行细节。

