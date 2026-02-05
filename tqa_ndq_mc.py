from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import sys
import argparse
import numpy as np
import random
import os
from transformers import RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
# 引入 AdaLoGN 组件
from aux_methods import get_data_ndq, process_data_ndq, training_ndq
from models.new_model import RobertaAdaLoGN


def load_model_weights(model, weight_path):
    """
    精简版权重加载：仅负责将预训练权重迁移到 AdaLoGN 新架构中
    """
    print(f"Loading weights from {weight_path} ...")
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return False

    try:
        # 1. 加载检查点 (在 v2.11 环境下，这里应该能直接成功)
        checkpoint = torch.load(weight_path, map_location='cpu')

        # 2. 提取参数字典 (state_dict)
        if hasattr(checkpoint, 'state_dict'):
            # 如果加载出来是一个模型对象(RobertaForMultipleChoice)，提取其参数
            print("Detected model object. Extracting state_dict for migration...")
            state_dict = checkpoint.state_dict()
        else:
            # 如果本身就是字典
            state_dict = checkpoint

        # 3. 迁移参数
        # strict=False 是必须的！
        # 因为 state_dict 里只有 RoBERTa 的参数，没有 AdaLoGN 的 GNN/Fusion 参数
        # 我们只加载匹配的部分 (Backbone)，剩下的 GNN 部分保持随机初始化
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"Weights loaded successfully.")
        print(f"  - Backbone loaded (RoBERTa weights).")
        print(f"  - Initialized randomly: {len(missing_keys)} layers (GNN/Fusion layers).")
        return True

    except Exception as e:
        print(f"Error loading weights: {e}")
        return False


def main(argv):
    parser = argparse.ArgumentParser(description='TQA Multiple Choice Task with AdaLoGN')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'], help='retrieval solver for the contexts.',
                          required=True)

    parser.add_argument('-t', '--dataset', default='ndq', choices=['ndq', 'dq'],
                        help='dataset to train the model with.')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device options: cpu or gpu.')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/pretrainings_e4.pth", help='path to pretrainings.')

    # --- 关键参数优化 ---
    parser.add_argument('-b', '--batchsize', default=2, type=int,
                        help='Physical batch size per step. Keep small (e.g. 2) for GNN.')
    parser.add_argument('-acc', '--accumulation_steps', default=8, type=int,
                        help='Gradient accumulation steps. Total Batch = b * acc.')
    # ------------------

    parser.add_argument('-x', '--maxlen', default=128, type=int, help='max sequence length.')
    parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate.')
    parser.add_argument('-e', '--epochs', default=4, type=int, help='number of epochs.')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training',
                        action='store_true')
    args = parser.parse_args()
    print(args)

    # 1. 初始化 AdaLoGN 模型 (MC任务 num_labels=1，用于给每个选项打分)
    print(f"Initializing RobertaAdaLoGN for {args.dataset} task...")
    # 注意：这里使用 ./checkpoints/roberta-large 以匹配你本地的路径设置，如果报错可改回 "roberta-large"
    try:
        model = RobertaAdaLoGN.from_pretrained("./checkpoints/roberta-large", num_labels=1)
    except:
        model = RobertaAdaLoGN.from_pretrained("roberta-large", num_labels=1)

    # 2. 加载权重 (直接加载 pretrainings_e4.pth)
    if args.pretrainings:
        success = load_model_weights(model, args.pretrainings)
        if not success:
            print("WARNING: Using random initialization for backbone!")

    # 初始化 Tokenizer
    try:
        print("Initializing RobertaTokenizerFast...")
        tokenizer = RobertaTokenizerFast.from_pretrained("./checkpoints/roberta-large")
    except:
        print("Warning: Local path failed, trying download...")
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    if args.device == "gpu":
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    # 3. 数据准备
    # 注意：aux_methods 已经修改为生成图数据
    raw_data_train = get_data_ndq(args.dataset, "train", args.retrieval, tokenizer, args.maxlen)
    raw_data_val = get_data_ndq(args.dataset, "val", args.retrieval, tokenizer, args.maxlen)

    train_dataloader = process_data_ndq(raw_data_train, args.batchsize, "train")
    val_dataloader = process_data_ndq(raw_data_val, args.batchsize, "val")

    # 4. 优化器
    # 总步数需要除以 accumulation_steps
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    total_steps = (len(train_dataloader) // args.accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 5. 训练 (传入 accumulation_steps)
    training_ndq(
        model, train_dataloader, val_dataloader, optimizer, scheduler,
        args.epochs, args.retrieval, device, args.save, args.dataset,
        accumulation_steps=args.accumulation_steps
    )


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    main(sys.argv[1:])