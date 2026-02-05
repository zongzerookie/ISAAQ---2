from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse
import os

# 这个是使用全部都是我们与训练的来训练dmc
from aux_methods import get_data_dq, training_dq, SpatiallyAwareISAAQ, ResnetRobertaBUTD


def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'],
                          help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')

    # [默认] 指向你刚刚训练好的 Unfrozen 权重
    parser.add_argument('-p', '--pretrainings', default="checkpoints/AI2D_spatial_best_NoFreezing.pth",
                        help='Path to YOUR SPATIAL checkpoint.')

    parser.add_argument('-b', '--batchsize', default=1, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')

    # [建议] 解冻微调阶段，学习率给小一点 (5e-6) 比较稳，防止破坏预训练知识
    parser.add_argument('-l', '--lr', default=5e-6, type=float, help='learning rate. Default: 5e-6')

    parser.add_argument('-e', '--epochs', default=4, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training',
                        action='store_true')

    parser.add_argument('--train_feats', default=None,
                        help='Path to extracted AdaLoGN features for TRAIN set')
    parser.add_argument('--val_feats', default=None,
                        help='Path to extracted AdaLoGN features for VAL set')

    args = parser.parse_args()
    print(args)

    # 确定设备
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 1. 实例化我们的新模型
    model = SpatiallyAwareISAAQ()

    # =================================================================================
    # 核心修改：只加载 NoFreezing 权重，不覆盖 RoBERTa
    # =================================================================================
    if args.pretrainings != "" and os.path.exists(args.pretrainings):
        print(f"Loading Unfrozen weights from: {args.pretrainings}")
        try:
            checkpoint = torch.load(args.pretrainings, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            # strict=False 是为了兼容 AdaLoGN 的新层 (它们会保持随机初始化)
            # 但关键的 RoBERTa 和 Spatial Encoder 必须匹配
            msg = model.load_state_dict(state_dict, strict=False)

            print(f"✅ Weights loaded successfully!")
            print(f"   Missing keys (Expected for new AdaLoGN layers): {len(msg.missing_keys)}")
            print("   (RoBERTa & Spatial Encoder are now loaded from your NoFreezing training)")

        except Exception as e:
            print(f"ERROR loading your weights: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Checkpoint not found at {args.pretrainings}")
        sys.exit(1)

    # 3. 冻结 ResNet (保持不变，只做特征提取)
    for param in model.resnet.parameters():
        param.requires_grad = False
    print("ResNet parameters frozen.")

    # 4. 解冻 RoBERTa (关键：允许 RoBERTa 在 TQA 上继续微调)
    # 你的 NoFreezing 权重里的 RoBERTa 已经适应了空间编码器，现在让它适应 TQA 任务
    for param in model.roberta.parameters():
        param.requires_grad = True
    print("RoBERTa parameters UN-FROZEN (Ready for Fine-tuning).")

    # -------------------------------------------------------------

    tokenizer = RobertaTokenizer.from_pretrained("./checkpoints/roberta-large")

    model.to(device)
    model.zero_grad()

    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save

    print("Loading Training Data...")
    # 数据加载依然使用原始的 retrieval (比如 "IR") 来寻找正确的数据文件
    raw_data_train = get_data_dq("train", retrieval_solver, tokenizer, max_len, feature_file=args.train_feats)

    print("Loading Validation Data...")
    raw_data_val = get_data_dq("val", retrieval_solver, tokenizer, max_len, feature_file=args.val_feats)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    total_steps = len(raw_data_train[-1]) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # =================================================================================
    # 文件名 Hack：构造一个新的 save_name 传给 training_dq
    # 这样生成的文件名就会包含 "UNFROZEN"，避免覆盖之前的 "SPATIAL" 权重
    # =================================================================================
    save_name_prefix = f"UNFROZEN_{retrieval_solver}"
    print(f"Training will save checkpoints as: dmc_dq_roberta_SPATIAL_{save_name_prefix}_eX.pth")

    training_dq(
        model,
        raw_data_train,
        raw_data_val,
        optimizer,
        scheduler,
        epochs,
        batch_size,
        save_name_prefix,  # <--- 这里传入修改后的名字
        device,
        save_model
    )


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])