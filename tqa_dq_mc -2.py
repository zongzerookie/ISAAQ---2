from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse
import os

# --- MODIFICATION START: 导入新旧两个模型类 ---
from aux_methods import get_data_dq, training_dq, SpatiallyAwareISAAQ, ResnetRobertaBUTD


# --- MODIFICATION END ---

def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'],
                          help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')

    # 这里默认加载你自己的 Spatial 预训练权重
    parser.add_argument('-p', '--pretrainings', default="checkpoints/AI2D_spatial_best.pth",
                        help='Path to YOUR SPATIAL checkpoint.')

    parser.add_argument('-b', '--batchsize', default=1, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')


    # parser.add_argument('-l', '--lr', default=1e-6, type=float, help='learning rate. Default: 2e-5')
    parser.add_argument('-l', '--lr', default=5e-6, type=float, help='learning rate. Default: 2e-5')

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
    # 第一步：加载你自己的 Spatial 预训练权重 (为了获取 Spatial Encoder)
    # =================================================================================
    if args.pretrainings != "" and os.path.exists(args.pretrainings):
        print(f"Step 1: Loading YOUR Spatial weights from: {args.pretrainings}")
        try:
            checkpoint = torch.load(args.pretrainings, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            model_dict = model.state_dict()
            # 筛选匹配参数
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            # 加载 (此时 RoBERTa 变成了你预训练时的通用版/冻结版)
            msg = model.load_state_dict(pretrained_dict, strict=False)

            if any('spatial_encoder' in k for k in pretrained_dict):
                print("✅ SUCCESS: Loaded YOUR Spatial Encoder weights!")
            else:
                print("⚠️ WARNING: Spatial Encoder NOT found in your checkpoint!")

        except Exception as e:
            print(f"ERROR loading your weights: {e}")
    else:
        print("No spatial checkpoint specified. Spatial Encoder will be random!")

    # =================================================================================
    # 第二步：【核心修改】加载原作者的 AI2D 权重 (为了获取 满级 RoBERTa)
    # =================================================================================
    author_checkpoint_path = "checkpoints/AI2D_e12.pth"  # 原作者权重路径 (请确认文件名)

    print(
        f"\nStep 2: [Hybrid Strategy] Attempting to overwrite RoBERTa with Author's weights from {author_checkpoint_path}...")

    if os.path.exists(author_checkpoint_path):
        try:
            # 加载原作者模型
            author_ckpt = torch.load(author_checkpoint_path, map_location='cpu')

            # 获取 state_dict
            if hasattr(author_ckpt, 'state_dict'):
                author_sd = author_ckpt.state_dict()
            elif isinstance(author_ckpt, dict):
                author_sd = author_ckpt
            else:
                # 兼容旧代码直接保存模型对象的情况
                author_sd = author_ckpt.state_dict()

            # 提取并清理 RoBERTa 权重
            # 原作者 RoBERTa 键名通常是 'roberta.embeddings.word_embeddings.weight' 等
            author_roberta_weights = {}
            for k, v in author_sd.items():
                if k.startswith('roberta.'):
                    # 去掉 'roberta.' 前缀，因为我们要把它加载到 model.roberta 子模块里
                    new_key = k.replace('roberta.', '', 1)
                    author_roberta_weights[new_key] = v

            # 强行覆盖
            if len(author_roberta_weights) > 0:
                # strict=True 保证完全匹配 RoBERTa 的每一层，不缺斤少两
                model.roberta.load_state_dict(author_roberta_weights, strict=True)
                print(
                    f"✅ SUCCESS: Overwrote RoBERTa with Author's 'Expert' weights! ({len(author_roberta_weights)} layers)")
                print("   Current Model Status: [Author's RoBERTa] + [Your Spatial Encoder] + [Random AdaLoGN]")
            else:
                print("⚠️ WARNING: No 'roberta.' keys found in author's checkpoint.")

        except Exception as e:
            print(f"❌ Failed to load Author's RoBERTa: {e}")
            print("   Using the RoBERTa from Step 1 (Generic/Frozen) instead.")
    else:
        print(f"⚠️ Author's checkpoint not found at {author_checkpoint_path}. Using Step 1 weights only.")

    # 3. 冻结 ResNet (保持不变)
    for param in model.resnet.parameters():
        param.requires_grad = False
    print("\nResNet parameters frozen.")

    # 确保 RoBERTa 是解冻的 (参与训练)
    for param in model.roberta.parameters():
        param.requires_grad = True

    # -------------------------------------------------------------

    tokenizer = RobertaTokenizer.from_pretrained("./checkpoints/roberta-large")

    # 移动到设备
    model.to(device)
    model.zero_grad()

    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save

    print("Loading Training Data...")
    raw_data_train = get_data_dq("train", retrieval_solver, tokenizer, max_len, feature_file=args.train_feats)

    print("Loading Validation Data...")
    raw_data_val = get_data_dq("val", retrieval_solver, tokenizer, max_len, feature_file=args.val_feats)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

    total_steps = len(raw_data_train[-1]) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # training_dq
    training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device,
                save_model)


if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])