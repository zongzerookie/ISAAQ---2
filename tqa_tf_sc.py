from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import sys
import argparse
import numpy as np
import random
import os

# 引入数据处理和训练函数
from aux_methods import get_data_tf, process_data_ndq, training_tf
# 引入 AdaLoGN 模型
from models.new_model import RobertaAdaLoGN
from transformers import RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup

# === 增强版权重加载函数 ===
def load_model_weights(model, weight_path):
    """
    鲁棒的权重加载函数：
    1. 支持加载 state_dict
    2. 支持加载旧版 transformers 的全模型对象 (通过 sys.modules hack)
    3. 自动忽略不匹配的键值 (strict=False)
    """
    print(f"Loading weights from {weight_path} ...")
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return False

    try:
        # 尝试直接加载
        checkpoint = torch.load(weight_path, map_location='cpu')

        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded successfully. Missing keys (GNN layers initialized randomly): {len(missing_keys)}")
        return True

    except Exception as e:
        print(f"Standard load failed, trying compatibility mode... Error: {e}")
        try:
            # 兼容性 Hack：解决旧版 checkpoint 依赖 transformers.modeling_roberta 的问题
            import sys, transformers
            if not hasattr(transformers, 'modeling_roberta'):
                # 将新版路径映射到旧版路径
                sys.modules['transformers.modeling_roberta'] = transformers.models.roberta.modeling_roberta
                sys.modules['transformers.configuration_roberta'] = transformers.models.roberta.configuration_roberta

            checkpoint = torch.load(weight_path, map_location='cpu')
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded using compatibility hack. Missing keys: {len(missing)}")
            return True
        except Exception as e2:
            print(f"Fatal error loading weights: {e2}")
            return False


def main(argv):
    parser = argparse.ArgumentParser(description='TQA True/False Task with AdaLoGN')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'], help='retrieval solver', required=True)

    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/pretrainings_e4.pth", help='path to pretrainings.')

    # --- 关键参数优化 ---
    # 建议: batchsize=4, accumulation_steps=4 (等效 batch 16) 或 batchsize=2, acc=8
    parser.add_argument('-b', '--batchsize', default=4, type=int, help='Physical batch size per step.')
    parser.add_argument('-acc', '--accumulation_steps', default=4, type=int, help='Gradient accumulation steps.')
    # ------------------

    parser.add_argument('-x', '--maxlen', default=64, type=int, help='max sequence length. Default: 64')
    parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate. Default: 1e-5')
    parser.add_argument('-e', '--epochs', default=2, type=int, help='number of epochs. Default: 2')
    parser.add_argument('-s', '--save', default=False, help='save model', action='store_true')
    args = parser.parse_args()
    print(args)

    # 1. 初始化 AdaLoGN 模型
    # TF 任务是二分类 (True/False)，所以 num_labels=2
    print("Initializing RobertaAdaLoGN for TF task...")
    model = RobertaAdaLoGN.from_pretrained("./checkpoints/roberta-large", num_labels=2)

    # 2. 加载权重
    if args.pretrainings:
        # 如果是旧文件名，尝试自动重定向到修复后的文件（如果存在）
        if args.pretrainings == "checkpoints/pretrainings_e4.pth" and not os.path.exists(args.pretrainings):
            fixed_path = "checkpoints/pretrainings_e4_state_dict.pth"
            if os.path.exists(fixed_path):
                print(f"Redirecting to fixed checkpoint: {fixed_path}")
                args.pretrainings = fixed_path

        success = load_model_weights(model, args.pretrainings)
        if not success:
            print("Warning: Training from random initialization (or partial pre-training)!")

    tokenizer = RobertaTokenizerFast.from_pretrained("./checkpoints/roberta-large")

    if args.device == "gpu":
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
        model.cpu()

    model.zero_grad()

    # 3. 数据准备
    # aux_methods.get_data_tf 现在会返回包含图结构的数据
    raw_data_train = get_data_tf("train", args.retrieval, tokenizer, args.maxlen)
    raw_data_val = get_data_tf("val", args.retrieval, tokenizer, args.maxlen)

    # aux_methods.process_data_ndq 会处理 TensorDataset 打包
    train_dataloader = process_data_ndq(raw_data_train, args.batchsize, "train")
    val_dataloader = process_data_ndq(raw_data_val, args.batchsize, "val")

    # 4. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    # 总步数 = (总样本数 / batchsize / accumulation_steps) * epochs
    total_steps = (len(train_dataloader) // args.accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 5. 开始训练
    # 传递 accumulation_steps 给训练函数
    training_tf(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        args.epochs,
        args.retrieval,
        device,
        args.save,
        accumulation_steps=args.accumulation_steps
    )


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    main(sys.argv[1:])