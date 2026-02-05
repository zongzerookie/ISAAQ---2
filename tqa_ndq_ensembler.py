from transformers import RobertaTokenizerFast
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse

# 引入必要的数据处理函数
from aux_methods import get_data_ndq, process_data_ndq, validation_ndq, get_upper_bound, ensembler
# 引入模型类以进行实例化
from models.new_model import RobertaAdaLoGN


def main(argv):
    parser = argparse.ArgumentParser(description='Ensemble logic for TQA NDQ')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')

    # === 修改点 1: 更新最佳权重的默认路径 (基于你的训练日志) ===
    # IR: Epoch 4 (0.6876)
    # NSP: Epoch 2 (0.6882)
    # NN: Epoch 1 (0.6399)
    parser.add_argument('-p', '--pretrainingslist',
                        default=[
                            "checkpoints/tmc_ndq_adalog_roberta_IR_e4.pth",
                            "checkpoints/tmc_ndq_adalog_roberta_NSP_e2.pth",
                            "checkpoints/tmc_ndq_adalog_roberta_NN_e2.pth"
                        ],
                        help='list of paths of the pretrainings model.')

    parser.add_argument('-x', '--maxlen', default=128, type=int,
                        help='max sequence length. Default: 128 (match training)')
    parser.add_argument('-b', '--batchsize', default=32, type=int, help='size of the batches. Default: 32')
    args = parser.parse_args()
    print(args)

    # 这里的列表顺序必须与下面的 retrieval_solvers 对应
    retrieval_solvers = ["IR", "NSP", "NN"]

    # === 修改点 2: 使用 Fast Tokenizer ===
    tokenizer = RobertaTokenizerFast.from_pretrained("./checkpoints/roberta-large")

    max_len = args.maxlen
    batch_size = args.batchsize
    dataset_name = "ndq"

    feats_train = []
    feats_test = []

    # 设置设备
    if args.device == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # === 修改点 3: 正确加载 AdaLoGN 模型权重 ===
    for model_path, retrieval_solver in zip(args.pretrainingslist, retrieval_solvers):
        print(f"\nLoading model for solver: {retrieval_solver} from {model_path}...")

        # 1. 实例化模型结构 (必须与训练时一致)
        # NDQ MC 任务 num_labels=1
        model = RobertaAdaLoGN.from_pretrained("./checkpoints/roberta-large", num_labels=1)

        # 2. 加载参数字典
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights for {retrieval_solver}: {e}")
            sys.exit(1)

        model.to(device)
        model.eval()

        # 获取验证集特征 (用于训练集成器)
        print(f"Extracting features for {retrieval_solver} - VAL set")
        raw_data_train = get_data_ndq(dataset_name, "val", retrieval_solver, tokenizer, max_len)
        train_dataloader = process_data_ndq(raw_data_train, batch_size, "val")
        feats_train.append(validation_ndq(model, train_dataloader, device))
        labels_train = raw_data_train[2]  # labels list index

        # 获取测试集特征 (用于评估集成效果)
        print(f"Extracting features for {retrieval_solver} - TEST set")
        raw_data_test = get_data_ndq(dataset_name, "test", retrieval_solver, tokenizer, max_len)
        test_dataloader = process_data_ndq(raw_data_test, batch_size, "test")
        feats_test.append(validation_ndq(model, test_dataloader, device))
        labels_test = raw_data_test[2]  # labels list index

        # 释放显存
        del model
        torch.cuda.empty_cache()

    print("\nCalculating Upper Bound...")
    upper_bound_train = get_upper_bound(feats_train, labels_train)
    print(f"Upper Bound on Validation Set: {upper_bound_train:.4f}")

    print("\nRunning Ensemble on TEST SET...")
    # 在验证集上训练集成逻辑回归，在测试集上评估
    res_test = ensembler(feats_train, feats_test, labels_train, labels_test)
    print(f"FINAL TEST ENSEMBLE ACCURACY: {res_test:.4f}")

    print("\nRunning Ensemble on VALIDATION SET (Sanity Check)...")
    # 仅仅为了检查过拟合情况，在验证集上训练并在验证集上测试
    res_val = ensembler(feats_train, feats_train, labels_train, labels_train)
    print(f"VALIDATION ENSEMBLE ACCURACY: {res_val:.4f}")


if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])