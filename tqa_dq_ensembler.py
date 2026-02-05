from transformers import RobertaTokenizer, RobertaForMultipleChoice
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse
import os

# 导入辅助函数和新模型类
from aux_methods import get_data_ndq, process_data_ndq, get_data_dq, validation_ndq, validation_dq
from aux_methods import get_upper_bound, superensembler, ensembler
from aux_methods import SpatiallyAwareISAAQ


def main(argv):
    parser = argparse.ArgumentParser(description='Ensembler for TQA Diagram Questions')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')

    # 默认权重路径更新为 Epoch 1 (根据你的训练日志，Epoch 1 效果最好)
    parser.add_argument('-p', '--pretrainingslist', default=[
        "checkpoints/tmc_dq_roberta_IR_e4.pth",
        "checkpoints/tmc_dq_roberta_NSP_e4.pth",
        "checkpoints/tmc_dq_roberta_NN_e2.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_IR_e3.pth",  # Updated based on logs
        "checkpoints/dmc_dq_roberta_SPATIAL_NSP_e3.pth",  # Updated based on logs
        "checkpoints/dmc_dq_roberta_SPATIAL_NN_e1.pth"  # Updated based on logs
    ], help='List of paths to pre-trained models (3 TMC, 3 DMC)')

    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default=32, type=int, help='size of the batches. Default: 32')

    # 新增特征文件路径参数
    parser.add_argument('--val_feats', default='features_cache_nodes/adalogn_nodes_val.pt',
                        help='Path to AdaLoGN features for Validation set')
    parser.add_argument('--test_feats', default='features_cache_nodes/adalogn_nodes_test.pt',
                        help='Path to AdaLoGN features for Test set')

    args = parser.parse_args()
    print(args)

    # 模型类型定义 (前3个是纯文本TMC，后3个是图表DMC)
    model_types = ["tmc", "tmc", "tmc", "dmc", "dmc", "dmc"]
    retrieval_solvers = ["IR", "NSP", "NN", "IR", "NSP", "NN"]

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-large')

    loaded_models = []

    print("\n=== Loading Models ===")
    for path, m_type in zip(args.pretrainingslist, model_types):
        print(f"Loading {m_type.upper()} model from {path}...")

        if m_type == "tmc":
            # TMC 模型通常是 RobertaForMultipleChoice
            # 兼容逻辑：尝试直接加载对象，如果失败则加载 state_dict
            try:
                loaded = torch.load(path, map_location='cpu')
                if isinstance(loaded, dict):  # 是 state_dict
                    model = RobertaForMultipleChoice.from_pretrained("./checkpoints/roberta-large")
                    model.load_state_dict(loaded)
                else:  # 是完整对象 (Legacy)
                    model = loaded
            except Exception as e:
                print(f"Error loading TMC model: {e}")
                # Fallback init
                model = RobertaForMultipleChoice.from_pretrained("./checkpoints/roberta-large")

        elif m_type == "dmc":
            # DMC 模型是我们新的 SpatiallyAwareISAAQ
            model = SpatiallyAwareISAAQ()
            try:
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading DMC state_dict: {e}")
                print("Make sure the model architecture matches the checkpoint.")
                sys.exit(1)

        model.to(device)
        model.eval()
        loaded_models.append(model)

    max_len = args.maxlen
    batch_size = args.batchsize

    feats_train = []  # 这里指 Validation 集的结果 (用于训练 Ensembler)
    feats_test = []  # 这里指 Test 集的结果
    labels_train = []
    labels_test = []

    print("\n=== Running Inference for Ensemble ===")

    # 遍历每个模型进行推理
    for i, (model, m_type, solver) in enumerate(zip(loaded_models, model_types, retrieval_solvers)):
        print(f"\nProcessing Model {i + 1}/6: Type={m_type}, Solver={solver}")

        if m_type == "dmc":
            # --- DMC (Diagram MC) 推理 ---

            # Validation Set (作为 Ensembler 的 Train 数据)
            print(f"  - Inference on VAL set (DMC)...")
            # 关键修改：传入 feature_file
            raw_data_val = get_data_dq("val", solver, tokenizer, max_len, feature_file=args.val_feats)
            # 此时 raw_data_val 包含了 adalogn_nodes
            val_logits = validation_dq(model, raw_data_val, batch_size, device)
            feats_train.append(val_logits)
            if i == 3:  # 只在第一次处理 DMC 时保存 labels，避免重复
                labels_train = raw_data_val[-1]  # labels_list is the last element

            # Test Set
            print(f"  - Inference on TEST set (DMC)...")
            raw_data_test = get_data_dq("test", solver, tokenizer, max_len, feature_file=args.test_feats)
            test_logits = validation_dq(model, raw_data_test, batch_size, device)
            feats_test.append(test_logits)
            if i == 3:
                labels_test = raw_data_test[-1]

        elif m_type == "tmc":
            # --- TMC (Text MC) 推理 ---

            # Validation Set
            print(f"  - Inference on VAL set (TMC)...")
            raw_data_val = get_data_ndq("dq", "val", solver, tokenizer, max_len)
            val_loader = process_data_ndq(raw_data_val, batch_size, "val")
            val_logits = validation_ndq(model, val_loader, device)
            feats_train.append(val_logits)
            if i == 0:
                labels_train = raw_data_val[2]  # labels_list is index 2 in get_data_ndq return

            # Test Set
            print(f"  - Inference on TEST set (TMC)...")
            raw_data_test = get_data_ndq("dq", "test", solver, tokenizer, max_len)
            test_loader = process_data_ndq(raw_data_test, batch_size, "test")
            test_logits = validation_ndq(model, test_loader, device)
            feats_test.append(test_logits)
            if i == 0:
                labels_test = raw_data_test[2]

    # 确保标签一致性 (简单的 sanity check)
    # 注意：TMC 和 DMC 加载的数据顺序必须一致，TQA 数据集通常是固定的，所以顺序应该没问题

    print("\n=== Calculating Ensemble Results ===")

    # 训练 Ensembler (Logistic Regression) 并评估
    # 注意：这里的 feats_train 对应的是 Validation 集的 Logits
    # feats_test 对应的是 Test 集的 Logits
    # 我们用 Val 集训练 ensemble 权重，在 Test 集上测试

    print("Calculating Test Set Results (Trained on Val)...")
    res = superensembler(feats_train, feats_test, labels_train, labels_test)
    print("\nFINAL RESULTS:")
    print("TEST SET ACCURACY: ", res)

    # 也可以看下在 Val 集上的自测效果 (Upper bound check)
    print("\nCalculating Validation Set Results (Self-check)...")
    res_val = superensembler(feats_train, feats_train, labels_train, labels_train)
    print("VALIDATION SET ACCURACY: ", res_val)


if __name__ == "__main__":
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    main(sys.argv[1:])