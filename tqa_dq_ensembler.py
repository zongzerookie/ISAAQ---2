from transformers import RobertaTokenizer, RobertaForMultipleChoice
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse
import os
import gc  # 引入垃圾回收

# 导入辅助函数和新模型类
from aux_methods import get_data_ndq, process_data_ndq, get_data_dq, validation_ndq, validation_dq
from aux_methods import get_upper_bound, superensembler, ensembler
from aux_methods import SpatiallyAwareISAAQ

# 尝试导入 RobertaAdaLoGN (虽然这次全是 DMC 用不到，但保留防止报错)
try:
    from models.new_model import RobertaAdaLoGN

    print("Success: RobertaAdaLoGN imported.")
except ImportError:
    print("Warning: Could not import RobertaAdaLoGN. (Safe to ignore for All-DMC ensemble)")


def main(argv):
    parser = argparse.ArgumentParser(description='Ensembler for TQA Diagram Questions')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')

    # [核心修改] 6 个模型全部替换为 DMC 模型
    # 前 3 个：你刚刚训练的 Unfrozen (解冻版)
    # 后 3 个：之前训练的 Frozen (冻结版)
    parser.add_argument('-p', '--pretrainingslist', default=[
        "checkpoints/dmc_dq_roberta_SPATIAL_UNFROZEN_IR_e4.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_UNFROZEN_NSP_e3.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_UNFROZEN_NN_e3.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_IR_e3.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_NSP_e4.pth",
        "checkpoints/dmc_dq_roberta_SPATIAL_NN_e1.pth"
    ], help='List of paths to pre-trained models (All 6 are DMC)')

    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default=32, type=int, help='size of the batches. Default: 32')

    parser.add_argument('--val_feats', default='features_cache_nodes/adalogn_nodes_val.pt',
                        help='Path to AdaLoGN features for Validation set')
    parser.add_argument('--test_feats', default='features_cache_nodes/adalogn_nodes_test.pt',
                        help='Path to AdaLoGN features for Test set')

    args = parser.parse_args()
    print(args)

    # [核心修改] 全部改为 "dmc"
    model_types = ["dmc", "dmc", "dmc", "dmc", "dmc", "dmc"]

    # 对应的 Solver 顺序 (与上面的权重文件顺序一一对应)
    retrieval_solvers = ["IR", "NSP", "NN", "IR", "NSP", "NN"]

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-large')

    max_len = args.maxlen
    batch_size = args.batchsize

    # 结果容器
    feats_train = []
    feats_test = []
    labels_train = []
    labels_test = []

    print("\n=== Running Sequential Inference (Load -> Infer -> Delete) ===")
    print("Strategy: All-DMC Ensemble (Unfrozen vs Frozen)")

    # 循环处理每个模型
    for i, (path, m_type, solver) in enumerate(zip(args.pretrainingslist, model_types, retrieval_solvers)):
        print(f"\n{'=' * 40}")
        print(f"Processing Model {i + 1}/6: Type={m_type}, Solver={solver}")
        print(f"Checkpoint: {path}")
        print(f"{'=' * 40}")

        # ---------------- 1. 加载单个模型 ----------------
        model = None

        # 因为全是 DMC，理论上只会进这个分支
        if m_type == "dmc":
            model = SpatiallyAwareISAAQ()
            try:
                if os.path.exists(path):
                    state_dict = torch.load(path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    print("  -> DMC Weights loaded successfully.")
                else:
                    print(f"  -> Error: Checkpoint not found at {path}")
                    # 这里可以选择 sys.exit(1) 或者跳过
                    sys.exit(1)
            except Exception as e:
                print(f"Error loading DMC state_dict: {e}")
                sys.exit(1)

        elif m_type == "tmc":
            # 保留此逻辑作为备份，但这次不会用到
            try:
                model = RobertaAdaLoGN.from_pretrained("./checkpoints/roberta-large", num_labels=1)
                if os.path.exists(path):
                    state_dict = torch.load(path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    print(f"  -> TMC Weights loaded.")
                else:
                    print(f"  -> Error: Checkpoint not found at {path}")
                    sys.exit(1)
            except Exception as e:
                print(f"Error loading TMC model: {e}")
                sys.exit(1)

        model.to(device)
        model.eval()

        # ---------------- 2. 执行推理 ----------------
        if m_type == "dmc":
            # Validation Set
            print(f"  - Inference on VAL set (DMC)...")
            raw_data_val = get_data_dq("val", solver, tokenizer, max_len, feature_file=args.val_feats)
            with torch.no_grad():
                val_logits = validation_dq(model, raw_data_val, batch_size, device)
            feats_train.append(val_logits)

            # 保存 Label (只存一次)
            if len(labels_train) == 0:
                labels_train = raw_data_val[-1]

            # Test Set
            print(f"  - Inference on TEST set (DMC)...")
            raw_data_test = get_data_dq("test", solver, tokenizer, max_len, feature_file=args.test_feats)
            with torch.no_grad():
                test_logits = validation_dq(model, raw_data_test, batch_size, device)
            feats_test.append(test_logits)

            if len(labels_test) == 0:
                labels_test = raw_data_test[-1]

        elif m_type == "tmc":
            # TMC 推理 (本次不执行)
            print(f"  - Inference on VAL set (TMC)...")
            raw_data_val = get_data_ndq("dq", "val", solver, tokenizer, max_len)
            val_loader = process_data_ndq(raw_data_val, batch_size, "val")
            with torch.no_grad():
                val_logits = validation_ndq(model, val_loader, device)
            feats_train.append(val_logits)
            if len(labels_train) == 0:
                labels_train = raw_data_val[2]

            print(f"  - Inference on TEST set (TMC)...")
            raw_data_test = get_data_ndq("dq", "test", solver, tokenizer, max_len)
            test_loader = process_data_ndq(raw_data_test, batch_size, "test")
            with torch.no_grad():
                test_logits = validation_ndq(model, test_loader, device)
            feats_test.append(test_logits)
            if len(labels_test) == 0:
                labels_test = raw_data_test[2]

        # ---------------- 3. 删除模型并清理显存 ----------------
        print(f"  -> Cleaning up GPU memory for Model {i + 1}...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  -> Memory cleared.\n")

    print("\n=== Calculating Ensemble Results ===")

    print("Calculating Test Set Results (Trained on Val)...")
    res = superensembler(feats_train, feats_test, labels_train, labels_test)
    print("\nFINAL RESULTS:")
    print("TEST SET ACCURACY: ", res)

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