import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer

# 引入你的 AdaLoGN 模型和数据处理工具
from models.new_model import RobertaAdaLoGN
from aux_methods import get_data_ndq, process_data_ndq


def extract_and_save(model, dataloader, device, output_path):
    print(f"Start extracting features to {output_path}...")
    model.eval()

    # 容器：存储节点特征和掩码
    # h_final 也可以存，作为备用或对比
    all_h_nodes = []
    all_node_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)

            # 解包数据 (支持 Hard Alignment Mask)
            if len(batch) >= 8:
                b_ids, b_mask, b_lbls, b_e_idx, b_e_type, b_n_mask, b_p_feats, b_nt_mask = batch

                outputs = model(
                    input_ids=b_ids,
                    attention_mask=b_mask,
                    labels=None,
                    graph_data={
                        'edge_index': b_e_idx,
                        'edge_type': b_e_type,
                        'node_mask': b_n_mask,
                        'pivot_features': b_p_feats,
                        'node_to_token_mask': b_nt_mask
                    }
                )

                # 获取字典
                vis_dict = outputs[-1]

                h_nodes = vis_dict['h_nodes']  # [Batch * 4, 32, 1024]
                node_mask = vis_dict['node_mask']  # [Batch * 4, 32]

                # 转回 CPU (显存极其宝贵，必须立即转)
                all_h_nodes.append(h_nodes.cpu())
                all_node_masks.append(node_mask.cpu())

            else:
                print("Error: Batch data incomplete, skipping...")
                continue

    # 1. 拼接
    tensor_nodes = torch.cat(all_h_nodes, dim=0)
    tensor_mask = torch.cat(all_node_masks, dim=0)

    # 2. 重塑为 [Total_Questions, 4, ...]
    num_options = 4
    total_samples = tensor_nodes.size(0)
    total_questions = total_samples // num_options
    hidden_size = tensor_nodes.size(2)
    max_nodes = tensor_nodes.size(1)

    # [N, 4, 32, 1024]
    tensor_nodes = tensor_nodes.view(total_questions, num_options, max_nodes, hidden_size)
    # [N, 4, 32]
    tensor_mask = tensor_mask.view(total_questions, num_options, max_nodes)

    print(f"Extraction Done.")
    print(f"Nodes Shape: {tensor_nodes.shape}")

    # 3. 保存
    # 注意：这个文件可能会比较大 (约 13GB+)，请确保磁盘空间充足
    torch.save({
        'h_nodes': tensor_nodes,
        'node_mask': tensor_mask
    }, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'])
    # 必须指定训练好的 AdaLoGN 权重路径
    parser.add_argument('-p', '--weights', required=True, help='Path to trained AdaLoGN weights')
    parser.add_argument('-r', '--retrieval', default='IR', choices=['IR', 'NSP', 'NN'])
    # 建议 batchsize 设小一点，防止 OOM (h_nodes 占用显存较大)
    parser.add_argument('-b', '--batchsize', default=8, type=int)
    parser.add_argument('-x', '--maxlen', default=128, type=int)
    parser.add_argument('--output_dir', default='./features_cache_nodes', help='Directory to save extracted features')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    # 注意使用 2.11 兼容的 Tokenizer 初始化
    tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-large')

    print("Loading AdaLoGN Model...")
    model = RobertaAdaLoGN.from_pretrained("./checkpoints/roberta-large", num_labels=1)

    print(f"Loading weights from {args.weights}...")
    try:
        state_dict = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device)

    # 提取 train, val, test
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\nProcessing {split} split...")

        # 加载数据
        raw_data = get_data_ndq("dq", split, args.retrieval, tokenizer, args.maxlen)

        # 强制 SequentialSampler (split="val")
        dataloader = process_data_ndq(raw_data, args.batchsize, split="val")

        output_file = os.path.join(args.output_dir, f"adalogn_nodes_{split}.pt")
        extract_and_save(model, dataloader, device, output_file)


if __name__ == "__main__":
    main()