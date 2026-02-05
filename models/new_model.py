import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertPreTrainedModel
import math




# =============================================================================
# [新增] 1. 结构感知图变换器层 (Structure-Aware Graph Transformer Layer)
# =============================================================================
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, num_relations=7, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # num_relations + 1 处理无边情况
        self.num_relations = num_relations
        self.edge_bias = nn.Embedding(num_relations + 1, num_heads)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, edge_index, edge_type):
        B, N, C = x.shape
        device = x.device

        # --- A. 确定性构建结构偏置矩阵 (Deterministic Bias Construction) ---

        # 1. 准备基础数据
        num_edges = edge_index.shape[2]
        batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, num_edges).reshape(-1)
        src_idx = edge_index[:, 0, :].reshape(-1)
        tgt_idx = edge_index[:, 1, :].reshape(-1)
        rel_types = edge_type.reshape(-1)

        # 2. 过滤有效边
        valid_mask = (src_idx < N) & (tgt_idx < N)
        b_valid = batch_idx[valid_mask]
        s_valid = src_idx[valid_mask]
        t_valid = tgt_idx[valid_mask]
        r_valid = rel_types[valid_mask]

        # 3. 排序与去重
        flat_indices = b_valid * N * N + s_valid * N + t_valid
        sort_key = flat_indices * (self.num_relations + 1) + r_valid

        sorted_order = torch.argsort(sort_key)
        sorted_flat_indices = flat_indices[sorted_order]
        sorted_rels = r_valid[sorted_order]

        if sorted_flat_indices.numel() > 0:
            unique_mask = torch.ones_like(sorted_flat_indices, dtype=torch.bool)
            unique_mask[:-1] = (sorted_flat_indices[1:] != sorted_flat_indices[:-1])

            final_flat_indices = sorted_flat_indices[unique_mask]
            final_rels = sorted_rels[unique_mask]

            final_b = torch.div(final_flat_indices, (N * N), rounding_mode='floor')
            rem = final_flat_indices % (N * N)
            final_s = torch.div(rem, N, rounding_mode='floor')
            final_t = rem % N

            final_biases = self.edge_bias(final_rels)

            attn_bias = torch.zeros(B, self.num_heads, N, N, device=device)
            attn_bias = attn_bias.permute(0, 2, 3, 1)
            attn_bias[final_b, final_s, final_t] = final_biases
            attn_bias = attn_bias.permute(0, 3, 1, 2)
        else:
            attn_bias = torch.zeros(B, self.num_heads, N, N, device=device)

        # --- B. Multi-Head Self-Attention ---
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores + attn_bias

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout1(attn_probs)

        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = residual + self.dropout1(out)

        # --- C. FFN ---
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = residual + self.dropout2(out)

        return out


# =============================================================================
# [原有] GatedFusion, NodeFeatureInitializer 类
# =============================================================================
class GatedFusion(nn.Module):
    def __init__(self, hidden_size, visual_dim=0, dropout_prob=0.1):
        super().__init__()
        self.use_visual = visual_dim > 0
        if self.use_visual:
            self.visual_proj = nn.Linear(visual_dim, hidden_size)

        input_dim = hidden_size * 2
        if self.use_visual:
            input_dim += hidden_size

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_net[-2].bias, -2.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, h_global, h_graph, h_visual=None):
        features = [h_global, h_graph]
        if self.use_visual and h_visual is not None:
            h_visual_proj = self.visual_proj(h_visual)
            features.append(h_visual_proj)

        combined = torch.cat(features, dim=-1)
        g = self.gate_net(combined)

        weighted_sum = (1 - g) * h_global + g * h_graph
        if self.use_visual and h_visual is not None:
            weighted_sum = weighted_sum + 0.1 * h_visual_proj

        return self.layer_norm(weighted_sum)


class NodeFeatureInitializer(nn.Module):
    def __init__(self, hidden_size, max_nodes=32):
        super().__init__()
        self.max_nodes = max_nodes
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output, node_to_token_mask=None):
        if node_to_token_mask is None:
            batch_size = sequence_output.size(0)
            return torch.zeros(batch_size, self.max_nodes, sequence_output.size(2), device=sequence_output.device)

        node_feats = torch.bmm(node_to_token_mask, sequence_output)

        sum_mask = node_to_token_mask.sum(dim=2, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        node_feats = node_feats / sum_mask

        return self.layer_norm(node_feats)



# =============================================================================
# [修改] RobertaAdaLoGN: 加入对比学习模块 (LACL)
# =============================================================================
class RobertaAdaLoGN(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.node_init = NodeFeatureInitializer(config.hidden_size, max_nodes=32)
        self.pivot_proj = nn.Linear(1, config.hidden_size)

        # Graph Transformer Layers
        self.gnn_layers = nn.ModuleList([
            GraphTransformerLayer(config.hidden_size, num_heads=8, num_relations=7),
            GraphTransformerLayer(config.hidden_size, num_heads=8, num_relations=7)
        ])

        # =========== [保留优化] 方案二：GAT Tanh Attention ===========
        # 我们保留这个优化，因为它对小数据集更鲁棒
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_vec = nn.Parameter(torch.empty(1, config.hidden_size))
        nn.init.xavier_normal_(self.attn_vec)
        # =========================================================

        # [删除] 对比学习投影头 (self.cl_head 被移除了)

        self.fusion = GatedFusion(config.hidden_size, visual_dim=0)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.init_weights()

    def _compute_gat_pooling(self, h_nodes, h_global, node_mask):
        """
        保留 GAT Style Pooling 逻辑
        """
        # 1. 线性变换
        h_nodes_trans = self.attn_proj(h_nodes)
        h_global_trans = self.attn_proj(h_global).unsqueeze(1)

        # 2. Tanh 激活
        energy = torch.tanh(h_nodes_trans + h_global_trans)

        # 3. 计算分数
        scores = (energy * self.attn_vec).sum(dim=-1).unsqueeze(1)

        # 4. 掩码
        if node_mask is not None:
            scores = scores.masked_fill(node_mask.unsqueeze(1) == 0, -1e9)

        # 5. Softmax & 聚合
        alphas = F.softmax(scores, dim=-1)
        h_graph = torch.bmm(alphas, h_nodes).squeeze(1)

        return h_graph

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, graph_data=None, visual_feats=None):

        # 1. 维度处理 (保持不变)
        num_choices = 1
        if input_ids.dim() == 3:
            batch_size, num_choices, seq_len = input_ids.shape
            input_ids = input_ids.view(-1, seq_len)
            if attention_mask is not None: attention_mask = attention_mask.view(-1, seq_len)
            if token_type_ids is not None: token_type_ids = token_type_ids.view(-1, seq_len)
            if position_ids is not None: position_ids = position_ids.view(-1, seq_len)

            if graph_data is not None:
                new_graph_data = {}
                for k, v in graph_data.items():
                    if v is not None:
                        new_graph_data[k] = v.view(-1, *v.shape[2:])
                    else:
                        new_graph_data[k] = None
                graph_data = new_graph_data
        else:
            batch_size = input_ids.shape[0]

        # 2. RoBERTa 编码
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        h_global = sequence_output[:, 0, :]  # [CLS]

        # 3. 图推理 (Main Graph Stream)
        h_nodes = None
        node_mask = None
        h_graph = torch.zeros_like(h_global)

        if graph_data is not None:
            edge_index = graph_data['edge_index']
            edge_type = graph_data['edge_type']
            pivot_vals = graph_data['pivot_features']
            nt_mask = graph_data.get('node_to_token_mask', None)

            if 'node_mask' in graph_data:
                node_mask = graph_data['node_mask']

            # 初始化节点
            h_nodes = self.node_init(sequence_output, nt_mask)
            h_pivot = self.pivot_proj(pivot_vals)
            h_nodes = h_nodes + h_pivot

            # Transformer 推理
            for layer in self.gnn_layers:
                h_nodes = layer(h_nodes, edge_index, edge_type)

            # [保留] 使用 GAT Pooling
            h_graph = self._compute_gat_pooling(h_nodes, h_global, node_mask)

        # =================================================================
        # [删除] 逻辑感知对比学习 (LACL) 模块的所有代码
        # 原本这里的几十行代码（正样本构建、负样本构建、InfoNCE Loss）全部被移除
        # =================================================================

        # 4. 融合
        h_final = self.fusion(h_global, h_graph, h_visual=visual_feats)
        logits = self.classifier(h_final)

        # 5. Output Packaging
        if num_choices > 1:
            if self.num_labels == 1:
                reshaped_logits = logits.view(batch_size, num_choices)
            else:
                reshaped_logits = logits.view(batch_size, num_choices, self.num_labels)
        else:
            reshaped_logits = logits

        outputs = (reshaped_logits,) + outputs[2:]

        if labels is not None:
            if num_choices > 1 and self.num_labels == 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)
            elif self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(reshaped_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(reshaped_logits.view(-1, self.num_labels), labels.view(-1))

            # [修改] 直接返回主任务 Loss，不再加 cl_loss
            outputs = (loss,) + outputs

        # 接口数据
        if h_nodes is None:
            h_nodes = torch.zeros(h_global.size(0), 32, self.config.hidden_size, device=h_global.device)
        if node_mask is None:
            node_mask = torch.zeros(h_global.size(0), 32, device=h_global.device)

        vis_interface_dict = {
            "h_final": h_final,
            "h_graph": h_graph,
            "h_nodes": h_nodes,
            "node_mask": node_mask
        }
        outputs = outputs + (vis_interface_dict,)

        return outputs