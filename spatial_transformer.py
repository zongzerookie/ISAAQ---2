# spatial_transformer.py
import math
import torch
from torch import nn
import torch.nn.functional as F


class SpatialBertConfig:
    def __init__(self, hidden_size=1024, num_attention_heads=16, num_hidden_layers=2,
                 intermediate_size=4096, dropout_prob=0.1, context_width=1):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.num_spatial_relations = 12
        # 关系ID定义: 1-12(空间), 13(隐式V-T), 14(全局T-T)
        self.relation_implicit = 13
        self.relation_global = 14
        self.relation_labeling = 15#加了
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.context_width = context_width  # 参数 c


class SpatialBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} must be multiple of heads {config.num_attention_heads}")

        self.num_attention_heads = config.num_attention_heads  # 16
        self.num_spatial_heads = 12  # 前12个头负责空间
        self.config = config

        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _make_spatial_mask(self, spatial_adj_matrix, device):
        """
        根据参数 c 和头索引构建 Attention Mask。
        spatial_adj_matrix: [bs, N, N] (包含 0-14 的关系ID)
        """
        bs, n_nodes, _ = spatial_adj_matrix.shape
        # 初始化为极小值 (Masked)
        mask = torch.full((bs, self.num_attention_heads, n_nodes, n_nodes), -1e9, device=device)

        c = self.config.context_width

        # 遍历所有头
        for h in range(self.num_attention_heads):
            allowed_relations = []
            allowed_relations.append(self.config.relation_labeling)#加了
            # --- A. 空间推理头 (0-11) ---
            if h < self.num_spatial_heads:
                # 1. 必须包含隐式关系 (Visual <-> Text)
                allowed_relations.append(self.config.relation_implicit)

                # 2. 必须包含自环
                allowed_relations.append(12)

                # 3. 空间关系 (受参数 c 控制)
                # 简化的映射：头 h 对应关系 ID h+1
                center_rel = h + 1

                if center_rel <= 3:
                    # 拓扑关系(1-3)：只关注自己
                    allowed_relations.append(center_rel)
                else:
                    # 方向关系 (4-11)：在 4-11 的圆环上取 [center-c, center+c]
                    direction_idx = center_rel - 4  # 0-7
                    for offset in range(-c, c + 1):
                        neighbor_idx = (direction_idx + offset) % 8
                        allowed_relations.append(neighbor_idx + 4)

            # --- B. 全局语义头 (12-15) ---
            else:
                # 关注所有文本相关关系 (Implicit, Global) 和 自环
                allowed_relations.append(self.config.relation_implicit)  # 13
                allowed_relations.append(self.config.relation_global)  # 14
                allowed_relations.append(12)  # 12 (Self)

            # 构建当前头的 Mask
            for rel in allowed_relations:
                # 将 adj_matrix 中等于 rel 的位置设为 0 (Unmasked)
                mask[:, h, :, :][spatial_adj_matrix == rel] = 0.0

        return mask

    def forward(self, hidden_states, spatial_adj_matrix):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # [bs, heads, N, N]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取动态掩码
        spatial_mask = self._make_spatial_mask(spatial_adj_matrix, hidden_states.device)

        # 应用掩码
        attention_scores = attention_scores + spatial_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SpatialBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SpatialBertSelfAttention(config)
        self.attention_output_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, spatial_adj_matrix):
        attention_output = self.attention(hidden_states, spatial_adj_matrix)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)

        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(attention_output + layer_output)

        return layer_output


class BertSpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([SpatialBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, spatial_adj_matrix):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, spatial_adj_matrix)
        return hidden_states