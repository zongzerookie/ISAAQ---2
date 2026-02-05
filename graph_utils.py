import torch
import numpy as np
import re

LOGIC_KEYWORDS = {
    'cause': ['because', 'since', 'therefore', 'leads to', 'result', 'cause', 'due to', 'implies', 'thus', 'so'],
    'contrast': ['however', 'but', 'although', 'contrast', 'while', 'unlike', 'instead', 'yet', 'nevertheless'],
    'condition': ['if', 'unless', 'assume', 'suppose', 'provided', 'given'],
    'parallel': ['and', 'also', 'similarly', 'furthermore', 'moreover', 'additionally']
}


class GraphBuilder:
    def __init__(self, max_nodes=32, max_edges=128):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.edge_type_map = {'self': 0, 'next': 1, 'match': 2, 'cause': 3, 'contrast': 4, 'condition': 5,
                              'parallel': 6}
        self.num_relations = len(self.edge_type_map)

    def split_into_edus_with_spans(self, text):
        """
        基于标点切分，并返回 (text, start_char, end_char)
        """
        if not text:
            return []

        # 使用 finditer 查找句子，避免 split 丢失位置信息
        # 匹配非空句子，以此结尾: . ? ! ; 或者 字符串末尾
        # 这是一个简化的正则，处理 "Fig. 1" 这种缩写可能会有问题，但比 split 好
        pattern = re.compile(r'[^.?!;]+[.?!;]?')

        edus = []
        for match in pattern.finditer(text):
            span_text = match.group().strip()
            if not span_text: continue
            # 记录字符级 span (Start, End)
            edus.append({
                'text': span_text,
                'start': match.start(),
                'end': match.end()
            })
        return edus

    def check_logic_keyword(self, text):
        text_lower = text.lower()
        for logic_type, keywords in LOGIC_KEYWORDS.items():
            for kw in keywords:
                # 简单的单词边界匹配
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    return logic_type, 1.0
        return None, 0.0

    def build_graph(self, context_text, question_text, option_text=None):
        """
        构建图并返回带 Span 信息的 EDU 列表
        """
        # 1. 提取带位置的 EDU
        # 注意：这里需要分别提取，因为它们是不同的字符串输入
        c_edus = self.split_into_edus_with_spans(context_text)
        q_edus = self.split_into_edus_with_spans(question_text)
        o_edus = self.split_into_edus_with_spans(option_text) if option_text else []

        all_edu_objects = c_edus + q_edus + o_edus
        all_text_list = [e['text'] for e in all_edu_objects]

        num_c = len(c_edus)
        num_q = len(q_edus)
        num_o = len(o_edus)
        total_nodes = len(all_edu_objects)

        # 节点截断逻辑 (优先保留 Q 和 O)
        if total_nodes > self.max_nodes:
            keep_c = max(0, self.max_nodes - num_q - num_o)
            # 截断 Context 后部
            c_edus = c_edus[:keep_c]
            all_edu_objects = c_edus + q_edus + o_edus
            all_text_list = [e['text'] for e in all_edu_objects]
            num_c = len(c_edus)
            total_nodes = len(all_edu_objects)

        edge_sources = []
        edge_targets = []
        edge_types = []
        pivot_features = np.zeros((self.max_nodes, 1), dtype=np.float32)

        # 构建边 (逻辑同前，保持不变)
        # A. Context 内部
        for i in range(num_c):
            l_type, p_val = self.check_logic_keyword(all_text_list[i])
            pivot_features[i] = p_val
            edge_sources.append(i);
            edge_targets.append(i);
            edge_types.append(self.edge_type_map['self'])
            if i < num_c - 1:
                edge_sources.append(i);
                edge_targets.append(i + 1);
                edge_types.append(self.edge_type_map['next'])
                edge_sources.append(i + 1);
                edge_targets.append(i);
                edge_types.append(self.edge_type_map['next'])
            if l_type and i > 0:
                e_type_id = self.edge_type_map.get(l_type, 0)
                edge_sources.append(i - 1);
                edge_targets.append(i);
                edge_types.append(e_type_id)
                edge_sources.append(i);
                edge_targets.append(i - 1);
                edge_types.append(e_type_id)

        # B. Q-C Match
        q_start_idx = num_c
        for i in range(num_c):
            for j in range(num_q):
                q_idx = q_start_idx + j
                c_words = set(re.findall(r'\w+', all_text_list[i].lower()))
                q_words = set(re.findall(r'\w+', all_text_list[q_idx].lower()))
                if len(c_words.intersection(q_words)) > 0:
                    edge_sources.append(i);
                    edge_targets.append(q_idx);
                    edge_types.append(self.edge_type_map['match'])
                    edge_sources.append(q_idx);
                    edge_targets.append(i);
                    edge_types.append(self.edge_type_map['match'])

        # C. C-O Connect
        if num_o > 0:
            o_start_idx = num_c + num_q
            for i in range(num_c):
                # 仅连接最后一句 Context
                if i == num_c - 1:
                    for k in range(num_o):
                        o_idx = o_start_idx + k
                        edge_sources.append(i);
                        edge_targets.append(o_idx);
                        edge_types.append(self.edge_type_map['next'])
                        edge_sources.append(o_idx);
                        edge_targets.append(i);
                        edge_types.append(self.edge_type_map['next'])

        # Padding
        num_actual_edges = len(edge_sources)
        if num_actual_edges > self.max_edges:
            edge_sources = edge_sources[:self.max_edges]
            edge_targets = edge_targets[:self.max_edges]
            edge_types = edge_types[:self.max_edges]
        else:
            pad_len = self.max_edges - num_actual_edges
            edge_sources += [0] * pad_len
            edge_targets += [0] * pad_len
            edge_types += [0] * pad_len

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        node_mask = torch.zeros(self.max_nodes, dtype=torch.long)
        node_mask[:total_nodes] = 1

        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_mask': node_mask,
            'pivot_features': torch.tensor(pivot_features),
            'edu_objects': all_edu_objects,  # 包含 start/end/text 的对象列表
            'split_indices': [num_c, num_c + num_q]
        }