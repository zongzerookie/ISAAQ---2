import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert_ai2d(raw_dir, output_dir):
    """
    将 AI2D 原始数据集 (questions/, annotations/) 转换为 ISAAQ 训练所需的 json 格式。
    (修正版：适配 questions key-value 结构，适配 answerTexts 列表转字典，适配 OCR 坐标格式)
    """
    questions_dir = os.path.join(raw_dir, "questions")
    annotations_dir = os.path.join(raw_dir, "annotations")

    if not os.path.exists(questions_dir) or not os.path.exists(annotations_dir):
        print(f"错误: 在 {raw_dir} 下找不到 questions 或 annotations 文件夹。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_data = []

    # 获取所有问题文件
    q_files = [f for f in os.listdir(questions_dir) if f.endswith(".json")]

    print(f"正在转换 {len(q_files)} 个样本...")

    for q_file in tqdm(q_files):
        try:
            # 这里的 imageName 通常就是文件名去掉 .json，例如 "0.png.json" -> "0.png"
            # 但 AI2D 有时文件名和 imageName 字段不一致，优先使用文件名推导 ID
            image_id = q_file.replace(".json", "")

            # 1. 读取问题数据
            with open(os.path.join(questions_dir, q_file), 'r', encoding='utf-8') as f:
                q_data = json.load(f)

            # 2. 读取标注数据 (布局、文本、多边形)
            anno_file = os.path.join(annotations_dir, q_file)
            if not os.path.exists(anno_file):
                continue
            with open(anno_file, 'r', encoding='utf-8') as f:
                a_data = json.load(f)

            # 3. 提取 OCR 文本和坐标
            ocr_texts = []
            ocr_coords = []

            if 'text' in a_data:
                for t_id, t_info in a_data['text'].items():
                    content = t_info.get('value', '')
                    # [新增] 获取 replacementText (例如 "A", "B")
                    replacement = t_info.get('replacementText', '')

                    rect = t_info.get('rectangle', [])

                    if content and len(rect) == 2:
                        x1, y1 = rect[0]
                        x2, y2 = rect[1]

                        # [核心修改] 将标签和内容拼接，例如 "A FACE"
                        # 这样模型既能匹配问题中的 "A"，也能匹配选项中的 "Face"
                        final_text = content
                        if replacement:
                            final_text = f"{replacement} {content}"

                        ocr_texts.append(final_text)
                        ocr_coords.append([x1, y1, x2, y2])

            # 4. 提取物体坐标 (Blobs)
            obj_coords = []
            if 'blobs' in a_data:
                for b_id, b_info in a_data['blobs'].items():
                    polygon = b_info.get('polygon', [])
                    if polygon:
                        # polygon 格式 [[x,y], [x,y]...] -> 转为 Box
                        poly_np = np.array(polygon)
                        x1 = int(np.min(poly_np[:, 0]))
                        y1 = int(np.min(poly_np[:, 1]))
                        x2 = int(np.max(poly_np[:, 0]))
                        y2 = int(np.max(poly_np[:, 1]))
                        obj_coords.append([x1, y1, x2, y2])

            # 5. 处理每一个问题
            if 'questions' in q_data:
                # [修正] q_data['questions'] 的 key 是问题文本，value 是信息
                for question_text, q_info in q_data['questions'].items():
                    # [修正] 将 answerTexts (List) 转换为 ISAAQ 需要的 Dict {"0": "A", "1": "B"}
                    answer_list = q_info.get('answerTexts', [])
                    answers_dict = {str(i): ans for i, ans in enumerate(answer_list)}

                    # 构建样本 Entry
                    entry = {
                        "question": question_text,  # [修正] 使用 Key 作为问题文本
                        "image_path": f"images/ai2d/{image_id}",
                        "image_id": image_id,
                        "answers": answers_dict,  # [修正] 使用转换后的字典
                        "correct_answer": str(q_info['correctAnswer']),
                        "ocr_texts": ocr_texts,
                        "ocr_coords": ocr_coords,
                        "obj_coords": obj_coords,
                        "split": "train"
                    }
                    processed_data.append(entry)

        except Exception as e:
            # 打印具体错误以便调试，但不要打断循环
            # print(f"Skipping {q_file}: {e}")
            pass

    if len(processed_data) == 0:
        print("错误: 没有成功转换任何数据！请检查路径结构或 JSON 格式。")
        return

    # 6. 划分训练集和验证集
    print(f"成功提取 {len(processed_data)} 个问题样本，正在保存...")

    # AI2D 比较特殊，通常按 Image ID 划分，但这里为了简单按问题划分 (Random Split)
    # 如果想更严谨，应该按 Image ID 划分，防止同一张图的问题泄露到验证集
    # 这里我们采用简单划分，对于预训练来说影响不大
    train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)

    for d in val_data: d['split'] = 'test'

    with open(os.path.join(output_dir, "ai2d_train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, "ai2d_test.json"), 'w', encoding='utf-8') as f:
        json.dump(val_data, f)

    print(f"转换完成！")
    print(f"训练集: {len(train_data)} -> jsons/ai2d_train.json")
    print(f"验证集: {len(val_data)} -> jsons/ai2d_test.json")


if __name__ == "__main__":
    # 配置路径
    raw_dataset_dir = "ai2d_raw"  # 确保这里面有 questions 和 annotations 文件夹
    output_json_dir = "jsons"

    convert_ai2d(raw_dataset_dir, output_json_dir)