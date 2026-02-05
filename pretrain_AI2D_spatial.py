# 加载ai2d的权重预训练ai2d，冻结robert，只训练空间编码器
import os
import random
import numpy as np
import torch
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from aux_methods import SpatiallyAwareISAAQ, get_data_AI2D_spatial, flat_accuracy

# ---------------- 配置日志 ----------------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, val_data, device, batch_size, max_ocr_num):
    model.eval()
    input_ids, att_mask, token_type, images, obj_coords, ocr_coords, ocr_ids, spatial_adj, labels = val_data
    total_points = 0;
    total_errors = 0;
    total_loss = 0;
    num_batches = 0
    indices = list(range(len(labels)))

    with torch.no_grad():
        for i in tqdm(range(0, len(indices), batch_size), desc="Validation"):
            batch_idx = indices[i: i + batch_size]
            if not batch_idx: break

            b_ids = torch.tensor([input_ids[k] for k in batch_idx]).to(device)
            b_mask = torch.tensor([att_mask[k] for k in batch_idx]).to(device)
            b_type = torch.tensor([token_type[k] for k in batch_idx]).to(device)
            b_imgs = [images[k] for k in batch_idx]
            b_obj_c = [obj_coords[k] for k in batch_idx]
            b_ocr_c = [ocr_coords[k][:max_ocr_num] for k in batch_idx]
            b_adj = [spatial_adj[k] for k in batch_idx]
            b_lbls = torch.tensor([labels[k] for k in batch_idx]).to(device)

            batch_ocr = [ocr_ids[k][:max_ocr_num] for k in batch_idx]
            curr_max_nodes = max([t.size(0) for t in batch_ocr if t.size(0) > 0] + [1])
            padded_ocr = torch.zeros(len(batch_idx), curr_max_nodes, 10, dtype=torch.long).to(device)
            for j, t in enumerate(batch_ocr):
                if t.size(0) > 0: padded_ocr[j, :t.size(0), :] = t.to(device)

            outputs = model(
                input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type,
                images=b_imgs, obj_coords_list=b_obj_c, ocr_coords_list=b_ocr_c,
                ocr_input_ids_list=padded_ocr, spatial_adj_matrix_list=b_adj,
                adalogn_nodes=None, labels=b_lbls
            )
            loss, logits = outputs[:2]
            total_loss += loss.item();
            num_batches += 1
            preds = logits.detach().cpu().numpy()
            lbls_np = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls_np)
            total_points += p;
            total_errors += e

    acc = total_points / (total_points + total_errors + 1e-9)
    avg_loss = total_loss / (num_batches + 1e-9)
    logger.info(f"Validation Result - Acc: {acc:.4f} | Loss: {avg_loss:.4f}")
    return acc, avg_loss


def main():
    SEED = 42;
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [参数配置]
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 4
    EPOCHS = 10
    LR = 1e-4  # 只训练 Spatial Encoder，可以用大火候
    MAX_OCR_NUM_LIMIT = 16

    logger.info(f"Using device: {device}")
    tokenizer = RobertaTokenizer.from_pretrained("./checkpoints/roberta-large")
    model = SpatiallyAwareISAAQ()

    # ==============================================================================
    # [核心策略] 加载原作者的 AI2D_e12.pth (专家权重)
    # ==============================================================================
    author_ckpt_path = "./checkpoints/AI2D_e12.pth"

    if os.path.exists(author_ckpt_path):
        logger.info(f"Loading Author's Expert Weights from {author_ckpt_path}...")
        try:
            author_ckpt = torch.load(author_ckpt_path, map_location='cpu')

            # 提取 RoBERTa 权重
            author_sd = author_ckpt.state_dict() if hasattr(author_ckpt, 'state_dict') else author_ckpt
            roberta_weights = {}
            for k, v in author_sd.items():
                if k.startswith('roberta.'):
                    new_key = k.replace('roberta.', '', 1)
                    roberta_weights[new_key] = v

            if roberta_weights:
                model.roberta.load_state_dict(roberta_weights, strict=True)
                logger.info("✅ Loaded Author's RoBERTa weights successfully!")
            else:
                logger.warning("❌ No 'roberta.' keys found in author checkpoint.")
        except Exception as e:
            logger.error(f"⚠️ Error loading author weights: {e}")
            return
    else:
        logger.error(f"❌ Critical: Author checkpoint AI2D_e12.pth not found!")
        return

    model.to(device)

    # [冻结策略] 冻结 RoBERTa 和 ResNet，只训练 Spatial Encoder
    logger.info("Freezing RoBERTa (Expert) and ResNet. Training ONLY Spatial Encoder.")

    # [关键] 冻结 RoBERTa，保护专家权重不被破坏
    for param in model.roberta.parameters(): param.requires_grad = False
    for param in model.resnet.parameters(): param.requires_grad = False

    # 确保 Spatial Encoder 是解冻的
    for name, param in model.spatial_encoder.named_parameters():
        param.requires_grad = True

    # 4. 加载数据
    train_data = get_data_AI2D_spatial("train", tokenizer, max_len=128)
    val_data = get_data_AI2D_spatial("test", tokenizer, max_len=128)

    # 5. 优化器 (只优化 requires_grad=True 的参数)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, eps=1e-8)
    total_steps = (len(train_data[-1]) // (BATCH_SIZE * ACCUMULATION_STEPS)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.1,
                                                num_training_steps=total_steps)

    best_val_acc = 0.0

    # 6. 训练循环
    for epoch in range(EPOCHS):
        logger.info(f"======== Epoch {epoch + 1} / {EPOCHS} ========")
        model.train()
        model.roberta.eval()  # 必须 eval
        model.resnet.eval()

        input_ids, att_mask, token_type, images, obj_coords, ocr_coords, ocr_ids, spatial_adj, labels = train_data
        indices = list(range(len(labels)));
        random.shuffle(indices)

        total_points = 0;
        total_errors = 0;
        current_loss_window = []
        optimizer.zero_grad()
        pbar = tqdm(range(0, len(indices), BATCH_SIZE), desc="Training")

        for step, i in enumerate(pbar):
            batch_idx = indices[i: i + BATCH_SIZE]
            if not batch_idx: break

            # (数据处理不变)
            b_ids = torch.tensor([input_ids[k] for k in batch_idx]).to(device)
            b_mask = torch.tensor([att_mask[k] for k in batch_idx]).to(device)
            b_type = torch.tensor([token_type[k] for k in batch_idx]).to(device)
            b_imgs = [images[k] for k in batch_idx]
            b_obj_c = [obj_coords[k] for k in batch_idx]
            b_ocr_c = [ocr_coords[k][:MAX_OCR_NUM_LIMIT] for k in batch_idx]
            b_adj = [spatial_adj[k] for k in batch_idx]
            b_lbls = torch.tensor([labels[k] for k in batch_idx]).to(device)

            batch_ocr = [ocr_ids[k][:MAX_OCR_NUM_LIMIT] for k in batch_idx]
            curr_max_nodes = max([t.size(0) for t in batch_ocr if t.size(0) > 0] + [1])
            padded_ocr = torch.zeros(len(batch_idx), curr_max_nodes, 10, dtype=torch.long).to(device)
            for j, t in enumerate(batch_ocr):
                if t.size(0) > 0: padded_ocr[j, :t.size(0), :] = t.to(device)

            outputs = model(
                input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type,
                images=b_imgs, obj_coords_list=b_obj_c, ocr_coords_list=b_ocr_c,
                ocr_input_ids_list=padded_ocr, spatial_adj_matrix_list=b_adj,
                adalogn_nodes=None, labels=b_lbls
            )

            loss, logits = outputs[:2]
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            preds = logits.detach().cpu().numpy()
            lbls_np = b_lbls.cpu().numpy()
            p, e = flat_accuracy(preds, lbls_np)
            total_points += p;
            total_errors += e
            current_loss_window.append(loss.item() * ACCUMULATION_STEPS)
            if len(current_loss_window) > 100: current_loss_window.pop(0)

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            avg_acc = total_points / (total_points + total_errors + 1e-9)
            pbar.set_description(f"Train Acc: {avg_acc:.4f} | Loss: {np.mean(current_loss_window):.4f}")

        val_acc, _ = validate(model, val_data, device, BATCH_SIZE, MAX_OCR_NUM_LIMIT)

        # 始终保存 best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/AI2D_spatial_best.pth")
            logger.info(f"🔥 New Best Model Saved! Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()