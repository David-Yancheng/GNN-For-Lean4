import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import time
import copy

# ==========================================
# 1. EMA (指数模型平均) 类定义
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 初始化时，拷贝一份当前模型的权重作为初始影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        # 每次 optimizer.step() 后调用，平滑更新影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        # 验证/测试前调用：将模型的真实权重备份，并替换为影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        # 验证/测试后调用：恢复模型的真实权重，继续训练
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==========================================
# 2. 模型与 Loss 定义
# ==========================================
class LeanRGCN(nn.Module):
    # 新增 in_channels 参数，匹配 ByT5-small 的 1472 维
    def __init__(self, in_channels=1472, hidden_channels=1024, dropout=0.256):
        super().__init__()
        self.dropout = dropout
        
        # --- 核心修复：特征降维投影层 ---
        self.proj_p = nn.Linear(in_channels, hidden_channels)
        self.proj_s = nn.Linear(in_channels, hidden_channels)
        
        # 阶段 1: Premise 节点内部的 2 层更新 (现在统一为 1024 -> 1024)
        self.conv1 = HeteroConv({
            ('premise', 'sig_hypo', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'sig_goal', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('premise', 'sig_hypo', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'sig_goal', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')

        # 阶段 2: State 节点的单步聚合
        self.state_conv = HeteroConv({
            ('premise', 'state_hypo', 'state'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'state_goal', 'state'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        # --- 核心修复：进入 GNN 前先通过线性层完成 1472 -> 1024 的降维 ---
        h_p = self.proj_p(x_dict['premise'])
        h_s_init = self.proj_s(x_dict['state'])
        
        # Layer 1 (Premise)
        h_p_out1 = self.conv1({'premise': h_p}, edge_index_dict)
        h_p_out1 = F.relu(h_p_out1['premise'])
        h_p_out1 = F.dropout(h_p_out1, p=self.dropout, training=self.training)
        h_p = h_p + h_p_out1  # 现在残差连接两边都是 1024，完美匹配！
        
        # Layer 2 (Premise)
        h_p_out2 = self.conv2({'premise': h_p}, edge_index_dict)
        h_p_out2 = F.relu(h_p_out2['premise'])
        h_p_out2 = F.dropout(h_p_out2, p=self.dropout, training=self.training)
        h_p_final = h_p + h_p_out2
        
        # Layer 3 (State 单向接收)
        state_out_dict = self.state_conv({'premise': h_p_final, 'state': h_s_init}, edge_index_dict)
        h_s_final = F.relu(state_out_dict['state'])
        h_s_final = h_s_init + h_s_final

        return h_p_final, h_s_final

def compute_advanced_loss(h_s_batch, h_p_all, positive_ids_list, tau=0.05, margin=0.1, triplet_weight=1.0):
    """
    联合损失函数：InfoNCE (全局) + Triplet Margin Loss (难负样本局部打击)
    """
    # 1. L2 归一化，准备计算余弦相似度
    h_s_norm = F.normalize(h_s_batch, p=2, dim=-1)
    h_p_norm = F.normalize(h_p_all, p=2, dim=-1)
    
    # 核心矩阵：包含了 batch 内每个 State 与全库 21063 个 Premise 的相似度
    sim_matrix = torch.matmul(h_s_norm, h_p_norm.transpose(0, 1))
    
    # --- InfoNCE 准备工作 ---
    logits = sim_matrix / tau
    log_denominator = torch.logsumexp(logits, dim=1)
    
    total_infonce_loss = 0.0
    total_triplet_loss = 0.0
    total_positives = 0
    valid_triplet_count = 0
    
    # 遍历当前 Batch 中的每一个 State
    for i, pos_indices in enumerate(positive_ids_list):
        if not pos_indices: continue
            
        # ==========================================
        # 模块 A: InfoNCE Loss 计算 (保持原有逻辑)
        # ==========================================
        pos_logits = logits[i, pos_indices]
        log_probs = pos_logits - log_denominator[i]
        total_infonce_loss -= torch.sum(log_probs)
        total_positives += len(pos_indices)
        
        # ==========================================
        # 模块 B: 难负样本挖掘 (Hard Negative Mining) 与 Triplet Loss
        # ==========================================
        # 1. 拷贝当前 State 的全部相似度得分
        sim_row = sim_matrix[i].clone()
        
        # 2. 物理隔离：把真正的答案（正样本）得分强行设为负无穷
        # 这样在挑“最像的错误答案”时，就不会把真答案挑进去了
        sim_row[pos_indices] = -float('inf')
        
        # 3. 挖掘！挑出得分最高的 Top-5 错误答案 (这就是难负样本 Hard Negatives)
        hard_neg_scores, _ = torch.topk(sim_row, k=5)
        
        # 4. 获取正确答案的平均得分作为基准 (Positive Score)
        pos_scores = sim_matrix[i, pos_indices]
        avg_pos_score = pos_scores.mean()
        
        # 5. 计算 Triplet Loss: max(0, 难负样本得分 - 正确答案得分 + 边界安全距离 margin)
        # 我们强迫模型：正确答案的分数 必须比 难负样本的分数 高出 margin (比如 0.1) 这么多！
        triplet_loss = F.relu(hard_neg_scores - avg_pos_score + margin).mean()
        
        total_triplet_loss += triplet_loss
        valid_triplet_count += 1
        
    # 计算均值
    final_infonce = total_infonce_loss / total_positives if total_positives > 0 else torch.tensor(0.0, device=h_s_batch.device)
    final_triplet = total_triplet_loss / valid_triplet_count if valid_triplet_count > 0 else torch.tensor(0.0, device=h_s_batch.device)
    
    # 将两种 Loss 加权组合返回
    return final_infonce + triplet_weight * final_triplet

@torch.no_grad()
def evaluate(h_s_eval, h_p_all, eval_pos_ids_list):
    h_s_norm = F.normalize(h_s_eval, p=2, dim=-1)
    h_p_norm = F.normalize(h_p_all, p=2, dim=-1)
    
    sim_matrix = torch.matmul(h_s_norm, h_p_norm.transpose(0, 1))
    _, top10_indices = torch.topk(sim_matrix, k=10, dim=1)
    top10_indices = top10_indices.cpu().numpy()
    
    r1, r10, mrr = 0.0, 0.0, 0.0
    valid_count = 0
    
    for i, pos_indices in enumerate(eval_pos_ids_list):
        if not pos_indices: continue
        valid_count += 1
        top10 = top10_indices[i].tolist()
        
        if top10[0] in pos_indices: r1 += 1.0
        if any(p in pos_indices for p in top10): r10 += 1.0
        
        rank = 0
        for rank_idx, pred_p in enumerate(top10):
            if pred_p in pos_indices:
                rank = rank_idx + 1
                break
        if rank > 0: mrr += 1.0 / rank
            
    if valid_count == 0: return 0.0, 0.0, 0.0
    return (r1 / valid_count) * 100, (r10 / valid_count) * 100, (mrr / valid_count)


# ==========================================
# 3. 主训练循环 (修复缩进版)
# ==========================================
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📖 加载图数据...")
    data = torch.load('lean_hetero_data_full_optimized.pt', weights_only=False)
    data['premise'].x = data['premise'].x.to(DEVICE)
    data['state'].x = data['state'].x.to(DEVICE)
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    
    NUM_MODELS = 6
    for model_idx in range(NUM_MODELS):
        print(f"\n{'='*40}")
        print(f"🚀 开始训练第 {model_idx + 1}/{NUM_MODELS} 个模型")
        print(f"{'='*40}")
        
        torch.manual_seed(model_idx + 42) 
        
        model = LeanRGCN().to(DEVICE)
        ema = EMA(model, decay=0.99) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00499, weight_decay=2.359e-5)
    
        # ⬇️ 注意：从这里开始，所有代码都向右缩进了 4 个空格
        EPOCHS = 200
        BATCH_SIZE = 1024
        
        train_indices = data['state'].train_mask.nonzero(as_tuple=True)[0]
        val_indices = data['state'].val_mask.nonzero(as_tuple=True)[0]
        pos_ids_list = data['state'].positive_premise_ids
        
        print(f"🚀 开始训练! Train States: {len(train_indices)}, Val States: {len(val_indices)}")
        
        best_val_r10 = 0.0
        
        # 内层的 epoch 循环也需要跟着向右缩进
        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()
            
            # --- 训练阶段 ---
            model.train()
            optimizer.zero_grad()
            
            h_p_final, h_s_final = model(data.x_dict, edge_index_dict)
            
            perm = torch.randperm(train_indices.size(0))
            shuffled_train_indices = train_indices[perm]
            
            total_epoch_loss = 0.0
            num_batches = (shuffled_train_indices.size(0) + BATCH_SIZE - 1) // BATCH_SIZE
            
            final_loss = 0.0
            for i in range(0, shuffled_train_indices.size(0), BATCH_SIZE):
                batch_idx = shuffled_train_indices[i:i + BATCH_SIZE]
                h_s_batch = h_s_final[batch_idx]
                batch_pos_ids = [pos_ids_list[idx.item()] for idx in batch_idx]
                
                # margin=0.1 表示我们要求正样本比难负样本至少高 0.1 分
                # triplet_weight=1.5 表示我们适度加大对错题的惩罚权重
                loss = compute_advanced_loss(h_s_batch, h_p_final, batch_pos_ids, tau=0.05, margin=0.1, triplet_weight=1.5)
                final_loss = final_loss + (loss / num_batches)
                total_epoch_loss += loss.item()
                
            final_loss.backward()
            optimizer.step()
            ema.update(model)
            
            # --- 验证阶段 ---
            model.eval()
            ema.apply_shadow(model)
            
            with torch.no_grad():
                h_p_eval, h_s_eval_all = model(data.x_dict, edge_index_dict)
                h_s_val = h_s_eval_all[val_indices]
                val_pos_ids = [pos_ids_list[idx.item()] for idx in val_indices]
                
                r1, r10, mrr = evaluate(h_s_val, h_p_eval, val_pos_ids)
                
            ema.restore(model)
            
            # 保存最佳模型 (注意这里的名字是用 model_idx 区分的)
            if r10 > best_val_r10:
                best_val_r10 = r10
                ema.apply_shadow(model)
                torch.save(model.state_dict(), f'best_rgcn_model_{model_idx}.pth')
                ema.restore(model)
                
            epoch_time = time.time() - start_time
            avg_loss = total_epoch_loss / num_batches
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val R@1: {r1:.2f}% | Val R@10: {r10:.2f}% | Val MRR: {mrr:.4f} | Time: {epoch_time:.1f}s")

if __name__ == "__main__":
    main()