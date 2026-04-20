import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import time

# ==========================================
# 1. EMA (指数模型平均) 类定义 (保留以控制变量)
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ==========================================
# 2. 模型与 Loss 定义 (移除难负样本)
# ==========================================
class LeanRGCN(nn.Module):
    def __init__(self, in_channels=1472, hidden_channels=1024, dropout=0.256):
        super().__init__()
        self.dropout = dropout
        self.proj_p = nn.Linear(in_channels, hidden_channels)
        self.proj_s = nn.Linear(in_channels, hidden_channels)
        
        self.conv1 = HeteroConv({
            ('premise', 'sig_hypo', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'sig_goal', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('premise', 'sig_hypo', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'sig_goal', 'premise'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')

        self.state_conv = HeteroConv({
            ('premise', 'state_hypo', 'state'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
            ('premise', 'state_goal', 'state'): SAGEConv(hidden_channels, hidden_channels, normalize=True),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        h_p = self.proj_p(x_dict['premise'])
        h_s_init = self.proj_s(x_dict['state'])
        
        h_p_out1 = self.conv1({'premise': h_p}, edge_index_dict)
        h_p_out1 = F.relu(h_p_out1['premise'])
        h_p_out1 = F.dropout(h_p_out1, p=self.dropout, training=self.training)
        h_p = h_p + h_p_out1  
        
        h_p_out2 = self.conv2({'premise': h_p}, edge_index_dict)
        h_p_out2 = F.relu(h_p_out2['premise'])
        h_p_out2 = F.dropout(h_p_out2, p=self.dropout, training=self.training)
        h_p_final = h_p + h_p_out2
        
        state_out_dict = self.state_conv({'premise': h_p_final, 'state': h_s_init}, edge_index_dict)
        h_s_final = F.relu(state_out_dict['state'])
        h_s_final = h_s_init + h_s_final

        return h_p_final, h_s_final

def compute_base_loss(h_s_batch, h_p_all, positive_ids_list, tau=0.05):
    """
    基础版损失函数：仅保留 InfoNCE，移除 Triplet Margin Loss
    """
    h_s_norm = F.normalize(h_s_batch, p=2, dim=-1)
    h_p_norm = F.normalize(h_p_all, p=2, dim=-1)
    
    sim_matrix = torch.matmul(h_s_norm, h_p_norm.transpose(0, 1))
    logits = sim_matrix / tau
    log_denominator = torch.logsumexp(logits, dim=1)
    
    total_infonce_loss = 0.0
    total_positives = 0
    
    for i, pos_indices in enumerate(positive_ids_list):
        if not pos_indices: continue
            
        pos_logits = logits[i, pos_indices]
        log_probs = pos_logits - log_denominator[i]
        total_infonce_loss -= torch.sum(log_probs)
        total_positives += len(pos_indices)
        
    return total_infonce_loss / total_positives if total_positives > 0 else torch.tensor(0.0, device=h_s_batch.device)

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
# 3. 主训练循环 (无集成版)
# ==========================================
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("📖 加载图数据...")
    data = torch.load('lean_hetero_data_full_optimized.pt', weights_only=False)
    data['premise'].x = data['premise'].x.to(DEVICE)
    data['state'].x = data['state'].x.to(DEVICE)
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    
    print(f"\n{'='*40}")
    print(f"🚀 开始单模型 Ablation 训练")
    print(f"{'='*40}")
    
    torch.manual_seed(42) 
    
    model = LeanRGCN().to(DEVICE)
    ema = EMA(model, decay=0.99) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00499, weight_decay=2.359e-5)

    EPOCHS = 200
    BATCH_SIZE = 1024
    
    train_indices = data['state'].train_mask.nonzero(as_tuple=True)[0]
    val_indices = data['state'].val_mask.nonzero(as_tuple=True)[0]
    pos_ids_list = data['state'].positive_premise_ids
    
    print(f"🚀 训练启动! Train States: {len(train_indices)}, Val States: {len(val_indices)}")
    
    best_val_r10 = 0.0
    
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
            
            # 使用基础版 InfoNCE Loss
            loss = compute_base_loss(h_s_batch, h_p_final, batch_pos_ids, tau=0.05)
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
        
        # 保存单个最佳模型
        if r10 > best_val_r10:
            best_val_r10 = r10
            ema.apply_shadow(model)
            torch.save(model.state_dict(), 'best_rgcn_model_single.pth')
            ema.restore(model)
            
        epoch_time = time.time() - start_time
        avg_loss = total_epoch_loss / num_batches
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val R@1: {r1:.2f}% | Val R@10: {r10:.2f}% | Val MRR: {mrr:.4f} | Time: {epoch_time:.1f}s")

if __name__ == "__main__":
    main()