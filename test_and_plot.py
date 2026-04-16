import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==========================================
# 1. 模型定义 (保持不变)
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
        h_p = h_p + h_p_out1  
        
        h_p_out2 = self.conv2({'premise': h_p}, edge_index_dict)
        h_p_out2 = F.relu(h_p_out2['premise'])
        h_p_final = h_p + h_p_out2
        
        state_out_dict = self.state_conv({'premise': h_p_final, 'state': h_s_init}, edge_index_dict)
        h_s_final = F.relu(state_out_dict['state'])
        h_s_final = h_s_init + h_s_final

        return h_p_final, h_s_final

# ==========================================
# 2. 评估函数 (新增：返回预测细节供分析)
# ==========================================
@torch.no_grad()
def evaluate_ensemble(models, x_dict, edge_index_dict, test_indices, pos_ids_list):
    print(f"⚙️ 正在使用 {len(models)} 个模型进行集成计算...")
    ensemble_sim_matrix = None
    test_pos_ids = [pos_ids_list[idx.item()] for idx in test_indices]
    
    for i, model in enumerate(models):
        model.eval()
        h_p_eval, h_s_eval_all = model(x_dict, edge_index_dict)
        h_s_test = h_s_eval_all[test_indices]
        
        h_s_norm = F.normalize(h_s_test, p=2, dim=-1)
        h_p_norm = F.normalize(h_p_eval, p=2, dim=-1)
        sim_matrix = torch.matmul(h_s_norm, h_p_norm.transpose(0, 1))
        
        if ensemble_sim_matrix is None:
            ensemble_sim_matrix = sim_matrix
        else:
            ensemble_sim_matrix += sim_matrix
            
    ensemble_sim_matrix = ensemble_sim_matrix / len(models)
    
    # 提取 Top-10 索引和对应的相似度分数
    top10_scores, top10_indices = torch.topk(ensemble_sim_matrix, k=10, dim=1)
    top10_indices = top10_indices.cpu().numpy()
    top10_scores = top10_scores.cpu().numpy()
    
    r1, r5, r10, mrr = 0.0, 0.0, 0.0, 0.0
    valid_count = 0
    
    # 记录详细的预测结果用于 Case Study
    case_studies = []
    
    for i, pos_indices in enumerate(test_pos_ids):
        if not pos_indices: continue
        valid_count += 1
        top10 = top10_indices[i].tolist()
        scores = top10_scores[i].tolist()
        
        hit_rank = -1
        if top10[0] in pos_indices: r1 += 1.0
        if any(p in pos_indices for p in top10[:5]): r5 += 1.0
        if any(p in pos_indices for p in top10): 
            r10 += 1.0
            
        for rank_idx, pred_p in enumerate(top10):
            if pred_p in pos_indices:
                hit_rank = rank_idx + 1
                mrr += 1.0 / hit_rank
                break
                
        # 保存案例信息：原 state 的绝对索引、正确答案、预测 Top10 及其分数、命中排名
        case_studies.append({
            'state_global_idx': test_indices[i].item(),
            'gt_premises': pos_indices,
            'pred_top10': top10,
            'pred_scores': scores,
            'hit_rank': hit_rank
        })
            
    if valid_count == 0: return 0.0, 0.0, 0.0, 0.0, []
    
    return (r1 / valid_count) * 100, (r5 / valid_count) * 100, (r10 / valid_count) * 100, (mrr / valid_count), case_studies

# ==========================================
# 3. 数据重载与案例导出函数 (论文神器)
# ==========================================
def load_jsonl_robust(filepath):
    data = []
    if not os.path.exists(filepath): return data
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='utf-16') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    return data

def export_case_studies(case_studies, filename="thesis_case_studies.md"):
    print("\n📝 正在从 JSONL 逆向解析文本，生成案例分析报告...")
    
    # 严格按照 build_graph 时的顺序重新加载数据，保证索引绝对对齐
    premises_data = load_jsonl_robust('premises.jsonl')
    
    # 注意：需要把 train, val, test 一起加载才能对齐 global_idx
    raw_train = load_jsonl_robust('states_train.jsonl')
    raw_val = load_jsonl_robust('states_val.jsonl')
    raw_test = load_jsonl_robust('states_test.jsonl')
    
    # 重现训练代码中的严格去重逻辑
    train_states, val_states, test_states = [], [], []
    seen = set()
    for s in raw_train:
        if s['id'] not in seen:
            seen.add(s['id'])
            train_states.append(s)
    for s in raw_val:
        if s['id'] not in seen:
            seen.add(s['id'])
            val_states.append(s)
    for s in raw_test:
        if s['id'] not in seen:
            seen.add(s['id'])
            test_states.append(s)
            
    all_states_data = train_states + val_states + test_states
    
    # 挑选具有代表性的案例
    success_cases = [c for c in case_studies if c['hit_rank'] == 1][:3] # 挑 3 个 Top-1 命中的
    hard_cases = [c for c in case_studies if 1 < c['hit_rank'] <= 10][:3] # 挑 3 个 Top-10 命中但非 Top-1 的
    fail_cases = [c for c in case_studies if c['hit_rank'] == -1][:2] # 挑 2 个完全失败的
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# GNN 模型 Premise Selection 案例分析报告 (用于论文与答辩)\n\n")
        
        def write_cases(cases, section_title, description):
            f.write(f"## {section_title}\n")
            f.write(f"*{description}*\n\n")
            for idx, case in enumerate(cases):
                state_node = all_states_data[case['state_global_idx']]
                state_text = state_node.get('signature', '').strip()
                
                f.write(f"### 案例 {idx+1}: 证明状态 (Proof State)\n")
                f.write("```lean\n" + state_text + "\n```\n\n")
                
                f.write("**✅ 真实标准答案 (Ground Truth Premises):**\n")
                for gt_idx in case['gt_premises']:
                    gt_id = premises_data[gt_idx].get('id', 'Unknown')
                    f.write(f"- `{gt_id}`\n")
                f.write("\n")
                
                f.write("**🤖 模型预测 Top-5 (Model Predictions):**\n")
                for i in range(5):
                    pred_idx = case['pred_top10'][i]
                    pred_score = case['pred_scores'][i]
                    pred_id = premises_data[pred_idx].get('id', 'Unknown')
                    
                    mark = "✅ [HIT]" if pred_idx in case['gt_premises'] else "❌"
                    f.write(f"{i+1}. {mark} `{pred_id}` (相似度得分: {pred_score:.4f})\n")
                f.write("---\n\n")

        write_cases(success_cases, "一、 完美命中案例 (Top-1 Success)", "模型精准理解了数学结构，在第一顺位给出了正确答案。可以在论文中证明图结构信息的有效性。")
        write_cases(hard_cases, "二、 困难检索案例 (Top-10 Success)", "正确答案没有排在第一。可以在答辩时展示模型找出的第一名通常是'语义极其相似'的干扰项（体现模型并不是在瞎猜）。")
        write_cases(fail_cases, "三、 失败案例分析 (Failures)", "模型完全没有在 Top-10 中找到答案。可以用于撰写论文的 'Error Analysis' 章节，探讨未来 GAT 注意力机制或加入基础图谱库的必要性。")
        
    print(f"🎉 案例报告已成功生成！快去查看 {filename} 吧！")

# ==========================================
# 4. 绘图函数 (保持不变)
# ==========================================
def plot_metrics(r1, r5, r10, mrr, save_path="test_results_plot.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    metrics = ['Recall@1', 'Recall@5', 'Recall@10']
    values = [r1, r5, r10]
    colors = ['#4C72B0', '#55A868', '#C44E52']
    
    bars = ax1.bar(metrics, values, color=colors, width=0.5)
    ax1.set_ylim(0, max(values) + 15)
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Set Recall Metrics', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.bar(['MRR'], [mrr], color=['#8172B3'], width=0.4)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Reciprocal Rank (MRR)', fontsize=14, fontweight='bold')
    
    ax2.annotate(f'{mrr:.4f}',
                 xy=(0, mrr),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 可视化结果图表已保存至: {save_path}")

# ==========================================
# 5. 主流程
# ==========================================
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {DEVICE}")
    
    data = torch.load('lean_hetero_data_full_optimized.pt', weights_only=False)
    data['premise'].x = data['premise'].x.to(DEVICE)
    data['state'].x = data['state'].x.to(DEVICE)
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    test_indices = data['state'].test_mask.nonzero(as_tuple=True)[0]
    
    models = []
    NUM_MODELS = 6
    for i in range(NUM_MODELS):
        model_path = f'best_rgcn_model_{i}.pth'
        try:
            model = LeanRGCN().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            models.append(model)
            print(f"✅ 加载模型: {model_path}")
        except FileNotFoundError:
            print(f"⚠️ 未找到 {model_path}，跳过该模型。")
            
    if len(models) == 0:
        print("❌ 没有找到任何可用的模型权重！")
        return
        
    print("📈 正在计算集成评估指标...")
    # 注意这里接收了新增的 case_studies 返回值
    r1, r5, r10, mrr, case_studies = evaluate_ensemble(models, data.x_dict, edge_index_dict, test_indices, data['state'].positive_premise_ids)
    
    print("\n" + "="*40)
    print(f"🏆 FINAL TEST SET RESULTS (Ensemble of {len(models)} models)")
    print("="*40)
    print(f"Recall@1  : {r1:.2f}%")
    print(f"Recall@5  : {r5:.2f}%")
    print(f"Recall@10 : {r10:.2f}%")
    print(f"MRR       : {mrr:.4f}")
    print("="*40)
    
    plot_metrics(r1, r5, r10, mrr, save_path="ensemble_test_plot.png")
    
    # 核心新增：调用案例导出函数
    export_case_studies(case_studies)

if __name__ == "__main__":
    main()