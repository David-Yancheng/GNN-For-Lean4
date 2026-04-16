import json
import torch
import os
from transformers import AutoTokenizer, T5EncoderModel
from torch_geometric.data import HeteroData
from tqdm import tqdm

# ==========================================
# 1. 配置参数
# ==========================================
PREMISES_FILE = 'premises.jsonl'
STATES_TRAIN_FILE = 'states_train.jsonl'
STATES_VAL_FILE = 'states_val.jsonl'
STATES_TEST_FILE = 'states_test.jsonl'

MODEL_NAME = "kaiyuy/leandojo-lean4-retriever-byt5-small" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32 # 半精度下可以适当调大
MAX_LENGTH = 1024
PT_FILE = 'lean_hetero_data_full_optimized.pt' 

print(f"🚀 使用设备: {DEVICE}")

# ==========================================
# 2. 基础读取与严格去重合并
# ==========================================
def load_jsonl_robust(filepath):
    # ... (保持原有的读取逻辑不变) ...
    data = []
    if not os.path.exists(filepath):
        return data
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='utf-16') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    return data

print("\n📖 读取 JSONL 数据...")
premises_data = load_jsonl_robust(PREMISES_FILE)
raw_train_states = load_jsonl_robust(STATES_TRAIN_FILE)
raw_val_states = load_jsonl_robust(STATES_VAL_FILE)
raw_test_states = load_jsonl_robust(STATES_TEST_FILE)

# --- 核心新增：有序去重与防数据泄露 ---
print("🛡️ 开始进行跨集合去重与防数据泄露处理...")
train_states, val_states, test_states = [], [], []
seen_state_ids = set()

# 1. 处理 Train (保留第一次出现的重复项)
for s in raw_train_states:
    if s['id'] not in seen_state_ids:
        seen_state_ids.add(s['id'])
        train_states.append(s)

# 2. 处理 Val (如果和 Train 重复，直接丢弃，防止模型在验证时作弊)
for s in raw_val_states:
    if s['id'] not in seen_state_ids:
        seen_state_ids.add(s['id'])
        val_states.append(s)

# 3. 处理 Test (如果和 Train/Val 重复，直接丢弃，保证 OOD 测试纯洁性)
for s in raw_test_states:
    if s['id'] not in seen_state_ids:
        seen_state_ids.add(s['id'])
        test_states.append(s)

print(f"  - Train 去重后: {len(train_states)} (原 {len(raw_train_states)})")
print(f"  - Val 去重后:   {len(val_states)} (原 {len(raw_val_states)})")
print(f"  - Test 去重后:  {len(test_states)} (原 {len(raw_test_states)})")

# 合并为去重后的全量数据
all_states_data = train_states + val_states + test_states

# 严格依赖 id 映射
premise_id2idx = {p['id']: i for i, p in enumerate(premises_data) if 'id' in p}
state_id2idx = {s['id']: i for i, s in enumerate(all_states_data)}

print(f"✅ 图规模: {len(premise_id2idx)} 个 Premise | {len(state_id2idx)} 个唯一 State")

# ==========================================
# 3. 特征提取 (融合半精度优化)
# ==========================================
def maybe_autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)

def get_text_embeddings(texts, max_length=MAX_LENGTH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
    text_encoder.eval()
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="向量化进度"):
            batch_texts = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                               max_length=max_length, return_tensors="pt").to(DEVICE)
            
            with maybe_autocast_context():
                outputs = text_encoder(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
                sum_embeddings = torch.sum(hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
            
            # 存回内存时务必转回 float32，防止后续 GNN 计算精度溢出
            embeddings.append(mean_pooled.detach().cpu().to(torch.float32))
            
    del text_encoder
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

if os.path.exists(PT_FILE):
    print(f"\n✅ 检测到本地特征文件 {PT_FILE}，直接复用 (跳过特征计算)...")
    data = torch.load(PT_FILE)
    x_p = data['premise'].x
    x_s = data['state'].x
else:
    print("\n🧠 提取 Premise 的特征...")
    premise_texts = [f"{p['id']} : {p.get('signature', '')}" for p in premises_data]
    x_p = get_text_embeddings(premise_texts)

    print("\n🧠 提取 State 的特征...")
    state_texts = [s.get('signature', '') for s in all_states_data]
    x_s = get_text_embeddings(state_texts)

# ==========================================
# 4. 构建异构图 (修复重复边隐患)
# ==========================================
print("\n🔗 构建去重的异构图边索引...")
data = HeteroData()
data['premise'].x = x_p
data['state'].x = x_s

def build_edge_index_dedup(target_list, src_idx_map, tgt_idx_map, edge_type_key):
    sources, targets = [], []
    for tgt_dict in target_list:
        tgt_id = tgt_dict['id']
        raw_deps = tgt_dict.get('dependencies', {}).get(edge_type_key, [])
        
        # 核心修复：使用 set() 对单个节点的依赖进行去重，防止生成重复边
        unique_deps = set(raw_deps) 
        
        for src_id in unique_deps:
            if src_id in src_idx_map and tgt_id in tgt_idx_map:
                sources.append(src_idx_map[src_id])
                targets.append(tgt_idx_map[tgt_id])
                
    if not sources: return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([sources, targets], dtype=torch.long)

data['premise', 'sig_hypo', 'premise'].edge_index = build_edge_index_dedup(
    premises_data, premise_id2idx, premise_id2idx, 'signature_hypothesis')
data['premise', 'sig_goal', 'premise'].edge_index = build_edge_index_dedup(
    premises_data, premise_id2idx, premise_id2idx, 'signature_goal')

data['premise', 'state_hypo', 'state'].edge_index = build_edge_index_dedup(
    all_states_data, premise_id2idx, state_id2idx, 'signature_hypothesis')
data['premise', 'state_goal', 'state'].edge_index = build_edge_index_dedup(
    all_states_data, premise_id2idx, state_id2idx, 'signature_goal')

# ==========================================
# 5. 多标签保留与严格的 Mask 生成
# ==========================================
print("\n🏷️ 提取多标签并生成严格的 Train/Val/Test Masks...")
positive_premise_ids_list = []
state_y = []
valid_state_mask = []

for s in all_states_data:
    raw_next_premises = s.get('dependencies', {}).get('next_tactic_premise', [])
    
    # 保持顺序的去重多标签提取
    mapped_ids = []
    seen = set()
    for tp in raw_next_premises:
        if tp in premise_id2idx and tp not in seen:
            seen.add(tp)
            mapped_ids.append(premise_id2idx[tp])
            
    positive_premise_ids_list.append(mapped_ids)
    
    if len(mapped_ids) > 0:
        state_y.append(mapped_ids[0]) # 保留首个作为单目标 y 兼容老代码
        valid_state_mask.append(True)
    else:
        state_y.append(-1)
        valid_state_mask.append(False)

# 将多标签列表挂载到 data 上供评估和 InfoNCE 使用
data['state'].positive_premise_ids = positive_premise_ids_list
data['state'].y = torch.tensor(state_y, dtype=torch.long)
valid_tensor = torch.tensor(valid_state_mask, dtype=torch.bool)

# 生成原始切片
n_tr, n_v, n_te = len(train_states), len(val_states), len(test_states)
raw_train_mask = torch.tensor([True]*n_tr + [False]*(n_v + n_te))
raw_val_mask = torch.tensor([False]*n_tr + [True]*n_v + [False]*n_te)
raw_test_mask = torch.tensor([False]*(n_tr + n_v) + [True]*n_te)

# 强制与 valid_tensor 取交集，彻底阻绝无标签节点流入计算图
data['state'].train_mask = raw_train_mask & valid_tensor
data['state'].val_mask = raw_val_mask & valid_tensor
data['state'].test_mask = raw_test_mask & valid_tensor

# ==========================================
# 6. 保存
# ==========================================
torch.save(data, PT_FILE)
print(f"\n🎉 完美！数据预处理完成，全量异构图已保存为 '{PT_FILE}'")
print(data)