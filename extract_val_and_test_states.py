import json
import re
import hashlib
import os
from collections import defaultdict
import chardet

# 1. 配置文件路径
KG_FILE = "premises.jsonl" 
# 输入：LeanDojo 原始的 val 和 test 文件路径
LEANDOJO_VAL_FILE = "leandojo_benchmark_4/random/val.json"
LEANDOJO_TEST_FILE = "leandojo_benchmark_4/random/test.json"

OUTPUT_DIR = "."
VAL_OUTPUT_FILE = f"{OUTPUT_DIR}/states_val.jsonl"
TEST_OUTPUT_FILE = f"{OUTPUT_DIR}/states_test.jsonl"

# 核心修改：只保留顶层目录，匹配所有线性代数领域文件
TARGET_FOLDERS = [
    "LinearAlgebra"
]

# 动态检测 KG_FILE 编码
with open(KG_FILE,'rb') as f:
    raw = f.read(10000)
encoding = chardet.detect(raw)['encoding']

def build_vocabulary():
    print(f"正在加载静态知识图谱词典 ({KG_FILE})，仅用于建立依赖索引...")
    vocab = set()
    short_to_full = defaultdict(list)
    
    if not os.path.exists(KG_FILE):
        print(f"错误：找不到文件 {KG_FILE}，请检查路径。")
        return vocab, short_to_full

    with open(KG_FILE, 'r', encoding=encoding) as f:
        for line in f:
            try:
                node = json.loads(line)
                full_name = node["id"] 
                vocab.add(full_name)
                
                short_name = full_name.split('.')[-1]
                short_to_full[short_name].append(full_name)
            except Exception as e:
                continue
                
    print(f"成功加载 {len(vocab)} 个静态节点索引。")
    return vocab, short_to_full

def extract_symbols_from_text(text, vocab, short_to_full):
    words = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text)
    matched_deps = set()
    
    for word in words:
        if word in vocab:
            matched_deps.add(word)
        elif word in short_to_full:
            matched_deps.add(short_to_full[word][0]) 
            
    return list(matched_deps)

def split_state(state_text):
    if '⊢' in state_text:
        parts = state_text.split('⊢')
        return parts[0], parts[1]
    elif '\u22a2' in state_text:
        parts = state_text.split('\u22a2')
        return parts[0], parts[1]
    else:
        return state_text, ""

def process_and_extract_states(input_json, output_jsonl, vocab, short_to_full, target_folders):
    if not os.path.exists(input_json):
        print(f"⚠️ 找不到输入文件: {input_json}，已跳过。")
        return

    print(f"正在解析 LeanDojo 状态数据 ({input_json})，目标目录: {target_folders}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        leandojo_data = json.load(f)

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        state_count = 0
        
        for thm in leandojo_data:
            file_path = thm.get("file_path", "")
            
            # 只要 file_path 包含 "LinearAlgebra" 即可命中
            if not any(folder in file_path for folder in target_folders):
                continue
                
            for tac in thm.get("traced_tactics", []):
                state_text = tac.get("state_before", "")
                
                state_id = "state_" + hashlib.md5(state_text.encode('utf-8')).hexdigest()[:16]
                
                hypo_text, goal_text = split_state(state_text)
                state_hypo = extract_symbols_from_text(hypo_text, vocab, short_to_full)
                state_goal = extract_symbols_from_text(goal_text, vocab, short_to_full)
                
                next_tactic_premise = []
                annotated = tac.get("annotated_tactic", [])
                if len(annotated) > 1 and isinstance(annotated[1], list):
                    for premise_info in annotated[1]:
                        if "full_name" in premise_info:
                            next_tactic_premise.append(premise_info["full_name"])
                
                state_node = {
                    "id": state_id,
                    "node_type": "state",        
                    "is_core": True,             
                    "signature": state_text,     
                    "dependencies": {
                        "signature_hypothesis": state_hypo, 
                        "signature_goal": state_goal,       
                        "next_tactic_premise": next_tactic_premise
                    }
                }
                
                fout.write(json.dumps(state_node, ensure_ascii=False) + '\n')
                state_count += 1
                
    print(f"处理完成！共提取了 {state_count} 个符合条件的状态节点。")
    print(f"文件已保存至: {output_jsonl}\n")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vocab, short_to_full = build_vocabulary()
    print("-" * 40)
    process_and_extract_states(LEANDOJO_VAL_FILE, VAL_OUTPUT_FILE, vocab, short_to_full,TARGET_FOLDERS)
    process_and_extract_states(LEANDOJO_TEST_FILE, TEST_OUTPUT_FILE, vocab, short_to_full, TARGET_FOLDERS)
    print("✅ 验证集和测试集数据提取完毕！")