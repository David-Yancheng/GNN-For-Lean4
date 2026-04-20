# GNN 模型 Premise Selection 案例分析报告 (用于论文与答辩)

## 一、 完美命中案例 (Top-1 Success)
*模型精准理解了数学结构，在第一顺位给出了正确答案。可以在论文中证明图结构信息的有效性。*

### 案例 1: 证明状态 (Proof State)
```lean
case h
R : Type u
K : Type u'
M : Type v
V : Type v'
M₂ : Type w
V₂ : Type w'
M₃ : Type y
V₃ : Type y'
M₄ : Type z
ι : Type x
M₅ : Type u_1
M₆ : Type u_2
S : Type u_3
inst✝¹³ : Semiring R
inst✝¹² : Semiring S
inst✝¹¹ : AddCommMonoid M
inst✝¹⁰ : AddCommMonoid M₂
inst✝⁹ : AddCommMonoid M₃
inst✝⁸ : AddCommMonoid M₄
inst✝⁷ : AddCommMonoid M₅
inst✝⁶ : AddCommMonoid M₆
inst✝⁵ : Module R M
inst✝⁴ : Module R M₂
inst✝³ : Module R M₃
inst✝² : Module R M₄
inst✝¹ : Module R M₅
inst✝ : Module R M₆
f : M →ₗ[R] M₂
x : M × M₂
⊢ x ∈ range (inl R M M₂) ↔ x ∈ ker (snd R M M₂)
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `LinearMap.mem_ker`
- `LinearMap.mem_range`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ✅ [HIT] `LinearMap.mem_ker` (相似度得分: 0.6098)
2. ❌ `LinearMap.snd_apply` (相似度得分: 0.5719)
3. ❌ `LinearMap.fst_apply` (相似度得分: 0.5671)
4. ❌ `LinearMap.inl_apply` (相似度得分: 0.5305)
5. ✅ [HIT] `LinearMap.mem_range` (相似度得分: 0.5141)
---

### 案例 2: 证明状态 (Proof State)
```lean
case h.mpr
R : Type u
K : Type u'
M : Type v
V : Type v'
M₂ : Type w
V₂ : Type w'
M₃ : Type y
V₃ : Type y'
M₄ : Type z
ι : Type x
M₅ : Type u_1
M₆ : Type u_2
S : Type u_3
inst✝¹³ : Semiring R
inst✝¹² : Semiring S
inst✝¹¹ : AddCommMonoid M
inst✝¹⁰ : AddCommMonoid M₂
inst✝⁹ : AddCommMonoid M₃
inst✝⁸ : AddCommMonoid M₄
inst✝⁷ : AddCommMonoid M₅
inst✝⁶ : AddCommMonoid M₆
inst✝⁵ : Module R M
inst✝⁴ : Module R M₂
inst✝³ : Module R M₃
inst✝² : Module R M₄
inst✝¹ : Module R M₅
inst✝ : Module R M₆
f : M →ₗ[R] M₂
x : M × M₂
h : (snd R M M₂) x = 0
⊢ ∃ y, (inl R M M₂) y = x
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Prod.ext`
- `rfl`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ✅ [HIT] `Prod.ext` (相似度得分: 0.6337)
2. ❌ `Prod.ext_iff` (相似度得分: 0.6032)
3. ✅ [HIT] `rfl` (相似度得分: 0.6018)
4. ❌ `LinearMap.snd_apply` (相似度得分: 0.5907)
5. ❌ `LinearMap.fst_apply` (相似度得分: 0.5876)
---

### 案例 3: 证明状态 (Proof State)
```lean
n✝ : Type u_1
p : Type u_2
R : Type u₂
𝕜 : Type u_3
inst✝³ : Field 𝕜
inst✝² : DecidableEq n✝
inst✝¹ : DecidableEq p
inst✝ : CommRing R
r : ℕ
M : Matrix (Fin r ⊕ Unit) (Fin r ⊕ Unit) 𝕜
i : Fin r ⊕ Unit
k : ℕ
hk : k ≤ r
n : ℕ
hn : n < r
a✝ : k ≤ n
IH : ((List.drop (n + 1) (listTransvecCol M)).prod * M) (inr ()) i = M (inr ()) i
⊢ n < (listTransvecCol M).length
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Matrix.Pivot.listTransvecCol`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ✅ [HIT] `Matrix.Pivot.listTransvecCol` (相似度得分: 0.6075)
2. ❌ `Matrix.Pivot.listTransvecRow` (相似度得分: 0.6065)
3. ❌ `List.length` (相似度得分: 0.5746)
4. ❌ `List.length_ofFn` (相似度得分: 0.5439)
5. ❌ `Matrix.Pivot.mul_listTransvecRow_last_col_take` (相似度得分: 0.5406)
---

## 二、 困难检索案例 (Top-10 Success)
*正确答案没有排在第一。可以在答辩时展示模型找出的第一名通常是'语义极其相似'的干扰项（体现模型并不是在瞎猜）。*

### 案例 1: 证明状态 (Proof State)
```lean
R : Type u1
inst✝⁴ : CommRing R
M : Type u2
inst✝³ : AddCommGroup M
inst✝² : Module R M
A : Type u_1
inst✝¹ : Semiring A
inst✝ : Algebra R A
⊢ Disjoint (LinearMap.range (ι R)) 1
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Submodule.disjoint_def`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `Submodule.mem_bot` (相似度得分: 0.6160)
2. ❌ `disjoint_iff_inf_le` (相似度得分: 0.5888)
3. ❌ `Submodule.eq_bot_iff` (相似度得分: 0.5687)
4. ❌ `disjoint_iff` (相似度得分: 0.5520)
5. ❌ `Submodule.mem_inf` (相似度得分: 0.5259)
---

### 案例 2: 证明状态 (Proof State)
```lean
case h.h
k : Type u_1
E : Type u_2
PE : Type u_3
inst✝³ : Field k
inst✝² : AddCommGroup E
inst✝¹ : Module k E
inst✝ : AddTorsor E PE
f : k → E
c : PE
a b : k
⊢ slope (fun x => f x +ᵥ c) a b = slope f a b
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `slope`
- `vadd_vsub_vadd_cancel_right`
- `vsub_eq_sub`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `smul_smul` (相似度得分: 0.5313)
2. ❌ `smul_sub` (相似度得分: 0.5127)
3. ❌ `vadd_eq_add` (相似度得分: 0.5082)
4. ❌ `one_smul` (相似度得分: 0.5051)
5. ❌ `sub_smul` (相似度得分: 0.4832)
---

### 案例 3: 证明状态 (Proof State)
```lean
case insert
R : Type uR
S : Type uS
ι : Type uι
n : ℕ
M : Fin n.succ → Type v
M₁ : ι → Type v₁
M₂ : Type v₂
M₃ : Type v₃
M' : Type v'
inst✝¹¹ : Semiring R
inst✝¹⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
inst✝⁹ : (i : ι) → AddCommMonoid (M₁ i)
inst✝⁸ : AddCommMonoid M₂
inst✝⁷ : AddCommMonoid M₃
inst✝⁶ : AddCommMonoid M'
inst✝⁵ : (i : Fin n.succ) → Module R (M i)
inst✝⁴ : (i : ι) → Module R (M₁ i)
inst✝³ : Module R M₂
inst✝² : Module R M₃
inst✝¹ : Module R M'
f f' : MultilinearMap R M₁ M₂
α✝ : ι → Type u_1
g✝ : (i : ι) → α✝ i → M₁ i
A : (i : ι) → Finset (α✝ i)
α : Type u_2
inst✝ : DecidableEq ι
i : ι
g : α → M₁ i
m : (i : ι) → M₁ i
a : α
t : Finset α
has : a ∉ t
ih : f (update m i (∑ a ∈ t, g a)) = ∑ a ∈ t, f (update m i (g a))
⊢ f (update m i (∑ a ∈ insert a t, g a)) = ∑ a ∈ insert a t, f (update m i (g a))
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Finset.sum_insert`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `Finset.subset_insert` (相似度得分: 0.5475)
2. ✅ [HIT] `Finset.sum_insert` (相似度得分: 0.5445)
3. ❌ `Finset.mem_insert_self` (相似度得分: 0.5435)
4. ❌ `Function.update` (相似度得分: 0.5376)
5. ❌ `Finset.sum_insert_zero` (相似度得分: 0.5329)
---

## 三、 失败案例分析 (Failures)
*模型完全没有在 Top-10 中找到答案。可以用于撰写论文的 'Error Analysis' 章节，探讨未来 GAT 注意力机制或加入基础图谱库的必要性。*

### 案例 1: 证明状态 (Proof State)
```lean
R : Type u_1
inst✝² : CommSemiring R
M : Type u_2
inst✝¹ : AddCommMonoid M
inst✝ : Module R M
m : M
⊢ toTensor (ι R m) = (TensorAlgebra.ι R) m
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `FreeAlgebra.toTensor`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `LinearMap.lTensor` (相似度得分: 0.5591)
2. ❌ `LinearMap.rTensor` (相似度得分: 0.5481)
3. ❌ `TensorProduct.lid` (相似度得分: 0.5462)
4. ❌ `LinearMap.coe_comp` (相似度得分: 0.5299)
5. ❌ `AlgHom.comp_toLinearMap` (相似度得分: 0.5200)
---

### 案例 2: 证明状态 (Proof State)
```lean
R : Type u1
inst✝⁴ : CommRing R
M : Type u2
inst✝³ : AddCommGroup M
inst✝² : Module R M
A : Type u_1
inst✝¹ : Semiring A
inst✝ : Algebra R A
⊢ ∀ x ∈ LinearMap.range (ι R), x ∈ 1 → x = 0
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `algebraMap`
- `ExteriorAlgebra`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `LinearMap.mem_range` (相似度得分: 0.6000)
2. ❌ `LinearMap.mem_ker` (相似度得分: 0.5732)
3. ❌ `Submodule.mem_bot` (相似度得分: 0.5689)
4. ❌ `LinearMap.ext_iff` (相似度得分: 0.5366)
5. ❌ `Submodule.eq_bot_iff` (相似度得分: 0.5117)
---

