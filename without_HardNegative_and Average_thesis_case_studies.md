# GNN 模型 Premise Selection 案例分析报告 (Ablation 结果)

## 一、 完美命中案例 (Top-1 Success)
*基础模型预测结果。*

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
1. ✅ [HIT] `LinearMap.mem_ker` (相似度得分: 0.6198)
2. ❌ `LinearMap.snd_apply` (相似度得分: 0.5835)
3. ❌ `LinearMap.fst_apply` (相似度得分: 0.5678)
4. ❌ `LinearMap.inl_apply` (相似度得分: 0.5396)
5. ✅ [HIT] `LinearMap.mem_range` (相似度得分: 0.5303)
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
1. ✅ [HIT] `Prod.ext` (相似度得分: 0.6486)
2. ❌ `Prod.ext_iff` (相似度得分: 0.6183)
3. ❌ `LinearMap.snd_apply` (相似度得分: 0.6069)
4. ✅ [HIT] `rfl` (相似度得分: 0.5968)
5. ❌ `LinearMap.fst_apply` (相似度得分: 0.5943)
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
1. ✅ [HIT] `Matrix.Pivot.listTransvecCol` (相似度得分: 0.6123)
2. ❌ `Matrix.Pivot.listTransvecRow` (相似度得分: 0.6093)
3. ❌ `List.length` (相似度得分: 0.5789)
4. ❌ `le_rfl` (相似度得分: 0.5549)
5. ❌ `List.length_ofFn` (相似度得分: 0.5466)
---

## 二、 困难检索案例 (Top-10 Success)
*基础模型预测结果。*

### 案例 1: 证明状态 (Proof State)
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
1. ❌ `smul_smul` (相似度得分: 0.5433)
2. ❌ `smul_sub` (相似度得分: 0.5237)
3. ❌ `vadd_eq_add` (相似度得分: 0.5198)
4. ❌ `one_smul` (相似度得分: 0.5103)
5. ❌ `sub_smul` (相似度得分: 0.4932)
---

### 案例 2: 证明状态 (Proof State)
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
1. ❌ `Finset.subset_insert` (相似度得分: 0.5558)
2. ✅ [HIT] `Finset.sum_insert` (相似度得分: 0.5516)
3. ❌ `Finset.sum_insert_zero` (相似度得分: 0.5322)
4. ❌ `Function.update` (相似度得分: 0.5283)
5. ❌ `Insert.insert` (相似度得分: 0.5233)
---

### 案例 3: 证明状态 (Proof State)
```lean
case refine_1
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
⊢ ((List.drop n (listTransvecCol M)).prod * M) (inr ()) i = M (inr ()) i
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Matrix.Pivot.listTransvecCol`
- `List.length`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `Matrix.Pivot.listTransvecRow` (相似度得分: 0.5594)
2. ❌ `List.prod` (相似度得分: 0.5475)
3. ✅ [HIT] `Matrix.Pivot.listTransvecCol` (相似度得分: 0.5433)
4. ❌ `le_rfl` (相似度得分: 0.5389)
5. ✅ [HIT] `List.length` (相似度得分: 0.5293)
---

## 三、 失败案例分析 (Failures)
*基础模型预测结果。*

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
1. ❌ `LinearMap.lTensor` (相似度得分: 0.5739)
2. ❌ `LinearMap.rTensor` (相似度得分: 0.5661)
3. ❌ `LinearMap.comp` (相似度得分: 0.5608)
4. ❌ `LinearMap.coe_comp` (相似度得分: 0.5563)
5. ❌ `TensorProduct.lid` (相似度得分: 0.5548)
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
⊢ Disjoint (LinearMap.range (ι R)) 1
```

**✅ 真实标准答案 (Ground Truth Premises):**
- `Submodule.disjoint_def`

**🤖 模型预测 Top-5 (Model Predictions):**
1. ❌ `Submodule.mem_bot` (相似度得分: 0.6288)
2. ❌ `disjoint_iff_inf_le` (相似度得分: 0.6085)
3. ❌ `disjoint_iff` (相似度得分: 0.5780)
4. ❌ `Submodule.eq_bot_iff` (相似度得分: 0.5754)
5. ❌ `Submodule.mem_inf` (相似度得分: 0.5301)
---

