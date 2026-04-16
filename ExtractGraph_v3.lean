import Lean
import Mathlib

open Lean Meta
set_option maxHeartbeats 0

-- 1. 定义核心域判定逻辑
def isLinearAlgebra (n : Name) : Bool :=
  match n with
  | .str p s => s == "LinearAlgebra" || isLinearAlgebra p
  | .num p _ => isLinearAlgebra p
  | .anonymous => false

def isCoreDomain (env : Environment) (n : Name) : Bool :=
  match env.getModuleIdxFor? n with
  | some idx => isLinearAlgebra env.header.moduleNames[idx]!
  | none => false

-- 2. 辅助函数：JSON 转义
def escapeJson (s : String) : String :=
  ((s.replace "\\" "\\\\").replace "\"" "\\\"").replace "\n" "\\n"

-- 3. 辅助函数：判断节点类型 (Theorem, Definition, Inductive 等)
def getNodeType (cinfo : ConstantInfo) : String :=
  match cinfo with
  | .thmInfo _ => "theorem"
  | .defnInfo _ => "definition"
  | .inductInfo _ => "inductive"
  | .axiomInfo _ => "axiom"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quotient"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

-- 4. 符号提取
def extractSymbols (e : Expr) : Array Name :=
  e.foldConsts #[] (fun n acc => if !acc.contains n then acc.push n else acc)

-- 5. 统一输出函数
def emitNode (name : Name) (isCore : Bool) (cinfo? : Option ConstantInfo) : MetaM Unit := do
  let isCoreStr := if isCore then "true" else "false"
  let mut json := "\"id\": \"" ++ toString name ++ "\", \"is_core\": " ++ isCoreStr

  let nodeTypeStr := match cinfo? with
    | some cinfo => getNodeType cinfo
    | none => "unknown"
  json := json ++ ", \"node_type\": \"" ++ nodeTypeStr ++ "\""

  match cinfo? with
  | some cinfo =>
    let fmtType ← ppExpr cinfo.type
    json := json ++ ", \"signature\": \"" ++ escapeJson (toString fmtType) ++ "\""

    let codeStr ← match cinfo.value? with
      | some val =>
        let fmtVal ← ppExpr val
        pure (escapeJson (toString fmtVal))
      | none => pure ""
    json := json ++ ", \"code\": \"" ++ codeStr ++ "\""

    let (hypoDeps, goalDeps) ← forallTelescope cinfo.type fun hyps goal => do
      let mut hAcc := #[]
      for h in hyps do
        let hType ← inferType h
        hAcc := hAcc ++ extractSymbols hType
      return (hAcc, extractSymbols goal)

    let hypoStr := ", ".intercalate (hypoDeps.toList.eraseDups.map (fun n => "\"" ++ toString n ++ "\""))
    let goalStr := ", ".intercalate (goalDeps.toList.eraseDups.map (fun n => "\"" ++ toString n ++ "\""))

    json := json ++ ", \"dependencies\": { " ++
            "\"signature_hypothesis\": [" ++ hypoStr ++ "], " ++
            "\"signature_goal\": [" ++ goalStr ++ "] }"

  | none =>
    json := json ++ ", \"signature\": \"\", \"code\": \"\", \"dependencies\": { \"signature_hypothesis\": [], \"signature_goal\": [] }"

  IO.println ("{ " ++ json ++ " }")

-- 6. 全新递归处理逻辑 (带深度限制的深度优先搜索)
partial def processNode (env : Environment) (name : Name)
    (visitedRef : IO.Ref (Lean.NameMap Nat))
    (emittedRef : IO.Ref Lean.NameSet)
    (depth : Nat)
    (maxOutDepth : Nat) : MetaM Unit := do
  -- 1. 防御：如果是内部生成变量，直接跳过
  if name.isInternal then return ()

  -- 2. 深度剪枝：如果已经以 ≥ 当前深度的层数访问过，则跳过（避免重复展开和死循环）
  let visited ← visitedRef.get
  if let some prevDepth := visited.find? name then
    if prevDepth >= depth then return ()

  -- 记录该节点当前访问到的最大深度
  visitedRef.modify (fun m => m.insert name depth)

  -- 获取节点信息
  let isCore := isCoreDomain env name
  let cinfo? := env.find? name

  -- 3. 输出控制：保证无论重新遍历多少次，每个节点只输出一次 JSON
  let emitted ← emittedRef.get
  if !emitted.contains name then
    emittedRef.modify (·.insert name)
    emitNode name isCore cinfo?

  -- 4. 计算传给下一层依赖的深度
  -- 核心节点永远享有满额的外延深度；非核心节点深度递减
  let nextDepth := if isCore then maxOutDepth else depth - 1

  -- 只要还在核心域内，或者外延深度还没扣完，就继续向下探索
  if isCore || depth > 0 then
    if let some cinfo := cinfo? then
      let allDeps := extractSymbols cinfo.type ++ (match cinfo.value? with | some v => extractSymbols v | none => #[])
      for dep in allDeps do
        processNode env dep visitedRef emittedRef nextDepth maxOutDepth

-- 7. Main 入口
def extractGraphWithProgress : MetaM Unit := do
  let env ← getEnv

  -- visitedRef 记录 [节点名称 -> 剩余向外探索的最大深度]
  let visitedRef : IO.Ref (Lean.NameMap Nat) ← IO.mkRef {}
  -- emittedRef 记录已经输出过 JSON 的节点名称
  let emittedRef ← IO.mkRef Lean.NameSet.empty

  let constants := env.constants.map₁
  let total := constants.size
  let mut count := 0
  let mut coreCount := 0

  -- ✨ 在这里修改你想在核心领域外扩展的层数 ✨
  let maxOutDepth := 0

  IO.eprintln s!"[开始提取] 总计扫描常数: {total} | 设定非核心域外延层数: {maxOutDepth}"

  for (_, cinfo) in constants do
    count := count + 1
    if count % 1000 == 0 then
      IO.eprintln s!"[进度] {count}/{total} | 已扫描核心节点起点: {coreCount}"

    -- 只有核心域节点才作为遍历的“起点”扔进递归函数
    if isCoreDomain env cinfo.name then
      coreCount := coreCount + 1
      -- 起点初始递归深度给满 maxOutDepth
      processNode env cinfo.name visitedRef emittedRef maxOutDepth maxOutDepth

  IO.eprintln "[完成]"

def main : IO Unit := do
  let leanPath ← Lean.findSysroot
  Lean.initSearchPath leanPath
  let env ← Lean.importModules #[{ module := `Mathlib }] {}

  let opts := Lean.Options.empty.insert `maxHeartbeats (Lean.DataValue.ofNat 0)

  let ctx : Core.Context := {
    fileName := "ExtractGraph_v2.lean"
    fileMap := default
    options := opts
  }

  let state : Core.State := {
    env := env
  }

  let _ ← (extractGraphWithProgress.run' {}).toIO ctx state

  IO.eprintln "[任务成功结束]"
