GNN-For-Lean4
本项目旨在利用图神经网络（GNN）探索 Lean 4 形式化数学证明中的推导逻辑。由于 GitHub 对单个文件及仓库总容量的限制，部分大型数据集和模型权重文件未直接上传至本仓库，已通过 .gitignore 进行忽略。

目录结构与数据说明
本仓库的完整运行需要以下数据支持，这些文件存放在本地目录 data/ 下：

1. 原始数据集 (Datasets)
leandojo_benchmark_4/:

包含 LeanDojo 生成的基准测试数据，主要文件为 train.json（约 700MB+）。

该数据用于提取 Lean 4 证明过程中的目标（Goal）和前提（Premise）。

2. 预处理图数据 (Processed Data)
lean_hetero_data_full_optimized.pt:

文件大小：~200MB。

说明：这是经过预处理后的异构图数据（Heterogeneous Graph Data），直接用于 PyTorch Geometric (PyG) 的模型输入。

3. 模型权重 (Model Checkpoints)
仓库中不包含训练生成的 .pth 模型文件，主要涉及以下两个实验变体：

best_rgcn_model_with_HardNegative/:

包含在引入硬负采样（Hard Negative Sampling）策略下训练出的最佳 R-GCN 模型权重。

best_rgcn_model_without_HardNegative/:

包含基准对比实验（无硬负采样）的 R-GCN 模型权重。
