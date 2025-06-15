# Tensor-GCN 项目结构与文件说明

## 根目录（一级）

| 路径 / 文件         | 类型       | 作用简介                                 | 常用命令 / 备注                       |
| ------------------- | ---------- | ---------------------------------------- | ------------------------------------- |
| **Tensor-GCN/**     | 代码仓库   | 主体算法与训练脚本                       | 详见下一节                            |
| **baselines/**      | 代码仓库   | 各种对比 / 基线方法实现                  | —                                     |
| **ZDataSet/**       | 数据集     | 原始、未经预处理的数据                   | —                                     |
| **Zdata/**          | 数据集     | 由 `dataLoader.py` 生成的缓存数据        | —                                     |
| **dataLoader.py**   | 脚本       | 预处理 `ZDataSet` 中的数据并写入 `Zdata` | `python dataLoader.py`                |
| **handle_USPS.py**  | 脚本       | 专门处理 USPS 数据集的 DataLoader        | `python handle_USPS.py`               |
| **Tensor-GCN.yaml** | Conda 环境 | 复现实验所需依赖列表                     | `conda env create -f Tensor-GCN.yaml` |

---

## Tensor-GCN 子目录（二级）

| 路径 / 文件       | 类型     | 作用简介                                    | 常用命令 / 备注                |
| ----------------- | -------- | ------------------------------------------- | ------------------------------ |
| **config.py**     | 配置     | 超参数与训练设置统一入口                    | 修改后 `run_GCN.py` 会自动读取 |
| **dataloader.py** | 数据     | 加载 `Zdata` 中的预处理数据并封装为 Dataset | 被 `run_GCN.py` 调用           |
| **GCN.py**        | 模型     | 图卷积网络（主干）实现                      | —                              |
| **metrics.py**    | 评估     | 计算指标、绘制热图、可视化等                | 训练时自动调用                 |
| **run_GCN.py**    | 训练入口 | 启动训练与测试流程                          | `python run_GCN.py`            |
| **tool.py**       | 工具     | 张量处理、归一化等辅助函数                  | —                              |
| **run_loop.sh**   | Shell    | （可选）批量运行 / 多实验循环脚本           | `bash run_loop.sh`             |
| **Result 存档/**  | 输出     | 保存了之前的部分实验结果                    | —                              |
| **Prev/**         | 备份     | 历史版本或中间文件                          | —                              |
| **__pycache__/**  | 缓存     | Python 编译缓存，可忽略                     | —                              |

