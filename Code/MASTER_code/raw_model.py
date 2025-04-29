import torch
import netron
import os
import sys
from MASTER_propose import (
    MASTER, MASTERModel, Gate,
    PositionalEncoding, TAttention, SAttention,
    TemporalAttention
)

# 1. 加载模型
model_path = '/Users/testkarma/Desktop/CEF3/CEF3_project/notebooks/models/MASTER_save_model/master0428_dmodel256_tn4_sn2_do0.2_bs16_lr8e-06_beta5.pt'
print(f"尝试加载模型: {model_path}")

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    sys.exit(1)

# 加载模型
model = torch.load(model_path, map_location='cpu')
print(f"模型类型: {type(model)}")

# 获取模型参数
d_feat = 79  # 基本特征维度
gate_input_start_index = getattr(model, 'gate_input_start_index', 79)
gate_input_end_index = getattr(model, 'gate_input_end_index', 89)

print(f"使用特征维度: {d_feat}")
print(f"门控输入范围: {gate_input_start_index} - {gate_input_end_index}")

# 创建足够大的模拟输入
batch_size = 1
seq_len = 10
input_dim = max(d_feat, gate_input_end_index)  # 确保维度足够大
dummy_input = torch.randn(batch_size, seq_len, input_dim)

# 导出ONNX
onnx_path = 'master_model.onnx'
torch.onnx.export(
    model,                   # 模型实例
    dummy_input,             # 模拟输入
    onnx_path,               # 输出文件路径
    export_params=True,      # 导出模型参数
    opset_version=12,        # ONNX操作集版本
    do_constant_folding=True,# 执行常量折叠优化
    input_names=['input'],   # 输入名称
    output_names=['output'], # 输出名称
    dynamic_axes={           # 动态轴
        'input': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size'}
    }
)

print(f"✅ 成功导出ONNX模型至：{onnx_path}")

# 4. 可视化
netron.start(onnx_path)