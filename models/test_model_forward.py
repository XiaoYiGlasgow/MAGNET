import torch
from MyTransformerNet.models.transformer_predictor import TransformerPredictor
from MyTransformerNet.utils.config import OUT_CHANNELS

def test_model_forward():
    # 设置模型输入维度
    batch_size = 4
    num_polylines = 66
    polyline_feature_dim = 10  # VectorNet结构规定为10维（最后一维为polyline_id）

    # 构造 dummy 输入，模拟真实 polyline 特征数据
    dummy_input = torch.randn(batch_size, num_polylines, polyline_feature_dim)

    # 为第10维 polyline_id 设置合法整数（用于 embedding）
    dummy_input[:, :, -1] = torch.randint(0, 1024, (batch_size, num_polylines))

    # 构建模型
    model = TransformerPredictor()

    # 前向传播
    output = model(dummy_input)

    # 打印输出维度
    print("✅ Input shape: ", dummy_input.shape)
    print("✅ Output shape:", output.shape)
    assert output.shape == (batch_size, OUT_CHANNELS), "❌ Output shape 不符合预期！"

if __name__ == "__main__":
    test_model_forward()
