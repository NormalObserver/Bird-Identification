import onnxruntime as ort
import numpy as np
import cv2

# 1. 加载 ONNX 模型
onnx_model_path = "Birds_ir8.onnx"  # 替换为你的模型路径
session = ort.InferenceSession(onnx_model_path)

# 2. 加载并预处理图片
image_path = "test_image.jpg"  # 替换为你要预测的图片路径
image = cv2.imread(image_path)

# 假设模型输入尺寸是 640x640
input_size = (640, 640)
image_resized = cv2.resize(image, input_size)

# 转换为 RGB（OpenCV 加载的是 BGR 图像）
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 归一化（假设模型要求的输入是 [0, 1] 范围）
image_normalized = image_rgb.astype(np.float32) / 255.0

# 维度变换：从 (height, width, channels) 转为 (batch_size, channels, height, width)
image_input = np.transpose(image_normalized, (2, 0, 1))  # (C, H, W)
image_input = np.expand_dims(image_input, axis=0)  # (1, C, H, W)

# 3. 运行推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_input})

# 4. 解析输出并应用后处理
# YOLOv8 的输出通常是一个三维张量 (batch_size, num_anchors * grid_size * grid_size, num_classes + 5)
# 我们假设这里输出的 shape 为 (1, 25200, 85)（与 YOLOv8 的一般设置一致）

output = outputs[0]  # 获取输出
batch_size, num_boxes, _ = output.shape

# 假设模型的输入尺寸为 640x640，目标图像尺寸为原始尺寸
original_height, original_width = image.shape[:2]

# 非极大值抑制 (NMS) 参数
confidence_threshold = 0.5
nms_threshold = 0.4

# 进行解码：YOLOv8 使用 sigmoid 来对中心坐标、置信度进行变换
sigmoid_output = 1 / (1 + np.exp(-output))  # sigmoid 应用到输出

# 获取坐标和置信度
box_coords = sigmoid_output[..., :4]  # (x_center, y_center, width, height)
confidence = sigmoid_output[..., 4]  # 置信度
class_probs = sigmoid_output[..., 5:]  # 类别概率

# 计算边界框的左上角和右下角坐标
x_center, y_center, width, height = box_coords[..., 0], box_coords[..., 1], box_coords[..., 2], box_coords[..., 3]
x_min = (x_center - width / 2) * original_width
y_min = (y_center - height / 2) * original_height
x_max = (x_center + width / 2) * original_width
y_max = (y_center + height / 2) * original_height

# 计算类别的概率
class_confidences = confidence[..., None] * class_probs  # 类别置信度 = 置信度 * 类别概率
# 对每个框应用 NMS
for i in range(num_boxes):
    # 假设 confidence 是 (num_boxes,) 的一维数组
    # 检查每个框的置信度
    if confidence[i] >= confidence_threshold:  # 单独比较每个置信度
        # 提取检测框信息
        box = [x_min[i], y_min[i], x_max[i], y_max[i], confidence[i]]
        class_id = np.argmax(class_confidences[i])
        class_conf = class_confidences[i][class_id]

        if class_conf >= confidence_threshold:
            # 可视化检测框
            color = (0, 255, 0)  # 设置框的颜色
            cv2.rectangle(image, (int(x_min[i]), int(y_min[i])), (int(x_max[i]), int(y_max[i])), color, 2)
            cv2.putText(image, f"Class {class_id}: {class_conf:.2f}",
                        (int(x_min[i]), int(y_min[i]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 5. 显示检测结果
cv2.imshow("Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
