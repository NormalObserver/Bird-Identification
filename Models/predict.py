from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

model = YOLO('./train4/weights/best.pt')
# model.export(format='onnx')  # 默认会导出到当前目录

folder_path = './CUB/CUB_200_2011/images/002.Laysan_Albatross'  # 替换为你的图片文件夹路径
output_folder = './res'  # 预测结果保存到 ./res 文件夹
# 遍历文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    # 获取文件的完整路径
    image_path = os.path.join(folder_path, filename)

    # 检查文件是否为图片（可以根据文件扩展名判断）
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {filename}...")

        # 对图片进行预测
        results = model(image_path)

        # 如果results是一个列表，遍历每个结果并打印
        for result in results:
            result.show()

            # os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
            #
            # # 使用 result.save() 保存带有预测框的图片
            # result.save()  # 结果会默认保存到 ./runs/detect 目录
            # saved_img_path = "results_"+filename
            # output_img_path = os.path.join(output_folder, f'pred_{filename}')
            # os.rename(saved_img_path, output_img_path)  # 移动保存的文件到 ./res 文件夹
            # print(f"Prediction for {filename} saved to {output_img_path}")