clear; clc; close all;
onnx_model_name = 'Birds.onnx';
img_path = 'test1.jpg';
confT = 0.3;  % 置信度阈值
iouT = 0.5;   % 交并比阈值
% classes = load_classes('./birds.yaml'); 
% disp(classes);

[classes,classesId] = predict_with_onnx(onnx_model_name, img_path, confT, iouT);

disp(classes{classesId});


function [classes, classesId] = predict_with_onnx(onnx_model_name, img_path, confT, iouT)
    onnxruntime = py.importlib.import_module('onnxruntime');
    session = onnxruntime.InferenceSession(onnx_model_name); 

    % 假设类别信息存在 birds.yaml 文件中
    classes = load_classes('birds.yaml'); 

    % 读取图像并进行预处理
    img = imread(img_path);
    img_data = preprocess_image(img);

    % 模型推理
    input_name = session.get_inputs{1}.name;
    % 确保 img_data 是正确的 Python 对象
    img_data_py = py.numpy.array(img_data);  % 将 img_data 转换为 numpy 数组
    
    % 调用模型进行推理，注意这里我们传递的是一个字典
    inputs_dict = py.dict(pyargs(input_name, img_data_py));  % 创建字典

    outputs = session.run(py.None, inputs_dict);  % 执行推理

    classesId = get_class_ids(outputs, confT);
end

function classes = load_classes(yaml_file)
    % 加载Python模块
    py.importlib.import_module('yaml');
   
    % 使用py.io.open读取yaml文件
    file_id = py.io.open(yaml_file, 'r');  % 打开文件对象
    yaml_data = py.yaml.safe_load(file_id);  % 使用safe_load而非load, 这个方法不需要指定Loader
    
    % 关闭文件对象
    file_id.close();
    classes = yaml_data{'names'}; 
end
function img_data = preprocess_image(img)
    % 对图像进行预处理以适应模型的输入要求
    % 假设输入大小为 (3, 640, 640)
    [img_height, img_width, ~] = size(img);

    % 将图像颜色空间从BGR转换为RGB (在MATLAB中读取的图像默认就是RGB)
    % 如果是BGR格式图像，可以用以下代码：
    % img = img(:,:, [3, 2, 1]); % 交换通道
    
    % 将图像调整为匹配输入形状 (640, 640, 3)
    img = imresize(img, [640, 640]); 

    img = single(img) / 255.0;  % 归一化
    
    % 转换为通道优先的格式 (C, H, W)
    img_data = permute(img, [3, 1, 2]);  % 将 (H, W, C) 转换为 (C, H, W)
    
    % 扩展维度为 (1, C, H, W) 以匹配模型的输入格式
    img_data = reshape(img_data, [1, size(img_data)]);
end


function targetId = get_class_ids(outputs, confT)

    output = outputs{1};  % output 是一个 numpy.ndarray（Python 类型）
    disp(class(output));
    output = py.numpy.squeeze(output);  % 移除单维度
    output_trans = py.numpy.transpose(output);

    output_list = output_trans.tolist();  % 使用 .tolist() 将其转换为 Python list
    
    % 获取输出的行数（即检测框的数量）
    num_boxes = double(py.len(output_list));  % 获取输出的行数，即检测框的数量

    class_ids = [];
    scores = [];
    
    % 遍历所有检测框
    for i = 1:num_boxes
        % 使用 MATLAB 的圆括号访问 Python list 中的第 i 行
        row = double(py.array.array('d', output_list{i}));  % 将每一行转换为 MATLAB 数组

        % 提取从第 5 列到末尾的得分部分
        scores_row  = row(5:end);  % 取出从第 5 列开始的得分

        
        [max_score, class_id] = max(scores_row);  % 找到最大得分及对应的类别 ID（MATLAB 中的索引从 1 开始）

        % 判断最大得分是否大于或等于置信度阈值
        if max_score >= confT
            % 由于 MATLAB 中的索引从 1 开始，Python 的索引从 0 开始，所以需要减去 1
            class_ids = [class_ids; class_id - 1];  % 保存类别 ID
            scores = [scores; max_score];  % 保存得分
        end
    end

%     disp(class_ids);
    targetId = mode(class_ids);
     if isempty(targetId)
        targetId = NaN;  % 返回 -1 表示没有有效的类别 ID
     end
%      disp(targetId);
end






% function class_ids = get_class_ids(outputs, confT)
% %     disp(outputs);
%     output = outputs{1}; 
%     output = double(output);  
%     disp(class(outputs));
%     disp(class(outputs{1}));
%     disp(class(output));
% %     disp(output);
%     output = squeeze(output);  % 去除单一维度
%     
%     % 确保输出是 [8400, 204] 的形状，如果不是的话进行转置
%     % 假设原始输出的形状为 [1, 8400, 204]
%     if size(output, 1) == 1
%         output = permute(output, [2,3, 1]);  % 转置为 [8400, 204]
%     end
%     
%     % 获取输出数组的行数（即检测框的数量）
%     rows = size(output, 1);  % 输出的行数即为检测框数量
%     disp(size(output));
%     disp(rows);
%     % 存储类别 ID 的数组
%     class_ids = [];
% 
%     % 遍历每个检测框
%     for i = 1:rows
%         % 提取当前检测框的类别得分
%         classes_scores = outputs(i, 5:end);  % 从索引 5 开始是类别的得分
%         [max_score, class_id] = max(classes_scores);  % 找到最大得分及其对应的类别 ID
% 
%         % 如果最大得分大于或等于置信度阈值
%         if max_score >= confT
%             % 存储该类别 ID
%             class_ids = [class_ids, class_id];
%         end
%     end
%     disp(class_ids);
% end







% function output_img = postprocess_image(img, outputs, classes, confT, iouT)
% %     outputs = outputs{1};
%    
%     boxes = outputs{1};   % 假设模型返回的是边界框信息
%     scores = outputs{2};  % 假设模型返回的是得分信息
%     class_ids = outputs{3};  % 假设模型返回的是类别 ID 信息
%     
%     % 遍历所有检测到的框
%     for i = 1:size(boxes, 1)
%         if scores(i) > confT  % 如果得分超过阈值
%             % 获取边界框坐标
%             box = boxes(i, :);
%             class_id = class_ids(i);
%             score = scores(i);
%             
%             % 绘制边界框
%             img = insertShape(img, 'Rectangle', box, 'Color', 'green', 'LineWidth', 3);
%             % 添加标签
%             label = sprintf('%s: %.2f', classes{class_id}, score);
%             img = insertText(img, box(1:2), label, 'BoxColor', 'yellow', 'TextColor', 'black');
%         end
%     end
%     
%     output_img = img;
% end

