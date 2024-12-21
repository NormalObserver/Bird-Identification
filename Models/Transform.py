import shutil
import cv2
path='./CUB/CUB_200_2011'
labpath='./datasets/cub/labels'
imapath='./datasets/cub/images'


def get_yaml():
    # get birds.yaml
    with open(path + '/classes.txt', 'r') as fo, open('birds.yaml', "a") as fi:  # r只读，w写，a追加写
        for num, line in enumerate(fo):  # enumerate为枚举，num为从0开始的序号，line为每一行的信息
            s = '  ' + str(num) + ': ' + line.split(" ")[-1]  # 以空格分隔，去掉末尾的换行
            fi.write(s)  # 追加写入目标文件


def get_alllab():
    dataall = {}  # 字典用于存放txt中的各种信息
    with open(path + '/images.txt', 'r') as imagesall, open(path + '/image_class_labels.txt', 'r') as classall, \
            open(path + '/train_test_split.txt', 'r') as splitall, open(path + '/bounding_boxes.txt', 'r') as boxall:
        for num, line in enumerate(imagesall):  # 值用列表存储，方便后续添加元素，-1去掉末尾的/n
            s = line.split(" ")
            dataall[s[0]] = [s[1][:-1]]
        for num, line in enumerate(classall):
            s = line.split(" ")
            dataall[s[0]].append(s[1][:-1])
        for num, line in enumerate(splitall):
            s = line.split(" ")
            dataall[s[0]].append(s[1][:-1])
        for num, line in enumerate(boxall):
            s = line.split(" ")
            dataall[s[0]].extend([s[1], s[2], s[3], s[4][:-1]])
    print('dataall have got...')
    for item in dataall:
        na = item.rjust(12, '0')  # item为字典的键，左侧扩充0，改为所需名字格式（未看到明确要求）
        image = cv2.imread(path + '/images/' + dataall[item][0])  # 读取图片，使用shape获取图片宽高
        # 这两行代码为验证boundingbox信息，手动画框，图片存储至test文件夹，左上角和右下角坐标,image.shape[1] 宽度  image.shape[0] 高度
        # cv2.rectangle(image, (int(float(dataall[item][3])), int(float(dataall[item][4]))), (int(float(dataall[item][3]))+\
        #             int(float(dataall[item][5])),int(float(dataall[item][4]))+int(float(dataall[item][6]))), (0, 0, 255), 3)
        # cv2.imwrite(imapath+'/test2017/' + na + '.jpg', image)  # 带小数的str需先转为float才能转为int
        x = (float(dataall[item][3]) + float(dataall[item][5]) / 2) / image.shape[1]
        y = (float(dataall[item][4]) + float(dataall[item][6]) / 2) / image.shape[0]
        w = float(dataall[item][5]) / image.shape[1]
        h = float(dataall[item][6]) / image.shape[0]
        s = str(int(dataall[item][1]) - 1) + ' ' + str('%.6f' % x) + ' ' + str('%.6f' % y) + \
            ' ' + str('%.6f' % w) + ' ' + str('%.6f' % h)  # 将cub的boundingbox转换为yolov5格式

        if dataall[item][2] == '1':  # 划分训练集验证集，shutil.copy（a，b）为复制图片，a为原路径，b为目标路径（带名字则自动重命名）
            with open(labpath + '/train/{}.txt'.format(na), 'w') as lab:
                lab.write(s)  # 写入文件 已存在就覆盖，没有就生成
            shutil.copy(path + '/images/' + dataall[item][0], imapath + '/train/' + na + '.jpg')

        elif dataall[item][2] == '0':
            with open(labpath + '/val/{}.txt'.format(na), 'w') as lab:
                lab.write(s)
            shutil.copy(path + '/images/' + dataall[item][0], imapath + '/val/' + na + '.jpg')


if __name__ =='__main__':
    # get_yaml()
    get_alllab()