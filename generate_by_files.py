import os
import numpy as np
#from scipy.io import scipy
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
from pyts.image import GramianAngularField

# 根据自己的需求修改一下变量的参数
dirname = r"F:\demo"  # 要处理的文件夹路径
savepath = r"F:\demo\data"  # GAF 图片保存的路径
img_sz = 550           # 确定生成的 GAF 图片的大小
method = 'summation'  # GAF 图片的类型， 'summation' 是GASF方法
#method = 'difference'   # 'difference' 是GADF方法


# 以下是 GAF 生成的代码
print("GAF 生成方法：%s，图片大小：%d * %d" % (method, img_sz, img_sz))
img_path = "%s/images" % savepath     # 可视化图片保存的文件夹
data_path = "%s/data_mat" % savepath  # 数据文件保存的文件夹
if not os.path.exists(img_path):
    os.makedirs(img_path)  # 如果文件夹不存在就创建一个
if not os.path.exists(data_path):
    os.makedirs(data_path)  # 如果文件夹不存在就创建一个

print("开始生成...")
print("可视化图片保存在文件夹 %s 中，数据文件保存在文件夹 %s 中。" % (img_path, data_path))
gaf = GramianAngularField(image_size=img_sz, method=method)
img_num = 0  # 计算生成的图片个数

for fname in os.listdir(dirname):
    filename, ext = os.path.splitext(fname)
    if ext != '.csv': continue  # 如果不是 csv 文件则跳过
    img_num += 1

    src_data = pd.read_csv("{}/{}".format(dirname, fname))  #读取数据
    src_data = np.array([src_data["Horizontal_vibration_signals"], src_data["Vertical_vibration_signals"]])  #跳过第一行的标签
    src_data = src_data.T  #转置

    src_data = np.loadtxt("{}/{}".format(dirname, fname), delimiter=",") #第一行没有标签用该代码加载数据

    print(img_num)
    x = src_data.reshape(1, -1)
    img_gaf = gaf.fit_transform(x)

    img_save_path = "%s/%s.png" % (img_path, filename)
    image.imsave(img_save_path, img_gaf[0])  # 保存图片

    data_save_path = "%s/%s.csv" % (data_path, filename)
    np.savetxt(data_save_path, img_gaf[0], delimiter=',')  # 保存数据为 csv 文件

print("生成完成，共处理 %d 个图片。" % img_num) 