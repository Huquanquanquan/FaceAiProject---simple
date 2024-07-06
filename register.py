# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os
import argparse
from configs import configs
from tkinter import Tk, filedialog
import shutil
import ui2
import ui3

def echo(param):
    return bool(param)


def parse_opt():
    # portrait = "./data/database/portrait"  # 人脸肖像图像路径
    # database = os.path.join(os.path.dirname(image_dir), "database.json")
    portrait = configs.portrait  # 人脸肖像图像路径
    database = configs.database  # 存储人脸数据库特征路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--portrait', type=str, default=portrait, help='人脸数据库目录')
    parser.add_argument('--database', type=str, default=database, help='存储人脸数据库特征路径')
    opt = parser.parse_args()
    print(opt)

    return opt

    # 面向结果
def check_error_image(portrait_dir):
    error_image_file = os.path.join(portrait_dir, "error-image.jpg")
    if os.path.exists(error_image_file):
        return echo(False)
    else:
        return echo(True)

def register():
    """
    注册人脸，生成人脸数据库
    portrait：人脸数据库图片目录，要求如下：
              (1) 图片按照[ID-XXXX.jpg]命名,如:张三-image.jpg，作为人脸识别的底图
              (2) 人脸肖像照片要求五官清晰且正脸的照片，不能出现多个人脸的情况
    @return:
    """
    # opt = parse_opt()
    portrait = configs.portrait  # 人脸肖像图像路径
    database = configs.database  # 存储人脸数据库特征路径

    # 避免循环导入
    from core.face_recognizer import FaceRecognizer

    fr = FaceRecognizer(database=database)
    # 生成人脸数据库
    fr.create_database(portrait=portrait, vis=False)
    # 测试人脸识别效果
    # fr.detect_image_dir(test_dir, vis=True)
    check_error_image(portrait)

def my_register(portrait, database, local_load=False, success=None):
    fr = FaceRecognizer(database=database, local_load=local_load)
    # 生成人脸数据库
    fr.create_database(portrait=portrait, vis=False)



##################################################################


# 添加图片到portait中，实现随意添加
def register2():
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title='选择人脸肖像图像', filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # 提取文件名和扩展名
        original_file_name = os.path.basename(file_path)  # 提取文件名部分
        file_name, file_ext = os.path.splitext(original_file_name)

        # 测试断点
        # print(file_name)
        # print(file_ext)

        # 构建新文件名
        new_file_name = f"{file_name}-image{file_ext}"
        #print(new_file_name)

        # 目标文件夹路径
        target_dir = os.path.join(os.path.dirname(configs.portrait), "portrait")

        # 构建完整的目标文件路径
        target_file_path = os.path.join(target_dir, new_file_name)

        # 复制选中的文件到目标文件路径，并重命名为新文件名
        shutil.copy(file_path, target_file_path)


#获取portait中的文件数量
def getnum():
    portrait_dir = "./data/database/portrait"  # 获取 portrait 文件夹路径，假设从 configs 中获取
    if not os.path.exists(portrait_dir):
        print(f"Error: {portrait_dir} 不存在")
        return -1

    file_count = len(os.listdir(portrait_dir))
    #print(file_count)
    return file_count


if __name__ == "__main__":
    #register2()
    getnum()
    #check_error_image(portrait_dir)