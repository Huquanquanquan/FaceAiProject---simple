# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""
import os, sys
import shutil
from tkinter import filedialog

import torch
import tkinter as tk

sys.path.insert(0, os.path.dirname(__file__))
import cv2
from core.detection import detector
from core.alignment.face_alignment import face_alignment

from pybaseutils import image_utils, file_utils

import configs



class FaceDetector(object):
    def __init__(self, net_name, input_size, conf_thresh=0.5, nms_thresh=0.3, device = torch.device('cuda:0')):
        self.detector = detector.FaceDetector(net_name,
                                              input_size=input_size,
                                              conf_thresh=conf_thresh,
                                              nms_thresh=nms_thresh,
                                              device=device)

    def detect_face_landmarks(self, bgr, vis=False):
        """
        :param bgr:
        :return:
        """
        bboxes, scores, landms = self.detector.detect_face_landmarks(bgr, vis)
        return bboxes, scores, landms


    def face_alignment(self, image, landmarks):
        """
        :param image: 
        :param landmarks: 
        :return:
        """
        alig_faces = face_alignment(image, landmarks)
        return alig_faces


    def crop_faces_alignment(self, image, bboxes, landmarks, alignment=True):
        face_size = [112, 112]
        if alignment:
            # 人脸数据预处理
            faces = self.face_alignment(image, landmarks)
            # cv2.imshow("image", faces[0]), cv2.waitKey(0)
        else:
            faces = image_utils.get_bboxes_image(image, bboxes, size=tuple(face_size))
        return faces

    def detect_crop_faces(self, bgr_image, alignment=True):
        """
        :param bgr_image:  input rgb-image
        :param face_size: return and scale face size
        :param alignment: True(default) or False
        :return:
        """
        if bgr_image is None:
            print("-----ERROR: input rgb_image is None")
            return None
        bboxes, scores, landmarks = self.detect_face_landmarks(bgr_image)
        # image_processing.show_landmark_boxes("detect", bgr_image, landmarks, bboxes)
        if len(bboxes) == 0:
            print("-----ERROR:no face")
            return None
        faces = self.crop_faces_alignment(bgr_image, bboxes, landmarks, alignment=alignment)
        return faces


    def detect_image_dir(self, image_dir, vis=True):
        image_list = file_utils.get_files_lists(image_dir)
        for image_file in image_list:
            image = cv2.imread(image_file)
            image = image_utils.resize_image(image, size=(None, 640))
            bboxes, scores, landms = self.detector.detect_face_landmarks(image, vis=vis)
            print("bboxes:\n{}\nscores:\n{}\nlandms:\n{}".format(bboxes, scores, landms))


    @staticmethod
    def show_landmark_boxes(win_name, image, bboxes, scores, landms):
        """
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        """
        image = image_utils.draw_landmark(image, landms, vis_id=True)
        image = image_utils.draw_image_bboxes_text(image, bboxes, scores, color=(0, 0, 255))
        cv2.imshow(win_name, image)
        cv2.waitKey(0)

# 添加图片到portait中，实现随意添加
def select_image(image_dir=None):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title='选择人脸肖像图像', filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # 检查文件扩展名，确保是图片文件
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 提取文件名和扩展名
            original_file_name = os.path.basename(file_path)  # 提取文件名部分
            file_name, file_ext = os.path.splitext(original_file_name)
        else:
            raise ValueError("请导入图片文件!")

        # 构建新文件名
        new_file_name = f"{file_name}-image{file_ext}"

        # 目标文件夹路径
        target_dir = image_dir

        # 构建完整的目标文件路径
        target_file_path = os.path.join(target_dir, new_file_name)

        # 复制选中的文件到目标文件路径，并重命名为新文件名
        shutil.copy(file_path, target_file_path)
        return True

def clear_images(image_dir):
    try:
        # 遍历目标文件夹
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            # 确保是文件而不是文件夹
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
        print(f"成功清空目录: {image_dir}")
    except Exception as e:
        print(f"清空目录时发生错误: {e}")

def face_detection(image_dir):
    input_size = [320, None]
    device = "cuda:0"
    detector = FaceDetector(net_name="MTCNN",
                            input_size=input_size,
                            device=device)
    detector.detect_image_dir(image_dir, vis=True)

if __name__ == '__main__':

    image_dir = "../data/test_image"
    #select_image(image_dir)
    face_detection(image_dir)
    '''input_size = [320, None]
    device = "cuda:0"
    detector = FaceDetector(net_name="MTCNN",
                            input_size=input_size,
                            device=device)
    detector.detect_image_dir(image_dir, vis=True)'''
    clear_images(image_dir)
    exit(0)
