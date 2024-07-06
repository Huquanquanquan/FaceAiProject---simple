# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author :  bingo
# @E-mail :
# @Date   : 2022-12-31 10:28:46
# --------------------------------------------------------
"""


import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from alignment import cv_face_alignment
from pybaseutils import image_utils


def show_landmark_boxes(title, image, landmarks, boxes, color=(0, 255, 0)):
    '''
    显示landmark和boxes
    :param title:
    :param image:
    :param landmarks: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    point_size = 1
    thickness = 8  # 可以为 0 、4、8
    for lm in landmarks:
        for landmark in lm:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, color, thickness * 2)
    for box in boxes:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, color, thickness=thickness)
    image_utils.cv_show_image(title, image, delay=0)


def face_alignment(image, landmarks, vis=False):
    """
    face alignment and crop face ROI
    :param image:输入RGB/BGR图像
    :param landmarks:人脸关键点landmarks(5个点)
    :param vis: 可视化矫正效果
    :return:
    """
    output_size = [112, 112]
    alig_faces = []
    kpts_ref = cv_face_alignment.get_reference_facial_points(square=True, vis=vis)
    # kpts_ref = align_trans.get_reference_facial_points(output_size, default_square=True)
    for landmark in landmarks:
        warped_face = cv_face_alignment.alignment_and_crop_face(np.array(image), output_size, kpts=landmark,
                                                                kpts_ref=kpts_ref)
        alig_faces.append(warped_face)
    if vis:
        for face in alig_faces: image_utils.cv_show_image("face_alignment", face)
    return alig_faces


if __name__ == "__main__":
    image_file = "test4.jpg"
    image = cv2.imread(image_file)
    # face detection from MTCNN
    boxes = np.asarray([ [676.4226804123712, 289.69072164948454, 811.4742268041238, 445.36082474226805] , [676.4226804123712, 289.69072164948454, 811.4742268041238, 445.36082474226805]])

    # landmarks = np.asarray([[[287.86636353, 306.13598633],
    #                          [399.58618164, 272.68032837],
    #                          [374.80252075, 360.95596313],
    #                          [326.71264648, 409.12332153],
    #                          [419.06210327, 381.41421509]]])
    landmarks = np.asarray([[[712.5083333333340, 331.2508333333334],
                             [775.5208333333334, 329.5208333333334,],
                             [748.4375, 371.875],
                             [715.625, 392.1875],
                             [777.0833333333334, 390.1041666666667]],

                            [[408.6816720257235, 172.34726688102896],
                             [484.2443729903537, 170.41800643086816, ],
                             [453.3762057877814, 221.2218649517685],
                             [417.363344051447, 249.5176848874598],
                             [480.06430868167206, 247.58842443729904]],


                            ])
    alig_faces = face_alignment(image, landmarks, vis=True)
    # show bbox and bounding boxes
    show_landmark_boxes("image", np.array(image), landmarks, boxes)
