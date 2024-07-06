from tkinter import filedialog

import cv2 as cv
import time
import tkinter as tk

import os, sys
import shutil
from tkinter import filedialog



# 检测人脸并绘制人脸bounding box
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #  blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval  返回值   # swapRB是交换第一个和最后一个通道   返回按NCHW尺寸顺序排列的4 Mat值
    net.setInput(blob)
    detections = net.forward()  # 网络进行前向传播，检测人脸
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                         8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    return frameOpencvDnn, bboxes


# 网络模型  和  预训练模型
script_dir = os.path.dirname(__file__)
faceProto = os.path.join(script_dir, "../age_gender/models/opencv_face_detector.pbtxt")
faceModel = os.path.join(script_dir, "../age_gender/models/opencv_face_detector_uint8.pb")

ageProto = os.path.join(script_dir, "../age_gender/models/age_deploy.prototxt")
ageModel = os.path.join(script_dir, "../age_gender/models/age_net.caffemodel")

genderProto = os.path.join(script_dir, "../age_gender/models/gender_deploy.prototxt")
genderModel = os.path.join(script_dir, "../age_gender/models/gender_net.caffemodel")

# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['male', 'female']


# 加载网络
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv.dnn.readNet(faceModel, faceProto)


def encode_gender(gender):
    if gender == 'male':
        return 1
    elif gender == 'female':
        return 2
    else:
        return 3


def get_gender(genderPreds):
    gender = genderList[genderPreds[0].argmax()]
    return gender

# 打开一个视频文件或一张图片或一个摄像头
#cap = cv.VideoCapture(0)   #参数为0表示调用本机摄像头

def face_detect(video_dir=None, camera=False):
    # 获取该路径下的所有文件
    if camera:
        cap = cv.VideoCapture(0)
    else:
        files = os.listdir(video_dir)

        # 过滤视频文件
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mkv'))]

        if not video_files:
            return False

        # 获取第一个视频文件的完整路径
        first_video_path = os.path.join(video_dir, video_files[0])
        cap = cv.VideoCapture(first_video_path)

    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)   # 取出box框住的脸部进行检测,返回的是脸部图片
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                   max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
            print("=======", type(face), face.shape)  #  <class 'numpy.ndarray'> (166, 154, 3)
            #
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            print("======", type(blob), blob.shape)  # <class 'numpy.ndarray'> (1, 3, 227, 227)
            genderNet.setInput(blob)   # blob输入网络进行性别的检测
            genderPreds = genderNet.forward()   # 性别检测进行前向传播
            print("++++++", type(genderPreds), genderPreds.shape, genderPreds)   # <class 'numpy.ndarray'> (1, 2)  [[9.9999917e-01 8.6268375e-07]]  变化的值
            gender = gender = get_gender(genderPreds)   # 分类  返回性别类型
            gender_code = encode_gender(gender)
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            #

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(agePreds[0].argmax())  # 3
            print("*********", agePreds[0])   #  [4.5557402e-07 1.9009208e-06 2.8783199e-04 9.9841607e-01 1.5261240e-04 1.0924522e-03 1.3928890e-05 3.4708322e-05]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                       cv.LINE_AA)  # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
            cv.imshow("Age Gender Model", frameFace)

        print("time : {:.3f} ms".format(time.time() - t))

        # 等待按键，设置适当的延迟
        key = cv.waitKey(1)
        if key == 27:  # 如果按下了esc键
            break
    # 释放视频对象和关闭窗口
    cap.release()
    cv.destroyAllWindows()

def select_video(video_dir=True):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title='选择视频文件', filetypes=[("Video files", "*.mp4;*.avi")])

    if file_path:
        # 检查文件扩展名，确保是视频文件
        if file_path.lower().endswith(('.mp4', '.avi')):
            # 复制选中的文件到目标文件夹路径
            shutil.copy(file_path, video_dir)
            return True
        else:
            raise ValueError("请导入视频文件!")

def clear_images():
    script_dir = os.path.dirname(__file__)
    video_dir = os.path.join(script_dir, "../data/video")
    try:
        # 遍历目标文件夹
        for filename in os.listdir(video_dir):
            file_path = os.path.join(video_dir, filename)
            # 确保是文件而不是文件夹
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
                #print(f"已删除文件: {file_path}")
        print(f"成功清空目录: {video_dir}")
    except Exception as e:
        print(f"清空目录时发生错误: {e}")

def open_camera():
    video_dir = "../data/video"
    camera = True
    face_detect(camera=True)

def not_open_camera():

    not_camera = False
    script_dir = os.path.dirname(__file__)
    video_dir = os.path.join(script_dir, "../data/video")

    select_video(video_dir)
    face_detect(video_dir, not_camera)
    clear_images()

if __name__=="__main__":

    #open_camera()
    #not_open_camera()
    pass
    #clear_images()