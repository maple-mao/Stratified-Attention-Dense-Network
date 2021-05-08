import os
from PIL import Image, ImageDraw
# import cv2
import argparse

# use:
# python img_rectangle.py -- mode 1 --img_path ./00.png
# python img_rectangle.py -- mode 2 --img_path G:/learngit/SR_DataSet/Set5/HR

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=int, default='1')
parser.add_argument("--img_path", type=str, default='./baby.png')

args = parser.parse_args()
path0 = './baby.png'
x0, x1, y0, y1 = 20, 20 + 40, 50, 50 + 100  # 坐标点


# 方法1###########################实现图片上画矩形框
# def cv_rectangle(path):
#     img = cv2.imread(path)
#     img_name = path.split('/')[-1].split('.')[-2]
#
#     cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1] 高和宽
#     cv2.imwrite(img_name + "_cropped.png", cropped)
#
#     r_img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
#     # 函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
#     cv2.imwrite(img_name + "_rectangle.png", r_img)
#
#     print(img_name)
#     print('处理完成,保存于当前文件夹')
    # cv2.imshow("draw", r_img)#显示画过矩形框的图片
    # cv2.waitKey(0)#等待按键结束
    # cv2.destroyWindow("draw")#销毁窗口释放内存


#####################################################


# 方法2#############################
def pil_rectangle(path):
    img = Image.open(path)
    cropped = img.crop((x0, y0, x1, y1))  # (left, upper, right, lower)

    draw = ImageDraw.Draw(img)

    draw.rectangle([(x0, y0), (x1, y1)], outline='red')
    # draw.rectangle([(x0-1,y0-1),(x1-1,y1-1)], outline='red') #增加边框线条宽度
    # draw.rectangle([(x0+1,y0+1),(x1+1,y1+1)], outline='red') #

    img_name = path.split('/')[-1].split('.')[-2]
    cropped.save(img_name + "_cropped.png")
    img.save(img_name + "_rectangle.png")

    print(img_name)
    print('处理完成，保存于当前文件夹')
    # img.show()


pil_rectangle(path0)
######################################

# 一个文件夹下批量进行
# def d_rectangle(path):
#     for file in os.listdir(path):
#         if file.split('.')[-1] == 'png':
#             im_path = path + '/' + file
#             cv_rectangle(im_path)
#             # pil_rectangle(im_path)
#
#
# if args.mode == 1:
#     cv_rectangle(args.img_path)
#
# elif args.mode == 2:
#     d_rectangle(args.img_path)

####################################################
# PIL.Image转换成OpenCV格式：
# img = Image.open(path).convert("RGB")#.convert("RGB")可不要，默认打开就是RGB
# img.show()
# #转opencv
# #img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
# img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
# cv2.imshow("OpenCV",img)

# OpenCV转换成PIL.Image格式：
# img = cv2.imread("plane.jpg") # opencv打开的是BRG
# cv2.imshow("OpenCV",img)
# image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# image.show()
#####################################################