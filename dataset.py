import cv2
import glob
import os
import numpy as np
from math import log10, sqrt

#展示未添加噪声的图片
def show_images(images_path, image_size):
    images = []
    path = os.path.join(images_path, '*g')
    files = glob.glob(path) #所有文件全路径列表
    for fl in files:
        image = cv2.imread(fl)  # 通过路径把图读进来
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR) #图片重塑,shape是(256, 256)
        #展示图片
        winname = 'Image ' + str(files.index(fl)+1)
        cv2.imshow(winname, image) #用窗口展示图片
        cv2.waitKey(0) #等待一个字符
        cv2.destroyWindow(winname) #销毁用来展示图片的窗口
        images.append(image)  # 将未经过处理的图片添加到列表中
    return images #没有噪声的(256, 256)灰度图片

#为图片添加椒盐噪声并保存
def add_sp_and_save(images):
    sp_noise_imgs = [] #添加完椒盐噪声之后的图片集
    index = 0 #图片索引
    for image in images: #对于images中保存的所有 没有噪声的(256, 256)灰度图片
        sp_noise_img = salt_and_pepper(image, 0.02)
        index += 1
        cv2.imwrite('Image_Salt_and_Pepper_Noise\\' + str(index) + '.jpg', sp_noise_img)
        sp_noise_imgs.append(sp_noise_img)
    return sp_noise_imgs #返回添加椒盐噪声之后的图片集

#定义添加椒盐噪声的函数,src灰度图片,per噪声比例
def salt_and_pepper(src, per):
    sp_noise_img = np.copy(src) #深拷贝
    sp_noise_num = int(per * src.shape[0] * src.shape[1]) #噪点数量
    for i in range(sp_noise_num):
        #生成闭区间[low,high]上离散均匀分布的整数值;若high=None，则取值区间变为[1,low]
        rand_x = np.random.randint(0, src.shape[0])
        rand_y = np.random.randint(0, src.shape[1])
        #随机将一些点变成白(255),或黑(0)
        if np.random.randint(0, 2) == 0:
            sp_noise_img[rand_x, rand_y] = 0
        else:
            sp_noise_img[rand_x, rand_y] = 255
    return sp_noise_img #添加椒盐噪声之后的图片

#对未添加噪声的图片进行数据预处理
def pretrain_images(images):
    images_data = []
    for image in images:
        image = image.astype(np.float32)  # 将图片像素点数据转化成float32类型
        image = np.multiply(image, 1.0 / 255.0)  # 每个像素点值都在0到255之间,进行归一化
        images_data.append(image)
    return images_data

#展示添加椒盐噪声之后的图片,并进行数据预处理
def show_sp_noise_images(sp_noise_imgs):
    sp_data = []
    #展示添加椒盐噪声之后的图片
    index = 0  # 图片索引
    for image in sp_noise_imgs:
        index += 1
        winname = 'Image ' + str(index)
        cv2.imshow(winname, image) #用窗口展示图片
        cv2.waitKey(0) #等待一个字符
        cv2.destroyWindow(winname) #销毁用来展示图片的窗口
        image = image.astype(np.float32)  # 将图片像素点数据转化成float32类型
        image = np.multiply(image, 1.0 / 255.0)  # 每个像素点值都在0到255之间,进行归一化
        sp_data.append(image)
    return sp_data

#展示添加高斯噪声之后的图片,并进行数据预处理
def show_gaussian_noise_images(gaussian_noise_imgs):
    gaussian_data = []
    #展示添加高斯噪声之后的图片
    index = 0 #图片索引
    for image in gaussian_noise_imgs:
        index += 1
        winname = 'Image ' + str(index)
        cv2.imshow(winname, image)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        gaussian_data.append(image)
    return gaussian_data

#定义添加高斯噪声的函数,src灰度图片,scale噪声标准差
def gaussian(src, scale):
    gaussian_noise_img = np.copy(src) #深拷贝
    noise = np.random.normal(0, scale, size=(src.shape[0], src.shape[1])) #噪声
    noise = noise[:, :, np.newaxis]
    add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #未经检查的图片
    add_noise_and_check += noise
    add_noise_and_check = add_noise_and_check.astype(np.int16)
    # #原来的错误算法
    # # gaussian_noise_num = int(per * src.shape[0] * src.shape[1])
    # # for i in range(gaussian_noise_num):
    # #     rand_x = np.random.randint(0, src.shape[0])
    # #     rand_y = np.random.randint(0, src.shape[1])
    # #     #添加高斯噪声
    # #     gaussian_noise_img[rand_x, rand_y] += int(10 * np.random.randn()) #要添加的噪声数值
    for i in range(len(add_noise_and_check)):
        for j in range(len(add_noise_and_check[0])):
            if add_noise_and_check[i][j] > 255:
                add_noise_and_check[i][j] = 255
            elif add_noise_and_check[i][j] < 0:
                add_noise_and_check[i][j] = 0
    '''
    uint8是无符号整数,0到255之间
    0黑,255白
    256等价于0,-1等价于255
    每256个数字一循环
    '''
    gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
    return gaussian_noise_img #返回添加了高斯噪声之后的图片

#为图片添加高斯噪声并保存
def add_gaussian_and_save(images):
    gaussian_noise_imgs = [] #添加完高斯噪声之后的图片集
    index = 0 #图片索引
    for image in images:
        gaussian_noise_img = gaussian(image, 20)
        index += 1
        cv2.imwrite('Image_Gaussian_Noise\\' + str(index) + '.jpg', gaussian_noise_img)
        gaussian_noise_imgs.append(gaussian_noise_img)
    return gaussian_noise_imgs #返回添加高斯噪声之后的图片集

#将(9, 256, 256)的数据集切分成8*8的patch块
#原始数据和噪音数据都使用这个方法进行patch块切分
def image_data_patch(data):
    image_data = np.copy(data) #将数据载入
    patch_size = 8 #patch块边长
    #shape=(9, 62001, 8, 8),dtype=np.float32
    patches = np.zeros(shape=(image_data.shape[0],
                              int((image_data.shape[1] - patch_size + 1) ** 2),
                              patch_size,
                              patch_size),
                       dtype=np.float32)
    for image_count in range(len(data)): #所有9张图片
        number = 0 #每一张图片当前patch块数量
        image = data[image_count] #其中一张图片(256, 256)
        for row in range(0, len(image) - 7, 1): #所有行,每1个格标注一次
            for col in range(0, len(image[0]) - 7,  1): #所有列
                patches[image_count][number] = image[row:row + 8, col:col + 8]
                number += 1
    return patches

#清空一个文件夹中所有文件
def remove_files(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

# 计算和原图之间的误差
#要求的shape是(256, 256, 1)
def cal_diff(noise_img, img):
    return np.sqrt(np.sum((noise_img - img) ** 2))

#psnr评分,PIXEL_MAX是最大像素点
def psnr(img1, img2, PIXEL_MAX=255):
    if PIXEL_MAX == 255:
        mse = np.mean( (img1/1.0 - img2/1.0) ** 2 )
    elif PIXEL_MAX == 1:
        mse = np.mean( (img1/255.0 - img2/255.0) ** 2)
    return 20 * log10(PIXEL_MAX / sqrt(mse))

#保存图片
def save_image(image, folder_name, image_name):
    cv2.imwrite(folder_name + image_name, image)