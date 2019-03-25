import dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset.remove_files('./graphs')
dataset.remove_files('./encode_model')
dataset.remove_files('./Image_Gaussian_Denoising')
dataset.remove_files('./Image_Salt_and_Pepper_Denoising')



print('开始数据预处理...')

#读取并展示未加噪声的图片
path_images_without_noise = 'Image_Denoising'
#指定将图片resize成256*256
images = dataset.show_images(path_images_without_noise, 256)
images = np.array(images)
#检验一下导入的是否是灰度图片
images_channel_1 = np.copy(images[:, :, :, 0])
images_channel_2 = np.copy(images[:, :, :, 1])
images_channel_3 = np.copy(images[:, :, :, 2])

print('开始检验数据...')
assert (images_channel_1 == images_channel_2).all() == True
assert (images_channel_2 == images_channel_3).all() == True
print('完成检验数据...')

#适合训练的原始数据形状(9, 256, 256, 1)
images = np.copy(images_channel_1[:, :, :, np.newaxis])

#对原始数据进行数据预处理
images_data = dataset.pretrain_images(images)
images_data = np.array(images_data) #(9, 256, 256, 1)
print('原始数据shape : ', images_data.shape)

#为图片添加椒盐噪声并保存
sp_noise_imgs = dataset.add_sp_and_save(images)

#展示添加椒盐噪声之后的图片,并返回数据预处理之后的图片
#经过了归一化:数据值在0到1之间
sp_data = dataset.show_sp_noise_images(sp_noise_imgs)
sp_data = np.array(sp_data) #(9, 256, 256, 1)
print('椒盐噪声数据shape : ', sp_data.shape)

#为图片添加高斯噪声并保存
gaussian_noise_imgs = dataset.add_gaussian_and_save(images)

#展示添加高斯噪声之后的图片,并返回数据预处理之后的图片
#经过了归一化:数据值在0到1之间
gaussian_data = dataset.show_gaussian_noise_images(gaussian_noise_imgs)
gaussian_data = np.array(gaussian_data) #(9, 256, 256, 1)
print('高斯噪声数据shape : ', gaussian_data.shape)

print('完成数据预处理...')



print('开始展示图片...')

#看看原图的一张图片是什么样的
plt.figure()
plt.imshow(np.squeeze(images_data[0]), cmap='gray')
plt.title('原图')
plt.show()
print(images_data[0].shape) #看看图片的形状

#看看添加高斯噪声之后的一张图片是什么样的
plt.figure()
plt.imshow(np.squeeze(gaussian_data[0]), cmap='gray')
gaussian_psnr_score = dataset.psnr(gaussian_data[0], images_data[0], PIXEL_MAX=1)
plt.title('添加了高斯噪声的图像\npsnr_score : ' + str(round(gaussian_psnr_score, 2)))
plt.show()
print(gaussian_data[0].shape) #看看图片的形状

#看看添加椒盐噪声之后的一张图片是什么样的
plt.figure()
plt.imshow(np.squeeze(sp_data[0]), cmap='gray')
sp_psnr_score = dataset.psnr(sp_data[0], images_data[0], PIXEL_MAX=1)
plt.title('添加了椒盐噪声的图像\npsnr_score : ' + str(round(sp_psnr_score, 2)))
plt.show()
print(sp_data[0].shape) #看看图片的形状

print('完成展示图片...')



print('开始训练...')

#lrelu函数
def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

#输入到网络的数据
#图片28*28,color_channel是1
inputs_ = tf.placeholder(tf.float32, [None, 256, 256, 1])
#原始数据
targets_ = tf.placeholder(tf.float32, [None, 256, 256, 1])
learning_rate = tf.placeholder(tf.float32)


### Encoder
'''
filters: 32个卷积核
kernel_size: 卷积核大小
strides: 长宽步长都是1
padding: 边缘填充
use_bias: 在卷积中添加偏置
activation: 激活函数
'''
with tf.name_scope('en-convolutions'):
    conv1 = tf.layers.conv2d(inputs_, filters=32,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             use_bias=True,
                             activation=lrelu, )
# 卷积时padding是SAME,所以仍然是28*28
# now 256x256x32
'''
将第一卷积层的结果输入进来
pool_size: 池化窗口
strides: 步长
'''
with tf.name_scope('en-pooling'):
    maxpool1 = tf.layers.max_pooling2d(conv1,
                                       pool_size=(2, 2),
                                       strides=(2, 2), )

# now 128x128x32
'''
第二卷积层
参数和第一卷积层相似
'''
with tf.name_scope('en-convolutions'):
    conv2 = tf.layers.conv2d(maxpool1,
                             filters=32, #32
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             use_bias=True,
                             activation=lrelu, )

'''
第二池化层
和上面的池化层参数基本一样
'''
#  now 128x128x64
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(conv2,
                                      pool_size=(2, 2),
                                      strides=(2, 2), )
#经过pooling层之后,大小是64x64x64,Encoding完成
# now 64x64x64

#编码之后进行解码
### Decoder
with tf.name_scope('decoder'):
    conv3 = tf.layers.conv2d(encoded,
                             filters=32, #32
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             use_bias=True,
                             activation=lrelu)
    #  现在是64x64x64
    #开始反卷积
    upsamples1 = tf.layers.conv2d_transpose(conv3,
                                            filters=32,
                                            kernel_size=3,
                                            padding='SAME',
                                            strides=2,
                                            name='upsample1')
    # now 128x128x32
    #反卷积
    upsamples2 = tf.layers.conv2d_transpose(upsamples1,
                                            filters=32,
                                            kernel_size=3,
                                            padding='SAME',
                                            strides=2,
                                            name='upsamples2')
    # now 256x256x32
    #卷积,filter映射成1
    logits = tf.layers.conv2d(upsamples2,
                              filters=1,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              name='logits',
                              padding='SAME',
                              use_bias=True)
    # 此时的数据是 256x256x1
    # 通过sigmoid传递logits以获得重建图像
    decoded = tf.sigmoid(logits, name='recon')
with tf.name_scope('cost'):
    # 定义损失函数和优化器
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=targets_)
    #误差
    cost = tf.reduce_mean(loss)
    tf.summary.scalar('cost', cost)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#class_name噪声种类
#graph_name sess的默认图graph存在哪个文件夹
def train_sp_data(noise_data, graph_name, class_name, epochs):
    if class_name.find('sp') != -1:
        cls = 'sp'
    else:
        cls = 'gaussian'

    # 训练
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        lr = 1e-3
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./' + graph_name, sess.graph) #再graph文件夹下使用默认图

        for e in range(epochs): #每一个epoch
            summary, train_loss, _ = sess.run([merged, cost, opt],
                     feed_dict={inputs_:noise_data, targets_:images_data, learning_rate:lr*(0.997)**e})
            #每个epoch展示一下loss
            print('Epoch : ', e, ' Training Cost : ', train_loss, ' Learning Rate : ', lr*(0.997)**e)
            saver.save(sess, './encode_model/')
            writer.add_summary(summary, e)

        #展示去噪之前和去噪之后的照片
        fig, axes = plt.subplots(nrows=2, ncols=9, sharex=True, sharey=True, figsize=(20, 4))
        in_imgs = np.copy(noise_data) #原图
        #去噪后重塑的图片
        reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs, targets_: images_data})

        for is_noise_or_recon, (images, row) in enumerate(zip([in_imgs, reconstructed], axes)):
            for index, (img, ax) in enumerate(zip(images, row)):
                if is_noise_or_recon == 0: #说明是原噪声图
                    psnr_score = dataset.psnr(noise_data[index], images_data[index], PIXEL_MAX=1)
                    ax.set_title('noise:' + str(index + 1) + ' psnr_score:' + str(round(psnr_score, 2)))
                else: #说明是还原图
                    psnr_score = dataset.psnr(reconstructed[index], images_data[index], PIXEL_MAX=1)
                    ax.set_title('recon:' + str(index + 1) + ' psnr_score:' + str(round(psnr_score, 2)))
                ax.imshow(img.reshape((256, 256)), cmap='Greys_r')
                #使用opencv保存的图片必须是0到255的uint8
                img_save = np.array(img.reshape((256, 256)) * 255.0, dtype=np.uint8)
                if cls == 'sp': #说明是对椒盐噪声进行去噪
                    dataset.save_image(img_save,
                                       'Image_Salt_and_Pepper_Denoising\\', str(index + 1) + '.jpg')
                else: #说明是对高斯噪声进行去噪
                    dataset.save_image(img_save,
                                       'Image_Gaussian_Denoising\\', str(index + 1) + '.jpg')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.01)
        plt.show()


#使用椒盐噪声进行去噪
train_sp_data(sp_data, 'graphs', 'sp', 1000) #(9, 256, 256, 1)
#使用高斯噪声进行去噪
train_sp_data(gaussian_data, 'graphs', 'gaussian', 1000) #(9, 256, 256, 1)

print('Finish')