from PIL import Image
import numpy as np

# 导入图片
im = Image.open('头像.jpg')
im.show()

# 将图片转化为数组
im = np.array(im)
print(im.shape)

# 访问像素点颜色
print(im[100, 100])

# 提取红色分量
im_r = im[:, :, 0]
Image.fromarray(im_r).show()

# 将图片按比例混合
im1 = np.array(Image.open('4.jpg'))
im2 = np.array(Image.open('5.jpg'))
im_blend = im1 * 0.4 + im2 * 0.6
im_blend = im_blend.astype(np.uint8)
Image.fromarray(im_blend).show()

# 反转图片
im_flipped = im[::-1,:,:]
Image.fromarray(im_flipped).show()

# 裁剪图片
im_cropped = im1[40:540,400:900,:]
Image.fromarray(im_cropped).show()