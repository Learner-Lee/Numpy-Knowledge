import numpy as np

# 创建数组
print(np.array([1, 2, 3, 4, 5]))

# 创建全零数组
print(np.zeros((3, 2)))

# 创建全一数组
print(np.ones((2, 4)))

# 获取数组的尺寸
a = np.zeros((3, 2))
print(a.shape)  # （行，列）

# 创建递增或递减的数组,类似range
print(np.arange(3, 7))

# 介于两个数之间，等间距的数
print(np.linspace(0, 1, 5))

# 生成随机数组
print(np.random.rand(2, 4))

# 数组的数据类型
a = np.zeros((3, 2))
print(a.dtype)

# 指定数组的数据类型
a = np.zeros((3, 2), dtype=np.int32)
print(a.dtype)

# 转换数据类型
a = np.zeros((3, 2))
b = a.astype(int)
print(b.dtype)