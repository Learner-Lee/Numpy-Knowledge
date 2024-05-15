import numpy as np

# 四则运算,将数组同位置的元素进行 加减乘除
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a / b)

# 将两个向量进行点乘运算
print(np.dot(a, b))

# 矩阵乘法运算
a = np.array([[1, 2], [3, 4]])
b = np.array([[4, 5], [7, 8]])
print(a @ b)
print(np.matmul(a, b))

# 求平方根
a = np.array([1, 2, 3])
print(np.sqrt(a))

# 三角函数运算
a = np.array([1, 2, 3])
print(np.sin(a))
print(np.cos(a))

# 对数与指数运算
a = np.array([1, 2, 3])
print(np.log(a))
print(np.power(a, 2))

# 广播
a = np.array([1, 2, 3])
print(a * 5)

# 不同尺寸的数组之间的运算
a = np.array([[1], [10], [20]])
b = np.array([4, 5, 6])
print(a + b)

# 获取最大最小值的数与索引
a = np.array([1, 2, 3, 4, 5])
print(a.min())
print(a.max())

print(a.argmin())
print(a.argmax())

# 求和，平均值，中位数，方差，标准方差
# 一维数组
a = np.array([1, 2, 3, 4, 5])
print(a.sum())
print(a.mean())
print(np.median(a))
print(a.var())
print(a.std())

# 多维数组
a = np.array([[1, 2, 3, 4, 5],
              [5, 6, 7, 8, 9]])
print(a.sum(axis=0))  # 每一列中对应的数相加
print(a.sum(axis=1))  # 每一行中对应的数相加

# 获取数组中的元素
a = np.array([[1, 2, 3, 4, 5],
              [5, 6, 7, 8, 9]])

print(a[0, 1])  # 第一行第二列
# 切片
print(a[0, 0:2])
print(a[0, :])
# 步长
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[0:9:2])
print(a[0:9:3])
print(a[4:1:-2]) # 反转

# 筛选指定的元素
a = np.arange(10)
print(a)
print(a[a < 3])
print(a[(a > 3) & (a % 2 == 0)])
