# coding=utf8
# 用matplotlib绘制一个柱状图分析3部电影3天的票房。
import matplotlib.pyplot as plt
import numpy as np
# 准备
real_names = ["人在囧途", "阿甘正传", "熊出没"]
real_num1 = [5453, 7548, 6543]  # 人在囧途3天票房数据
real_num2 = [1840, 4013, 3421]  # 阿甘正传3天票房数据
real_num3 = [1080, 1673, 2342]  # 熊出没3天票房数据

# 绘制
plt.figure(figsize=(10, 5))
x = np.arange(3)
bar_width = 0.25
plt.bar(x - bar_width, real_num1, bar_width, label=real_names[0], color='#ff6b6b')
plt.bar(x, real_num2, bar_width, label=real_names[1], color='#4ecdc4')
plt.bar(x + bar_width, real_num3, bar_width, label=real_names[2], color='#45b7d1')
plt.xticks(x, ['第一天', '第二天', '第三天'])
plt.yticks(np.arange(-2000, 10000, 1000))
plt.grid(axis='y', linestyle='--')
plt.title('3部电影3天票房对比')
plt.xlabel('天数')
plt.ylabel('票房(万)')
plt.legend(title='电影名称')
plt.show()