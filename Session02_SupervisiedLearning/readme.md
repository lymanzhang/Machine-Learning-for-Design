## Session02 Supervisied Learning 监督学习

### 线性回归

本章介绍用线性模型处理回归问题。从简单问题开始，先处理一个响应变量和一个解释变量的一元问题。然后，我们介绍多元线性回归问题（multiple linear regression），线性约束由多个解释变量构成。紧接着，我们介绍多项式回归分析（polynomial regression问题），一种具有非线性关系的多元线性回归问题。最后，我们介绍如果训练模型获取目标函数最小化的参数值。在研究一个大数据集问题之前，我们先从一个小问题开始学习建立模型和学习算法。

#### 一元线性回归

上一章我们介绍过在监督学习问题中用训练数据来估计模型参数。训练数据由解释变量的历史观测值和对应的响应变量构成。模型可以预测不在训练数据中的解释变量对应的响应变量的值。回归问题的目标是预测出响应变量的连续值。本章我们将学习一些线性回归模型，后面会介绍训练数据，建模和学习算法，以及对每个方法的效果评估。首先，我们从简单的一元线性回归问题开始。

假设你想计算高铁票的价格，可以用机器学习方法建一个线性回归模型，通过分析行驶里程与高铁票价的数据的线性关系，来预测任意历程的高铁票的价格。我们先用scikitlearn写出回归模型，然后我们介绍模型的用法，以及将模型应用到具体问题中。假设我们查到了部分匹萨的直径与价格的数据，这就构成了训练数据，如下表所示：

|训练样本|特征1\_行驶里程（公里））|特征2\_高铁票价（人民币元）|
|--|--|--|
|1|60|42|
|2|140|72|
|3|210|96|
|4|380|186|
|5|860|360|

我们可以用matplotlib画出图形：

In [1]:
```python
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
```

In [2]:
```python
def generateplt():
    plt.figure()
    plt.title('高铁价格与里程数据',fontproperties=font)
    plt.xlabel('里程（公里）',fontproperties=font)
    plt.ylabel('票价（人民币元）',fontproperties=font)
    plt.axis([0, 1000, 0, 500])
    plt.grid(True)
    return plt

plt = generateplt()
X = [[60], [140], [210], [380], [860]]
y = [[42], [72], [96], [186], [360]]
plt.plot(X, y, 'k.')
plt.show()
```

![img](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/Session02_SupervisiedLearning/img_LinearRegression/%E9%AB%98%E9%93%81%E7%A5%A8%E4%BB%B7.png)

上图中，'x'轴表示行驶里程（公里），'y'轴表示高铁票价（人民币元）。能够看出，高铁票价与行驶里程正相关，这与我们的日常经验也比较吻合，票价自然是越远越贵。下面我们就用scikit-learn来构建模型。

```python
from sklearn.linear_model import LinearRegression
# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)
print('预测一张600公里里程的高铁票的价格：￥%.0f' % model.predict([600])[0])
print('预测一张600公里里程的高铁票的价格：￥%.0f' % model.predict([720])[0])
```

预测一张600公里里程的高铁票的价格：￥260

预测一张600公里里程的高铁票的价格：￥308

一元线性回归假设解释变量和响应变量之间存在线性关系；这个线性模型所构成的空间是一个超平面（hyperplane）。超平面是n维欧氏空间中余维度等于一的线性子空间，如平面中的直线、空间中的平面等，总比包含它的空间少一维。在一元线性回归中，一个维度是响应变量，另一个维度是解释变量，总共两维。因此，其超平面只有一维，就是一条线。

上述代码中sklearn.linear_model.LinearRegression类是一个估计器（estimator）。

估计器依据观测值来预测结果。在scikit-learn里面，所有的估计器都带有fit()和predict()方法。

fit()用来分析模型参数，predict()是通过fit()算出的模型参数构成的模型，对解释变量进行预测获得的值。

因为所有的估计器都有这两种方法，所有scikit-learn很容易实验不同的模型。

LinearRegression类的fit()方法学习下面的一元线性回归模型：

y = a + bx

y表示响应变量的预测值，本例指高铁票价格预测值，是解释变量，本例指高铁行驶里程。截距和相关系数是线性回归模型最关心的事情。下图中的直线就高铁行驶里程与票价的线性关系。用这个模型，你可以计算不同里程的高铁票的价格，600公里票价为￥260元，720公里的票价为￥308元。

```python
plt = generateplt()
plt.plot(X, y, 'k.')
X2 = [[0], [180], [360], [600], [720], [850], [1000]]
model = LinearRegression()
model.fit(X, y)
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
plt.show()
```

