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




