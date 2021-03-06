## 从线性回归到逻辑回归

在第2章，线性回归里面，我们介绍了一元线性回归，多元线性回归和多项式回归。这些模型都是广义线性回归模型的具体形式，广义线性回归是一种灵活的框架，比普通线性回归要求更少的假设。这一章，我们讨论广义线性回归模型的具体形式的另一种形式，逻辑回归（logistic regression）。

和前面讨论的模型不同，逻辑回归是用来做分类任务的。分类任务的目标是找一个函数，把观测值匹配到相关的类和标签上。学习算法必须用成对的特征向量和对应的标签来估计匹配函数的参数，从而实现更好的分类效果。在二元分类（binary classification）中，分类算法必须把一个实例配置两个类别。二元分类案例包括，预测患者是否患有某种疾病，音频中是否含有人声，杜克大学男子篮球队在NCAA比赛中第一场的输赢。多元分类中，分类算法需要为每个实例都分类一组标签。本章，我们会用逻辑回归来介绍一些分类算法问题，研究分类任务的效果评价，也会用到上一章学的特征抽取方法。
