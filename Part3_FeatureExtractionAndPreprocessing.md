## 特征提取与处理

上一章案例中的解释变量都是数值，比如匹萨的直接。而很多机器学习问题需要研究的对象可能是分类变量、文字甚至图像。本章，我们介绍提取这些变量特征的方法。这些技术是数据处理的前提 - 序列化，更是机器学习的基础，影响到本书的所有章节。

### 分类变量特征提取

许多机器学习问题都有分类的、标记的变量，不是连续的。例如，一个应用是用分类特征比如工作地点来预测工资水平。分类变量通常用独热编码（One-of-K or One-Hot Encoding），通过二进制数来表示每个解释变量的特征。

例如，假设city变量有三个值：New York, San Francisco, Chapel Hill。独热编码方式就是用三位二进制数，每一位表示一个城市。

scikit-learn里有DictVectorizer类可以用来表示分类特征：

```python
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [{'city': 'New York'},{'city': 'San Francisco'}, {'city': 'Ch
apel Hill'}]
print(onehot_encoder.fit_transform(instances).toarray())
```

[[ 0. 1. 0.]
[ 0. 0. 1.]
[ 1. 0. 0.]]

会看到，编码的位置并不是与上面城市一一对应的。第一个city编码New York是[ 0. 1. 0.]，用第二个元素为1表示。相比用单独的数值来表示分类，这种方法看起来很直观。New York, San Francisco, Chapel Hill可以表示成1，2，3。数值的大小没有实际意义，城市并没有自然数顺序。

## 文字特征提取

很多机器学习问题涉及自然语言处理（NLP），必然要处理文字信息。文字必须转换成可以量化的特征向量。下面我们就来介绍最常用的文字表示方法：词库模型（Bag-of-words model）。

### 词库表示法

词库模型是文字模型化的最常用方法。对于一个文档（document），忽略其词序和语法，句法，将其仅仅看做是一个词集合，或者说是词的一个组合，文档中每个词的出现都是独立的，不依赖于其他词是否出现，或者说当这篇文章的作者在任意一个位置选择一个词汇都不受前面句子的影响而独立选择的。词库模型可以看成是独热编码的一种扩展，它为每个单词设值一个特征值。词库模型依据是用类似单词的文章意思也差不多。词库模型可以通过有限的编码信息实现有效的文档分类和检索。

一批文档的集合称为文集（corpus）。让我们用一个由两个文档组成的文集来演示词库模型：

```python
corpus = ['UNC played Duke in basketball','Duke lost the basketball game']
```

文集包括8个词：UNC, played, Duke, in, basketball, lost, the, game。文件的单词构成词汇表（vocabulary）。词库模型用文集的词汇表中每个单词的特征向量表示每个文档。我们的文集有8个单词，那么每个文档就是由一个包含8位元素的向量构成。构成特征向量的元素数量称为维度（dimension）。用一个词典（dictionary）来表示词汇表与特征向量索引的对应关系。
在大多数词库模型中，特征向量的每一个元素是用二进制数表示单词是否在文档中。例如，第一个文档的第一个词是UNC，词汇表的第一个单词是UNC，因此特征向量的第一个元素就是1。词汇表的最后一个单词是game。第一个文档没有这个词，那么特征向量的最后一个元素就是0。CountVectorizer类会把文档全部转换成小写，然后将文档词块化（tokenize）。文档词块化是把句子分割成词块（token）或有意义的字母序列的过程。词块大多是单词，但是他们也可能是一些短语，如标点符号和词缀。CountVectorizer类通过正则表达式用空格分割句子，然后抽取长度大于等于2的字母序列。scikit-learn实现代码如下：

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['UNC played Duke in basketball','Duke lost the basketball game']
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

[[1 1 0 1 0 1 0 1]  
[1 1 1 0 1 0 1 0]]  
{'unc': 7, 'played': 5, 'game': 2, 'in': 3, 'basketball': 0, 'the': 6, 'duke': 1, 'lost': 4}

让我们再增加一个文档到文集里：

corpus = ['UNC played Duke in basketball','Duke lost the basketball game','I ate a sandwich'] 
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

[[0 1 1 0 1 0 1 0 0 1]
[0 1 1 1 0 1 0 0 1 0]
[1 0 0 0 0 0 0 1 0 0]]
{'unc': 9, 'played': 6, 'game': 3, 'in': 4, 'ate': 0, 'basketball': 1, 'the': 8, 'sandwich': 7, 'duke': 2, 'lost': 5}

通过CountVectorizer类可以得出上面的结果。词汇表里面有10个单词，但a不在词汇表里面，是因为a的长度不符合CountVectorizer类的要求。

对比文档的特征向量，会发现前两个文档相比第三个文档更相似。如果用欧氏距离（Euclidean distance）计算它们的特征向量会比其与第三个文档距离更接近。两向量的欧氏距离就是两个向量欧氏范数（Euclidean norm）或L2范数差的绝对值：

d = ||x<sub>0</sub> - x<sub>1</sub>||

向量的欧氏范数是其元素平方和的平方根：

||x|| = sqrt((x<sub>1</sub>)<sup>2</sup> + (x<sub>2</sub>)<sup>2</sup> + (x<sub>3</sub>)<sup>2</sup> + ... + (x<sub>n</sub>)<sup>2</sup>)

scikit-learn里面的euclidean_distances函数可以计算若干向量的距离，表示两个语义最相似的文档其向量在空间中也是最接近的。


```python
from sklearn.metrics.pairwise import euclidean_distances
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))
```
文档0与文档1的距离[[ 2.44948974]]
文档0与文档2的距离[[ 2.64575131]]
文档1与文档2的距离[[ 2.64575131]]

如果我们用新闻报道内容做文集，词汇表就可以用成千上万个单词。每篇新闻的特征向量都会有成千上万个元素，很多元素都会是0。体育新闻不会包含财经新闻的术语，同样文化新闻也不会包含财经新闻的术语。有许多零元素的高维特征向量成为稀疏向量（sparse vectors）。

用高维数据可以量化机器学习任务时会有一些问题，不只是出现在自然语言处理领域。第一个问题就是高维向量需要占用更大内存。NumPy提供了一些数据类型只显示稀疏向量的非零元素，可以有效处理这个问题。

第二个问题就是著名的维度灾难（curse of dimensionality，Hughes effect），维度越多就要求更大的训练集数据保证模型能够充分学习。如果训练样本不够，那么算法就可以拟合过度导致归纳失败。

下面，我们介绍一些降维的方法。在第7章，PCA降维里面，我们还会介绍用数值方法降维。

## 停用词过滤
特征向量降维的一个基本方法是单词全部转换成小写。这是因为单词的大小写一般不会影响意思。而首字母大写的单词一般只是在句子的开头，而词库模型并不在乎单词的位置和语法。

另一种方法是去掉文集常用词。这里词称为停用词（Stop-word），像a，an，the，助动词do，be，will，介词on，around，beneath等。停用词通常是构建文档意思的功能词汇，其字面意义并不体现。CountVectorizer类可以通过设置stop_words参数过滤停用词，默认是英语常用的停用词。

```python
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'] 
vectorizer =
CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

[[0 1 1 0 0 1 0 1]
[0 1 1 1 1 0 0 0]
[1 0 0 0 0 0 1 0]]
{'unc': 7, 'played': 5, 'game': 3, 'ate': 0, 'basketball': 1, ' sandwich': 6, 'duke': 2, 'lost': 4}

这样停用词就没有了，前两篇文档依然相比其与第三篇的内容更接近。

### 词根还原与词形还原

停用词去掉之后，可能还会剩下许多词，还有一种常用的方法就是词根还原（stemming ）与词形还原（lemmatization）。

特征向量里面的单词很多都是一个词的不同形式，比如jumping和jumps都是jump的不同形式。词根还原与词形还原就是为了将单词从不同的时态、派生形式还原。

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['He ate the sandwiches', 'Every sandwich was eaten by him'] 
vectorizer = CountVectorizer(binary=True, stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

[[1 0 0 1]
[0 1 1 0]]
{'sandwich': 2, 'sandwiches': 3, 'ate': 0, 'eaten': 1}

这两个文档意思差不多，但是其特征向量完全不同，因为单词的形式不同。两个单词都是有一个动词eat和一个sandwich，这些特征应该在向量中反映出来。词形还原就是用来处理可以表现单词意思的词元（lemma）或形态学的词根（morphological root）的过程。词元是单词在词典中查询该词的基本形式。词根还原与词形还原类似，但它不是生成单词的形态学的词根。而是把附加的词缀都去掉，构成一个词块，可能不是一个正常的单词。词形还原通常需要词法资料的支持，比如WordNet和单词词类（part of speech）。词根还原算法通常需要用规则产生词干（stem）并操作词块，不需要词法资源，也不在乎单词的意思。

让我们分析一下单词gathering的词形还原：

```python
corpus = ['I am gathering ingredients for the sandwich.','There were many wizards at the gathering.']
```

第一句的gathering是动词，其词元是gather。后一句的gathering是名词，其词元是gathering。我们用Python的NLTK（Natural Language Tool Kit）
(http://www.nltk.org/install.html)库来处理。

```python
import nltk
nltk.download()
```

showing info http://www.nltk.org/nltk_data/

True

NLTK的WordNetLemmatizer可以用gathering的词类确定词元。

```python
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))
```

gather
gathering

我们把词形还原应用到之前的例子里：

```python
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
wordnet_tags = ['n', 'v']
corpus = ['He ate the sandwiches','Every sandwich was eaten by him']
stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])
```

Stemmed: [['He', 'ate', 'the', 'sandwich'], ['Everi', 'sandwich', 'wa', 'eaten', 'by', 'him']]

```python
def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
```

Lemmatized: [['He', 'eat', 'the', 'sandwich'], ['Every', 'sandwich', 'be', 'eat', 'by', 'him']]

通过词根还原与词形还原可以有效降维，去掉不必要的单词形式，特征向量可以更有效的表示文档的意思。

### 带TF-IDF权重的扩展词库
前面我们用词库模型构建了判断单词是个在文档中出现的特征向量。这些特征向量与单词的语法，顺序，频率无关。不过直觉告诉我们文档中单词的频率对文档的意思有重要作用。一个文档中某个词多次出现，相比只出现过一次的单词更能体现反映文档的意思。现在我们就将单词频率加入特征向量，然后介绍由词频引出的两个问题。

我们用一个整数来代码单词的频率。代码如下：

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

[[2 1 3 1 1]]
{'wizard': 4, 'transfigured': 3, 'sandwich': 2, 'dog': 1, 'ate': 0}

结果中第一行是单词的频率，dog频率为1，sandwich频率为3。注意和前面不同的是，binary=True没有了，因为binary默认是False，这样返回的是词汇表的词频，不是二进制结果[1 1 1 1 1]。这种单词频率构成的特征向量为文档的意思提供了更多的信息，但是在对比不同的文档时，需要考虑文档的长度。

很多单词可能在两个文档的频率一样，但是两个文档的长度差别很大，一个文档比另一个文档长很多倍。scikit-learn的TfdfTransformer类可以解决这个问题，通过对词频（term frequency）特征向量归一化来实现不同文档向量的可比性。默认情况下，TfdfTransformer类用L2范数对特征向量归一化：

