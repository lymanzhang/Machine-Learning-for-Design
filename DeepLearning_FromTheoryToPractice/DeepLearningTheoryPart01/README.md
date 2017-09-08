## Deep Neural Networks. Theory. Part 1.

[!Deep Neural Network](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Deep%20Neural%20Network.png)

Deep Neural Network

This time we will talk about Artificial Intelligence. Deep Neural Network — is a part of Machine Learning techniques stack. Briefly — it simulates brain neurons behavior to make a decision based on network’s “experience”.

So let’s dive into Deep Learning. We have 3 layers on the image above. Left one is an input layer, right one — output layer. All in-between — hidden layers. There can be any number of hidden layers, this is why neural network is deep.

To make neural network generate predictions it should be trained. What does it mean? DNN looks like a graph. Each graph knob(perceptron) is an action. Each perceptron has inputs and outputs. All operations inside a perceptron happen only on values from input and go right to the next perceptron according to the output connections. No side effects or magic. Each connection can apply an operation on a passing value. It can be multiplication with some weight. To train a network we should configure weights in a such way that result is correct as much as possible. So our starting point is:

- Let’s define our goal — that one output from the right-most perceptron — imagine we have a self-driving vehicle and DNN should decide to move forward or maybe stop.

- Incoming values should pass into the first layer. Let’s assume this values are traffic signs only. Stop, speed limit etc.

- We define only one hidden layer for the sake of simplicity. This layer will decide based on it’s inputs what is more preferable for us from incoming data.

- All connections have it’s own weights. They can be positive or negative.

- We have an initial dataset — training dataset. It contains a lot of different combinations of inputs(features) and correct outputs.

### DATA
To be sure our prediction is correct let’s take a part from our dataset, 10% for example. This 10% will be a test dataset, we will not use it to train the network, but will use to test it after a training. Other 90% is a training dataset.

One more thing. Always check a dataset manually before using it. There are always human mistakes, odd outliers, weird issues from sensors etc. You need to be sure your dataset is representing incoming data well. Also don’t take as much data as you can, you will simply overfit your predictions, result will be fine on a training set, but not on a real data. And don’t take to less number of values for a training set. Otherwise your network will not predict well at all. Same with a number of features. Don’t take too small number of features which will not describe your overall population but don’t take too much, or your network will become too sensitive. Just sit in the middle.

### PERCEPTRONS IN ACTION

[!Perceptron in Action](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Perceptron%20in%20Action.png)

Perceptron in Action

So how do we use all that weights? The simplest approach is a linear combination of all incoming weights multiplied by their inputs. We will use w for an individual weight and W for a matrix of weights. We express linear combination as a sum:

[!sum](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/sum.png)

sum

Then we should emit an output signal from a perceptron. We can use simple step function as a trigger — Heaviside Step Function — we call it Activation Function

[!Heaviside step function](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Heaviside%20step%20function.png)

Heaviside step function

[!Yet another way to define Heaviside step function](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Yet%20another%20way%20to%20define%20Heaviside%20step%20function.png)

Yet another way to define Heaviside step function

So when we have a sum of weights*inputs less than zero — perceptron should say No. If more than 1 — Yes. That’s all we need from perceptron for now. Yes or No. But what if we need to move a step to bigger or smaller values? We shoud add Bias. This is just a value which will move our function higher or lower. Now Perceptron formula is. It looks like a linear regression, yeah?

[!Perceptron Formula](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Perceptron%20Formula.png)

Perceptron Formula

There are more different activation functions like tanh, sigmoid, softmax etc. Let’s look at sigmoid. We will use. Why? Because in compare to Heaviside function we can take a derivative from sigmoid.

[!Sigmoid function](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/Sigmoid%20function.png)

Sigmoid function

### UNKNOWN WEIGHTS

Probably you want to say — we don’t know weights. How do we calculate all this stuff? Yes, we need to define a metrics of how wrong the prediction is. A common metrics is a Sum of Squared Error (SSE)

[!SSE](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/SSE.png)

SSE

Where y — is a real value and y-hat — is a prediction, j — all output units and mu — all data points. Formula is quite easy. We find the difference between real value and predicated value, then square it, then sum it. That’s all. Theoretically we can take an absolute value, but square is better — all errors are positive and larger errors penalized out than lower errors.

We remember that prediction is a function of sum of multiplications of weights and inputs(say, depends on weights), than finally error is

[!SSE2](https://github.com/lymanzhang/Machine-Learning-for-Design/blob/master/DeepLearning_FromTheoryToPractice/DeepLearningTheoryPart01/images/SSE2.png)

and it should be as low as possible.
