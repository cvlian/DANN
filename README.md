# DANN

A Tensorflow implementation of **Domain Adversarial Neural Networks** (DANN, in short), introduced in paper titled [**"Unsupervised Domain Adaptation by Backpropagation"**](https://arxiv.org/pdf/1409.7495.pdf). 

---
## What is Domain Adapatation?

**Domain adaptation** is a field associated with **transfer learning**. The goal of domain adaptation is to address the discrepancy in domain distribution between source and target dataset. When the feature spaces of the two domains are different, your model trained on one domain (source) could be biased toward that domain, eventually could not be fit another domain (target). Especially, when we aim at employing a pre-trained model for the prediction of an **unlabeled data**, it won't guarantee good performance. 

This scenario is quite common in practice. For example, you may train your algorithm on MNIST (source) and evaluate it on MNIST-M (target), a modified version of MNIST. As you can see from the below images, the samples from MNIST-M have both similarities and differences compared to the MNIST sample images. The MNIST-M images keep the original digit shape, so it can be regarded as a domain-independent feature. On the contrary, the non-uniform background, which makes the appearances of MNIST-M samples significantly different from those of MNIST images can be seen as domain-dependent information. For a human, the digits from the target dataset are still clearly distinguishable. For a model trained on MNIST, however, it perceives the given target data in different fashions since the background pixels are no longer constant. The background color brings clouded judgment, consequently, the model performs poorly. To cope with change in the data distribution, a model demands a capability of picking up the domain-independent features beneficial for adapting unseen data. Devising models that can handle this kind of challenge is exactly the problem called domain adaptation.

<p align="center"><img src="./img/idea.png" width="480" height="360"/></p></br>

Perhaps the most famous example of such a methodology is the **Domain-Adversarial Neural Network (DANN)**. This network simultaneously does the two things: **label classification** and the **domain classification**. The main purpose of calculating the domain loss is to force the feature extractor to exclude the domain-dependent features since they are uninformative in a similar problem space, so that the model automatically figure out the essential spots in the feature map. It contains a gradient reversal layer, and this makes sure that samples are mutually indistinguishable for the domain classificatory. So the best model will be the one with minimizing the classification loss while maximizing the domain confusion loss for all samples as possible.

<p align="center"><img src="./img/architecture.png" width="600" height="450"/></p></br>

---
## Preparation

### Installing prerequisites

To install the prerequisite, simply type in the shell prompt the following:

```
$ pip install -r requirements.txt
```

You can use TensorFlow libraries with GPU support (See the guide at [TensorFlow tutorials](https://www.tensorflow.org/guide/gpu?hl=en)) to accelerate your code with CUDA back-end.

### Dataset

We prepared 2 different handwritten digit datasets, one for the source and the other for the target domain:

<img src="./img/data.png" width="600" height="100" />

* [MNIST](http://yann.lecun.com/exdb/mnist/) : A well-known monochrome digit image set to be utilized to build the initial digit classifier.  
* [MNIST-M](http://yaroslav.ganin.net/) : MNIST-M is created by combining MNIST digits with the patches randomly extracted from color photos of BSDS500 as their background.

### Note

Since the size of the MNIST-M dataset is too large to upload here, you want to click this [link](https://drive.google.com/file/d/1tOMWeDpuRFWLJWKswDliMlxl7UkfFaye/view) to download the MNIST-M dataset. Once you obtain `mnistm_data.pkl`, move it to the directory named `dataset`. 

---
## Experiment

The effectiveness of DANN was verified in below. With the different digit image sets, you can test (See `test.ipynb`) how much performance DANN method can improve from the baseline results (source only). Compared to the results (0.966) with target image/label, which can be regarded as the optimal performances, the accuracies of 'source only' model (0.469) are fallen to near half of the previous cases. But, you can see the improvement in accuracies after employing domain adaptation skills.

| Training Mode | Target Accuracy |
| ------ | ------------------ |
| Train on target | 0.966 |
| Source Only | 0.469 |
| DANN | **0.847** |

</br></br>Also, check the t-SNE visualization results on the extracted features, and the effect of adaptation on the distribution of the extracted features. Blue points correspond to the source domain examples, while red ones correspond to the target domain. The adaptation makes the two distributions of features much closer.

<img src="./img/effect.png" width="800" height="300" />
