# Datasets
This document provides information and samples of different datasets available in our shared datasets directory.

Code examples for loading and manipulating the dataseets are provided in the interactive data explorer notebook on PACE: 

```/storage/ice-shared/cs8903onl/lw-batch-selection/datasets/HAAG_BS_DatasetExplorer.ipynb```

## Vision Datasets


### CIFAR
CIFAR-10, CIFAR-100: train/test \
CIFAR-100-LT: training set derived from CIFAR-100 \
Source: torchvision.datasets \
Cite: CIFAR-100, CIFAR-10: Krizhevsky, A. and Hinton, G. [Learning multiple layers of
features from tiny images. Master’s thesis](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Department of Computer Science, University of Toronto, 2009\
Cite CIFAR-LT: Cui, Yin, et al. "[Class-balanced loss based on effective number of samples](https://arxiv.org/pdf/1901.05555)." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

#### CIFAR10
**CIFAR-10 (train)** \
Number of samples: 50000 \
Number of classes: 10 \
Example classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

**CIFAR-10 (test)** \
Number of samples: 10000 \
Number of classes: 10 \
Example classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

**Examples** 
| frog | truck | truck | deer | automobile |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/cifar10/000_frog.png"><img src="./static_samples/cifar10/000_frog.png" width="96"></a> | <a href="./static_samples/cifar10/001_truck.png"><img src="./static_samples/cifar10/001_truck.png" width="96"></a> | <a href="./static_samples/cifar10/002_truck.png"><img src="./static_samples/cifar10/002_truck.png" width="96"></a> | <a href="./static_samples/cifar10/003_deer.png"><img src="./static_samples/cifar10/003_deer.png" width="96"></a> | <a href="./static_samples/cifar10/004_automobile.png"><img src="./static_samples/cifar10/004_automobile.png" width="96"></a> |

#### CIFAR100
**CIFAR-100 (train)** \
Number of samples: 50000 \
Number of classes: 100 \
Example classes: ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle']

**CIFAR-100 (test)** \
Number of samples: 10000 \
Number of classes: 100 \
Example classes: ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle']

**Examples**

| cattle | dinosaur | apple | boy | aquarium fish |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/cifar100/000_cattle.png"><img src="./static_samples/cifar100/000_cattle.png" width="96"></a> | <a href="./static_samples/cifar100/001_dinosaur.png"><img src="./static_samples/cifar100/001_dinosaur.png" width="96"></a> | <a href="./static_samples/cifar100/002_apple.png"><img src="./static_samples/cifar100/002_apple.png" width="96"></a> | <a href="./static_samples/cifar100/003_boy.png"><img src="./static_samples/cifar100/003_boy.png" width="96"></a> | <a href="./static_samples/cifar100/004_aquarium_fish.png"><img src="./static_samples/cifar100/004_aquarium_fish.png" width="96"></a> |

#### CIFAR-100-LT
This "long-tail" subset is dynamically generated to be imbalanced. While the training subset is purposefully balanced in these experiments, the test subset is kept the same (i.e. balanced).  Sample code for generating the unbalanced dataset under different specifications is provided in the data explorer.

**CIFAR-100-LT Train** \
Number of samples: 10847 \
Imbalance ratio: 100 \
Examples per class (first 10): [500, 477, 455, 434, 415, 396, 378, 361, 344, 328] 

**Examples**

| apple | apple | apple | apple | apple |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/cifar100-lt/000_orig26042_apple.png"><img src="./static_samples/cifar100-lt/000_orig26042_apple.png" width="96"></a> | <a href="./static_samples/cifar100-lt/001_orig29211_apple.png"><img src="./static_samples/cifar100-lt/001_orig29211_apple.png" width="96"></a> | <a href="./static_samples/cifar100-lt/002_orig05649_apple.png"><img src="./static_samples/cifar100-lt/002_orig05649_apple.png" width="96"></a> | <a href="./static_samples/cifar100-lt/003_orig32507_apple.png"><img src="./static_samples/cifar100-lt/003_orig32507_apple.png" width="96"></a> | <a href="./static_samples/cifar100-lt/004_orig12208_apple.png"><img src="./static_samples/cifar100-lt/004_orig12208_apple.png" width="96"></a> |

### CINIC-10
CINIC-10 was designed as a bridge between CIFAR-10 and ImageNet.  It includes the same 10 semantic categories as CIFAR-10.  It combines all of ICFAR-10 train and test images, and additional ImageNet images mapped into the same 10 classes.  This makes it a much larger dataset than CIFAR-10.

Splits: train/validation/test \
Source: https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz \
Cite: Darlow, Luke N; Crowley, Elliot J; Antoniou, Antreas; Storkey, Amos.(2018). [CINIC-10 Is Not ImageNet or CIFAR-10](https://datashare.ed.ac.uk/handle/10283/3192), [dataset]. University of Edinburgh

**CINIC-10 (train)** \
Number of samples: 90000 \
Number of classes: 10 \
Example classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

**CINIC-10 (val)**  \
Number of samples: 90000 \
Number of classes: 10 \
Example classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

**CINIC-10 (test)** \
Number of samples: 90000 \
Number of classes: 10 \
Example classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


**Examples**
| 7 | 7 | 1 | 9 | 2 |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/CINIC-10/000_7.png"><img src="./static_samples/CINIC-10/000_7.png" width="96"></a> | <a href="./static_samples/CINIC-10/001_7.png"><img src="./static_samples/CINIC-10/001_7.png" width="96"></a> | <a href="./static_samples/CINIC-10/002_1.png"><img src="./static_samples/CINIC-10/002_1.png" width="96"></a> | <a href="./static_samples/CINIC-10/003_9.png"><img src="./static_samples/CINIC-10/003_9.png" width="96"></a> | <a href="./static_samples/CINIC-10/004_2.png"><img src="./static_samples/CINIC-10/004_2.png" width="96"></a> |

### ImageNet-2012
Splits: train/val \
Source: 
```
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar 
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar 
```
Cite: J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, [ImageNet: A Large-Scale Hierarchical Image Database](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5206848). IEEE Computer Vision and Pattern Recognition (CVPR), 2009. \
Sample code for loading and manipulating ImageNet data is provided in the data explorer.

**ImageNet2012 (train)**\
Number of samples: 1281167 \
Number of classes: 1000

**ImageNet-2012 (val)** \
Number of samples: 50000 \
Number of classes: 1000

**Examples:**
| soap dispenser | warplane military plane | Dandie Dinmont (terrier) | lacewing (lacewing fly) | fire screen (fireguard) |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/imagenet-2012/000_soap_dispenser.png"><img src="./static_samples/imagenet-2012/000_soap_dispenser.png" width="160"></a> | <a href="./static_samples/imagenet-2012/001_warplane_military_plane.png"><img src="./static_samples/imagenet-2012/001_warplane_military_plane.png" width="160"></a> | <a href="./static_samples/imagenet-2012/002_Dandie_Dinmont_Dandie_Dinmont_terrier.png"><img src="./static_samples/imagenet-2012/002_Dandie_Dinmont_Dandie_Dinmont_terrier.png" width="160"></a> | <a href="./static_samples/imagenet-2012/003_lacewing_lacewing_fly.png"><img src="./static_samples/imagenet-2012/003_lacewing_lacewing_fly.png" width="160"></a> | <a href="./static_samples/imagenet-2012/004_fire_screen_fireguard.png"><img src="./static_samples/imagenet-2012/004_fire_screen_fireguard.png" width="160"></a> |

### MNIST / QMNIST

QMNIST extends MNIST with an extra 50,000 test images.  Splits test10k, test50k (aka "extra"), and nist can be exposed if you load QMNIST through torchvision.datasets.QMNIST with the *what* parameter. These splits are derived from the downloaded files.  

In other words: 

- **train** - 60k (MNIST train)
- **test10k** - 10k (the classic MNIST test set)
- **test50k** - the “extra” 50k (the remainder of the QMNIST test set, sometimes called "extra" in papers)
- **test** - 60k total = test10k + test50k
- **nist** - 60k additional digits sourced from NIST

**Classes:** \
['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

Source: torchvision.datasets \
Cite QMNIST: Yadav, C. and Bottou, L. [Cold case: The lost mnist digits.](https://proceedings.neurips.cc/paper_files/paper/2019/file/51c68dc084cb0b8467eafad1330bce66-Paper.pdf).
In Advances in Neural Information Processing Systems
32, 2019. \
Cite MNIST: LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. [Gradient-based learning applied to document recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf). Proceedings of the IEEE, 86(11):2278–2324, November 1998.


#### QMNIST

**QMNIST (train)** \
Number of samples: 60000 

**QMNIST (test)** \
Number of samples: 60000

**QMNIST (test50k)** \
Number of samples: 50000

**QMNIST (test10k)** \
Number of samples: 60000

**QMNIST (NIST)** \
Number of samples: 402953

**Examples** 

| 5 (five) | 0 (zero) | 4 (four) | 1 (one) | 9 (nine) |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/qmnist/000_5_five.png"><img src="./static_samples/qmnist/000_5_five.png" width="96"></a> | <a href="./static_samples/qmnist/001_0_zero.png"><img src="./static_samples/qmnist/001_0_zero.png" width="96"></a> | <a href="./static_samples/qmnist/002_4_four.png"><img src="./static_samples/qmnist/002_4_four.png" width="96"></a> | <a href="./static_samples/qmnist/003_1_one.png"><img src="./static_samples/qmnist/003_1_one.png" width="96"></a> | <a href="./static_samples/qmnist/004_9_nine.png"><img src="./static_samples/qmnist/004_9_nine.png" width="96"></a> |


#### MNIST

**MNIST (train)** \
Number of samples: 60000 \
Number of classes: 10 

**MNIST (test)** \
Number of samples: 10000 \
Number of classes: 10 

| 1 (one) | 6 (six) | 8 (eight) | 4 (four) | 6 (six) |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/mnist/000_1_one.png"><img src="./static_samples/mnist/000_1_one.png" width="96"></a> | <a href="./static_samples/mnist/001_6_six.png"><img src="./static_samples/mnist/001_6_six.png" width="96"></a> | <a href="./static_samples/mnist/002_8_eight.png"><img src="./static_samples/mnist/002_8_eight.png" width="96"></a> | <a href="./static_samples/mnist/003_4_four.png"><img src="./static_samples/mnist/003_4_four.png" width="96"></a> | <a href="./static_samples/mnist/004_6_six.png"><img src="./static_samples/mnist/004_6_six.png" width="96"></a> |


### SVHN (Street View House Numbers)

Splits: train/test \
Source: torchvision.datasets \
Cite: Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. [Reading digits in natural images with
unsupervised feature learning](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37648.pdf). NIPS workshop on deep learning and unsupervised feature learning. Vol. 2011. No. 5. 2011.

**Note** \
Unlike MNIST or QMNIST, the **SVHN dataset** does not create `raw/` and `processed/` subdirectories.  
It ships as ready-to-use MATLAB `.mat` files.  The `torchvision.datasets.SVHN` class knows how to read these `.mat` files directly.  Sample code for loading the SVHN data is provided in the data explorer.

Additionally, note that the SVHN set provided by torchvision.datasets is the "cropped" version of the [original Stanford dataset](http://ufldl.stanford.edu/housenumbers/).  The resolution is lower, and the class labels are a **single** digit - so if the visible street number is 714, the label will likely be the single digit 1.  The source files of the full dataset, with multi-digit labels and bounding boxes, are available in our shared datasets directory, but the loading is more complex, and as most studies seem to reference the cropped set, a dataloader has not yet been provided in this notebook.

Train samples: 73257 \
Test samples: 26032 \
Number of classes: 10


| 0 | 3 | 5 | 1 | 3 |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/svhn/000_0.png"><img src="./static_samples/svhn/000_0.png" width="96"></a> | <a href="./static_samples/svhn/001_3.png"><img src="./static_samples/svhn/001_3.png" width="96"></a> | <a href="./static_samples/svhn/002_5.png"><img src="./static_samples/svhn/002_5.png" width="96"></a> | <a href="./static_samples/svhn/003_1.png"><img src="./static_samples/svhn/003_1.png" width="96"></a> | <a href="./static_samples/svhn/004_3.png"><img src="./static_samples/svhn/004_3.png" width="96"></a> |

### Tiny ImageNet
Splits: train/val/test \
Source: torchvision.datasets \
Cite Tiny ImageNet challenge: Le, Yann, and Xuan Yang. "[Tiny imagenet visual recognition challenge](https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf)." CS 231N 7.7 (2015): 3 \
Cite Tiny ImageNet original:  Deng, Jia, et al. "[ImageNet: A Large-Scale Hierarchical Image Database](https://ieeexplore.ieee.org/document/5206848)." 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255. 

Train samples: 100000 \
Validation samples: 10000 \
Test samples: 10000 \
Number of classes: 200

**Examples:**
| basketball | beaker | stopwatch / stop watch | limousine / limo | acorn |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/tiny-imagenet-200/000_basketball.png"><img src="./static_samples/tiny-imagenet-200/000_basketball.png" width="128"></a> | <a href="./static_samples/tiny-imagenet-200/001_beaker.png"><img src="./static_samples/tiny-imagenet-200/001_beaker.png" width="128"></a> | <a href="./static_samples/tiny-imagenet-200/002_stopwatch+stop_watch.png"><img src="./static_samples/tiny-imagenet-200/002_stopwatch+stop_watch.png" width="128"></a> | <a href="./static_samples/tiny-imagenet-200/003_limousine+limo.png"><img src="./static_samples/tiny-imagenet-200/003_limousine+limo.png" width="128"></a> | <a href="./static_samples/tiny-imagenet-200/004_acorn.png"><img src="./static_samples/tiny-imagenet-200/004_acorn.png" width="128"></a> |

### PASCAL VOC

Splits: train/validation \
Source: tensorflow datasets, e.g. \
`python -c "import tensorflow_datasets as tfds; voc, info = tfds.load('voc/2012', data_dir='$DATASETS_DIR/vision/voc2012', with_info=True); print(info)"` \
Cite: Everingham, M., Van Gool, L., Williams, C.K.I. et al. [The Pascal Visual Object Classes (VOC) Challenge](https://doi.org/10.1007/s11263-009-0275-4). Int J Comput Vis 88, 303–338 (2010). 

VOC images have multiple labels, meaning that, for example, an image of a plant shop could have 10 "potted plant" labels.  Additionally, VOC images come in a variety of resolutions. To handle them in PyTorch, we need to apply a transform. We permute the images to channel-last format for visualization.  A custom collator is needed to address the different image sizes (torch.stack fails).  Sample code for this is provided in the data explorer.

Train samples: 5717 \
Validation samples: 5823 \
Number of classes: 20 (but could be multiple examples in an image)

**Examples:**
| bus + car ×2 | train | bird | bird | dining table + chair ×4 |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/voc2012/000_bus+carx2.png"><img src="./static_samples/voc2012/000_bus+carx2.png" width="160"></a> | <a href="./static_samples/voc2012/001_train.png"><img src="./static_samples/voc2012/001_train.png" width="160"></a> | <a href="./static_samples/voc2012/002_bird.png"><img src="./static_samples/voc2012/002_bird.png" width="160"></a> | <a href="./static_samples/voc2012/003_bird.png"><img src="./static_samples/voc2012/003_bird.png" width="160"></a> | <a href="./static_samples/voc2012/004_diningtable+chairx4.png"><img src="./static_samples/voc2012/004_diningtable+chairx4.png" width="160"></a> |

### Wikipedia-Multimedia
[Hong](https://arxiv.org/abs/2406.04872) references the "Wikpedia dataset".  However, the citations are Rasiwasia et al., 2010; Hu et al.,
2021, which refer to a ~3000 item text-image dataset described here: http://www.svcl.ucsd.edu/projects/crossmodal/
and not the "Wikipedia" dataset available from HuggingFace.  
Source: URL=http://www.svcl.ucsd.edu/projects/crossmodal/wikipedia_dataset.zip

This shows the cited Rasiwaisa wikipedia version.  

Train samples: 2173 \
Test samples: 693 \
Number of classes: 10 \
Categories: ['art', 'biology', 'geography', 'history', 'literature', 'media', 'music', 'royalty', 'sport', 'warfare']

**Examples:**
| sport | geography | biology | warfare | sport |
|:---:|:---:|:---:|:---:|:---:|
| <a href="./static_samples/wikipedia-vision/001_sport.png"><img src="./static_samples/wikipedia-vision/001_sport.png" width="160"></a> | <a href="./static_samples/wikipedia-vision/002_geography.png"><img src="./static_samples/wikipedia-vision/002_geography.png" width="160"></a> | <a href="./static_samples/wikipedia-vision/003_biology.png"><img src="./static_samples/wikipedia-vision/003_biology.png" width="160"></a> | <a href="./static_samples/wikipedia-vision/004_warfare.png"><img src="./static_samples/wikipedia-vision/004_warfare.png" width="160"></a> | <a href="./static_samples/wikipedia-vision/005_sport.png"><img src="./static_samples/wikipedia-vision/005_sport.png" width="160"></a> |

Text snippets: \
Morris started the 1947-48 Australian season strongly, scoring 162 in his second match as New South Wales crushed the t... \
On July 16, 1990, the major 1990 Luzon earthquake of magnitude 7.7 struck central Luzon., USGS This was the largest ear... \
Like with most exotic pets, owning a raccoon often takes a significant amount of time and patience.http://www.filthyluc... \
Sixty-one thousand Puerto Ricans served in the Korean War, including 18,000 Puerto Ricans who enlisted in the continent... \
The following season, Olympic again reached the semi-finals of the FA Cup, as did Blackburn Rovers.  When the draw for ... 

### Clothing-1M
This is a really big dataset, so it hasn't been expanded just yet.  Has been downloaded and can be expanded if necessary.

Cite: Xiao, T., Xia, T., Yang, Y., Huang, C., and Wang, X. [Learning from massive noisy labeled data for image classification](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf). In CVPR, 2015.

## NLP Datasets

### CoLA
The Corpus of Linguistic Acceptability (CoLA) labels sentences for grammatical acceptability \
Source: HuggingFace Datasets (GLUE) \
Cite: [Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural Network Acceptability Judgments. TACL (2019)](https://aclanthology.org/Q19-1040/)

train: 8551 samples \
validation: 1043 samples \
test: 1063 samples 

Label meanings: 0 = unacceptable, 1 = acceptable

**Examples:**
| sentence | label | idx |
|:--|:--:|:--:|
| Sodium is a little too peppy for me to want to... | 0 | 1792 |
| I am having eaten seaweed. | 0 | 7950 |
| She's enough tall. | 0 | 5436 |
| Sandy sang me a song. | 1 | 2959 |
| I know I should go to the dentist's, but I jus... | 1 | 3581 |


### SST2
The Stanford Sentiment Treebank labels movie reviews for positive or negative sentiment \
Source: HuggingFace Datasets (GLUE) \
Cite (Original dataset) Richard Socher, Alex Perelygin, Jean Y. Wu, et al.
[Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf). EMNLP 2013.

train: 67349 samples \
validation: 872 samples \
test: 1821 samples

Label meanings: 0 = negative, 1 = positive

**Examples:**
| sentence | label | idx |
|:--|:--:|:--:|
| klein , charming in comedies like american pie... | 1 | 32326 |
| be fruitful | 1 | 27449 |
| soulful and | 1 | 60108 |
| the proud warrior that still lingers in the so... | 1 | 23141 |
| covered earlier and much better | 0 | 35226 |


### E2E NLG

The task is to generate natural-sounding restaurant descriptions directly from structured meaning representations (MRs).  This is not a classification task but is included here because it was one of the test sets listed in [Hong](https://arxiv.org/abs/2406.04872). 

Cite: Novikova, J., Dušek, O., Curry, A. C., & Rieser, V. (2017).
[The E2E Dataset: New Challenges For End-to-End Generation](https://arxiv.org/abs/1706.09254).
EMNLP 2017

Note that this dataset contains two testsets. testset.csv has a single human reference per MR. testset_w_refs has MULTIPLE human references for the same MR.

train: 42061 samples \
validation: 4672 samples \
test: 630 samples

Field meanings:
 - mr  = meaning representation (structured input)
 - ref = reference text (human-written realization)

 **Examples:**
 | mr | ref |
|:--|:--|
| name[Clowns], eatType[pub], priceRange[more than £30], area[city centre], familyFriendly[no], near[The Portland Arms] | Located near the city centre is Clowns a low rated pub near The Portland Arms. |
| name[The Eagle], eatType[coffee shop], food[French], area[riverside], near[Burger King] | Near Burger King, at Riverside, there's a coffee shop called The Eagle that serves French food. |
| name[The Eagle], eatType[coffee shop], food[Indian], area[riverside], familyFriendly[yes] | The Eagle, a coffee shop that serves Indian food, is family-friendly and located at Riverside. |
| name[Cocum], eatType[pub], priceRange[cheap], familyFriendly[yes], near[The Rice Boat] | Cocum is a cheap pub. The restaurant is family-friendly and near The Rice Boat. |
| name[Alimentum], food[Italian], priceRange[moderate], familyFriendly[yes], near[Café Adriatic] | There is a kid friendly Italian restaurant named Alimentum near Café Adriatic. |
