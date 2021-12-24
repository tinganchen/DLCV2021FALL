# DLCV2021FALL
Deep Learning for Computer Vision

### [hw0. Face recognition with PCA](./hw0)

1. Mean face & eigenfaces
2. Project onto the eigenspace and reconstruct the images
3. Reconstruction error in MSE
4. 3-fold cross-validation + KNN for testing
5. Recognition rate

[Results](./hw0/report.pdf)

### [hw1. Image classification & Segmentation](./hw1)

i. Image classification
  1. CNN [VGG-16, ResNet-110]
  2. Multi-class problem 
  3. Cross entropy loss
  4. Evaluation: Accuracy
  5. Visualization: t-SNE for visualizing the classification results

ii. Image segmentation on satellite map
  1. CNN + FCN [VGG-16+FCN32, ResNet-50+FCN]
  2. Multi-class problem (One dimension per pixel)
  3. Cross entropy loss 2D + Dice loss
  4. Evaluation: mIOU

[Results](./hw1/hw1_d09921014.pdf)
 

### [hw2. GAN & Unsupervised Domain Adaptation (UDA)](./hw2)

i. GAN (Face generation)
  1. DCGAN
  2. Unsupervised learning problem 
  3. Binary cross entropy loss (fake vs. real images)
  4. Evaluation: IS & FID score

ii. ACGAN (Digit number generation)
  1. ACGAN
  2. Supervised learning problem (One dimension per pixel)
  3. Binary cross entropy loss (fake vs. real images)
  4. Evaluation: Accuracy

iii. UDA (DANN & DSAN)
  1. DANN & DSAN
  2. Supervised learning problem
  3. Binary cross entropy loss (fake vs. real images)
  4. Evaluation: Accuracy
  5. Visualization: t-SNE for visualizing the classification results (both DANN & DSAN) & domain discrimination results (DANN only)

[Results](./hw2/hw2_d09921014.pdf)

### [hw3. Transformer on Image Classification & Captioning](./hw3)

i. Vision Transformer (ViT)
  1. Image classification
  2. Image patches
  3. Sequential learning
  4. Cross entropy loss
  5. Evaluation: Accuracy
  6. Visualization: Multi-head attention

ii. Caption Transformer (CATR)
  1. Image captioning
  2. BERT tokenizer
  3. Transformer encoder (encode as image features) & decoder (decode as sequence of words)
  4. Sequential learning
  5. Cross entropy loss (predicted caption vs. encoded caption ground truth)
  6. Visualization: Multi-head attention & Caption words

[Results](./hw3/hw3_d09921014.pdf)

### [hw4. Few-shot learning (FSL) & Self-supervised Learning (SSL)](./hw4)

i. Few-shot learning (FSL) - Prototypical Net
  1. Image classification
  2. N-way K-shot sampling

ii. Self-supervised Learning (SSL) - BYOL
  1. Image classification
  2. Unsupervised learning
  3. Data augmentation
  4. EMA - online & target model

[Results](./hw4/hw4_d09921014.pdf)
