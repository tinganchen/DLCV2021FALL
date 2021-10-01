from sklearn.decomposition import PCA as pca
import numpy as np
from os import listdir
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt
from scipy import stats

IMG_DIR = 'p1_data'
PERSON = 40
FACE = 10
W = 56
H = 46

img_files = listdir(IMG_DIR)

# 0. dataset 
train_list = []
test_list = []

for p in range(PERSON):
    for f in range(FACE-1):
        for img_path in glob.glob(f"{IMG_DIR}/{p+1}_{f+1}.png"):
            #print(img_path)
            img = Image.open(img_path)
            img = np.array(img.getdata())
            train_list.append(img)
    
    for img_path in glob.glob(f"{IMG_DIR}/{p+1}_10.png"):
        #print(img_path)
        img = Image.open(img_path)
        img = np.array(img.getdata())
        test_list.append(img)
    
train_data = np.array(train_list) # size (360, 2576)
test_data = np.array(test_list) # size (40, 2576)

'''
# 1. Perform PCA on the training set. 
#    Plot the mean face and the first four eigenfaces.
'''

## mean face
mean_face = np.mean(train_data, 0)


if not os.path.isdir('mean_face'):
  os.makedirs('mean_face')
  
plt.imshow(mean_face.reshape([W, H]), cmap = plt.cm.gray)
plt.axis('off')
plt.title('Mean Face')
plt.savefig('mean_face/mean_face.png')
plt.show()

## first four eigenfaces
PCA = pca()
PCA.fit(train_data)
#print(PCA.mean_)
#mean_face-PCA.mean_

#PCA.singular_values_
#plt.plot(PCA.singular_values_)

eigen_faces = PCA.components_
singular_values = PCA.singular_values_

if not os.path.isdir('eigenfaces'):
  os.makedirs('eigenfaces')

for i in range(4):
    eigen_face = eigen_faces[i]
    plt.imshow(eigen_face.reshape((W, H)), cmap = plt.cm.gray)
    plt.axis('off')
    plt.title(f'Eigenface {i+1}\nSingular value = {np.round(singular_values[i], 4)}')
    plt.savefig(f'eigenfaces/eigen_face_{i+1}.png')
    plt.show()

#X_train_pca = PCA.transform(train_data)
#X_train_pca2 = (train_data - PCA.mean_).dot(PCA.components_.T)
#np.testing.assert_array_almost_equal(X_train_pca, X_train_pca2)

'''
# 2. Project person 8 image 1 onto the PCA eigenspace you obtained above. 
#    Reconstruct this image using the first n = 3, 50, 170, 240, 345 eigenfaces. 
#    Plot the five reconstructed images.
'''

## Project person 8 image 1 onto the PCA eigenspace you obtained above
p8img1 = train_data[(FACE-1)*7].reshape([1, -1]) 
#plt.imshow(p8img1.reshape((W, H)), cmap = plt.cm.gray)

n = [3, 50, 170, 240, 345]


if not os.path.isdir('reconstruct_imgs'):
  os.makedirs('reconstruct_imgs')
  
for c in n:
    PCA = pca(n_components = c)
    PCA.fit(train_data)
    
    p8img1_pca = PCA.transform(p8img1)
     
    ## Reconstruct this image using the first n = 3, 50, 170, 240, 345 eigenfaces.
    p8img1_projected = PCA.inverse_transform(p8img1_pca)
    #p8img1_projected2 = p8img1_pca.dot(PCA.components_) + PCA.mean_
    #np.testing.assert_array_almost_equal(p8img1_projected, p8img1_projected2)
    
    '''
    # 3. For each of the five images you obtained in 2., 
    # compute the mean squared error (MSE)
    # between the reconstructed image and the original image. 
    # Record the corresponding MSE values.
    '''
    mse = np.mean((p8img1 - p8img1_projected) ** 2)
    
    plt.imshow(p8img1_projected.reshape((W, H)), cmap = plt.cm.gray)
    plt.axis('off')
    plt.title(f'Reconstructed image with {c} components\nMSE = {np.round(mse, 4)}')
    
    plt.savefig(f'reconstruct_imgs/reconstruct_img_components_{c}.png')
    plt.show()
     
    #with open('mse.txt', 'a') as f:
    #    f.write(str(mse)+'\n')

'''
# 4. Now, apply the k-nearest neighbors algorithm to classify the testing set images. 
#    First, you will need to determine the best k and n values 
#    by 3-fold cross-validation. 
#    For simplicity, the choices for such hyperparameters are k = {1, 3, 5} 
#    and n = {3, 50, 170}. 
#    Show the cross-validation results and explain your choice for (k, n).
'''
# 3-fold cross validation
n_train = train_data.shape[0]

cv1, cv2, cv3 = [], [], [] # validate data idx

for i in range(n_train):
    if i % (FACE-1) == 0:
        for k in range(3):
            cv1.append(i+k)
    elif i % (FACE-1) == 3:
        for k in range(3):
            cv2.append(i+k)
    elif i % (FACE-1) == 6:
        for k in range(3):
            cv3.append(i+k)

valid1 = train_data[cv1]
train1 = train_data[cv2+cv3]

valid2 = train_data[cv2]
train2 = train_data[cv1+cv3]

valid3 = train_data[cv3]
train3 = train_data[cv1+cv2]

train_sets = [train1, train2, train3]
valid_sets = [valid1, valid2, valid3]

# labels
train_labels = np.repeat(np.arange(PERSON), FACE-1)
test_labels = np.arange(PERSON)

valid1_labels = train_labels[cv1]
train1_labels = train_labels[cv2+cv3]

valid2_labels = train_labels[cv2]
train2_labels = train_labels[cv1+cv3]

valid3_labels = train_labels[cv3]
train3_labels = train_labels[cv1+cv2]

train_set_labels = [train1_labels, train2_labels, train3_labels]
valid_set_labels = [valid1_labels, valid2_labels, valid3_labels]

# train
def pca_model(train_data, test_data, train_labels, test_labels, k, n):
    PCA = pca(n_components = n)
    PCA.fit(train_data)
    
    train_pca = PCA.transform(train_data)
    test_pca = PCA.transform(test_data)
    
    n_train_acc = 0
    n_test_acc = 0
    
    for train_id, train in enumerate(train_pca):
        #train_projected = PCA.inverse_transform(train)
        distances = np.mean((train - train_pca)**2, 1)
        k_nearest_img_ids = np.argsort(distances)[:k]
        
        possible_person = train_labels[k_nearest_img_ids]
        
        pred_person = stats.mode(possible_person)[0]
        
        if pred_person == min(possible_person):
            pred_person = possible_person[0]
        
        actual_person = train_labels[train_id]
        
        if pred_person == actual_person:
            n_train_acc += 1
            
    for test_id, test in enumerate(test_pca):
        #test_projected = PCA.inverse_transform(test)
        distances = np.mean((test - train_pca)**2, 1)
        k_nearest_img_ids = np.argsort(distances)[:k]
        
        possible_person = train_labels[k_nearest_img_ids]

        pred_person = stats.mode(possible_person)[0]
        
        if pred_person == min(possible_person):
            pred_person = possible_person[0]
        
        actual_person = test_labels[test_id]
        
        if pred_person == actual_person:
            n_test_acc += 1
    
    train_acc = n_train_acc / train_pca.shape[0]
    test_acc = n_test_acc / test_pca.shape[0]
    
    #print(f'\nK = {k}, N = {n}\nTrain_acc: {train_acc}\nTest_acc: {test_acc}')
    
    return train_acc, test_acc


# results of cross-validation

K = [1, 3, 5] # k-nn
N = [3, 50, 170] # n components

best_acc = 0
best_k = K[0]
best_n = N[0]

for k in K:
    for n in N:
        train_mean_acc, valid_mean_acc = 0, 0
        
        for i in range(3):
            train, valid, tr_labels, val_labels = train_sets[i], valid_sets[i], train_set_labels[i], valid_set_labels[i]
            train_acc, valid_acc = pca_model(train, valid, tr_labels, val_labels, k, n)
            train_mean_acc = (train_mean_acc * i + train_acc) / (i+1)
            valid_mean_acc = (valid_mean_acc * i + valid_acc) / (i+1)
        
        if valid_mean_acc > best_acc:
            best_acc = valid_mean_acc
            best_k = k
            best_n = n

        print(f'\nK = {k}, N = {n}\nTrain_acc: {np.round(train_mean_acc, 4)}\nValid_acc: {np.round(valid_mean_acc, 4)}')
            
print(f'\nBest K = {best_k}, Best N = {best_n}\nBest_acc: {np.round(best_acc, 4)}')
        
'''
# 5. Use your hyperparameter choice in 4. 
#    and report the recognition rate of the testing set.
'''
# results of testing

k = best_k
n = best_n

train, valid, tr_labels, val_labels = train_data, test_data, train_labels, test_labels
train_acc, test_acc = pca_model(train, valid, tr_labels, val_labels, k, n)

print(f'\nK = {k}, N = {n}\nTrain_acc: {np.round(train_acc, 4)}\nValid_acc: {np.round(test_acc, 4)}')
    
