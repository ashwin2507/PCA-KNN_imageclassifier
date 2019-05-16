# Ashwin Babu
# Machine Learning
# Image recognition using PCA and KNN


import numpy as np
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
from sklearn.decomposition import PCA # To check if the output of using .pca() matches my implementaion
from sklearn.preprocessing import StandardScaler # For using sklearn.pca()
# from sklearn.model_selection import train_test_split # For testing accuracy by splitting the training data
import math
import operator

print('Compiled')



# Training Data set
fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("/Users/ashwinbabu/Downloads/fruits-proj/Training/*"):
    # Extracting the labels from the folder name
    fruit_label = fruit_dir_path.split("/")[-1]
    print(fruit_label)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        # Loads a color image 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Resize the image
        image = cv2.resize(image, (28, 28))
        # Changing the color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)
print('Total Training data images: ',len(fruit_images))


# In[3]:


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])


# In[4]:


# Function to plot images
def plot_image_grid(images, nb_rows, nb_cols, figsize=(5, 5)):
    assert len(images) == nb_rows*nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            # axs[i, j].xaxis.set_ticklabels([])
            # axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1
            
# Plotting some images from the training data
plot_image_grid(fruit_images[0:100], 10, 10)



# using PCA() library to get the output in order to cross check with my pca() implementation
scaler = StandardScaler()
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])



# PCA implemtation
def pca(data):
    # First Flattening the image data to 1899 by 2352(28*28*3)
    image_flattened = ([i.flatten() for i in data])
    image_flattened = np.asarray(image_flattened,dtype=float)

    # Standardizing the data by calculating the mean and standard deviation
    # Subtracting with the mean to center the data and dividing by the standard deviation
    for x in range(len(image_flattened[0])):
        mean = np.mean(image_flattened[:,x])
        std = np.std(image_flattened[:,x])
        for y in range(len(image_flattened)):
            if std != 0.0:

                image_flattened[y][x] = (image_flattened[y][x] - mean)/std
            else:
                image_flattened[y][x] = (image_flattened[y][x]- mean)



    # Finding the co-variance matrix
    co_var = np.cov(image_flattened.T)


    # Computing the eigen-value and eigen-vector from co-variance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(co_var)
    



    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]


    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)


    # Taking the top 2 features (components)
    matrix_w = np.hstack((eig_pairs[0][1].reshape(2352,1), eig_pairs[1][1].reshape(2352,1)))

    # Training data converted to 2-D space
    transformed = matrix_w.T.dot(image_flattened.T)

    # print(len(transformed[0]))
    # print(transformed.T)

    final_data = transformed.T
    final_data = final_data.real
    final_data[:,1] = np.multiply(final_data[:,1],-1)
    
    """""
     Plottling the propotion of variance graph and keeping the threshold 40% and adding features until 40% of total
    variability which is PC1 and PC2 together in this case (45%)

    """""
    eig_val_cov = np.real(eig_val_cov)
    eig_val_sorted = np.sort(eig_val_cov)[::-1]
    eig_val = []
    total = np.sum(eig_val_sorted)

    for x in range(len(eig_val_sorted)):
        temp = []
        if eig_val_cov[x] !=0:
            temp.append(eig_val_cov[x]/(total))
            temp.append(x+1)
            eig_val.append(temp)

    eig_val = np.asarray(eig_val)

    for x in range(len(eig_val)):
        plt.plot(eig_val[x][1],eig_val[x][0],'ro-',linewidth=2)
    plt.title("The proportion of variance plot")
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()
    
    
    return final_data


# In[7]:


# output from training data when given as an argument to my pca() implementation
after_pca_training = pca(fruit_images)
train = np.column_stack((after_pca_training,label_ids))
print('Using my pca function',after_pca_training)


# In[8]:


# output from training data when given as an argument to standard sklearn PCA() library implementation
prca = PCA(n_components=2)
pca_result = prca.fit_transform(images_scaled)

print('Using sklearn PCA',(pca_result))




# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=42)


# In[9]:


# graphing the new 2-d Training Data
for x in range((len(train))):
    if train[x][-1] == 0.0:
        plt.scatter(train[x][0], train[x][1], s=10, c='b', marker="o", label='first')
    elif train[x][-1] == 1.0:
        plt.scatter(train[x][0], train[x][1], s=10, c='r', marker="o", label='second')
    elif train[x][-1] == 2.0:
        plt.scatter(train[x][0], train[x][1], s=10, c='g', marker="o", label='third')
    else:
        plt.scatter(train[x][0], train[x][1], s=10, c='y', marker="o", label='fourth')
plt.title("Training Data")
plt.show()


# In[10]:


#Testing Data set
fruit_test_images = []
labels_test = [] 
for fruit_dir in glob.glob("/Users/ashwinbabu/Downloads/fruits-proj/Test/*"):
    fruit_label_test = fruit_dir.split("/")[-1]
    print(fruit_label_test)
    for image_path_t in glob.glob(os.path.join(fruit_dir, "*.jpg")):
        image = cv2.imread(image_path_t, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_test_images.append(image)
        labels_test.append(fruit_label_test)
fruit_test_images = np.array(fruit_test_images)
labels_test = np.array(labels_test)






label_to_id_dict_test = {v:i for i,v in enumerate(np.unique(labels_test))}
id_to_label_dict_test = {v: k for k, v in label_to_id_dict_test.items()}
label_test_ids = np.array([label_to_id_dict_test[x] for x in labels_test])



# PCA implemtation
after_pca_test = pca(fruit_test_images)
# adding labels to the data 
test = np.column_stack((after_pca_test,label_test_ids))
print('Testing data after pca',after_pca_test)


# In[13]:


# Visualizing the testing data
for x in range((len(test))):
    if test[x][-1] == 0.0:
        plt.scatter(test[x][0], test[x][1], s=10, c='b', marker="o", label='first')
    elif train[x][-1] == 1.0:
        plt.scatter(test[x][0], test[x][1], s=10, c='r', marker="o", label='second')
    elif train[x][-1] == 2.0:
        plt.scatter(test[x][0], test[x][1], s=10, c='g', marker="o", label='third')
    else:
        plt.scatter(test[x][0], test[x][1], s=10, c='y', marker="o", label='fourth')
plt.title("Testing data")
plt.show()





# K-Nearest Neighbours Implementation
def euclideanDistance(item1, item2, length):
    cal_dist = 0
    for x in range(length):
        cal_dist += pow((item1[x] - item2[x]), 2)
    return math.sqrt(cal_dist)


def checkNeighbors(trainingData, test, k):
    distance_measure =[]
    length = len(test)-1
    for x in range(len(trainingData)):
        dist = euclideanDistance(test, trainingData[x], length)
        distance_measure.append(( dist, trainingData[x]))
    distance_measure.sort(key=operator.itemgetter(0))
    
    neighbors =[]
    for x in range(k):
        neighbors.append(distance_measure[x][1])
 
    return neighbors

def determineClass(neighbors):
    classMajority = {}
    for x in range(len(neighbors)):
        classification = str(neighbors[x][-1])

        if classification in classMajority:
            classMajority[classification] +=1
        else:
            classMajority[classification] = 1
    
    sortedMajority = max(classMajority.items(), key=operator.itemgetter(1))[0]
    return sortedMajority[0][0]


def accuracy(test, prediction):
    identified=0

    for x in range(len(test)):
        
        
        b = float(prediction[x])
        
        if test[x][-1] == b:
            identified +=1
#     print(identified)
    return (identified/float(len(test))) * 100.0





# calling the functions of KNN --> predicting labels --> accuracy
def main():
    # New data that has been conerted to 2-D space
    trainingData = train
    testData = test
    print('Training Data: ' + repr(len(trainingData)))
    print('Test Data: ' + repr(len(testData)))
    # Predicting the class
    predictions = []
    k=input('Enter value for k: ')
    k = int(k)
    
    for x in range(len(testData)):
        neighbors = checkNeighbors(trainingData, testData[x], k)
        result = determineClass(neighbors)
        predictions.append(result)
#         print('--> predicted=' + repr(result) + '-->actual=' + repr(testData[x][-1]))
    accuracy_1 = accuracy(testData, predictions)
    print('Accuracy:' + repr(accuracy_1) + '%')
    
main()







