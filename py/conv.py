#!/usr/bin/env python
# coding: utf-8

# # Convolutional Shoe Classifier
# 
# **Goal**: Build a convolutional neural network that can predict 
# whether two shoes are from the same pair or from two different pairs.
# This kind of application can have real-world applications: for example to help
# people who are visually impaired to have more independence.

# ## 1. Data Processing

# In[3]:


import pandas
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# Men and women's shoes tend to be somewhat different. We separate the test set to make sure the model does not end up being much more effective for one gender than it is for the other. Since we might have a dataset that is more male-dominated, our model could be much better at making predictions on men's shoes, and this is why we have to make sure that the women's shoes are recognized just as well as the men's shoes. We don't want to end up deploying a model that performs great for men but not for women.

# In[ ]:


TRAIN_PATH = "data/train/*.jpg"
TEST_M_PATH = "data/test_m/*.jpg"
TEST_W_PATH = "data/test_w/*.jpg"

import glob
def create_dataset(path):
    """ 
    Takes in the path of a folder containing some image data, and returns a 
    dataset in the form of a (N, 3, 2, 224, 224, 3) NumPy array.
    
    We will use the three *_PATH constants above to differentiate between train
    and test data paths.

    @param :str: path 
        the location of the images we are parsing
    @returns (N, 3, 2, 224, 224, 3) numpy.ndarray
    """ 
    images = {}
    for file in glob.glob(path):
        filename = file.split("/")[-1]   # get the name of the .jpg file
        img = plt.imread(file)           # read the image as a numpy array
        images[filename] = img[:, :, :3] # remove the alpha channel

    students = list(sorted(images.keys()))
    students = list(filter(lambda x : not "(1)" in x, students)) # eliminate duplicate shoes

    dataset = []
    for i in range(0, len(students), 6): 
        student_shoes = students[i:i+6]
        student_pairs = []
        for j in range(0, len(student_shoes), 2):
            left = -0.5 + images[student_shoes[j]] / 225
            right = -0.5 + images[student_shoes[j+1]] / 255
            student_pairs += [[left, right]]
        dataset += [student_pairs]

    return np.array(dataset)


# In[4]:


print("Generating data from images...")
data = create_dataset(TRAIN_PATH)
pivot = round(len(data) * 0.8)

train_data, valid_data = data[:pivot], data[pivot:]
test_m_data = create_dataset(TEST_M_PATH)
test_w_data = create_dataset(TEST_W_PATH)
print("Done!")


# In[5]:


plt.figure()
plt.imshow(train_data[4,0,0,:,:,:]) # left shoe of first pair submitted by 5th student
plt.figure()
plt.imshow(train_data[4,0,1,:,:,:]) # right shoe of first pair submitted by 5th student
plt.figure()
plt.imshow(train_data[4,1,1,:,:,:]) # right shoe of second pair submitted by 5th student


# First, we must create some labelled training data, so we need to generate pairs of images where
# both shoes are from the same pair, and some other pairs where they are not.

# In[6]:


def generate_same_pair(dataset):
    """ For each pair of shoes in our dataset, concatenate each image into one
    new NumPy array.

    @param :numpy.ndarray: dataset
        the dataset to be modified
    @returns (N*3, 448, 224, 3) numpy.ndarray
    """
    concats = []
    for student in dataset:
        for pair in student:
            concats += [np.vstack((pair[0], pair[1]))]
    return np.array(concats)

print(train_data.shape) # if this is [N, 3, 2, 224, 224, 3]
print(generate_same_pair(train_data).shape) # should be [N*3, 448, 224, 3]
plt.imshow(generate_same_pair(train_data)[0]) # should show 2 shoes from the same pair


# In[7]:


def generate_different_pair(dataset):
    concats = []
    for student in dataset:
        concats += [np.vstack((student[0][0], student[1][1]))]
        concats += [np.vstack((student[1][0], student[2][1]))]
        concats += [np.vstack((student[2][0], student[0][1]))]
    return np.array(concats)

print(train_data.shape) # if this is [N, 3, 2, 224, 224, 3]
print(generate_different_pair(train_data).shape) # should be [N*3, 448, 224, 3]
plt.imshow(generate_different_pair(train_data)[0]) # should show 2 shoes from different pairs


# The reason we want to group pairs from the same student in the same set to avoid the producer effect. Images submitted by the same student have many things in common - the angle might be similiar, the shoe size, the background, etc... If a student has one of their pairs appears in the training set, then performance on their other pairs of shoes in the test set is not representative of performance on entirely new data. 
# 

# ## 2. Convolutional Neural Networks
# 
# In this section, we will build two CNN models in PyTorch.

# First, we implement a CNN model in PyTorch called `CNN` that will take images of size
# $3 \times 448 \times 224$, and classify whether the images contain shoes from
# the same pair or from different pairs.
# 
# The model architecture is as follows:
# 
# - A convolution layer that takes in 3 channels, and outputs $n$ channels.
# - A $2 \times 2$ downsampling (either using a strided convolution in the previous step, or max pooling)
# - A second convolution layer that takes in $n$ channels, and outputs $n \times 2$ channels.
# - A $2 \times 2$ downsampling (either using a strided convolution in the previous step, or max pooling)
# - A third convolution layer that takes in $n \times 2$ channels, and outputs $n \times 4$ channels.
# - A $2 \times 2$ downsampling (either using a strided convolution in the previous step, or max pooling)
# - A fourth convolution layer that takes in $n \times 4$ channels, and outputs $n \times 8$ channels.
# - A $2 \times 2$ downsampling (either using a strided convolution in the previous step, or max pooling)
# - A fully-connected layer with 100 hidden units
# - A fully-connected layer with 2 hidden units

# In[ ]:


class CNN(nn.Module):
    def __init__(self, n=4):
        super(CNN, self).__init__()
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=n,
                               kernel_size=3,
                               #stride=2,
                               padding=1)
        self.batch_norm1 = nn.BatchNorm2d(n)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=n,
                               out_channels=n*2,
                               kernel_size=3,
                               #stride=2,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(n*2)
        self.conv3 = nn.Conv2d(in_channels=n*2,
                               out_channels=n*4,
                               kernel_size=3,
                               padding=1)
        self.batch_norm3 = nn.BatchNorm2d(n*4)
        self.conv4 = nn.Conv2d(in_channels=n*4,
                               out_channels=n*8,
                               kernel_size=3,
                               padding=1)
        self.batch_norm4 = nn.BatchNorm2d(n*8)
        self.fc1 = nn.Linear(n*8*28*14, 100)
        self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout2d(p=0.8)

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        x = self.pool(self.batch_norm1(torch.relu(self.conv1(img))))
        x = self.pool(self.batch_norm2(torch.relu(self.conv2(x))))
        x = self.pool(self.batch_norm3(torch.relu(self.conv3(x))))
        x = self.pool(self.batch_norm4(torch.relu(self.conv4(x))))
        x = x.view(-1, self.n*8*28*14)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# Next, we implement a CNN model called `CNNChannel`, where instead of starting with an image
# of shape $3 \times 448 \times 224$, we will concatentae the images the channel dimension:

# In[ ]:


class CNNChannel(nn.Module):
    def __init__(self, n=4):
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        self.n = n
        self.conv1 = nn.Conv2d(in_channels=6,
                               out_channels=n,
                               kernel_size=3,
                               padding=1)
        self.batch_norm1 = nn.BatchNorm2d(n)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=n,
                               out_channels=n*2,
                               kernel_size=3,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(n*2)
        self.conv3 = nn.Conv2d(in_channels=n*2,
                               out_channels=n*4,
                               kernel_size=3,
                               padding=1)
        self.batch_norm3 = nn.BatchNorm2d(n*4)
        self.conv4 = nn.Conv2d(in_channels=n*4,
                               out_channels=n*8,
                               kernel_size=3,
                               padding=1)
        self.batch_norm4 = nn.BatchNorm2d(n*8)
        self.fc1 = nn.Linear(self.n*8*14*14, 100)
        self.fc2 = nn.Linear(100, 2)        

    def forward(self, img):
        img1, img2 = img[:, :, :224, :], img[:, :, 224:, :]
        img = torch.cat((img1, img2), dim=1)
        x = self.pool(self.batch_norm1(torch.relu(self.conv1(img))))
        x = self.pool(self.batch_norm2(torch.relu(self.conv2(x))))
        x = self.pool(self.batch_norm3(torch.relu(self.conv3(x))))
        x = self.pool(self.batch_norm4((torch.relu(self.conv4(x)))))
        x = self.fc2(self.fc1(x.view(-1, self.n*8*14*14)))
        return x


# We will see later that due to `CNN` taking one input containing two shoes on the same dimension, it is not as good at consolidating enough relevant information to make good predictions. For `CNNChannel`, stacking both images along the channel dimension helps the model consolidate the information it needs for each channel, and make a direct side-by-side comparison. This allows it to make much better predictions.

# 
# 
# The function `get_accuracy` written below will separately compute the model accuracy on the
# positive and negative samples.  
# 
# The test accuracy helps us determine how efficient our model is at making accurate predictions. However, we don't always care as much about the general accuracy, and we might be more interested in knowing how many false positives and false negatives our model produces. Sometimes, we would rather minimize the rate of false positives and take a hit on the general accuracy, because the former can be costly. 
# 
# In this case, it's more concerning to have a high false positive rate because if we were to develop a tool to help visually impaired people put on their shoes in the morning, we would rather our model falsely predict shoes from the same pair to be incompatible than falsely associate two shoes from different pairs. If the latter case were to be true, our client would show up to work in a rather embarrassing outfit whereas if the former were to occur, we could always classify a new pair correctly.

# In[ ]:


def get_accuracy(model, data, targets, batch_size=64):
    """ Compute the overall model accuracy on the given dataset and targets.

    Example Usage:

    >>> model = CNN() # create untrained model
    >>> valid_accuracy = get_accuracy(model, valid_data)
    
    """

    model.eval()
    n = data.shape[0]

    correct = 0
    for i in range(0, len(data), batch_size):
        xs = data[i:i+batch_size]
        ts = targets[i:i+batch_size]
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1].view(len(zs)) # get the index of the max logit
        correct += len((pred-ts).nonzero())

    return round(correct / n, 2)


# In[ ]:


def get_test_accuracy(model, data, batch_size=64):
    """Compute the model accuracy on the data set. This function returns two
    separate values: the model accuracy on the positive samples,
    and the model accuracy on the negative samples.

    Example Usage:

    >>> model = CNN() # create untrained model
    >>> pos_acc, neg_acc= get_accuracy(model, valid_data)
    >>> false_positive = 1 - pos_acc
    >>> false_negative = 1 - neg_acc
    """

    model.eval()
    n = data.shape[0]

    data_pos = generate_same_pair(data)      # should have shape [n * 3, 448, 224, 3]
    data_neg = generate_different_pair(data) # should have shape [n * 3, 448, 224, 3]

    pos_correct = 0
    for i in range(0, len(data_pos), batch_size):
        xs = torch.Tensor(data_pos[i:i+batch_size]).permute(0, 3, 1, 2)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().numpy()
        pos_correct += (pred == 1).sum()
    
    neg_correct = 0
    for i in range(0, len(data_neg), batch_size):
        xs = torch.Tensor(data_neg[i:i+batch_size]).permute(0, 3, 1, 2)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().numpy()
        neg_correct += (pred == 0).sum()

    return pos_correct / (n * 3), neg_correct / (n * 3)


# ## 3. Training
# 
# We write the functions required to train the model.

# In[ ]:


CHECKPOINT_PATH = 'checkpoints/checkp-{0}'


# In[ ]:


def train_model(model, train_data, validation_data=valid_data, batch_size=64, weight_decay=0.0,
          optimizer="sgd", learning_rate=0.1, momentum=0.9,
          checkpoint_path=None, log_acc=True, num_epochs=10):
    """
    @param nn.Module model: 
        the model we are training
    @param numpy.ndarray train_data: 
        our training data
    @param int batch_size: 
        the size of our mini batches
    @param str optimizer: 
        allows us to choose the optimizer from ["sgd", "adam"]
    @param float learning_rate:
        the learning rate constant
    @param float momentum: 
        allows us to implement SGD with momentum
    @param checkpoint_path:
        the location of our weights checkpoints
    @param num_epochs:
        the number of epochs
    """
    criterion = nn.CrossEntropyLoss()
    assert optimizer in ("sgd", "adam")
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    # track learning curve
    iters, losses, train_acc, val_acc = [], [], [], []

    pos_train_data = generate_same_pair(train_data)
    neg_train_data = generate_different_pair(train_data)

    pos_valid_data = generate_same_pair(validation_data[:])
    neg_valid_data = generate_different_pair(validation_data[:])    
    
    val_xs = np.vstack((pos_valid_data, neg_valid_data))
    val_ts = np.hstack((np.ones(len(pos_valid_data)), 
                        np.zeros(len(neg_valid_data))))

    n = 0 # for plotting
    for epoch in range(num_epochs+1):
        pos_train_copy = pos_train_data.copy()
        pos_ts = np.ones(len(pos_train_copy))
        neg_train_copy = neg_train_data.copy()
        neg_ts = np.zeros(len(neg_train_copy))
        reindex_p = np.random.permutation(len(pos_train_copy))
        reindex_n = np.random.permutation(len(neg_train_copy))

        pos_train_data = pos_train_copy[reindex_p]
        neg_train_data = neg_train_copy[reindex_n]

        xs = np.vstack((pos_train_data, neg_train_data))
        ts = np.hstack((pos_ts, neg_ts))

        X_train = xs.copy()      
        t_train = ts.copy()      
        reindex = np.random.permutation(len(X_train))     
        xs = X_train[reindex]     
        ts = t_train[reindex]   

        for i in range(0, xs.shape[0], batch_size):
            if (i + batch_size) > xs.shape[0]:
                break

            batch = torch.Tensor(xs[i:i+batch_size]).permute(0, 3, 1, 2)
            targets = torch.Tensor(ts[i:i+batch_size]).long()

            model.train() # annotate model for training
            zs = model(batch)
            loss = criterion(zs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            

        # save the current training information
        iters.append(n)
        losses.append(float(loss)/batch_size)    # compute average loss
        train_acc.append(get_accuracy(model,
                                    torch.Tensor(xs).permute(0,3,1,2), 
                                    torch.Tensor(ts).long(), 
                                    batch_size=64)) 
        val_acc.append(get_accuracy(model, 
                                  torch.Tensor(val_xs).permute(0,3,1,2), 
                                  torch.Tensor(val_ts).long(), 
                                  batch_size=len(val_xs))) 
        n += 1

        if log_acc:
            print("[Epoch {0}] [Train Acc {1}%] [Val Acc {2}%] [Loss {3}]".format(
                epoch, 100*train_acc[-1], 100*val_acc[-1], losses[-1]
            ))

        if (checkpoint_path is not None) and n > 0:
            torch.save(model.state_dict(), checkpoint_path.format(n))

    # plotting
    plt.title("Learning Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


# Now, we perform a sanity check on our code for the `CNN` and `CNNChannel` models by showing that they can
# can memorize a very small subset of the training set (e.g. 5 images).
# We should be able to achieve 90%+ accuracy relatively quickly (within ~30 or so iterations).

# In[ ]:


cnn = CNN(2)
print("Overfitting the CNN model: ")
train_model(cnn, train_data[:1, :, :, :, :, :], batch_size=2, optimizer="sgd", 
      learning_rate=0.001, momentum=0.7, weight_decay=0.5, num_epochs=35)


# In[ ]:


cnn = CNNChannel()
print("Overfitting the CNNChannel model: ")
train_model(cnn, train_data[:1, :, :, :, :, :], batch_size=2, optimizer="sgd", 
      learning_rate=0.001, momentum=0.6, weight_decay=0.5, num_epochs=35)


# 
# Now we train both models. We will explore the effects of a few hyperparameters, including the learning rate, batch size, choice of $n$, and potentially the kernel size. 
# 
# We use a grid search stratey to tune our hyperparameters. We moved our best performing models at the very end of each section. Our detailed strategy for tuning the hyperparameters is also included at the beginning of each section.
# 

# #### Training CNN

# Overall, we failed to obtain a high validation acuracy with the `CNN` model. We used a grid search strategy to get an intuition of what each hyperparameter does. For `batch_size`, we used the values `[16, 32, 64, 128]`. For larger values of $n$, we found that a larger batch size was needed (`64` or `128`) but with $n=2$ and $n=4$, we found that `32` worked the best, and `64` worked well enough. For $n=4$, any batch size smaller than `128` led to some serious issues in training - the model would fail to generalize and would "cheat" by returning an array of all zeros or all ones to earn a 50% accuracy.
# 
# We also tried using different optimizers. We noticed that `SGD` with momentum of `0.7` or `0.8` performed slightly better on average than `Adam` in terms of accuracy, but `Adam` was faster to converge. 
# 
# For our learning rate, we experimented with `[0.1, 0.05, 0.01, 0.005]`. The learning rates of `0.005` and `0.001` were training way too slowly and led to a constant learning curve, so we discarded these values. `0.1` was too large and led to instability in the network. `0.05` and `0.01` had variable degrees of success and were mostly performing the same.
# 
# For the choice of $n$, we tried `[1,2,4,6]`. $n=6$ performed similiarly to $n=4$ but was crashing the kernel too often by depleting it of RAM, so we decided to discard it.

# In[14]:


cnn = CNN(4)
train_model(cnn, train_data, batch_size=32, optimizer="sgd", 
      learning_rate=0.005, momentum=0.8, weight_decay=0.7, num_epochs=40)


# In[ ]:


# model learns, but not well enough
# alpha=0.01, batch_size=128
cnn = CNN(4)
train_model(cnn, train_data, batch_size=128, optimizer="adam", 
      learning_rate=0.01, num_epochs=40)


# In[ ]:


cnn = CNN(2)
train_model(cnn, train_data, batch_size=32, optimizer="adam", 
      learning_rate=0.05, num_epochs=40)


# In[ ]:


cnn = CNN(2)
train_model(cnn, train_data, batch_size=32, optimizer="sgd", 
      learning_rate=0.01, num_epochs=10)


# In[ ]:


cnn = CNN(2)
train_model(cnn, train_data, batch_size=32, optimizer="sgd", 
            learning_rate=0.01, num_epochs=35)


# In[ ]:


# This is our best performing model
cnn_n2 = CNN(2)
train_model(cnn_n2, train_data, batch_size=32, optimizer="sgd", 
      learning_rate=0.001, momentum=0.7, weight_decay=0.5, num_epochs=35)


# #### Training CNNChannel

# We experimented with several values for each hyperparameter. Our main method was using a grid search strategy. Our learning rate values were `[0.5, 0.1, 0.05, 0.01, 0.005, 0.001]`. It turns out `0.5` and `0.1` are too large for this model and lead to fluctuations in the loss. We are showcasing the values `0.01` and `0.05` for the learning rate.
# 
# For the batch size, we tried the values `[16, 32, 64]`. All these values had mostly good results, though a batch size of `16` led to more overfitting. We found that `32` and `64` had the best results, but `64` converged to a good validation accuracy faster.
# 
# We also experimented with different optimizers - namely `SGD` and `Adam`. We ended up keeping `Adam` since it seemed less susceptible to exploding and vanishing gradients, whereas `SGD` performed well on some iterations but would vanish on others. 
# 
# For the value of $n$, we used the values `[2,4,6,8]`. $n=2$ yielded acceptable results (around $70\%$ accuracy), and $n=4$ yielded very good results, with a validation accuracy of around $83\%$. As for $n=6$, it performed slightly better than $n=4$ but was so slow we decided to keep $n=4$ as our go-to value. We did not show $n=6$ and $n = 8$ below because they have a nasty habit of using too much RAM and crashing the Colab kernel. 

# In[ ]:


# Testing learning rate of 0.1
cnn_c_n2 = CNNChannel(2)
train_model(cnn_c_n2, train_data, batch_size=64, optimizer="adam", 
      learning_rate=0.1, num_epochs=10)


# In[ ]:


cnn_c_n2 = CNNChannel(2)
train_model(cnn_c_n2, train_data, batch_size=64, optimizer="adam", 
      learning_rate=0.01, num_epochs=20)


# In[ ]:


# Using a smaller batch size
cnn_c_n2 = CNNChannel(2)
train_model(cnn_c_n2, train_data, batch_size=32, optimizer="adam", 
      learning_rate=0.005, num_epochs=20)


# In[25]:


cnn_c_n4 = CNNChannel(4)
train_model(cnn_c_n4, train_data, batch_size=64, optimizer="adam", 
      learning_rate=0.005, checkpoint_path=CHECKPOINT_PATH, num_epochs=25)


# Looking at the training curves and the reported validation accuracies, the best `CNN` model is `cnn_n2`, with a maximum validation accuracy of 56%, and the best `CNNChannel` model is `cnn_c_n4`, achieving a maximum validation accuracy of 88%.

# ## 4. Testing

# Here we report the test accuracies of our single best model, separately for the two test sets.
# We do this by choosing the checkpoint of the model architecture that produces the best validation accuracy.

# The epoch with the highest validation accuracy is epoch 23. However, the difference between training and validation accuracies leads us to believe that it is overfitting
# We load epoch 11 instead, with validation accuracy of 88%

# In[44]:


checkpt = torch.load(CHECKPOINT_PATH.format(11)) # load checkpoint for epoch 11
cnn_c_n4.load_state_dict(checkpt)


# In[45]:


# (Pos Accuracy, Neg Accuracy)
get_test_accuracy(cnn_c_n4, test_m_data)


# In[46]:


# (Pos Accuracy, Neg Accuracy)
get_test_accuracy(cnn_c_n4, test_w_data)


# We see a positive accuracy of 83% for both men and women's shoes. The negative accuracy is 70% for men's shoes and 73% for women's shoes.

# Here is a set of men's shoes that our model correctly classified:

# In[47]:


test_m_same_pairs = generate_same_pair(test_m_data)
plt.figure()
plt.imshow(test_m_same_pairs[0])


# We can also find a pair that our model classified incorrectly:

# In[48]:


pred_0 = cnn_c_n4(torch.Tensor(test_m_same_pairs).permute(0,3,1,2))[0]
is_from_same_pair = (pred_0[1] > pred_0[0]).item()
is_from_same_pair


# In[49]:


plt.figure()
plt.imshow(test_m_same_pairs[-4])


# In[50]:


pred_n4 = cnn_c_n4(torch.Tensor(test_m_same_pairs).permute(0,3,1,2))[-4]
is_from_same_pair = (pred_n4[1] > pred_n4[0]).item()
is_from_same_pair


# We repeat the same process with women's shoes. Here are pairs of shoes that were classified correctly and incorrectly, respectively:

# In[51]:


test_w_same_pairs = generate_same_pair(test_w_data)
plt.figure()
plt.imshow(test_w_same_pairs[0])


# In[52]:


pred_0 = cnn_c_n4(torch.Tensor(test_w_same_pairs).permute(0,3,1,2))[0]
is_from_same_pair = (pred_0[1] > pred_0[0]).item()
is_from_same_pair


# In[53]:


plt.figure()
plt.imshow(test_w_same_pairs[10])


# In[54]:


pred_10 = cnn_c_n4(torch.Tensor(test_w_same_pairs).permute(0,3,1,2))[10]
is_from_same_pair = (pred_10[1] > pred_10[0]).item()
is_from_same_pair

