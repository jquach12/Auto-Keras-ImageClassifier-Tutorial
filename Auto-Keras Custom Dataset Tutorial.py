
# coding: utf-8

# # Auto-Keras, the Open Source Neural Architecture Search 
# Website: https://autokeras.com/ Citation: arXiv:1806.10282 Paper: https://arxiv.org/abs/1806.10282
# 
# (Version 0.2.13 as of 9/8/18)

# In[5]:


from autokeras.image_supervised import ImageClassifier, load_image_dataset


# ## Loading the Images and Labels
# 
# Images should be squares. To avoid tedious errors, they should also be either gray scale (1 color channel) or sRGB (3 color channels).
# 
# The labels are in a .csv file with the format
# 
# img0, 0 <br>
# img1, 1 <br>
# img2, 1 <br>
# img3, 2 <br>
# . <br>
# . <br>
# . <br>
# imgN, 0

# In[6]:


train_path = 'datasets/dogsCats/dogCat128rgb_train'
train_labels = 'datasets/dogsCats/dogsCats_train.csv'

validation_path = 'datasets/dogsCats/dogCat128rgb_val'
validation_labels = 'datasets/dogsCats/dogsCats_val.csv'

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(train_path) if isfile(join(train_path,f))]
print(files)


# ## Using load_image_dataset for Your Own Image Dataset (arguably the hardest part)
# All images of `x_train` should be the same height and same width and have same color channels to each other.
# The shape of `x_train[0]` and `y_train[0]` should be the same as they both represent the amount of samples.
# 
# Similarly, images in `x_val` should have the same height and same width and have same color channels to each other.
# The shape of `x_val[0]` and `y_val[0]` should be the same as they both represent the amount of samples.
# 
# The .csv file does not have to be in the same order in which the image files are arranged (the images may not be read in order to begin with); you should be fine so long as an image exists in the image path provided and its name is somewhere in the csv file along with its label.
# 
# Use ImageMagick if you have trouble formatting your images! https://www.imagemagick.org/script/index.php

# In[7]:


x_train, y_train = load_image_dataset(csv_file_path=train_labels,
                                    images_path=train_path)
print(x_train.shape)
print(y_train.shape)

x_val, y_val = load_image_dataset(csv_file_path=validation_labels,images_path=validation_path)

print(x_val.shape)
print(y_val.shape)


# ## Searching for the Best Model
# These models can get large, and exponentially get larger as the image sizes get larger. 
# Thus, it is best to scale your images smaller (e.g. 32x32, 64x64, 128x128) to avoid receiving the error: __RuntimeError: CUDA error: out of memory__. This is also good practice to deal with the Curse of Dimensionality.
# 
# Furthermore, you should consider letting `fit()` run for several hours to avoid receiving the error: __TimeoutError: Search Time too short. No model was found during the search time__. I personally feel that `max_iter_num` should be at least 25. Otherwise, any candidate model will have limited training time before the Searcher searches for another model. Auto-Keras will train the model up to `max_iter_num` or if no incremental loss is being made after 5 epochs (whichever one comes first) before training another candidate model.

# In[8]:


clf = ImageClassifier(verbose = True, searcher_args = {'trainer_args':{'max_iter_num': 25}})
clf.fit(x_train,y_train, time_limit = 4 * 60 * 60) # default time_limit is 24 hours 


# ## Fitting the Best Model Found During Search
# NOTE: In Jupyter Notebooks, even if the program crashes during the search process, `final_fit()` will still evaluate the best model amongst the models that were searched before the crash (i.e. `final_fit()` will still run as intended, but with probably fewer models). This process may take several minutes.
# 
# Setting the *retrain* boolean to `True` will maintain the architecture of the model with the highest accuracy, but will reset its weights. Setting it to `False` will not only maintain the architecture but effectively resume its training with the weights it learned during `fit()`
# 

# In[9]:


clf.final_fit(x_train,y_train,x_val,y_val,retrain = True, trainer_args={'max_iter_num':10})


# ## Evaluating the Model

# In[10]:


print(clf.evaluate(x_val,y_val))


# ## Loading the Best Model Found
# 
# By default, load_best_model() will find the best model based on the metric of highest __accuracy__.

# In[11]:


best = clf.load_searcher().load_best_model()
best_torchModel = best.produce_model()
best_kerasModel = best.produce_keras_model()


# ## More Info about the Best Model Found

# In[12]:


print(best.n_layers)


# In[13]:


print(best_torchModel)


# In[14]:


print(best_kerasModel)


# ## Saving the Model to be used in other Projects

# In[15]:


import torch
torch.save(best.produce_model(),'dogsAndCats_model.pt')
loadedTorchModel = torch.load('dogsAndCats_model.pt')

from keras.models import load_model
best_kerasModel.save('dogsAndCats_model.h5')
loadedKerasModel = load_model('dogsAndCats_model.h5')


# ## Sanity Check by Visualizing the Model Predictions

# In[26]:


import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'datasets/dogsCats'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = ['dog','cat']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[27]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[34]:


loadedTorchModel = loadedTorchModel.to(device) # allows model to use GPU
visualize_model(loadedTorchModel,4)

