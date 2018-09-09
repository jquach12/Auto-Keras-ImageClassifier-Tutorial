# coding: utf-8

# Auto-Keras, the Open Source Neural Architecture Search 
# Website: https://autokeras.com/ Citation: arXiv:1806.10282 Paper: https://arxiv.org/abs/1806.10282
# (Version 0.2.13 as of 9/8/18)

from autokeras.image_supervised import ImageClassifier, load_image_dataset

train_path = 'datasets/dogsCats/dogCat128rgb_train'
train_labels = 'datasets/dogsCats/dogsCats_train.csv'

validation_path = 'datasets/dogsCats/dogCat128rgb_val'
validation_labels = 'datasets/dogsCats/dogsCats_val.csv'

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(train_path) if isfile(join(train_path,f))]
print(files)

x_train, y_train = load_image_dataset(csv_file_path=train_labels,
                                    images_path=train_path)
print(x_train.shape)
print(y_train.shape)

x_val, y_val = load_image_dataset(csv_file_path=validation_labels,images_path=validation_path)

print(x_val.shape)
print(y_val.shape)

# Searching for the Best Model
clf = ImageClassifier(verbose = True, searcher_args = {'trainer_args':{'max_iter_num': 25}})
clf.fit(x_train,y_train, time_limit = 4 * 60 * 60) # default time_limit is 24 hours 

# Fitting the Best Model Found During Search
clf.final_fit(x_train,y_train,x_val,y_val,retrain = True, trainer_args={'max_iter_num':10})

# Evaluating the Model
print(clf.evaluate(x_val,y_val))

# Loading the Best Model Found
best = clf.load_searcher().load_best_model()
best_torchModel = best.produce_model()
best_kerasModel = best.produce_keras_model()

print(best.n_layers)
print(best_torchModel)
print(best_kerasModel)

# Saving the Model to be used in other Projects
import torch
torch.save(best.produce_model(),'dogsAndCats_model.pt')
loadedTorchModel = torch.load('dogsAndCats_model.pt')

from keras.models import load_model
best_kerasModel.save('dogsAndCats_model.h5')
loadedKerasModel = load_model('dogsAndCats_model.h5')

# Sanity Check by Visualizing the Model Predictions
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

loadedTorchModel = loadedTorchModel.to(device) # allows model to use GPU
visualize_model(loadedTorchModel,4)

