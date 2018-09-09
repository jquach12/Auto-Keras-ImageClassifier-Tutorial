# Auto-Keras-ImageClassifier-Tutorial


## Installation


To install Auto-Keras please use the `pip` installation as follows:

    pip install autokeras
    
**Note:** currently, Auto-Keras is only compatible with: **Python 3.6**.

## Example

Here is a short example of using the package.


    import autokeras as ak

    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)
    
## The Dataset

The dataset contains images of dogs and cats. They are 128x128 and are sRGB.
The train dataset has 10,000 dogs and 10,000 cats.
The validation dataset has 2000 dogs and 2000 cats.
The test dataset has 21 cats and 11 dogs.
