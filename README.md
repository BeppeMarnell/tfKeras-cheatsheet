# tfKeras-cheatsheet
I have been recently working intensively with tensorflow/keras for my master's thesis and I have gathered code that if I had beforehand, it could have saved me some time. I will try to update this repository constantly with good and useful stuff. Contribuitions are welcomed.

- More is to come, sorry if the following is a little bit messy.


# Table of contents
1. [Create dataset from numpy](#dataset_numpy)
2. [Visualize dataset samples](#visualize_samples)
3. [Custom Callback for keras](#custom_callbacks)


## Quickly create dataset from numpy <a name="dataset_numpy"></a>
Something that I love doing is to preprocess all the data on my computer and upload it onto my google drive (this because I use Colab). This avoids excessive loading times when in a hurry, but it may overload RAM and GPU, depending on how big is your dataset. I usually can fit 10GB of samples.

```python
    import numpy as np
    import tensorflow as tf
    
    BATCH=16
    BUFFER=200

    # load the training dataset
    X_train = np.load('dataset.npy')
    Y_train = np.load('dataset_labels.npy')
    
    # load the evaluation dataset
    X_val = np.load('dataset_val.npy')
    Y_val = np.load('dataset_val_labels.npy')
    
    # load datasets 
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    train_dataset = train_dataset.shuffle(BUFFER).batch(BATCH)
    val_dataset = val_dataset.batch(BATCH)

```

## Quickly create dataset from numpy <a name="dataset_numpy"></a>
Ready copy paste code to check whether your dataset has been loaded correctly. This one is a must check before building the model.
Note, <em>your_dataset</em> needs to be <em>tf.data.Dataset</em>. Be aware that you need to index the variable <em>image</em> to the right size/dimension.

```python
image_batch, label_batch = next(iter(your_dataset))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)

    image = image_batch[i].numpy()

    plt.imshow(image)
```


## Custom Callback for keras - [link documentation](https://www.tensorflow.org/guide/keras/custom_callback) <a name="custom_callbacks"></a>
For instance my model kept being stuck in some sort of local minima. With this callback I saved a lot of time in grid search.

```python
from tensorflow import keras

class yourCustomCallback(keras.callbacks.Callback):

    def __init__(self):
        # initialise variables for the next methods

    def on_epoch_end(self, epoch, logs=None):
        # this method gets called when the epoch ends
```
      
