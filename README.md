# tfKeras-cheatsheet
I have been recently working intensively with tensorflow/keras for my master's thesis and I have gathered code that if I had beforehand, it could have saved me some time. I will try to update this repository constantly with good and useful stuff. Contribuitions are welcomed.

- More is to come, sorry if the following is a little bit messy.


## Custom Callback for keras - [link documentation](https://www.tensorflow.org/guide/keras/custom_callback)
For instance my model kept being stuck in some sort of local minima. With this callback I saved a lot of time in grid search.

```python
class yourCustomCallback(keras.callbacks.Callback):

    def __init__(self):
        # initialise variables for the next methods

    def on_epoch_end(self, epoch, logs=None):
        # this method gets called when the epoch ends
```
      
