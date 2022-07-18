# Traffic

Traffic Signal classifier is an Artificial Intelligence model that can detect the traffic signals from the boards.

## Getting Started

```
pip install -r requirements.txt
```

Running the Program!

```
python traffic.py
```

Boom! you got it working!

### Problems

In `load_data` function i wasn't able to read the image, I tried `os.read`, and after sometime with `mode='rb'` i got the same error.
then after a lot of search, i found that the image is in the form of a `numpy array` and i tried to read it using `cv2.imread` and finally it worked!
then slowly I created that function and it worked!
In `get_model` function, I was using gtsrb-small model from the start because I thought it would be easier to train. but then as I did it I didn't know where the error was comming from because I did everything right, applied a lot of filters, pooling, but still got a error with this name:

```
ValueError: Shapes (None, 3) and (None, 43) are incompatible
```

but then after a lot of time i found that the error was because of the shape of the image and the model. so i changed the model to the gtsrb model and it worked!