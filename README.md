# Wide-Residual-Network-Tensorflow

This is an implementation of Wide Residual Network using [Tensorflow](https://www.tensorflow.org/) library for image classification The model was trained and tested on [cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The model was then used to make predictions on some images downloaded from Google.

Some sample <b>Cifar10</b> training images (after some processing):

![png](/Data_images/output_4_1.png)

Details on Wide Residual Network can be found [here](https://arxiv.org/abs/1605.07146) and [here](https://github.com/szagoruyko/wide-residual-networks)

I trained a 16-2 WRN 3,3 type block (with dropout) with batch size 120. Any L layered WRN 3,3 type block with K widening factor block can be constructed by simply changing the value of the variables 'layers' and 'K' in the code.

I acheived an accuracy of about 90.15% which is quite low compared to the state of art performance an WRN is supposed to acheive.



![png](/Data_images/output_13_0.png)



![png](/Data_images/output_13_1.png)


There are various potential reasons for the low accuracy:

1) Due to various circustantial issues, I could train the model for relative few iterations (approximately 100 epochs only)
2) Performance might be better on wider and deeper WRNs
3) I implemented very light augmentations (only horizontal flips). Cropping with mirror padding may be worth a try. 
4) For preprocessing I only used global contrast normalization without ZCA whitening. I didn't use meanstd either. Different preprocessing steps may produce noticeably different results.
5) No L2 regularization. There seems to be some overfitting. 
6) Certain hyperparameters may need to be further optimized.

It may be wortthy to try this network out with [ELUs](https://arxiv.org/abs/1511.07289) (removing BN and ReLu).

# File Descriptions:

**DataProcessing.ipynb:** This consists of the code for performing some basic preprocessing on the cifar10 data and saving the processed data in an hdf5 file.

**Model(WRN).ipynb:** This consists of the code for retreiving the processed data, functions for creating unbiased and augmented training batches during training at realtime, model definition and construction, training (along with checkpoints and model saving) and plots.

**Predict.ipynb:** This file is for restoring the saved model and using the model for making new predictions on any images in a specified directory. I tested the model by making it predict the class of several images downloaded through Google.

**Predict-lite.ipynb:** Same as Predict.ipynb but with less downloaded pictures being tested. Try this file if Predict.ipynb takes too much time to load or open. 

The **Model_Backup** folder contains files for the trained model which can be loaded for prediction or further training.

# Some example Predictions:

Note: The followed Images are downloaded through google-images. I don't know the exact sources. I am not sure if there are any license\copyright issues associated with these images. 

    
    Enter relative path to the image: cartoonaeroplane.png



![png](/Images/output_5_3.png)


    
    There's about a 70.990% chance that there is at least one airplane in the image
    
    The whole probability distribution:
    
    airplane: 70.990%
    automobile: 28.987%
    bird: 0.019%
    cat: 0.001%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.002%
    truck: 0.000%
    
    Enter relative path to the image: car1.jpg



![png](/Images/output_5_5.png)


    
    There's about a 97.054% chance that there is at least one automobile in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 97.054%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 2.945%
    
    Enter relative path to the image: car2.jpg



    Enter relative path to the image: bird1.jpg



![png](/Images/output_5_13.png)


    
    There's about a 100.000% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 100.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: bird2.jpg



![png](/Images/output_5_15.png)


    
    There's about a 99.986% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 0.014%
    automobile: 0.000%
    bird: 99.986%
    cat: 0.001%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: bird3.jpg



![png](/Images/output_5_17.png)


    
    There's about a 99.551% chance that there is at least one horse in the image
    
    The whole probability distribution:
    
    airplane: 0.001%
    automobile: 0.000%
    bird: 0.021%
    cat: 0.002%
    deer: 0.373%
    dog: 0.053%
    frog: 0.001%
    horse: 99.551%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: cat1.jpg



![png](/Images/output_5_19.png)


    
    There's about a 95.607% chance that there is at least one cat in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.052%
    cat: 95.607%
    deer: 0.001%
    dog: 4.319%
    frog: 0.022%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: cat2.jpeg



![png](/Images/output_5_21.png)


    
    There's about a 96.047% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.453%
    cat: 3.372%
    deer: 0.122%
    dog: 96.047%
    frog: 0.003%
    horse: 0.000%
    ship: 0.003%
    truck: 0.000%
    
    Enter relative path to the image: cat3.jpg



![png](/Images/output_5_23.png)


    
    There's about a 99.943% chance that there is at least one cat in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 99.943%
    deer: 0.000%
    dog: 0.057%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: deer1.jpg



![png](/Images/output_5_25.png)


    
    There's about a 100.000% chance that there is at least one deer in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 100.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    
    
    Enter relative path to the image: dog1.jpg



![png](/Images/output_5_31.png)


    
    There's about a 99.901% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.080%
    cat: 0.018%
    deer: 0.000%
    dog: 99.901%
    frog: 0.001%
    horse: 0.001%
    ship: 0.000%
    truck: 0.000%
    
    
    Enter relative path to the image: cartoondog.png



![png](/Images/output_5_41.png)


    
    There's about a 67.493% chance that there is at least one cat in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 67.493%
    deer: 0.000%
    dog: 32.504%
    frog: 0.000%
    horse: 0.004%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: frog1.jpg



![png](/Images/output_5_43.png)


    
    There's about a 93.749% chance that there is at least one frog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 6.251%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 93.749%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    
    Enter relative path to the image: frog3.jpg



![png](/Images/output_5_47.png)


    
    There's about a 96.741% chance that there is at least one frog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 2.681%
    bird: 0.576%
    cat: 0.002%
    deer: 0.000%
    dog: 0.000%
    frog: 96.741%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: horse1.jpg



![png](/Images/output_5_49.png)


    
    There's about a 100.000% chance that there is at least one horse in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 100.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: horse2.jpg



![png](/Images/output_5_51.png)


    
    There's about a 94.826% chance that there is at least one horse in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.011%
    deer: 0.001%
    dog: 0.003%
    frog: 0.000%
    horse: 94.826%
    ship: 0.000%
    truck: 5.159%
    
    Enter relative path to the image: ship2.jpg



![png](/Images/output_5_55.png)


    
    There's about a 95.020% chance that there is at least one ship in the image
    
    The whole probability distribution:
    
    airplane: 4.504%
    automobile: 0.000%
    bird: 0.244%
    cat: 0.230%
    deer: 0.000%
    dog: 0.002%
    frog: 0.000%
    horse: 0.000%
    ship: 95.020%
    truck: 0.000%
    
    Enter relative path to the image: ship3.jpeg



![png](/Images/output_5_57.png)


    
    There's about a 100.000% chance that there is at least one ship in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 100.000%
    truck: 0.000%
    
    Enter relative path to the image: truck2.jpg



![png](/Images/output_5_61.png)


    
    There's about a 100.000% chance that there is at least one truck in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 100.000%
    
    Enter relative path to the image: truck3.jpg


    Enter relative path to the image: STOP



```python

```

