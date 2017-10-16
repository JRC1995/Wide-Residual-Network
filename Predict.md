

```python
import h5py
import os
import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = h5py.File('processed_data.h5','r+') 

#Retrieves all the preprocessed training and validation\testing data from a file

X_train = file['X_train'][...]
Y_train = file['Y_train'][...]
X_test = file['X_test'][...]
Y_test = file['Y_test'][...]

# Unpickles and retrieves class names and other meta informations of the database
classes = unpickle('cifar-10-batches-py/batches.meta') #keyword for label = label_names

# The steps below are completely unncessary. A long time ago, I put some modified version of X_train and X_test 
# into the X_train_feed and X_test_feed variables but things changed since then (as in I discarded those modifications). 
# But even though things changedit was difficult to change back all variables X_train_feed and X_test_feed 
# that are used later on the code to X_train and X_test which is why I made these unnecessary steps to put X_train and X_test
# into X_train_feed and X_test_feed

X_train_feed = X_train
X_test_feed = X_test
```


```python
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
```


```python
# Copy pasted the model from Model(WRN).ipynb file to initialize all the variables properly which needs to be restored from 
# the saved model file.
# (There's probably a better way to do this without rewriting all these steps but this is basically a quick fix I managed 
# to use to properly restore a saved model and use it for making predictions)

#Hyper Parameters!
learning_rate = 0.01
batch_size = 120
training_iters = 200*(int(len(X_train)/batch_size))
layers = 16

# 1 conv + 3 convblocks*(3 conv layers *1 group for each block + 2 conv layers*(N-1) groups for each block [total 1+N-1 = N groups]) = layers
# 3*2*(N-1) = layers - 1 - 3*3
# N = (layers -10)/6 + 1

N = ((layers-10)/6)+1
K = 4 #(deepening factor)

#(N and K are used in the same sense as defined here: https://arxiv.org/abs/1605.07146)

n_classes = len(classes['label_names']) # another useless step that I made because of certain reasons. 

# tf Graph input

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
phase = tf.placeholder(tf.bool, name='phase') 
#(Phase = true means training is undergoing. The contrary is ment when Phase is false.)

# Create some wrappers for simplicity

def conv2d(x,shape,strides):
    # Conv2D wrapper
    W = tf.Variable(tf.truncated_normal(shape=shape,stddev=5e-2))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    # Didn't add bias because I read somewhere it's not necessary to add a bias if batch normalization is to be performed later
    # May be add L2 regularization or something here if you wish to.
    return x

def activate(x,phase):
    #wrapper for performing batch normalization and relu activation
    x = tf.contrib.layers.batch_norm(x, center=True, scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"],updates_collections=None, decay=0.9,is_training=phase,zero_debias_moving_mean=True, fused=True)
    return tf.nn.relu(x,'relu')


def wideres33block(X,N,K,iw,bw,s,dropout,phase):
    
    # Creates N no. of 3,3 type residual blocks with dropout that consitute the conv2/3/4 blocks
    # with widening factor K and X as input. s is stride and bw is base width (no. of filters before multiplying with k)
    # iw is input width.
    # (see https://arxiv.org/abs/1605.07146 paper for details on the block)
    # In this case, dropout = probability to keep the neuron enabled.
    # phase = true when training, false otherwise.
    
    conv33_1 = conv2d(X,[3,3,iw,bw*K],s)
    conv33_1 = activate(conv33_1,phase)
    
    conv33_1 = tf.nn.dropout(conv33_1,dropout)
    
    conv33_2 = conv2d(conv33_1,[3,3,bw*K,bw*K],1)
    conv_s_1 = conv2d(X,[1,1,iw,bw*K],s) #shortcut connection
    
    caddtable = tf.add(conv33_2,conv_s_1)
    
    #1st of the N blocks for conv2/3/4 block ends here. The rest of N-1 blocks will be implemented next with a loop.

    for i in range(0,N-1):
        
        C = caddtable
        Cactivated = activate(C,phase)
        
        conv33_1 = conv2d(Cactivated,[3,3,bw*K,bw*K],1)
        conv33_1 = activate(conv33_1,phase)
        
        conv33_1 = tf.nn.dropout(conv33_1,dropout)
            
        conv33_2 = conv2d(conv33_1,[3,3,bw*K,bw*K],1)
        caddtable = tf.add(conv33_2,C)
    
    return activate(caddtable,phase)


    
def WRN(x, dropout, phase): #Wide residual network

    conv1 = conv2d(x,[3,3,3,16],1)
    conv1 = activate(conv1,phase)

    conv2 = wideres33block(conv1,N,K,16,16,1,dropout,phase)
    conv3 = wideres33block(conv2,N,K,16*K,32,2,dropout,phase)
    conv4 = wideres33block(conv3,N,K,32*K,64,2,dropout,phase)

    pooled = tf.nn.avg_pool(conv4,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
    
    #Initialize weights and biases for fully connected layers
    wd1 = tf.Variable(tf.truncated_normal([1*1*64*K, 64*K],stddev=5e-2))
    bd1 = tf.Variable(tf.constant(0.1,shape=[64*K]))
    wout = tf.Variable(tf.random_normal([64*K, n_classes]))
    bout = tf.Variable(tf.constant(0.1,shape=[n_classes]))

    # Fully connected layer
    # Reshape pooling layer output to fit fully connected layer input
    
    fc1 = tf.reshape(pooled, [-1, wd1.get_shape().as_list()[0]])   
    fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
    fc1 = tf.nn.relu(fc1)

    #fc1 = tf.nn.dropout(fc1, dropout) #Not sure if I should or should not apply dropout here.
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, wout), bout)
    
    return out

# Construct model
model = WRN(x,keep_prob,phase)

# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
prediction = tf.nn.softmax(logits=model)

global_step = tf.Variable(0)

#learning_rate = tf.train.exponential_decay(init_lr,global_step*batch_size, decay_steps=len(X_train), decay_rate=0.95, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = 0.9, use_nesterov=True).minimize(cost,global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
```


```python
with tf.Session() as sess: # Begin session
    
    # Test the avergae accuracy of the model on the testing batch to be sure that model is loaded properly
    
    print 'Loading pre-trained weights for the model...'
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    print '\nRESTORATION COMPLETE\n'
    
    print 'Testing Model Performance...'
    total_val_loss=0
    total_val_acc=0
    val_loss=0
    val_acc=0
    avg_val_loss=0
    avg_val_acc=0
            
    for i in xrange(0,len(X_test_feed)):
        val_loss, val_acc = sess.run([cost, accuracy], feed_dict={x: X_test_feed[i].reshape((1,32,32,3)),
                                                                  y: Y_test[i].reshape((1,n_classes)),
                                                                  keep_prob: 1,
                                                                  phase: False})

        total_val_loss= total_val_loss+val_loss
        total_val_acc = total_val_acc+val_acc
                
    avg_val_loss = total_val_loss/len(X_test_feed)
    avg_val_acc = total_val_acc/len(X_test_feed)
    
    print "Validation Loss = " + \
          "{:.3f}".format(avg_val_loss) + ", validation Accuracy = " + \
          "{:.3f}%".format(avg_val_acc*100)
```

    Loading pre-trained weights for the model...
    
    RESTORATION COMPLETE
    
    Testing Model Performance...
    Validation Loss = 0.446, validation Accuracy = 90.150%



```python

# Converts any image into a square by cropping
def ToSquare(img):
    if img.shape[0]>img.shape[1]:
        extra=img.shape[0]-img.shape[1]
        if extra%2==0:
            crop = img[extra//2:(-extra//2):]
        else:
            crop = img[max(0,1+extra//2):min(-1,(-extra//2)),:]
    elif img.shape[1]>img.shape[0]:
        extra=img.shape[1]-img.shape[0]
        if extra%2==0:
            crop = img[:,extra//2:(-extra//2)]
        else:
            crop = img[:,max(0,1+extra//2):min(-1,(-extra//2))]
    elif img.shape[1]==img.shape[0]:
        crop=img
    return crop

# Applies global contrast normalization.
def global_contrast_normalization(img,s, lmda, epsilon):
    X = np.array(img)
    X_average = np.mean(X)
    X = X - X_average
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)
    return X

# Performs all necessary preprocessing on an image before feeding it to the network for prediction.
def imgprocess(img):
    SqrImg = ToSquare(img)
    TinyImg = imresize(SqrImg,(32,32))
    GcnImg =  global_contrast_normalization(TinyImg,1, 10, 0.000000001)
    return GcnImg
```


```python
import matplotlib.pyplot as plt
from scipy.misc import toimage
from scipy.misc import imresize
%matplotlib inline

filename = ""

with tf.Session() as sess: # Begin Session
    
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    
    while True:
        
        filename = raw_input("Enter relative path to the image: ")
        if filename == "STOP": # Entering "STOP" as a filename will break the loop. 
            break
        img = open(filename,'r')
        img = plt.imread(img)

        plt.imshow(img)
        plt.show()

        TestImg = imgprocess(img)
        
        """print "After Processing: "
        plt.imshow(toimage(TestImg))
        plt.show()"""

        result = sess.run([prediction],feed_dict={x: TestImg.reshape((1,32,32,3)),keep_prob: 1,phase: False})
    
        model_prediction = result[0][0]
        
        most_probable_class = np.argmax(model_prediction)
        
        print "\nThere's about a %.3f%%"%(model_prediction[most_probable_class]*100) + \
              " chance that there is at least one "+classes['label_names'][most_probable_class]+" in the image\n"
        
        print "The whole probability distribution:\n"

        for i in xrange(0,10):
            print str(classes['label_names'][i])+": %.3f%%"%(model_prediction[i]*100)
        
        print("")
        
```

    Enter relative path to the image: aeroplane1.jpg



![png](output_5_1.png)


    
    There's about a 100.000% chance that there is at least one airplane in the image
    
    The whole probability distribution:
    
    airplane: 100.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: cartoonaeroplane.png



![png](output_5_3.png)


    
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



![png](output_5_5.png)


    
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



![png](output_5_7.png)


    
    There's about a 100.000% chance that there is at least one automobile in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 100.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: car3.jpg



![png](output_5_9.png)


    
    There's about a 99.126% chance that there is at least one automobile in the image
    
    The whole probability distribution:
    
    airplane: 0.005%
    automobile: 99.126%
    bird: 0.000%
    cat: 0.001%
    deer: 0.000%
    dog: 0.000%
    frog: 0.002%
    horse: 0.001%
    ship: 0.003%
    truck: 0.864%
    
    Enter relative path to the image: car4.jpg



![png](output_5_11.png)


    
    There's about a 100.000% chance that there is at least one automobile in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 100.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: bird1.jpg



![png](output_5_13.png)


    
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



![png](output_5_15.png)


    
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



![png](output_5_17.png)


    
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



![png](output_5_19.png)


    
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



![png](output_5_21.png)


    
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



![png](output_5_23.png)


    
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



![png](output_5_25.png)


    
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
    
    Enter relative path to the image: deer2.jpg



![png](output_5_27.png)


    
    There's about a 72.604% chance that there is at least one bird in the image
    
    The whole probability distribution:
    
    airplane: 0.422%
    automobile: 0.000%
    bird: 72.604%
    cat: 0.000%
    deer: 25.153%
    dog: 0.000%
    frog: 1.821%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: deer3.jpg



![png](output_5_29.png)


    
    There's about a 99.965% chance that there is at least one deer in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.034%
    cat: 0.000%
    deer: 99.965%
    dog: 0.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: dog1.jpg



![png](output_5_31.png)


    
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
    
    Enter relative path to the image: dog2.jpg



![png](output_5_33.png)


    
    There's about a 100.000% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 100.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: dog3.jpg



![png](output_5_35.png)


    
    There's about a 99.999% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.001%
    deer: 0.000%
    dog: 99.999%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: cartoondog.png



![png](output_5_37.png)


    
    There's about a 57.800% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 2.312%
    automobile: 0.000%
    bird: 14.730%
    cat: 0.510%
    deer: 14.800%
    dog: 57.800%
    frog: 0.317%
    horse: 9.531%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: catdog1.jpg



![png](output_5_39.png)


    
    There's about a 100.000% chance that there is at least one dog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 100.000%
    frog: 0.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: catdog2.jpg



![png](output_5_41.png)


    
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



![png](output_5_43.png)


    
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
    
    Enter relative path to the image: frog2.jpg



![png](output_5_45.png)


    
    There's about a 100.000% chance that there is at least one frog in the image
    
    The whole probability distribution:
    
    airplane: 0.000%
    automobile: 0.000%
    bird: 0.000%
    cat: 0.000%
    deer: 0.000%
    dog: 0.000%
    frog: 100.000%
    horse: 0.000%
    ship: 0.000%
    truck: 0.000%
    
    Enter relative path to the image: frog3.jpg



![png](output_5_47.png)


    
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



![png](output_5_49.png)


    
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



![png](output_5_51.png)


    
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
    
    Enter relative path to the image: ship1.jpg



![png](output_5_53.png)


    
    There's about a 99.971% chance that there is at least one airplane in the image
    
    The whole probability distribution:
    
    airplane: 99.971%
    automobile: 0.000%
    bird: 0.017%
    cat: 0.001%
    deer: 0.000%
    dog: 0.002%
    frog: 0.000%
    horse: 0.001%
    ship: 0.008%
    truck: 0.000%
    
    Enter relative path to the image: ship2.jpg



![png](output_5_55.png)


    
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



![png](output_5_57.png)


    
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
    
    Enter relative path to the image: truck1.jpg



![png](output_5_59.png)


    
    There's about a 96.710% chance that there is at least one airplane in the image
    
    The whole probability distribution:
    
    airplane: 96.710%
    automobile: 0.000%
    bird: 0.048%
    cat: 0.044%
    deer: 0.000%
    dog: 0.000%
    frog: 0.000%
    horse: 0.021%
    ship: 0.343%
    truck: 2.833%
    
    Enter relative path to the image: truck2.jpg



![png](output_5_61.png)


    
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



![png](output_5_63.png)


    
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
    
    Enter relative path to the image: STOP



```python

```
