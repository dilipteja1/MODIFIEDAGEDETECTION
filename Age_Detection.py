
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import os
import random

import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer


# In[ ]:


root_dir = os.path.abspath(r"C:\Users\Hello\Desktop\age-detection-using-CNN-master")
data_dir = r'C:\Users\Hello\Desktop\age-detection-using-CNN-master\data'

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))


# In[ ]:



i = random.choice(train.index)


img_name = train.ID[i]
img = imread(os.path.join(data_dir, 'Train', img_name))


imshow(img)
print('Age: ', train.Class[i])


# In[ ]:


from scipy.misc import imresize

temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    img = img.astype('float32') 
    temp.append(img)
    #print(1)

train_x = np.stack(temp)


# In[ ]:


temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)


# In[ ]:


train_x = train_x / 255.
test_x = test_x / 255.


# In[ ]:


train.Class.value_counts(normalize=True)


# In[ ]:


import keras
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)


# In[ ]:



train_x_temp = train_x.reshape(-3, 32, 32, 3)

input_shape = (1024,)
input_reshape = (32, 32, 3)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden_num_units = 1536
output_num_units = 3

epochs = 7
batch_size = 250

model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(96, (3, 3), activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(256, (2, 2), activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(256, (2, 2), activation='relu'),
 MaxPooling2D(pool_size=pool_size),
        
 Convolution2D(384, (2, 2), activation='relu'),

 Flatten(),
 Dense(units=1536, activation='relu'),
 Dense(units=output_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.3)


# In[ ]:


sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))


# In[ ]:


test_x_temp = test_x.reshape(-3, 32, 32, 3)
pred = model.predict_classes(test_x_temp)

pred.shape


# In[ ]:


pred_f = lb.inverse_transform(pred)


# In[ ]:


sample_submission.ID = test.ID; sample_submission.Class = pred_f
sample_submission.to_csv(os.path.join(data_dir, 'sub.csv'), index=False)


# In[ ]:


sample_submission.shape

