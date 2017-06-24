
# coding: utf-8

# In[1]:

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
import os


# In[ ]:

def vgg16(input=None, classes=1000):
    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1', trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1', trainable=False)
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    # change the structure, now fc only has 2048, leass parameters, which is enough for this task
    x = tflearn.fully_connected(x, 2048, activation='relu', scope='fc7', restore=False)
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, classes, activation='softmax', scope='fc8', restore=False)

    return x


# In[ ]:

# mean pixel value of r, g, b channels of ImageNet dataset
MEAN_VALUE = [123.68, 116.779, 103.939]

def train():
    model_path = '.'
    file_list = './train_fvgg_emo.txt'
    X, Y = image_preloader(file_list, image_shape=(224,224), mode='file', categorical_labels=True,
                           normalize=False, files_extension=['.jpg', '.png'], filter_channel=True)

    classes = 7

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=MEAN_VALUE, per_channel=True)

    x = tflearn.input_data(shape=[None, 224, 224, 3], name='input', data_preprocessing=img_prep)
    softmax = vgg16(x, classes)
    # default optimizer='adam', loss='categorical_crossentropy'
    regression = tflearn.regression(softmax, learning_rate=0.0001, restore=False)
    # tensorboard_verbose=3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.(Best visualization)
    model = tflearn.DNN(regression, checkpoint_path='./logs/vgg-finetuning/checkpoints/', max_checkpoints=3, tensorboard_verbose=2, tensorboard_dir='./logs/vgg-finetuning/summaries/')
    model_file = os.path.join(model_path, 'vgg16.tflearn')
    model.load(model_file, weights_only=True)

    # start finetuning
    model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_epoch=False, snapshot_step=200, run_id='vgg-finetuning')
    model.save('./logs/vgg-finetuning/vgg_finetune_emo.tfmodel')



if __name__ == '__main__':
    train()
