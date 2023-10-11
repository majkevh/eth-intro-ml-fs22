import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Model
#a = list(tfds.as_numpy(train_dataset))
#b = int(np.ceil(len(a) / 32))
#Parameters & load FILE
SHAPE = 224
BATCH = 32
EPOCH = 10
triplets = pd.read_csv (r'train_triplets.txt',sep=' ',names=['A', 'B','C'], dtype = str)
#triplets= triplets.head(1000)
test_triplets = pd.read_csv (r'test_triplets.txt',sep=' ',names=['A', 'B','C'], dtype = str)
number_triplets_train=len(triplets)
number_triplets_test=len(test_triplets)

########################################################################################################################

#Import functions and preprocessing functions
def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (SHAPE, SHAPE))
    #image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image



def preprocess_triplets(anchor, positive, negative):
    return tf.stack([preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative)], axis=0), 1
def preprocess_triplets_test(anchor, positive, negative):
    return tf.stack([preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative)], axis=0)

########################################################################################################################

#Import Data
def import_data(train):
    if train:
        anchor_set = tf.data.Dataset.from_tensor_slices([r'food/'+ triplets.at[i, 'A'] + '.jpg' for i in range(number_triplets_train)])
        positive_set = tf.data.Dataset.from_tensor_slices([r'food/'+ triplets.at[i, 'B'] + '.jpg' for i in range(number_triplets_train)])
        negative_set = tf.data.Dataset.from_tensor_slices([r'food/'+ triplets.at[i, 'C'] + '.jpg' for i in range(number_triplets_train)])
    else:
        anchor_set = tf.data.Dataset.from_tensor_slices([r'food/'+ test_triplets.at[i, 'A'] + '.jpg' for i in range(number_triplets_test)])
        positive_set = tf.data.Dataset.from_tensor_slices([r'food/'+ test_triplets.at[i, 'B'] + '.jpg' for i in range(number_triplets_test)])
        negative_set = tf.data.Dataset.from_tensor_slices([r'food/'+ test_triplets.at[i, 'C'] + '.jpg' for i in range(number_triplets_test)])
    return (anchor_set, positive_set, negative_set)


########################################################################################################################
#Preprocess
#Train
dataset = tf.data.Dataset.zip(import_data(True))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

percentage_train = 0.8
train_dataset = dataset.take(round(number_triplets_train * percentage_train))
train_dataset = train_dataset.batch(BATCH, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(round(number_triplets_train * percentage_train))
val_dataset = val_dataset.batch(BATCH, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

#Test
data_test = tf.data.Dataset.zip(import_data(False))
data_test = data_test.map(preprocess_triplets_test)
data_test = data_test.batch(1, drop_remainder=False)
data_test = data_test.prefetch(tf.data.AUTOTUNE)


########################################################################################################################
#Create Model
def triplets_model():
    trainable = False
    triplet_input = tf.keras.Input(shape=(3, SHAPE, SHAPE, 3))
    base_cnn = tf.keras.applications.EfficientNetV2L(weights = 'imagenet', input_shape=(SHAPE, SHAPE, 3), include_top=False)
    base_cnn.trainable = trainable
    
    trainmodel = tf.keras.Sequential([   
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dropout(0.5),#REMOVE
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),#REMOVE
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    
    (model_anchor, model_positive, model_negative) = (triplet_input[:, 0, ...], triplet_input[:, 1, ...], triplet_input[:, 2, ...])
    anchor_embedded = trainmodel(base_cnn(model_anchor))
    positive_embedded = trainmodel(base_cnn(model_positive))
    negative_embedded = trainmodel(base_cnn(model_negative))
    Embedded = tf.stack([anchor_embedded, positive_embedded, negative_embedded], axis=-1)
    
    Triplets_Siamese_Network = Model(inputs=triplet_input, outputs=Embedded)
    Triplets_Siamese_Network.summary()
    return Triplets_Siamese_Network





#Use euclidian distance to mesure distance from anchor
def euclidian_distance(Embedded):
    model_anchor, model_positive, model_negative = Embedded[..., 0], Embedded[..., 1], Embedded[..., 2]
    distance1 = tf.reduce_sum(tf.square(model_anchor - model_positive), 1)
    distance2 = tf.reduce_sum(tf.square(model_anchor - model_negative), 1)
    return (distance1, distance2)

#Closest distance
def closest_distance(distance1, distance2, type_variable):
    return tf.cast(tf.less_equal(distance1, distance2), type_variable)

#Define Loss
def triplet_loss(self, Embedded):
    distance1, distance2 = euclidian_distance(Embedded)
    margin = 0.5
    return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))

#Accuracy: positive must be closer to anchor than negative
def metrics(self, Embedded):
    distance1, distance2 = euclidian_distance(Embedded)
    return tf.reduce_mean(closest_distance(distance1, distance2, tf.float32))

#Use trained model to evaluate test model
def test_model(model):
    distance1, distance2 = euclidian_distance(model.output)
    return Model(inputs=model.inputs, outputs=closest_distance(distance1, distance2, tf.int8))

########################################################################################################################
#Evaluate Model
#Train
model = triplets_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss, metrics=metrics)
history = model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset)
#Test && Save
model_test = test_model(model)
result = model_test.predict(data_test, verbose=1)
np.savetxt('result.csv', result, fmt='%i')