import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet




# mask_train_data_generator=ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     rotation_range=90,
#     zoom_range=(0.95, 0.95), 
#     fill_mode='nearest',
#     # preprocessing_function=preprocess_input

# )

# mask_test_data_generator=ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     rotation_range=90,
#     zoom_range=(0.95, 0.95), 
#     fill_mode='nearest',
#     # preprocessing_function=preprocess_input

# )

training_directory="dataset/mask_nomask/train"
testing_directory="dataset/mask_nomask/test"
batch_size=32
img_height = 224
img_width = 224

# mob_net = tf.keras.applications.MobileNetV2(input_shape=(img_height,img_width,3),include_top=False,weights='imagenet')
# for layer in mob_net.layers:
#     mob_net.trainable=False


# mask_train_data=mask_train_data_generator.flow_from_directory(
#     training_directory,
#     target_size=(img_height,img_width),
#     batch_size=batch_size,
#     class_mode="sparse",
#     color_mode="rgb",
#     seed=1
# )

# mask_test_data=mask_test_data_generator.flow_from_directory(
#     testing_directory,
#     target_size=(img_height,img_width),
#     batch_size=batch_size,
#     class_mode="sparse",
#     color_mode="rgb",
#     seed=1   
# )

# print("Image Generator step over, heading to model building")
# _____________________________________________Model Builing________________________

def mask_model_build():
    return tf.keras.models.Sequential([
    ###  Convolute

    # mob_net,
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),

    ###  ANN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=32,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dense(units=32,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dense(units=2,activation="softmax")
        
    ])

def hn_model_build():
    return tf.keras.models.Sequential([
    ###  Convolute

    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),

    ###  ANN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=32,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dense(units=32,kernel_initializer=tf.keras.initializers.he_uniform,activation="relu"),
    tf.keras.layers.Dense(units=2,activation="softmax")
        
    ])

def model_compile(model):
     model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

def model_train(model,train_gen,epoch,steps_per_ep,validation_gen,validation_steps):
     return model.fit(train_gen,epochs=epoch,steps_per_epoch=steps_per_ep,validation_data=validation_gen,validation_steps=validation_steps  )


# mask_train_steps_per_epoch=np.floor(len(mask_train_data.classes)/mask_train_data.batch_size)  # steps taken per epoch-> len(train_dataset)/batch_size
# mask_test_steps_per_epoch=np.floor(len(mask_test_data.classes)/mask_test_data.batch_size)

# print("Starting model Building")
# mask_model_obj=mask_model_build()
# model_compile(mask_model_obj)
# mask_history=model_train(mask_model_obj,train_gen=mask_train_data,epoch=10,steps_per_ep=mask_train_steps_per_epoch,validation_gen=mask_test_data,validation_steps=mask_test_steps_per_epoch)

# print("Model building over")

# mask_model_obj.save("artifacts/mask_nomask_model.h5")

# mask_train_loss=mask_history.history["loss"]
# mask_train_acc=mask_history.history["accuracy"]
# mask_test_loss=mask_history.history["val_loss"]
# mask_test_acc=mask_history.history["val_accuracy"]


# # 
# fig,axis=plt.subplots(1,2,figsize=(10,5))
# sns.lineplot(mask_train_loss,ax=axis[0],label="Training Loss")
# sns.lineplot(mask_test_loss,ax=axis[0],label="Testing Loss")
# axis[0].set_title("Traing and Testing Loss")
# axis[0].legend()
# sns.lineplot(mask_train_acc,ax=axis[1],label="Training Accuracy")
# axis[0].grid(True)
# axis[1].set_title("Traing and Testing Accuracy")
# sns.lineplot(mask_test_acc,ax=axis[1],label="Testing Accuracy")
# axis[1].legend()
# plt.suptitle("Results of CNN Model For Mask Detection")
# axis[1].grid(True)
# plt.show()

#_____________________________Human_nohuman model______________________________________________

hn_training_directory=r"dataset\human_detection\train"
hn_testing_directory=r"dataset\human_detection\test"
batch_size=32
img_height = 224
img_width = 224


hn_train_data_generator=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=90,
    zoom_range=(0.95, 0.95), 
    fill_mode='nearest',
)

hn_test_data_generator=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=90,
    zoom_range=(0.95, 0.95), 
    fill_mode='nearest',
)


train_hn=hn_train_data_generator.flow_from_directory(
    hn_training_directory,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode="sparse",
    color_mode="rgb",
    seed=1
)

test_hn=hn_test_data_generator.flow_from_directory(
    hn_testing_directory,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode="sparse",
    color_mode="rgb",
    seed=1   
)


hn_train_steps_per_epoch=np.floor(len(train_hn.classes)/train_hn.batch_size)  # steps taken per epoch-> len(train_dataset)/batch_size
hn_test_steps_per_epoch=np.floor(len(test_hn.classes)/test_hn.batch_size)

print("Starting model Building")
hn_model_obj=hn_model_build()
model_compile(hn_model_obj)
hn_history=model_train(hn_model_obj,train_gen=train_hn,epoch=10,steps_per_ep=hn_train_steps_per_epoch,validation_gen=test_hn,validation_steps=hn_test_steps_per_epoch)

print("Human or No human Model building over")

hn_model_obj.save("artifacts/human_nohuman_model.h5")

hn_train_loss=hn_history.history["loss"]
hn_train_acc=hn_history.history["accuracy"]
hn_test_loss=hn_history.history["val_loss"]
hn_test_acc=hn_history.history["val_accuracy"]


fig,axis=plt.subplots(1,2,figsize=(10,5))
sns.lineplot(hn_train_loss,ax=axis[0],label="Training Loss")
sns.lineplot(hn_test_loss,ax=axis[0],label="Testing Loss")
axis[0].set_title("Traing and Testing Loss")
axis[0].legend()
sns.lineplot(hn_train_acc,ax=axis[1],label="Training Accuracy")
axis[0].grid(True)
axis[1].set_title("Traing and Testing Accuracy")
sns.lineplot(hn_test_acc,ax=axis[1],label="Testing Accuracy")
axis[1].legend()
plt.suptitle("Results of CNN Network For Human and No Human Detection")
axis[1].grid(True)
plt.show()