import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir=None, val_dir=None, test_dir=None, batch_size=64, target_size=(48, 48)):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')


    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    if os.path.isdir(train_dir):
        train_gen = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size,
                                                      class_mode='categorical', color_mode='grayscale')
    else:
        train_gen = None 

    if os.path.isdir(val_dir):
        val_gen = val_test_datagen.flow_from_directory(val_dir, target_size=target_size, batch_size=batch_size,
                                                      class_mode='categorical', color_mode='grayscale')
    else:
        val_gen = None  

    if test_dir and os.path.isdir(test_dir):
        test_gen = val_test_datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size,
                                                       class_mode='categorical', color_mode='grayscale')
    else:
        test_gen = None  

    return train_gen, val_gen, test_gen
