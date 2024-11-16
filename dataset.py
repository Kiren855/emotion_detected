import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, test_dir, batch_size=64, target_size=(48, 48)):
  
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    val_test_datagen = ImageDataGenerator(rescale=1./255)  

    # Tạo generators
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size,
                                                  class_mode='categorical', color_mode='grayscale')

    val_gen = val_test_datagen.flow_from_directory(val_dir, target_size=target_size, batch_size=batch_size,
                                                  class_mode='categorical', color_mode='grayscale')

    test_gen = val_test_datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size,
                                                   class_mode='categorical', color_mode='grayscale')

    return train_gen, val_gen, test_gen
