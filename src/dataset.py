from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_generators(train_dir=None, test_dir=None, batch_size=64, target_size=(48, 48)):
    train_datagen = ImageDataGenerator(
        width_shift_range = 0.1,       
        height_shift_range = 0.1,   
        horizontal_flip = True,      
        rescale = 1./255,            
        validation_split = 0.2 
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split = 0.2 
    ) 
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    if os.path.isdir(train_dir):
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=target_size, batch_size=batch_size,
            class_mode='categorical', color_mode='grayscale', subset='training' 
        )
        
        val_gen = validation_datagen.flow_from_directory(
            train_dir, target_size=target_size, batch_size=batch_size,
            class_mode='categorical', color_mode='grayscale', subset='validation'  
        )
    else:
        train_gen = None
        val_gen = None

    if test_dir and os.path.isdir(test_dir):
        test_gen = test_datagen.flow_from_directory(
            test_dir, target_size=target_size, batch_size=batch_size,
            class_mode='categorical', color_mode='grayscale'
        )
    else:
        test_gen = None  

    return train_gen, val_gen, test_gen
