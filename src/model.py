from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D, Conv2D

def build_model(input_shape=(48, 48, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="Same", activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3,3), padding="Same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2))) 
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3,3), padding="Same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((2,2))) 
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    
    return model
