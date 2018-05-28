import h5py
import numpy
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
from sklearn.metrics import regression
from keras.applications import ResNet50
import numpy as np
import config as config

def train_FBP5500(db_file, model_file, batch_size=16, lr=0.0001, epochs=20, input_shape=(350, 350, 3), loss="mean_squared_error"):
    train_db = h5py.File(db_file, "r")
    features = np.array(train_db["features"])
    labels = np.array(train_db["labels"])

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.1, shuffle=True)

    model = build_model((features.shape[-1],))    
    model.compile(loss=loss, 
                  optimizer=Adam(lr=lr, decay=lr/epochs), 
                  metrics=["acc"])
   
    H = model.fit(trainX, trainY, 
                  epochs=epochs, 
                  batch_size=batch_size,
                  validation_split=0.1,
                  callbacks=[ModelCheckpoint(model_file, monitor="loss", save_best_only=True)])

    print("[INFO] Saving Model....") 
    #load_resnet
    resnet = ResNet50(weights="imagenet", 
                    include_top=False, 
                    input_shape=input_shape,
                    pooling="avg")
    trained_model = load_model(model_file)

    #Keeping it all together 

    final = Sequential()
    final.add(resnet)
    final.add(trained_model.layers[0])
    final.save(model_file)

    if loss == 'mean_squared_error':
        print("[INFO] Regression score (MSE) on test data {}".format(regression.mean_squared_error(testY, model.predict(testX))))
    else:
        print("[INFO] Regression score (MAE) on test data {}".format(regression.mean_absolute_error(testY, model.predict(testX))))
    
    return H

def build_model(input_shape):
    model = Sequential()
    # model.add(Dense(128, input_shape=input_shape, activation="relu"))
    # model.add(Dense(1))
    model.add(Dense(1, input_shape=input_shape))
    return model


#train_FBP5500(config.DATABSE_FILE, config.MODEL_FILE, epochs=20, lr=0.01)