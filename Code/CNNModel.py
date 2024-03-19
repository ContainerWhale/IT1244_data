import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras import layers, models, optimizers
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, concatenate, Input
from keras.models import Model


class CNNModel:
    def __init__(self, model_name,
                 X_train_img, X_val_img,
                 y_train, y_val,
                 X_train_ftr = None, X_val_ftr = None) -> None:

        '''Let there be T training samples and V validation samples and n features
        model_name: name of model as string (e.g. "VGG16", "RESNET50V2)
        X_train_img: numpy array of shape (T, 128, 128, 3)
        X_train_ftr: numpy array of shape (T, n)
        y_train: numpy array of shape (T,), with binary values
        X_val_img: numpy array of shape (V, 128, 128, 3)
        X_val_ftr: numpy array of shape (V, n)
        y_val: numpy array of shape (V,), with binary values
        '''

        self.X_train_img, self.X_val_img = X_train_img, X_val_img
        self.y_train, self.y_val = y_train, y_val

        if X_train_ftr is None or X_val_ftr is None:
            self.X_train_ftr = np.empty((X_train_img.shape[0], 0))
            self.X_val_ftr = np.empty((X_val_img.shape[0], 0))
        else:
            self.X_train_ftr = X_train_ftr
            self.X_val_ftr = X_val_ftr

        if model_name == "VGG16":
            self.model = self._create_VGG16_model(X_train_img.shape[1:], self.X_train_ftr.shape[1:])
        else:
            self.model = None


    def _create_VGG16_model(self, image_shape, feature_shape):
        # Defining inputs
        image_input = Input(shape=image_shape, name="image_input")
        feature_input = Input(shape=feature_shape, name='feature_input')

        # Constructing VGG16 model
        base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
        base_model.trainable = False

        # out = Flatten()(base_model.output)
        out = GlobalAveragePooling2D()(base_model.output)


        # Introducing features
        out = concatenate([out, feature_input])
        #
        ## Fully connected layers that are trainable
        out = Dense(128, activation="relu")(out)
        out = Dropout(0.5)(out)
        out = Dense(64, activation="relu")(out)
        out = Dropout(0.5)(out)
        out = Dense(1, activation="sigmoid")(out)

        model = Model(inputs = [image_input, feature_input], outputs = out)

        return model

    def summarise_model(self):
        if self.model is None:
            print("No model")
            return
        else:
            model = self.model
        print(f"Model input names: {model.input_names}")
        print()
        for i, layer in enumerate(model.layers):
            print(f"{i}\t{layer.__class__.__name__}   \t{layer.trainable}")

    def run_model(self,
              epochs, batch_size):
        '''
        epochs
        batch_size

        returns history
        '''
        if self.model is None:
            print("No model")
            return
        else:
            model = self.model

        X_train = {'image_input': self.X_train_img, 'feature_input': self.X_train_ftr}
        X_val = {'image_input': self.X_val_img, 'feature_input': self.X_val_ftr}

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(x = X_train, y = self.y_train,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, self.y_val))

        return history
