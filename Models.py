from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Add

def create_model(model_name, inputs_shape):
        if model_name is 'Track_classifier':
            inp = Input(shape=(inputs_shape[1],))
            dense = Dense(32, kernel_initializer='glorot_normal', activation='relu')(inp)
            dense = Dense(32, kernel_initializer='glorot_normal', activation='relu')(dense)
            dense = Dense(32, kernel_initializer='glorot_normal', activation='relu')(dense)
            dense = Dense(32, kernel_initializer='glorot_normal', activation='relu')(dense)
            dense = Dense(32, kernel_initializer='glorot_normal', activation='relu')(dense)
            out = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(dense)

            model = Model(inputs=inp, outputs=out)
            model.compile(loss='logcosh', optimizer=optimizers.Adam(lr=1e-3))

            return model

        else:
            print model_name
            raise ValueError('Model not found')
