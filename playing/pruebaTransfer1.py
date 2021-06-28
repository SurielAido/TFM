import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


def vgg_16():
    b_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    x = b_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=b_model.input, outputs=predictions)

    for layer in model.layers[:15]:
        layer.trainable = False

    opt = SGD(lr=0.003, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


main_model = vgg_16()
# main_model.fit(...)

pretrained_model = Model(main_model.input, main_model.layers[-2].output)


def combined_net():
    inp_u = Input((224, 224, 3))  # the same input dim of pretrained_model
    inp_v = Input((224, 224, 3))  # the same input dim of pretrained_model

    u_output = pretrained_model(inp_u)
    v_output = pretrained_model(inp_v)

    concat = concatenate([u_output, v_output])
    main_output = Dense(1, activation='sigmoid', name='main_output')(concat)

    model = Model(inputs=[inp_u, inp_v], outputs=main_output)
    opt = SGD(lr=0.001, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


base_model = combined_net()
base_model.summary()