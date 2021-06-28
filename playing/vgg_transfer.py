from keras import applications, Input, layers, Model, optimizers, losses, metrics

base_model = applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top= False
)

base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(),
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=metrics.binary_accuracy
              )
model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
