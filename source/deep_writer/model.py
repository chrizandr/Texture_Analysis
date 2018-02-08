"""CNN-LSTM Model for writer identification."""

from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers import Input
from keras.engine.topology import get_source_inputs
from keras.layers import Flatten, Reshape
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from read import DataGenerator, partition_data, getIds
from keras.models import load_model
import pdb


def VGG_Writer(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='', classes=150):
    """VVG16 without final dense layers."""
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)
    # pdb.set_trace()
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    feature_part = Model(inputs, x, name='vgg16')

    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    feature_part.load_weights(weights_path)

    for layer in feature_part.layers:
        layer.trainable = False

    return feature_part


if __name__ == "__main__":
    # (N, sent_len, 224, 224, 3)
    OUTPUT_FILE = "deep_writer.hd5"
    PATH = "/home/chris/new_data/"
    ID_FILE = "/home/chris/IAM_writerids.csv"
    classes = 657
    slen = 41
    x_dim = 224
    y_dim = 224
    z_dim = 3
    batch_size = 5

    print("Parsing files...")
    writer_ids = getIds(ID_FILE)
    partition, ids = partition_data(PATH, writer_ids)

    print("Making generators....")
    training_generator = DataGenerator(slen, x_dim, y_dim,
                                       z_dim, batch_size,
                                       shuffle=True, n_classes=classes).generate(partition['train'], ids["train"])

    validation_data = DataGenerator(slen, x_dim, y_dim,
                                    z_dim, batch_size,
                                    shuffle=True, n_classes=classes).generate(partition['validation'], ids["validation"])

    test_generator = DataGenerator(slen, x_dim, y_dim,
                                   z_dim, batch_size,
                                   shuffle=True, n_classes=classes).generate(partition['test'], ids["test"])

    checkpoint = ModelCheckpoint(filepath=OUTPUT_FILE, monitor='val_loss')

    print("Making model....")
    VGG_model = VGG_Writer(include_top=False, weights='imagenet')
    model = Sequential()
    model.add(TimeDistributed(VGG_model, input_shape=(slen, x_dim, y_dim, z_dim)))
    model.add(TimeDistributed(Reshape((49, 512))))
    print(model.output_shape)
    model.add(TimeDistributed(LSTM(50)))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1000, activation="relu"))
    print(model.output_shape)
    model.add(Dense(classes, activation="softmax"))

    print("Compiling model....")
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training.....")
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(partition['train'])//batch_size + 1,
                        validation_data=validation_data,
                        validation_steps=len(partition['validation'])//batch_size + 1,
                        callbacks=[checkpoint],
                        epochs=20)

    model = load_model(OUTPUT_FILE)
    results = model.evaluate_generator(generator=test_generator, steps=len(partition["test"])//batch_size + 1)
    print(results)
