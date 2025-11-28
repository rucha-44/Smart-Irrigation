import tensorflow as tf
from keras import layers, models, regularizers, Input

def build_advanced_dnn(input_dim, output_type='classification'):
    """
    Advanced DNN with GaussianNoise, Gelu, and Residuals.
    """
    inputs = Input(shape=(input_dim,))
    
    # Noise for robustness
    x = layers.GaussianNoise(0.05)(inputs)

    # Block 1
    x = layers.Dense(512, activation="gelu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    # Block 2
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual
    shortcut = layers.Dense(128, activation="gelu", kernel_initializer="he_normal")(inputs)

    # Block 3
    x = layers.Dense(
        128,
        activation="gelu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Add()([x, shortcut])

    # Tapering
    x = layers.Dense(96, activation="gelu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(48, activation="gelu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    if output_type == 'classification':
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        # Linear output (scaled 0-1 externally)
        outputs = layers.Dense(1, activation='linear')(x)
        loss = 'mse'
        metrics = ['mae']

    model = models.Model(inputs, outputs, name="advanced_dnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model