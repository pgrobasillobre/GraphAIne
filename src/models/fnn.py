import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def masked_loss(y_true, y_pred, mask_value=-9999.99):
    """
    Custom loss function that ignores predictions where the input contained a padding value.

    Args:
        y_true (Tensor): True charge values, containing real values and padding.
        y_pred (Tensor): Predicted charge values from the model.
        mask_value (float): The specific value used to indicate padding in y_true.

    Returns:
        Tensor: The masked loss value, computed only on non-padding elements.
    """

    # Step 1: Create a mask for valid (non-padding) values
    # The mask will be 1 for real values and 0 for padding values
    mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)

    # Step 2: Compute the Mean Squared Error (MSE) loss
    # We only compute the loss where mask=1 (real data) and ignore masked (padded) values
    squared_error = K.square(y_pred - y_true)  # Element-wise squared difference
    masked_loss = K.mean(mask * squared_error)  # Only considers non-padded values

    return masked_loss


def build_ffn(input_geometry_shape, input_frequency_shape):
    """
    Build a Feedforward Neural Network (FFN) for molecular data,
    handling NaN padding in the input by replacing it with -9999.99.

    Args:
        input_geometry_shape (tuple): The shape of the geometry input (N, 3), 
                                      where N is the number of atoms and 3 represents (x, y, z) coordinates.
        input_frequency_shape (tuple): The shape of the frequency input (e.g., (1,)).

    Returns:
        tf.keras.Model: The compiled FFN model ready for training.
    """

    # Step 1: Define Model Inputs
    # The network has two input branches: one for atomic geometries and one for frequency data.
    geometry_input = layers.Input(shape=input_geometry_shape, name="geometry_input")
    frequency_input = layers.Input(shape=input_frequency_shape, name="frequency_input")

    # Step 2: Handle NaN Values in the Geometry Input
    # Replace NaN values with a constant (-9999.99) so TensorFlow can process the data properly.
    geometry_input_cleaned = tf.where(
        tf.math.is_nan(geometry_input),  # Check where values are NaN
        tf.fill(tf.shape(geometry_input), -9999.99),  # Replace NaN with -9999.99
        geometry_input  # Keep original values where no NaN is found
    )

    # Step 3: Process the Geometry Input Using 1D Convolutions
    # Here, we use Conv1D layers to learn local atomic interactions and extract features.
    
    # First Conv1D layer: Extracts basic atomic relationships
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(geometry_input_cleaned)

    # Second Conv1D layer: Expands feature representation and learns deeper atomic interactions
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)

    # Global Average Pooling: Reduces dimensions by computing the average feature value per atom
    # This ensures the model can handle molecules of different sizes robustly.
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected layer: Further processing to extract high-level molecular features
    x = layers.Dense(256, activation='relu')(x)

    # Step 4: Process the Frequency Input
    # The frequency input is a scalar or small vector, processed through dense layers.

    # First Dense Layer for frequency input: Expands the representation of the frequency value
    y = layers.Dense(32, activation='relu')(frequency_input)

    # Second Dense Layer for frequency input: Extracts refined features from the frequency branch
    y = layers.Dense(16, activation='relu')(y)

    # Step 5: Combine the Geometry and Frequency Branches
    # We concatenate the extracted features from both branches to form a single feature vector.
    combined = layers.Concatenate()([x, y])

    # Fully connected layers for combined representation
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dense(64, activation='relu')(combined)

    # Step 6: Output Layer
    # The model predicts `N * 6` values, where N is the number of atoms.
    # Each atom has 6 predicted properties (e.g., charge distributions).
    outputs = layers.Dense(input_geometry_shape[0] * 6, activation='linear')(combined)

    # Reshape the output to match the expected shape: (N, 6)
    outputs = layers.Reshape((input_geometry_shape[0], 6))(outputs)

    # Step 7: Build the Model
    # We define the model with two inputs (geometry and frequency) and one output (predicted atomic properties).
    model = models.Model(inputs=[geometry_input, frequency_input], outputs=outputs)

    # Step 8: Compile the Model with the Custom Masked Loss
    # Using Adam optimizer and our custom loss function to ignore padding effects.
    model.compile(optimizer='adam', loss=masked_loss)

    return model

