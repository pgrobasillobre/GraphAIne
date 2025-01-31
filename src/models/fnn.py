import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def masked_loss(y_true, y_pred, mask_value=1.0e6):
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
    Build a Feedforward Neural Network (FFN) for molecular data.

    This model consists of:
    - A **geometry branch** that processes atomic coordinates (x, y, z) using 1D convolutional layers.
    - A **frequency branch** that processes a scalar frequency value using dense layers.
    - A **fully connected layer** that combines both branches to predict atomic charge distributions.
    
    Args:
        input_geometry_shape (tuple): The shape of the geometry input (N, 3), 
                                      where N is the number of atoms, and 3 represents (x, y, z) coordinates.
        input_frequency_shape (tuple): The shape of the frequency input (e.g., (1,)).

    Returns:
        tf.keras.Model: The compiled FFN model, ready for training.
    """

    # Step 1: Define Model Inputs
    # ===================================
    # The model expects two different inputs:
    #   1. **Geometric Input**: A tensor of shape (N, 3), where each row contains (x, y, z) atomic coordinates.
    #   2. **Frequency Input**: A tensor of shape (1,), representing a scalar frequency value.
    geometry_input = layers.Input(shape=input_geometry_shape, name="geometry_input")
    frequency_input = layers.Input(shape=input_frequency_shape, name="frequency_input")

    # Step 2: Process the Geometry Input Using 1D Convolutions
    # ===================================
    # The **geometry branch** will apply 1D convolutions to learn local atomic interactions.
    # The first convolution extracts **basic atomic relationships**.
    x = layers.Conv1D(
        filters=64,           # Number of filters (64 distinct patterns learned)
        kernel_size=3,        # Looks at 3 neighboring atoms at a time
        activation='relu',     # Applies non-linearity to capture complex relationships
        padding='same'         # Ensures the output size remains (N, 64)
    )(geometry_input)

    # The second convolution expands the feature representation and **learns deeper molecular patterns**.
    x = layers.Conv1D(
        filters=128,          # Doubles the number of filters for richer feature extraction
        kernel_size=3,        # Again, looks at 3-atom neighborhoods
        activation='relu',    # Uses ReLU for non-linearity
        padding='same'        # Keeps dimensions unchanged
    )(x)

    # The `GlobalAveragePooling1D` layer ensures **variable-length molecules can be processed**.
    # It **compresses all atomic features into a single fixed-size vector** by averaging them.
    x = layers.GlobalAveragePooling1D()(x)

    # A fully connected (Dense) layer refines the extracted features.
    x = layers.Dense(
        256,               # 256 neurons for more complex feature interactions
        activation='relu'   # Non-linear activation function
    )(x)

    # Step 3: Process the Frequency Input
    # ===================================
    # The **frequency branch** processes a single scalar value through fully connected layers.
    # First, it expands into a 128-neuron representation.
    y = layers.Dense(
        128,              # Expands single value to a more expressive feature space
        activation='relu'  # ReLU ensures the model can learn complex patterns
    )(frequency_input)
    
    # A second layer reduces the representation size.
    y = layers.Dense(
        64,               # Reduces to a more compact representation
        activation='relu'
    )(y)
    
    # A third layer further refines the frequency-based representation.
    y = layers.Dense(
        32,               # Further compresses the feature space
        activation='relu'
    )(y)

    # Step 4: Combine the Geometry and Frequency Branches
    # ===================================
    # The two feature representations are **concatenated into a single vector**.
    combined = layers.Concatenate()([x, y])

    # Fully connected layers further refine the combined representation.
    combined = layers.Dense(
        128,               # 128 neurons to process the joint feature space
        activation='relu'
    )(combined)

    combined = layers.Dense(
        64,               # Reducing feature space further
        activation='relu'
    )(combined)

    # Step 5: Output Layer
    # ===================================
    # The output is a prediction of `N * 6` values, where:
    #   - `N` is the number of atoms
    #   - Each atom has **6 charge-related predictions**.
    # This is a **regression task** (continuous values), so we use a **linear activation**.
    outputs = layers.Dense(
        input_geometry_shape[0] * 6,  # Total outputs: N atoms * 6 features per atom
        activation='linear'           # Linear activation for continuous value prediction
    )(combined)

    # The output is **reshaped** to `(N, 6)`, where:
    #   - `N` represents atoms.
    #   - Each atom gets **6 output values** (predictions).
    outputs = layers.Reshape((input_geometry_shape[0], 6))(outputs)

    # Step 6: Build the Model
    # ===================================
    # We define the model with two inputs (geometries and frequency) and one output (charge predictions).
    model = models.Model(
        inputs=[geometry_input, frequency_input],  # Takes both inputs
        outputs=outputs,                           # Produces charge predictions
        name="GraphAIne_FFN_Model"                 # Custom model name
    )

    # Step 7: Compile the Model with the Custom Masked Loss
    # ===================================
    # We use:
    #   - The **Adam optimizer** for efficient training.
    #   - A **custom loss function (`masked_loss`)** that ignores padded atoms (if present).
    model.compile(
        optimizer='adam',  # Adaptive learning rate optimization
        loss=masked_loss   # Custom loss that ignores padding
    )

    return model

