import tensorflow as tf

def parse_ffn_tfrecord(example_proto):
    """
    Parses a single FFN-style data record from a TFRecord file, ensuring NaN values are handled correctly.

    Args:
        example_proto: A single serialized TFRecord protobuf string.

    Returns:
        A tuple containing:
            - Input features as a dictionary with keys:
                'geometries': A dense tensor of shape (max_atoms, 3), replacing NaNs with -9999.99.
                'frequency': A tensor of shape (1,).
            - Target labels ('charges') as a dense tensor of shape (max_atoms, 6), replacing NaNs with 0.0.
    """
    
    # Step 1: Define the feature structure inside the TFRecord
    feature_description = {
        'unique_id': tf.io.FixedLenFeature([], tf.int64),          # Unique identifier for the record
        'structure_id': tf.io.FixedLenFeature([], tf.int64),       # Identifier for the molecular structure
        'frequency': tf.io.FixedLenFeature([], tf.float32),        # Scalar frequency value
        'geometries': tf.io.VarLenFeature(tf.float32),             # Sparse feature for atomic geometries
        'charges': tf.io.VarLenFeature(tf.float32),                # Sparse feature for atomic charges
    }

    # Step 2: Parse the raw serialized example into a dictionary of features
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Step 3: Extract 'frequency' (scalar) and ensure it's properly shaped
    frequency = tf.expand_dims(example['frequency'], axis=0)  # Shape: (1,)

    # Step 4: Extract 'geometries' and 'charges' as dense tensors
    geometries = tf.sparse.to_dense(example['geometries'])  # Convert from sparse to dense tensor
    charges = tf.sparse.to_dense(example['charges'])  # Convert from sparse to dense tensor

    # Step 5: Calculate number of atoms
    # Geometries is a flat array storing (x, y, z) coordinates, so dividing by 3 gives the number of atoms
    max_atoms = tf.shape(geometries)[0] // 3

    # Step 6: Reshape tensors to (max_atoms, 3) for geometries and (max_atoms, 6) for charges
    geometries = tf.reshape(geometries, (max_atoms, 3))
    charges = tf.reshape(charges, (max_atoms, 6))

    # 🔥 Step 7: Handle NaN values 🔥
    # Replace NaNs in geometries with -9999.99 (so they can be masked in training)
    geometries = tf.where(tf.math.is_nan(geometries), tf.fill(tf.shape(geometries), -9999.99), geometries)

    # Replace NaNs in charges with 0.0 (so they don't affect loss computation)
    charges = tf.where(tf.math.is_nan(charges), tf.zeros_like(charges), charges)

    # Return processed input features and target labels
    return {'geometries': geometries, 'frequency': frequency}, charges


def create_ffn_dataset(tfrecord_file, batch_size=32, shuffle=True):
    """
    Create a TensorFlow dataset for FFN training from a TFRecord file.

    Args:
        tfrecord_file: Path to the TFRecord file containing the dataset.
        batch_size: Number of samples per batch for training.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A TensorFlow dataset object ready for training.
    """
    # Step 1: Load the TFRecord file into a dataset
    # Each element in the dataset is a raw serialized TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Step 2: Parse the raw TFRecord into structured data
    # The `map` function applies `parse_ffn_tfrecord` to each element in the dataset
    dataset = dataset.map(
        parse_ffn_tfrecord,  # Function to parse each record
        num_parallel_calls=tf.data.AUTOTUNE  # Use optimal parallelism for faster processing
    )

    # Step 3: Shuffle the dataset (if enabled)
    # Shuffling randomizes the order of records to improve training performance
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)  # Shuffle with a buffer of 1000 records

    # Step 4: Batch the dataset
    # Group consecutive elements into batches of size `batch_size`
    dataset = dataset.batch(batch_size)

    # Step 5: Prefetch the dataset
    # Prefetching allows the pipeline to prepare the next batch while the current batch is being processed
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Return the fully prepared dataset
    return dataset
