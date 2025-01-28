import tensorflow as tf

def parse_ffn_tfrecord(example_proto):
    """
    Parse a single FFN-style data record from a TFRecord file.
    This function defines the structure of the input TFRecord, extracts the features,
    and reshapes the data into usable formats for training.

    Args:
        example_proto: A single serialized TFRecord protobuf string.

    Returns:
        A tuple containing:
            - Input features as a dictionary with keys:
                'geometries': A dense tensor of shape (max_atoms, 3).
                'frequency': A tensor of shape (1,).
            - Target labels ('charges') as a dense tensor of shape (max_atoms, 6).
    """
    # Define the structure of the features in the TFRecord
    feature_description = {
        'unique_id': tf.io.FixedLenFeature([], tf.int64),          # Unique identifier for the record
        'structure_id': tf.io.FixedLenFeature([], tf.int64),       # Identifier for the molecular structure
        'frequency': tf.io.FixedLenFeature([], tf.float32),        # Scalar frequency value
        'geometries': tf.io.VarLenFeature(tf.float32),             # Sparse feature for atomic geometries
        'charges': tf.io.VarLenFeature(tf.float32),                # Sparse feature for atomic charges
    }

    # Parse the raw serialized example into a dictionary of features
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Extract the 'frequency' feature and expand dimensions
    # (converting a scalar to a rank-1 tensor for compatibility)
    frequency = example['frequency']  # Scalar tensor
    frequency = tf.expand_dims(frequency, axis=0)  # Shape becomes (1,)

    # Extract the 'geometries' and 'charges' features (stored as sparse tensors)
    # Convert them to dense tensors so they are easier to work with
    geometries = example['geometries']  # Sparse tensor
    geometries = tf.sparse.to_dense(geometries)  # Dense tensor

    charges = example['charges']  # Sparse tensor
    charges = tf.sparse.to_dense(charges)  # Dense tensor

    # Calculate the number of atoms in the molecule
    # The geometries tensor is a flattened array of x, y, z coordinates, so dividing by 3 gives the number of atoms
    max_atoms = tf.shape(geometries)[0] // 3

    # Reshape the 'geometries' tensor into (max_atoms, 3), where each row is an (x, y, z) coordinate
    geometries = tf.reshape(geometries, (max_atoms, 3))

    # Reshape the 'charges' tensor into (max_atoms, 6), where each row contains 6 charge-related features
    charges = tf.reshape(charges, (max_atoms, 6))

    # Return the parsed data:
    # - A dictionary of input features: 'geometries' and 'frequency'
    # - The target labels: 'charges'
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
