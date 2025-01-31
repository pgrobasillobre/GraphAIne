import tensorflow as tf

def parse_ffn_tfrecord(example_proto):
    """
    Parses a single FFN-style data record from a TFRecord file, ensuring NaN values are handled correctly.

    Args:
        example_proto: A single serialized TFRecord protobuf string.

    Returns:
        A tuple containing:
            - Input features as a dictionary with keys:
                'geometries': A dense tensor of shape (max_atoms, 3), replacing NaNs with 1.0e6
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
    # Replace NaNs in geometries with 1.0e6 (so they can be masked in training)
    geometries = tf.where(tf.math.is_nan(geometries), tf.fill(tf.shape(geometries), 1.0e6), geometries)

    # Replace NaNs in charges with 0.0 (so they don't affect loss computation)
    charges = tf.where(tf.math.is_nan(charges), tf.zeros_like(charges), charges)

    # Return processed input features and target labels
    return {'geometries': geometries, 'frequency': frequency}, charges


def create_ffn_dataset(tfrecord_file, batch_size=32, shuffle=True, val_split=0.1, test_split=0.1):
    """
    Create TensorFlow datasets for FFN training, validation, and testing.

    Args:
        tfrecord_file (str): Path to the TFRecord file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        val_split (float): Fraction for validation set.
        test_split (float): Fraction for test set.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Step 1: Load TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Step 2: Parse the dataset
    dataset = dataset.map(parse_ffn_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Step 3: Shuffle if enabled
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)  # Ensures randomness

    # Step 4: Count the number of records **before batching**
    total_size = sum(1 for _ in dataset)  # Count total elements
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    print(f"📊 Total Samples: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Step 5: Split the dataset correctly
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)

    # Step 6: Batch after splitting!
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"📊 Final Split: Train={sum(1 for _ in train_dataset)}, Validation={sum(1 for _ in val_dataset)}, Test={sum(1 for _ in test_dataset)}")

    return train_dataset, val_dataset, test_dataset
