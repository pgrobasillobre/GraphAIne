import tensorflow as tf
import csv

def train_ffn(model, train_dataset, val_dataset, epochs=20, learning_rate=1e-3, checkpoint_path=None, log_file="training_log.csv"):
    """
    Train the FFN model with validation loss tracking and save loss values to a file.

    Args:
        model (tf.keras.Model): The FFN model to be trained.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        checkpoint_path (str, optional): Path to save training checkpoints.
        log_file (str): File to save loss values.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = model.loss  # Use the custom masked loss function

    train_loss = tf.keras.metrics.Mean(name="train_loss")  
    val_loss = tf.keras.metrics.Mean(name="val_loss")  

    # Step 1: Create CSV file & Write Header
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])  # CSV Header

    if checkpoint_path:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs} - Training {model.name}")

        # Training Phase
        for step, (inputs, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model([inputs['geometries'], inputs['frequency']], training=True)
                loss = loss_fn(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)

        # Validation Phase
        for val_inputs, val_labels in val_dataset:
            val_predictions = model([val_inputs['geometries'], val_inputs['frequency']], training=False)
            val_loss.update_state(loss_fn(val_labels, val_predictions))

        # Log results to CSV file
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss.result().numpy(), val_loss.result().numpy()])

        print(f"📊 Epoch {epoch + 1} - Train Loss: {train_loss.result().numpy()} | Val Loss: {val_loss.result().numpy()}")

        if checkpoint_path:
            manager.save()
            print(f"💾 Checkpoint saved at {manager.latest_checkpoint}")

        train_loss.reset_states()
        val_loss.reset_states()

    print("\n✅ Training Complete!")

