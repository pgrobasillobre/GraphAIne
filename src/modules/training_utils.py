import tensorflow as tf

def train_ffn(model, dataset, epochs=10, learning_rate=1e-3, checkpoint_path=None):
    """
    Train the Graphene Feedforward Neural Network (FFN) using a masked loss function.

    Args:
        model (tf.keras.Model): The FFN model to be trained.
        dataset (tf.data.Dataset): The training dataset.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        checkpoint_path (str, optional): Path to save training checkpoints.
    """
    
    # Step 1: Define Optimizer & Loss
    # We could also use model.optimizer (but doing as below allow as to define learning rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use the custom masked loss function (handles padding correctly)
    loss_fn = model.loss  # Ensures we use `masked_loss` from the model

    # Metric to track training loss
    # Computes the running average of the loss over all batches in an epoch
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    # Step 2: Setup Checkpointing
    if checkpoint_path:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    # Step 3: Training Loop
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs} - Training {model.name}")

        for step, (inputs, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # Forward pass: Get model predictions
                predictions = model([inputs['geometries'], inputs['frequency']], training=True)
                
                # Compute loss using the custom masked loss function
                loss = loss_fn(labels, predictions)

            # Compute gradients and update weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update loss metric (average) after each batch
            train_loss.update_state(loss)

            # Print loss every 10 steps
            # train_loss.result().numpy() gives the current average loss
            if step % 10 == 0:
                print(f"Step {step}, Loss: {train_loss.result().numpy()}")

        # Step 4: Save Model Checkpoint
        if checkpoint_path:
            manager.save()
            print(f"💾 Checkpoint saved at {manager.latest_checkpoint}")

        # Reset training loss metric
        train_loss.reset_states()

    print("\n✅ Training Complete!")


