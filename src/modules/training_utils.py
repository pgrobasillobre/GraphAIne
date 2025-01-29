import tensorflow as tf

def train_ffn(model, dataset, epochs=10, learning_rate=1e-3, checkpoint_path=None):
    """
    Train the Graphene Feedforward Neural Network (FFN) using a masked loss function.

    Args:
        model (tf.keras.Model): The FFN model to be trained.
        dataset (tf.data.Dataset): The training dataset.
        epochs (int): Number of training epochs (how many times the dataset is passed through the model).
        learning_rate (float): Learning rate for the optimizer (controls weight updates).
        checkpoint_path (str, optional): Path to save training checkpoints for resuming later.
    """

    # Step 1: Define Optimizer & Loss
    # =======================================
    # The optimizer is responsible for updating model weights based on gradients.
    # Here, we use the Adam optimizer, which adapts learning rates per parameter for stability.
    # Note: We could also use model.optimizer, but defining it here allows us to control the learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Retrieve the loss function used by the model (masked loss is already assigned at model compilation)
    loss_fn = model.loss  # Ensures we consistently use `masked_loss`

    # Define a metric to track the average loss during training
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    # Step 2: Setup Checkpointing (Optional)
    # =======================================
    # If a checkpoint path is provided, we enable checkpointing to save model progress.
    if checkpoint_path:
        # A Checkpoint object stores the optimizer state and model weights
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # CheckpointManager handles automatic checkpoint saving and retains the last 3 checkpoints
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    # Step 3: Training Loop (Epochs & Batches)
    # =======================================
    # We iterate over the number of epochs (full passes over the dataset)
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs} - Training {model.name}")

        # Iterate through the dataset batch by batch
        for step, (inputs, labels) in enumerate(dataset):
            # `inputs` is a dictionary with:
            #   - `inputs['geometries']`: Atomic coordinates (N, 3)
            #   - `inputs['frequency']`: Frequency value (1,)
            # `labels` contains the expected output charges (N, 6)

            # Step 3.1: Record Operations for Gradient Calculation
            # ----------------------------------------------------
            # `GradientTape()` records all operations for automatic differentiation.
            # This enables TensorFlow to compute gradients for weight updates.
            with tf.GradientTape() as tape:
                # Step 3.2: Forward Pass - Make Predictions
                # -------------------------------------------------
                # The model takes the geometries and frequency as input and predicts charges.
                # `training=True` ensures any dropout or batch normalization (if used) behaves correctly.
                predictions = model([inputs['geometries'], inputs['frequency']], training=True)

                # Step 3.3: Compute Loss
                # -------------------------------------------------
                # Compare predictions to true values using the custom masked loss function.
                loss = loss_fn(labels, predictions)

            # Step 3.4: Compute Gradients & Update Weights
            # -------------------------------------------------
            # Compute the gradients of the loss with respect to model parameters
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply the computed gradients to update model weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Step 3.5: Track Loss for Monitoring
            # -------------------------------------------------
            # `train_loss.update_state(loss)` updates the running average loss.
            train_loss.update_state(loss)

            # Step 3.6: Print Progress Every 10 Steps
            # -------------------------------------------------
            # We print the loss every 10 batches to monitor training progress.
            if step % 10 == 0:
                print(f"Step {step}, Loss: {train_loss.result().numpy()}")

        # Step 4: Save Model Checkpoint at End of Each Epoch
        # =======================================
        if checkpoint_path:
            manager.save()
            print(f"💾 Checkpoint saved at {manager.latest_checkpoint}")

        # Step 5: Reset Loss Metric for Next Epoch
        # =======================================
        # The running loss average is reset at the end of each epoch to ensure a clean start.
        train_loss.reset_states()

    # Training complete
    print("\n✅ Training Complete!")

