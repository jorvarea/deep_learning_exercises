# %%

import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
print(tf.__version__)

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Datos de entrenamiento", x_train.shape)
print("Datos de test", x_test.shape)
print("Primeros dígitos: ", y_train[:5])

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train.dtype

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# %%

epochs: int = 10
results: list = []
train_times: list = []
train_losses: list = []
train_accuracies: list = []
test_losses: list = []
test_accuracies: list = []
num_classes: int = 10
avg_proba_matrices: dict[str, np.ndarray] = {}

for optimizer in ['sgd', 'adagrad', 'rmsprop', 'adam']:
    print(f"Training with {optimizer}")
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    start_time: float = time.time()
    h = model.fit(x_train, y_train, epochs=epochs)
    end_time: float = time.time()
    elapsed_time: float = end_time - start_time
    print(f"Training time with {optimizer}: {elapsed_time:.2f} seconds")
    train_times.append(elapsed_time)
    print(f"Evaluation in training")
    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Evaluation in test")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    results.append(h)
    # Average probability matrix on test set (rows=true 0-9, cols=pred 0-9)
    probs: np.ndarray = model.predict(x_test, verbose=0)
    avg_matrix: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.float64)
    for true_label in range(num_classes):
        mask: np.ndarray = (y_test == true_label)
        if np.any(mask):
            avg_matrix[true_label] = probs[mask].mean(axis=0)
        else:
            avg_matrix[true_label] = 0.0
    avg_proba_matrices[optimizer] = avg_matrix

# %%

optimizers = ['sgd', 'adagrad', 'rmsprop', 'adam']
for i, h in enumerate(results):
    print(optimizers[i])
    print(f"train_times[i]: {train_times[i]:.2f} seconds")
    print(f"train_losses[i]: {train_losses[i]:.2f}")
    print(f"train_accuracies[i]: {train_accuracies[i]:.2f}")
    print(f"test_losses[i]: {test_losses[i]:.2f}")
    print(f"test_accuracies[i]: {test_accuracies[i]:.2f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h.history['loss'])
    plt.title('Pérdida del Modelo')
    plt.ylabel('Loss')
    plt.xlabel('Época')

    plt.subplot(1, 2, 2)
    plt.plot(h.history['accuracy'])
    plt.title('Precisión del Modelo')
    plt.ylabel('Accuracy')
    plt.xlabel('Época')

    plt.show()

    M: np.ndarray = avg_proba_matrices[optimizers[i]]
    print("Matriz de probabilidades promedio (filas=true, columnas=pred):")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='Prob. promedio')
    plt.title(f'Matriz de probabilidad promedio - {optimizers[i]}')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta verdadera')
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.tight_layout()
    plt.show()
