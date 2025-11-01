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
y_train = (y_train % 2).astype(np.int32)
y_test = (y_test % 2).astype(np.int32)
print("Etiquetas paridad (0=par, 1=impar):", y_train[:5])
x_train.dtype

# %%


def build_model(hidden_units: int) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    normalized_name: str = name.lower()
    if normalized_name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if normalized_name == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    if normalized_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if normalized_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Optimizer {name} not supported")


experiments: list[dict[str, object]] = [
    {
        'name': 'hidden_64',
        'hidden_units': 64,
        'epochs': 10,
        'batch_size': 32,
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'validation_split': 0.1,
    },
    {
        'name': 'hidden_128',
        'hidden_units': 128,
        'epochs': 10,
        'batch_size': 32,
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'validation_split': 0.1,
    },
    {
        'name': 'hidden_256',
        'hidden_units': 256,
        'epochs': 10,
        'batch_size': 32,
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'validation_split': 0.1,
    },
    {
        'name': 'hidden_512',
        'hidden_units': 512,
        'epochs': 10,
        'batch_size': 32,
        'loss': 'binary_crossentropy',
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'validation_split': 0.1,
    },
]

# %%

history_objects: list[tf.keras.callbacks.History] = []
train_times: list[float] = []
train_losses: list[float] = []
train_accuracies: list[float] = []
test_losses: list[float] = []
test_accuracies: list[float] = []
experiment_names: list[str] = []
num_classes: int = 2
class_labels: list[str] = ['Par', 'Impar']
avg_proba_matrices: dict[str, np.ndarray] = {}

for experiment in experiments:
    experiment_name: str = str(experiment['name'])
    hidden_units: int = int(experiment['hidden_units'])
    epoch_count: int = int(experiment['epochs'])
    batch_size: int = int(experiment['batch_size'])
    loss_identifier: str = str(experiment['loss'])
    optimizer_name: str = str(experiment['optimizer'])
    learning_rate_value: float = experiment.get('learning_rate')
    validation_split_raw: float = experiment.get('validation_split', 0.0)
    validation_split_value: float = float(validation_split_raw)

    print(f"Iniciando experimento: {experiment_name}")
    print(
        f"Config -> hidden_units={hidden_units}, epochs={epoch_count}, batch_size={batch_size}, "
        f"optimizer={optimizer_name}, learning_rate={learning_rate_value}, loss={loss_identifier}, "
        f"validation_split={validation_split_value}"
    )

    model = build_model(hidden_units)
    optimizer_instance = get_optimizer(optimizer_name, learning_rate_value)
    model.compile(
        optimizer=optimizer_instance,
        loss=loss_identifier,
        metrics=['accuracy']
    )

    start_time: float = time.time()
    fit_kwargs: dict[str, object] = {
        'epochs': epoch_count,
        'batch_size': batch_size,
        'verbose': 1,
    }
    fit_kwargs['validation_split'] = validation_split_value
    history = model.fit(x_train, y_train, **fit_kwargs)
    end_time: float = time.time()
    elapsed_time: float = end_time - start_time

    print(f"Training time for {experiment_name}: {elapsed_time:.2f} seconds")
    train_times.append(elapsed_time)
    experiment_names.append(experiment_name)
    history_objects.append(history)

    print("Evaluación en entrenamiento")
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print("Evaluación en test")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    probs: np.ndarray = model.predict(x_test, verbose=0).reshape(-1, 1)
    binary_probs: np.ndarray = np.concatenate((1.0 - probs, probs), axis=1)
    avg_matrix: np.ndarray = np.zeros((num_classes, binary_probs.shape[1]), dtype=np.float64)
    for true_label in range(num_classes):
        mask: np.ndarray = (y_test == true_label)
        if np.any(mask):
            avg_matrix[true_label] = binary_probs[mask].mean(axis=0)
        else:
            avg_matrix[true_label] = 0.0
    avg_proba_matrices[experiment_name] = avg_matrix

# %%

for index, history in enumerate(history_objects):
    experiment_name = experiment_names[index]
    print(experiment_name)
    print(f"train_times[{index}]: {train_times[index]:.2f} seconds")
    print(f"train_losses[{index}]: {train_losses[index]:.2f}")
    print(f"train_accuracies[{index}]: {train_accuracies[index]:.2f}")
    print(f"test_losses[{index}]: {test_losses[index]:.2f}")
    print(f"test_accuracies[{index}]: {test_accuracies[index]:.2f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val')
    plt.title('Pérdida del Modelo')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    if 'val_loss' in history.history:
        plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Precisión del Modelo')
    plt.ylabel('Accuracy')
    plt.xlabel('Época')
    if 'val_accuracy' in history.history:
        plt.legend()

    plt.show()

    M: np.ndarray = avg_proba_matrices[experiment_name]
    print("Matriz de probabilidades promedio (columnas: Par, Impar):")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='Prob. promedio')
    plt.title(f'Matriz de probabilidad promedio - {experiment_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta verdadera')
    plt.xticks(range(num_classes), class_labels)
    plt.yticks(range(num_classes), class_labels)
    plt.tight_layout()
    plt.show()

# %%


def plot_metric(metric: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for index, history in enumerate(history_objects):
        experiment_name = experiment_names[index]
        plt.plot(history.history[metric], label=f"{experiment_name} ({metric} train)")
        if f"val_{metric}" in history.history:
            plt.plot(history.history[f"val_{metric}"], linestyle='--', label=f"{experiment_name} ({metric} val)")
    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_metric('loss', 'Comparativa de pérdida', 'Loss')
plot_metric('accuracy', 'Comparativa de accuracy', 'Accuracy')
