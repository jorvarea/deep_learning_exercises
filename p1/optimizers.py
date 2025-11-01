# %%
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

# %% [markdown]
# # 1. Carga y exploración de datos

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de test:          {x_test.shape}")
print(f"Etiquetas train:        {y_train.shape}")
print(f"Etiquetas test:         {y_test.shape}")
print(f"\nPrimeras 10 etiquetas:  {y_train[:10]}")
print(f"Rango original:         [{x_train[0].min()}, {x_train[0].max()}]")

# %% [markdown]
# # 2. Preprocesamiento

# %%
# Normalización
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"Rango normalizado: [{x_train[0].min():.2f}, {x_train[0].max():.2f}]")

# %% [markdown]
# # 3. Configuraciones a evaluar

# %%
configurations = [
    {
        'name': 'SGD',
        'layers': [128],
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'SGD_two_layers',
        'layers': [256, 128],
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'Adagrad',
        'layers': [128],
        'optimizer': 'adagrad',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'Adagrad_two_layers',
        'layers': [256, 128],
        'optimizer': 'adagrad',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'RMSprop',
        'layers': [128],
        'optimizer': 'rmsprop',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'RMSprop_two_layers',
        'layers': [256, 128],
        'optimizer': 'rmsprop',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'Adam',
        'layers': [128],
        'optimizer': 'adam',
        'lr': 0.01,
        'batch_size': 128
    },
    {
        'name': 'Adam_two_layers',
        'layers': [256, 128],
        'optimizer': 'adam',
        'lr': 0.01,
        'batch_size': 128
    },
]

# %% [markdown]
# # 4. Función para crear modelos

# %%


def create_model(layers=[128]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    for neurons in layers:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

# %% [markdown]
# # 5. Función de evaluación completa

# %%


def evaluate_model(model: tf.keras.models.Sequential, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    results = {}

    # Accuracy y loss en test
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Predicciones y número de errores
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    errors = np.sum(y_pred != y_test)

    print(f"Errors:  {errors}/{len(y_test)} ({errors/len(y_test)*100:.2f}%)")

    # Matrices de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Guardar resultados
    results['loss'] = loss
    results['accuracy'] = accuracy
    results['errors'] = errors
    results['cm'] = cm
    results['predictions'] = y_pred

    return results

# %% [markdown]
# # 6. Entrenamiento de todas las configuraciones


# %%
epochs = 10
all_results = {}
all_histories = {}

for config in configurations:
    name = config['name']
    print(f"\n{'▶'*3} Configuración: {name}")
    print(f"    Capas: {config['layers']}")
    print(f"    Optimizer: {config['optimizer']} (lr={config['lr']})")
    print(f"    Batch size: {config['batch_size']}")

    # Crear modelo
    model = create_model(layers=config['layers'])

    # Compilar con configuración específica
    if config['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['lr'])
    elif config['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['lr'])
    elif config['optimizer'] == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=config['lr'])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar y medir tiempo
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=config['batch_size'],
        validation_split=0.1,
        verbose=1
    )
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Tiempo de entrenamiento: {elapsed_time:.2f} segundos")

    # Evaluar modelo
    results = evaluate_model(model, x_test, y_test)
    results['training_time'] = elapsed_time
    results['config'] = config

    # Guardar
    all_results[name] = results
    all_histories[name] = history

# %% [markdown]
# # 7. Tabla comparativa de resultados

# %%

comparison_data = []
for name, results in all_results.items():
    comparison_data.append({
        'Configuración': name,
        'Optimizer': results['config']['optimizer'],
        'LR': results['config']['lr'],
        'Batch Size': results['config']['batch_size'],
        'Capas': str(results['config']['layers']),
        'Test Acc': f"{results['accuracy']:.4f}",
        'Test Loss': f"{results['loss']:.4f}",
        'Test Errors': results['errors'],
        'Tiempo (s)': f"{results['training_time']:.2f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Guardar a CSV
os.makedirs('results', exist_ok=True)
df_comparison.to_csv('results/comparison_table.csv', index=False)
print("\n💾 Tabla guardada en: results/comparison_table.csv")

# Identificar mejor configuración
best_config = max(all_results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 MEJOR CONFIGURACIÓN: {best_config[0]}")
print(f"   Test Accuracy: {best_config[1]['accuracy']:.4f}")
print(f"   Test Errors: {best_config[1]['errors']}")
print(f"   Training Time: {best_config[1]['training_time']:.2f}s")

# %% [markdown]
# # 8. Visualizaciones comparativas

# %%
# Gráfica comparativa: Loss durante entrenamiento
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for name, history in all_histories.items():
    plt.plot(history.history['loss'], label=name, linewidth=2, alpha=0.8)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Comparación de Loss en Entrenamiento', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for name, history in all_histories.items():
    plt.plot(history.history['accuracy'], label=name, linewidth=2, alpha=0.8)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparación de Accuracy en Entrenamiento', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Gráfica de barras: Accuracy final
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Test Accuracy
config_names = list(all_results.keys())
test_accs = [all_results[name]['accuracy'] for name in config_names]

axes[0].bar(range(len(config_names)), test_accs, color='#3498db', alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(config_names)))
axes[0].set_xticklabels(config_names, rotation=45, ha='right')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Test Accuracy por Configuración', fontsize=12, fontweight='bold')
axes[0].set_ylim([0.95, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Test Errors
test_errors = [all_results[name]['errors'] for name in config_names]
axes[1].bar(range(len(config_names)), test_errors, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(config_names)))
axes[1].set_xticklabels(config_names, rotation=45, ha='right')
axes[1].set_ylabel('Número de Errores', fontsize=12)
axes[1].set_title('Errores en Test', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Training Time
train_times = [all_results[name]['training_time'] for name in config_names]
axes[2].bar(range(len(config_names)), train_times, color='#2ecc71', alpha=0.8, edgecolor='black')
axes[2].set_xticks(range(len(config_names)))
axes[2].set_xticklabels(config_names, rotation=45, ha='right')
axes[2].set_ylabel('Tiempo (segundos)', fontsize=12)
axes[2].set_title('Tiempo de Entrenamiento', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# # 9. Matrices de confusión

# %%
# Visualizar matrices de confusión para cada configuración
for name, results in all_results.items():
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Test confusion matrix
    sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Oranges',
                ax=ax, cbar_kws={'label': 'Cantidad'})
    ax.set_xlabel('Predicción', fontsize=11)
    ax.set_ylabel('Etiqueta Verdadera', fontsize=11)
    ax.set_title(f'Matriz de Confusión\n{name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
