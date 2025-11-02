# %%
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

# %% [markdown]
# # 1. Carga de datos

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# %% [markdown]
# # 2. Preprocesamiento

# %%
# Normalización
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir etiquetas a par (0) / impar (1)
y_train = (y_train % 2).astype(np.float32)
y_test = (y_test % 2).astype(np.float32)

print(f"\nDistribución en train:")
print(f"  Pares: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Impares: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")

# %% [markdown]
# # 3. Configuraciones a evaluar

# %%
# Valores posibles para cada parámetro
param_values = {
    'hidden_neurons': [512],
    'epochs': [15],
    'loss': ['binary_crossentropy'],
    'batch_size': [256],
    'lr': [0.001],
    'validation_split': [0.1]
}

param_names = list(param_values.keys())
param_combinations = list(product(*param_values.values()))

configurations = []
for i, combination in enumerate(param_combinations, 1):
    config = {'name': f'Config_{i}'}
    for param_name, param_value in zip(param_names, combination):
        config[param_name] = param_value
    configurations.append(config)

for config in configurations:
    print(f"{config['name']:>10} | "
          f"neurons={config['hidden_neurons']:>3} | "
          f"epochs={config['epochs']:>2} | "
          f"batch={config['batch_size']:>3} | "
          f"lr={config['lr']:.4f} | "
          f"val_split={config['validation_split']:.1f}")

# %% [markdown]
# # 4. Función para crear modelos

# %%


def create_model(hidden_neurons: int = 128) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(hidden_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# %% [markdown]
# # 5. Función de evaluación completa

# %%


def evaluate_model(model: tf.keras.models.Sequential, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    results = {}

    # Accuracy y loss en test
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Predicciones (para clasificación binaria con sigmoid)
    y_pred_probs = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    errors = np.sum(y_pred != y_test)

    print(f"Errors:  {errors}/{len(y_test)} ({errors/len(y_test)*100:.2f}%)")

    # Matrices de confusión (par vs impar)
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
all_results = {}
all_histories = {}

for config in configurations:
    name = config['name']
    print(f"\n{'▶'*3} Configuración: {name}")
    print(f"    Neuronas capa oculta: {config['hidden_neurons']}")
    print(f"    Épocas: {config['epochs']}")
    print(f"    Loss: {config['loss']}")
    print(f"    Batch size: {config['batch_size']}")
    print(f"    Learning rate: {config['lr']}")
    print(f"    Validation split: {config['validation_split']}")
    print(f"    Optimizer: Adam (fijo)")

    model = create_model(hidden_neurons=int(config['hidden_neurons']))
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])

    model.compile(
        optimizer=optimizer,
        loss=config['loss'],
        metrics=['accuracy']
    )

    # Entrenar y medir tiempo
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        verbose=1
    )
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Tiempo de entrenamiento: {elapsed_time:.2f} segundos")

    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    # Guardar
    all_histories[name] = history
    all_results[name] = {
        'config': config,
        'model': model,
        'training_time': elapsed_time,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }

# %% [markdown]
# # 7. Tabla comparativa de resultados

# %%

comparison_data = []
for name, results in all_results.items():
    comparison_data.append({
        'Configuración': name,
        'Neuronas': results['config']['hidden_neurons'],
        'Épocas': results['config']['epochs'],
        'Loss': results['config']['loss'],
        'Batch Size': results['config']['batch_size'],
        'LR': results['config']['lr'],
        'Val Split': results['config']['validation_split'],
        'Val Loss': f"{results['val_loss']:.4f}",
        'Val Accuracy': f"{results['val_accuracy']:.4f}",
        'Tiempo (s)': f"{results['training_time']:.2f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Guardar a CSV
os.makedirs('results', exist_ok=True)
df_comparison.to_csv('results/comparison_table.csv', index=False)
print("\n💾 Tabla guardada en: results/comparison_table.csv")

# Identificar mejor configuración
best_config = max(all_results.items(), key=lambda x: x[1]['val_accuracy'])
print(f"\n🏆 MEJOR CONFIGURACIÓN: {best_config[0]}")
print(f"   Val Accuracy: {best_config[1]['val_accuracy']:.4f}")
print(f"   Val Loss: {best_config[1]['val_loss']:.4f}")
print(f"   Training Time: {best_config[1]['training_time']:.2f}s")

# %% [markdown]
# # 8. Evaluación de la mejor configuración

# %%
results_test = evaluate_model(all_results[best_config[0]]['model'], x_test, y_test)

print(f"\n✅ RESULTADOS FINALES DEL MEJOR MODELO:")
print(f"   Config: {best_config[0]}")
print(f"   Test Accuracy: {results_test['accuracy']:.4f}")
print(f"   Test Loss: {results_test['loss']:.4f}")
print(f"   Test Errors: {results_test['errors']}/{len(y_test)}")

# %% [markdown]
# # 9. Visualizaciones comparativas

# %%
# Gráfica comparativa: Loss durante entrenamiento
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for name, history in all_histories.items():
    plt.plot(history.history['val_loss'], label=name, linewidth=2, alpha=0.8)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Comparación de Loss en Entrenamiento', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for name, history in all_histories.items():
    plt.plot(history.history['val_accuracy'], label=name, linewidth=2, alpha=0.8)
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

# Validation Accuracy
config_names = list(all_results.keys())
val_accs = [all_results[name]['val_accuracy'] for name in config_names]

axes[0].bar(range(len(config_names)), val_accs, color='#3498db', alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(config_names)))
axes[0].set_xticklabels(config_names, rotation=45, ha='right')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Validation Accuracy por Configuración', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Validation Loss
val_losses = [all_results[name]['val_loss'] for name in config_names]

axes[1].bar(range(len(config_names)), val_losses, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].set_xticks(range(len(config_names)))
axes[1].set_xticklabels(config_names, rotation=45, ha='right')
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Validation Loss por Configuración', fontsize=12, fontweight='bold')
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
# # 10. Matriz de Confusión del Mejor Modelo

# %%

best_name = best_config[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(results_test['cm'], annot=True, fmt='d', cmap='Oranges',
            ax=ax, cbar_kws={'label': 'Cantidad'},
            xticklabels=['Par (0)', 'Impar (1)'],
            yticklabels=['Par (0)', 'Impar (1)'])
ax.set_xlabel('Predicción', fontsize=11)
ax.set_ylabel('Etiqueta Verdadera', fontsize=11)
ax.set_title(f'Matriz de Confusión - {best_name}\n(Test Set: Par vs Impar)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'results/confusion_matrix_best_{best_name}.png', dpi=150, bbox_inches='tight')
plt.show()
