# %%
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

#%% [markdown]
# # 1. Carga y exploración de datos

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de test:          {x_test.shape}")
print(f"Etiquetas train:        {y_train.shape}")
print(f"Etiquetas test:         {y_test.shape}")
print(f"\nPrimeras 10 etiquetas:  {y_train[:10]}")
print(f"Rango original:         [{x_train[0].min()}, {x_train[0].max()}]")
print(f"Dtype original:         {x_train.dtype}")

# Visualizar ejemplos
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].set_title(f'Label: {y_train[i]}', fontsize=12)
    axes[i].axis('off')
plt.suptitle('Ejemplos del dataset MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dataset_examples.png', dpi=150, bbox_inches='tight')
plt.show()

#%% [markdown]
# # 2. Preprocesamiento

# %%
# Normalización
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"Rango normalizado: [{x_train[0].min():.2f}, {x_train[0].max():.2f}]")

#%% [markdown]
# # 3. Configuraciones a evaluar

# %%
configurations = [
    {
        'name': 'Baseline',
        'layers': [128],
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 32
    },
    {
        'name': 'SGD_optimized',
        'layers': [128],
        'optimizer': 'sgd',
        'lr': 0.1,
        'batch_size': 128
    },
    {
        'name': 'Adam_simple',
        'layers': [128],
        'optimizer': 'adam',
        'lr': 0.001,
        'batch_size': 32
    },
    {
        'name': 'Adam_large',
        'layers': [256, 128],
        'optimizer': 'adam',
        'lr': 0.001,
        'batch_size': 128
    },
    {
        'name': 'RMSprop',
        'layers': [128],
        'optimizer': 'rmsprop',
        'lr': 0.001,
        'batch_size': 64
    },
    {
        'name': 'Adagrad',
        'layers': [128],
        'optimizer': 'adagrad',
        'lr': 0.01,
        'batch_size': 64
    },
]

#%% [markdown]
# # 4. Función para crear modelos

# %%
def create_model(layers=[128]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    for neurons in layers:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

#%% [markdown]
# # 5. Función de evaluación completa

# %%
def evaluate_model(model, x_train, y_train, x_test, y_test, config_name):
    results = {}
    
    # Accuracy y loss en train
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Accuracy y loss en test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Predicciones y número de errores
    y_train_pred = np.argmax(model.predict(x_train, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    train_errors = np.sum(y_train_pred != y_train)
    test_errors = np.sum(y_test_pred != y_test)
    
    print(f"\n❌ Errores en train: {train_errors}/{len(y_train)} ({train_errors/len(y_train)*100:.2f}%)")
    print(f"❌ Errores en test:  {test_errors}/{len(y_test)} ({test_errors/len(y_test)*100:.2f}%)")
    
    # Matrices de confusión
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Guardar resultados
    results['train_loss'] = train_loss
    results['train_acc'] = train_acc
    results['test_loss'] = test_loss
    results['test_acc'] = test_acc
    results['train_errors'] = train_errors
    results['test_errors'] = test_errors
    results['train_cm'] = train_cm
    results['test_cm'] = test_cm
    results['y_train_pred'] = y_train_pred
    results['y_test_pred'] = y_test_pred
    
    return results

#%% [markdown]
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
    results = evaluate_model(model, x_train, y_train, x_test, y_test, name)
    results['training_time'] = elapsed_time
    results['config'] = config
    
    # Guardar
    all_results[name] = results
    all_histories[name] = history

#%% [markdown]
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
        'Train Acc': f"{results['train_acc']:.4f}",
        'Test Acc': f"{results['test_acc']:.4f}",
        'Train Loss': f"{results['train_loss']:.4f}",
        'Test Loss': f"{results['test_loss']:.4f}",
        'Train Errors': results['train_errors'],
        'Test Errors': results['test_errors'],
        'Tiempo (s)': f"{results['training_time']:.2f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Guardar a CSV
os.makedirs('results', exist_ok=True)
df_comparison.to_csv('results/comparison_table.csv', index=False)
print("\n💾 Tabla guardada en: results/comparison_table.csv")

# Identificar mejor configuración
best_config = max(all_results.items(), key=lambda x: x[1]['test_acc'])
print(f"\n🏆 MEJOR CONFIGURACIÓN: {best_config[0]}")
print(f"   Test Accuracy: {best_config[1]['test_acc']:.4f}")
print(f"   Test Errors: {best_config[1]['test_errors']}")
print(f"   Training Time: {best_config[1]['training_time']:.2f}s")

#%% [markdown]
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
test_accs = [all_results[name]['test_acc'] for name in config_names]

axes[0].bar(range(len(config_names)), test_accs, color='#3498db', alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(config_names)))
axes[0].set_xticklabels(config_names, rotation=45, ha='right')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Test Accuracy por Configuración', fontsize=12, fontweight='bold')
axes[0].set_ylim([0.95, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Test Errors
test_errors = [all_results[name]['test_errors'] for name in config_names]
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

#%% [markdown]
# # 9. Matrices de confusión

# %%
# Visualizar matrices de confusión para cada configuración
for name, results in all_results.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train confusion matrix
    sns.heatmap(results['train_cm'], annot=True, fmt='d', cmap='Blues',
                ax=axes[0], cbar_kws={'label': 'Cantidad'})
    axes[0].set_xlabel('Predicción', fontsize=11)
    axes[0].set_ylabel('Etiqueta Verdadera', fontsize=11)
    axes[0].set_title(f'Train - Matriz de Confusión\n{name}', fontsize=12, fontweight='bold')
    
    # Test confusion matrix
    sns.heatmap(results['test_cm'], annot=True, fmt='d', cmap='Oranges',
                ax=axes[1], cbar_kws={'label': 'Cantidad'})
    axes[1].set_xlabel('Predicción', fontsize=11)
    axes[1].set_ylabel('Etiqueta Verdadera', fontsize=11)
    axes[1].set_title(f'Test - Matriz de Confusión\n{name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()

#%% [markdown]
# # 10. Análisis detallado del mejor modelo

# %%
best_name = best_config[0]
best_results = best_config[1]

# Classification report
print("\n📊 TRAIN SET:")
print(classification_report(y_train, best_results['y_train_pred'], 
                           target_names=[str(i) for i in range(10)]))

print("\n📊 TEST SET:")
print(classification_report(y_test, best_results['y_test_pred'],
                           target_names=[str(i) for i in range(10)]))

#%% [markdown]
# # 11. Conclusiones

# %%

print(f"\n1. MEJOR OPTIMIZADOR:")
optimizer_accs = {}
for name, results in all_results.items():
    opt = results['config']['optimizer']
    if opt not in optimizer_accs:
        optimizer_accs[opt] = []
    optimizer_accs[opt].append(results['test_acc'])

for opt, accs in optimizer_accs.items():
    print(f"   {opt}: Accuracy promedio = {np.mean(accs):.4f} (±{np.std(accs):.4f})")

best_optimizer = max(optimizer_accs.items(), key=lambda x: np.mean(x[1]))
print(f"\n   ⭐ Mejor optimizador: {best_optimizer[0]}")

print(f"\n2. IMPACTO DEL BATCH SIZE:")
for name, results in all_results.items():
    print(f"   {name} (bs={results['config']['batch_size']}): "
          f"Test Acc={results['test_acc']:.4f}, Time={results['training_time']:.2f}s")

print(f"\n3. TRADE-OFF ACCURACY vs TIEMPO:")
for name in sorted(all_results.keys(), key=lambda x: all_results[x]['training_time']):
    results = all_results[name]
    print(f"   {name}: {results['test_acc']:.4f} accuracy en {results['training_time']:.2f}s")
