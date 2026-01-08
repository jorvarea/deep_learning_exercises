## Task 1

Analizamos los siguientes resultados. Primero, hemos cogido varias analogías que ponen de manifiesto la relación semántica que capturan los embeddings.

Para king - man + woman, vemos que para la palabra queen aparece en general en segundo puesto (la primera es king). Esto es matemáticamente esperable ya que el vector resultante permanece muy próximo al original. La aparición de queen en segundo lugar confirma que la dirección del cambio semántico es correcta, aunque la magnitud del desplazamiento no sea suficiente para alejarlo del punto de partida king.

Al aumentar la dimensión del embedding, vemos que el valor de la distancia coseno disminuye, debido a la esparsidad de los vectores, mientras que el del producto aumenta, debido a un módulo mayor. Esto se repite para las otras analogías.

Es curioso que para la analogía colder - cold + hot, cuando usamos el dot product, la palabra hotter no aparece entre las 10 primeras para el embedding de 50 dimensiones. Las que sí aparecen son colder, petruno, temperatures, music/club.

## Analogías por dimensión (Cosine y Dot Product)

| Analogía | Dim | Target | Rank Cos | Rank Dot | Cosine | Dot |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| king - man + woman | 50d | queen | 2 | 3 | 0.8610 | 24.64 |
|  | 100d | queen | 2 | 2 | 0.7834 | 29.91 |
|  | 300d | queen | 2 | 2 | 0.6896 | 38.17 |
| colder - cold + hot | 50d | hotter | 3 | >10 | 0.8125 | 22.55 |
|  | 100d | hotter | 2 | 2 | 0.7616 | 28.47 |
|  | 300d | hotter | 1 | 1 | 0.7156 | 41.49 |
| paris - france + spain | 50d | madrid | 3 | 3 | 0.8017 | 23.49 |
|  | 100d | madrid | 1 | 1 | 0.7962 | 29.49 |
|  | 300d | madrid | 1 | 1 | 0.7379 | 39.67 |
| bigger - big + small | 50d | smaller | 4 | 3 | 0.8655 | 22.29 |
|  | 100d | smaller | 2 | 2 | 0.8575 | 26.74 |
|  | 300d | smaller | 2 | 1 | 0.7753 | 31.71 |

## Pares Relacionados

Ahora analizamos la diferencia entre usar el dot product y la similitud coseno al calcular la similitud entre pares relacionados y no relacionados.

Es difícil hacer una comparación justa entre ambos, ya que utilizan escalas diferentes. La similitud coseno es más fácil de interpretar, ya que está acotada entre 0 y 1, mientras que el dot product no.

Notamos lo que ya mencionábamos antes, que al aumentar la dimensión del embedding, el valor de la distancia coseno disminuye, debido a la esparsidad de los vectores, mientras que el del producto aumenta, debido a un módulo mayor.

| Palabra 1 | Palabra 2 | 50d Cos | 50d Dot | 100d Cos | 100d Dot | 300d Cos | 300d Dot |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| king | queen | 0.7839 | 21.88 | 0.7508 | 27.59 | 0.6336 | 30.78 |
| cat | dog | 0.9218 | 19.74 | 0.8798 | 25.00 | 0.6817 | 28.78 |
| happy | joyful | 0.5550 | 11.04 | 0.5260 | 12.32 | 0.4751 | 15.16 |
| car | vehicle | 0.8834 | 27.46 | 0.8631 | 32.49 | 0.7655 | 35.86 |
| doctor | nurse | 0.7977 | 19.29 | 0.7522 | 22.12 | 0.5860 | 23.54 |
| sun | moon | 0.6543 | 17.12 | 0.6138 | 22.21 | 0.4807 | 24.72 |
| coffee | tea | 0.8080 | 20.95 | 0.7733 | 26.54 | 0.6692 | 32.34 |
| france | paris | 0.8025 | 26.57 | 0.7482 | 31.54 | 0.6581 | 34.27 |
| **PROMEDIO** | | **0.7758** | **20.51** | **0.7384** | **24.98** | **0.6187** | **28.18** |

## Pares No Relacionados

| Palabra 1 | Palabra 2 | 50d Cos | 50d Dot | 100d Cos | 100d Dot | 300d Cos | 300d Dot |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| king | banana | 0.2207 | 5.76 | 0.1609 | 5.35 | 0.0666 | 3.06 |
| cat | democracy | 0.0368 | 0.97 | 0.0840 | 2.69 | 0.0007 | 0.03 |
| happy | refrigerator | 0.1862 | 4.43 | 0.1670 | 4.95 | 0.0350 | 1.27 |
| car | philosophy | 0.0427 | 1.38 | 0.0919 | 3.45 | 0.0126 | 0.58 |
| doctor | volcano | 0.1447 | 4.09 | 0.0665 | 2.31 | -0.0126 | -0.62 |
| sun | keyboard | 0.1990 | 5.67 | 0.0584 | 2.13 | 0.0711 | 3.75 |
| coffee | elephant | 0.2532 | 6.02 | 0.1829 | 5.35 | 0.1410 | 6.46 |
| france | purple | 0.1403 | 4.29 | 0.1283 | 4.55 | 0.0616 | 3.03 |
| **PROMEDIO** | | **0.1529** | **4.08** | **0.1175** | **3.85** | **0.0470** | **2.20** |
