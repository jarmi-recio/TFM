# Model-Agnostic Meta-Learning

Este directorio contiene los archivos necesarios para implementar el algoritmo maml en el problema de datación de estrellas.

### Data
En `TFM/Code/maml/datasets/` se tienen dos conjuntos de datos, el archivo `gyro_tot_v20180801.txt` contiene el total de estrella con sus correspondientes característica, mientras que, el archivo `test_gyro.txt` guarda las estrellas correspondientes al conjunto de prueba.

### Scripts
- Para ejecutar el proyecto será necesario correr el código `main_jarmi.py`, teniendo este las intrucciones de uso al principio del script.
- En el archivo `data_generator_jarmi.py` se desarrolla la carga de los conjuntos de datos de las estrellas, así como las transformaciones necesarias y la generación de los batches para la fase de entrenamiento y test.
- El archivo `maml_jarmi.py` presenta el desarrollo del algoritmo, por un lado, mediante la definición de su funcionamiento (descenso de gradiente y optimización) y por otro, definiendo su estructura de capas.
- El script `grph.py` sirve para plotear las líneas resultantes tras ejecutar la fase de entrenamiento y test del algoritmo.

