# Pointer Networks

### Contenido del Repositorio

En este repositorio se implementan Redes Punteras(Pointer-Networks)[1], así como también su entrenamiento y evaluación. Además se agrega la solución al problema de Convex Hull.

### Breve descripción Teórica

Las redes punteras se concibieron en consecuencia a la necesidad de procesar datos secuenciales que poseen un grado de condicionalidad probabilistica. Algunos problemas de tal índole
son Convex-Hull, triangulación y TSP. La arquitecura de la red neuronal se presenta a continuación:

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/PtrNeural.png)

Una ventaja que posee este tipo de modelos con respecto a modelos secuencia-a-secuencia (sequence-to-sequence) es la no-dependencia del largo de la entrada con respecto al largo de la salida,
pudiendo diferenciarse entre ambas variables. Dicha caracteristica se posibilita gracias a la modificacion del model de atención que realizó Oriol Vinyals en el paper original.

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/eqPtr.png)

En donde e y d son los estados escondidos del enconder y decoder, respectivamente. v, W1 y W2 son parametros de aprendizaje
 y C,P son los indices y el conjunto de vectores de la entrada.

### Consideraciones

* Para hacer funcionar el codigo se debe leer un dataset de Convex-Hull. Oriol proporcionó un dataset de Convex-Hull y TSP el cual se puede descargar [aquí](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU).
* Ambas imagenes puestas en este repositorio forman parte de las referencias adjuntas más abajo.

### Resultados

En las siguientes tablas se presentan algunos resultados correspondientes a la solución del problema de TSP para con dos distintos tipos de entrenamiento: Supervisado y Reforzado. 
Además se utilizan las siguientes configuraciones con fines de estudio:

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| input size 	| 2 	| 2 	| 2 	| 2 	| 2 	| 2 	| 2 	| 128 	|
| hidden size 	| 256 	| 256 	| 256 	| 256 	| 256 	| 256 	| 512 	| 512 	|
| bidirectional 	| False 	| False 	| False 	| False 	| False 	| False 	| False 	| False 	|
| mask_bool 	| False 	| False 	| True 	| True 	| False 	| False 	| False 	| False 	|
| hidden_att_bool 	| False 	| False 	| False 	| False 	| False 	| True 	| False 	| False 	|
| first city fixed 	| False 	| True 	| False 	| False 	| False 	| False 	| False 	| False 	|
| C 	| None 	| None 	| None 	| 2 	| 2 	| None 	| None 	| None 	|
| normalization loss 	| True 	| True 	| True 	| True 	| True 	| True 	| True 	| True 	|

En donde las variables indexadas en la primera columna significan:

* **input size**: Dimensión de los nodos de entrada. La dimensión minima es dos y no conlleva ningun procesamiento, mientras que para mayores dimensionalides se debe implementar embedding.
* **hidden size**: Numero de neurona de la LSTM
* **bidirectional**: LSTM bidireccional o no
* **mask_bool**: Enmascarar las probabilidades de los nodos ya elegidos.
* **hidden_att_bool**: Usar el estado latente del mecanismo de atención
* **first_city_fixed**: Añadir un primer nodo fijo para todos lo viajes. Especificamente, [0, 0]
* **C**: Parametro que controla el rango de los logits (Bello, et al. 2017). Si es None se utiliza el mecanismo original.
* **normalization loss**: Normalización de la función de perdida.
* **Teachig forcing**: Probabilidad de asignar el correcto nodo a la entrada del decoder en un paso cualquiera sin importar la elección del mecanismo de atención. 
			Para todos los resultados se asigno cero por lo que no se tabulo. 
			Pero cabe aclarar que el utilizar esta variable sobreentreno la red neuronal, no pudiendo conseguir resultados destacables en el conjunto de validación. Como justificación se podría pensar en el aprendizaje de patrones incorrectos entre la entrada anterior, la elección por parte del mecanismo de atención y la entreda siguiente del decoder. 

#### Supervisado

A modo de primer resultado se presentan aquello obtenidos en la resolución de un viaje de 5 nodos.

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| Accuracy 	| 0.5 	| 0.485 	| 0.549 	| 0.06 	| 0 	| 0.418 	| 0.504 	| 0.498 	|
| Number of Invalid tours 	| 2451 	| 2699 	| 0 	| 0 	| 0 	| 3442 	| 2440 	| 2314 	|
|           Avg Tour Length 	| 1.998 	| 1.932 	| 2.6 	| 2.6 	| 0 	| 1.742 	| 1.999 	| * 	|
| Training time 	| 35:54.53 	| 36:00.00 	| 43:22.41 	| 42:31.77 	| 37:13.89 	| 37:10.23 	| 37:33.9 	| 41:10.55 	|


Estos resultados corresponden a la evaluación de la red neuronal en el conjunto de validación. 

Con el objetivo de mejorar los resultados de precisión en los viajes predichos, se implementa beam search para aumentar la cantidad de viajes validos. Aunque esto trae consigo el aumento del largo promedio de los viajes. Los resultados se presentan a continuación:

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| Accuracy 	| 0.604 	| 0.606 	| 0.512 	| * 	| * 	| 0.556 	| 0.604 	| 0.586 	|
| Avg Tour Length 	| 2.602 	| 2.602 	| 2.602 	| * 	| * 	| 2.692 	| 2.602 	| 2.602 	|

Además se incluyen las siguientes curvas de perdida y radio de viajes validos:

<p align="middle">
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Loss.png" height="50%" width="40%" />
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Ratio.png" height="50%" width="40%" />
</p>

A pesar que la red neuronal demuestra aprendizaje, los resultados en sí no son muy alentadores. Citando el paper de Bello en donde mencionan:

".., we implement and train a pointer network with supervised learning, similarly to (Vinyals et al., 2015b). While our supervised data 
consists of one million optimal tours, we find that our supervised learning results are not as good as those reported in by (Vinyals et al., 2015b). We suspect that learning from optimal tours is harder for
supervised pointer networks due to subtle features that the model cannot figure out only by looking at given supervised targets"[[2]](#2).

Tampoco pudieron lograr la implementación original. Podría ser que el paper original de Pointers Networks no documento todo los pormenores del entrenamiento supervisado.

#### Reforzado

Los modelos implementados para el entrenamiento Reforzado se muestran a continuación:


* **multinomial-RL**: Establece la elección del siguiente nodos muestreando desde las probabilidades obtenidas.
* **Greedy-RL**: Establece la elección del siguiente nodo elegiendo el de mayor probabilidad.
* **Sampling-RL**: Estrategia de validación mediante el muestreo del menor tour.
* **Active-Search-RL**: Estrategia de validación mediante Active Search.
* **Bello's Paper**: Mejor resultado proveniente de este paper para cada largo de viaje.
* **Supervised**: Resultados provenientes del paper original de Pointer-Networks.
* **Optimial**: Largo optimial del viaje.

Los resultados obtenidos del largo promedio de viaje son los siguientes:

| Configuración del modelo 	| Multinomial-RL 	| Greedy-RL 	| Sampling-RL 	| Active-Search-RL 	| Bello's Paper 	| Supervised 	| Optimal 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| 5 	| 2.13	|  	|  	|  	| * 	| 2.12 	| 2.12 	|
| 10 	| 2.91 	|  	|  	|  	| * 	| 2.88 	| 2.87 	|
| 20 	| 3.95 	|  	|  	|  	| 3.82 	| 3.88 	| 3.82 	|

Es necesario mencionar que se entrenó con viajes generados aleatoriamente con pytorch ocupando la semilla por defecto para el entrenamiento y la semilla 666 para la validación del modelo.
Además, el entrenamiento del modelo es sensible a la elección de esta semilla, en donde para algunas puede que el modelo no funcione o al menos no converge al mejor resultado. Claramente, esto es consecuencia directa de la complejidad
que se podrían generar en los viajes para la obtención de estos mismo de forma aleatoria. 

A modo de ejemplo se infieren dos viajes: Un primer de 10 nodos y un segundo de 20 nodos.

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/examples_10_20.png)

### Referencias
<a id="1">[1]</a>
O. Vinyals, M. Fortunato, and N. Jaitly, “Pointer networks,” in Proc. Adv. Neural Inf. Process. Syst., Montreal, QC, Canada, Dec. 2015, pp. 2692–2700.

<a id="2">[2]</a>
I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in Proc. Int. Conf. Learn. Represent., Toulon, France, Apr. 2017, Art. no. 09940.


 

 