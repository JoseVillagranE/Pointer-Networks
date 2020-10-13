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

* input size: Dimensión de los nodos de entrada. La dimensión minima es dos y no conlleva ningun procesamiento, mientras que para mayores dimensionalides se debe implementar embedding.
* hidden size: Numero de neurona de la LSTM
* bidirectional: LSTM bidireccional o no
* mask_bool: Enmascarar las probabilidades de los nodos ya elegidos.
* hidden_att_bool: Usar el estado latente del mecanismo de atención
* first_city_fixed: Añadir un primer nodo fijo para todos lo viajes. Especificamente, [0, 0]
* C: Parametro que controla el rango de los logits (Bello, et al. 2017). Si es None se utiliza el mecanismo original.
* normalization loss: Normalización de la función de perdida.

#### Supervisado

A modo de primer resultado se presentan aquello obtenidos en la resolución de un viaje de 5 nodos.

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| Accuracy 	| 0.5 	| 0.485 	| 0.549 	| 0.06 	| 0 	| 0.418 	| 0.504 	| 0.498 	|
| Number of Invalid tours 	| 2451 	| 2699 	| 0 	| 0 	| 0 	| 3442 	| 2440 	| 2314 	|
|           Avg Tour Length 	| 1.998 	| 1.932 	| 2.6 	| 2.6 	| 0 	| 1.742 	| 1.999 	| * 	|
| Training time 	| 35:54.53 	| 36:00.00 	| 43:22.41 	| 42:31.77 	| 37:13.89 	| 37:10.23 	| 37:33.9 	| 41:10.55 	|

Además se incluyen las siguientes curvas de perdida y radio de viajes validos:

<p align="middle">
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Loss.png" height="100%" width="100%" />
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Ratio.png" height="100%" width="100%" />
</p>

#### Reforzado

### Referencias

[1]  O. Vinyals, M. Fortunato, and N. Jaitly, “Pointer networks,” in Proc. Adv. Neural Inf. Process. Syst., Montreal, QC, Canada, Dec. 2015, pp. 2692–2700.

[2]  I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in Proc. Int. Conf. Learn. Represent., Toulon, France, Apr. 2017, Art. no. 09940.


 

 