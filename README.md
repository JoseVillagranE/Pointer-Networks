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

* Para hacer funcionar el codigo se debe leer un dataset de Convex-Hull. Oriol proporcionó un dataset de Convex-Hull y TSP el cual se puede dercargar [aquí](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU).
* Ambas imagenes puestas en este repositorio forman parte de las referencias adjuntas más abajo.
### Referencias

[1]  O. Vinyals, M. Fortunato, and N. Jaitly, “Pointer networks,” in Proc. Adv. Neural Inf. Process. Syst., Montreal, QC, Canada, Dec. 2015, pp. 2692–2700.

[2]  I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in Proc. Int. Conf. Learn. Represent., Toulon, France, Apr. 2017, Art. no. 09940.


 

 