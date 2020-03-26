# Pointer Networks

### Contenido del Repositorio

En este repositorio se implementan Redes Punteras(Pointer-Networks)[1], así como también su entrenamiento y evaluación. Además se agrega la solución al problema de Convex Hull.

### Breve descripción Teórica

Las redes punteras se concibieron en consecuencia a la necesidad de procesar datos secuenciales que poseen un grado de condicionalidad probabilistica. Algunos problemas de tal índole
son Convex-Hull, triangulación y TSP. La arquitecura de la red neuronal se presenta a continuación:

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/PtrNeural.png)

Una ventaja que posee este tipo de modelos con respecto a modelos secuencia-a-secuencia (sequence-to-sequence) es la no-dependencia del largo de la entrada con respecto al largo de la salida,
pudiendo diferenciarse entre ambas variables. Dicha caracteristica se posibilita gracias a la modificacion del model de atención que realizó Oriol Vinyals en el paper original.

$$ u_{i}^{j} = v^{T}tanh(W_{1}e_j + W_{2}d_i) \quad j \in (1,\ldots, n)$$

 

 