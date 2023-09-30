# Pointer Networks

### Repository Content

In this repository you can find a Pointer Network implementation[1], and also its training and evaluation. Also, the solution to the Convex Hull Problem.

### Briefly theorical Description

The Pointer Networks were coinceived in consequences to the necessities to process sequential data that in some part posses a probabilistic conditionality. Some such problems like are Convex Hull, Triangulation and TSP. In the following figure the architecture of the Neural Network is presented:

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/PtrNeural.png)

One advantage that these models have over sequence-to-sequence models is that the input size is not dependent on the output size, which allows the two variables to be distinguisehd. Such a feature is made possible by the modification of the attentiion model that Oriol Vynials made in his original paper.

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/eqPtr.png)

Where e and d are the hidden states of the encoder and decoder, respectively. v, W1 and W2 are learning parameters, and C, P are the set index and input vectors.

### Considerations

* In order for the code to work, a Convex Hull dataset must be read. Oriol provided a Convex Hull and TSP dataset, which can be downloaded from [here](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU).

* Both images posted to this repository are part of the references attached below.

### Results

The following tables present some results corresponding to the solution of the TSP problem with two different types of training: Supervised and Reinforcement.
Also, the following configurations are used for study purposes:

| Model Configuration 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| input size 	| 2 	| 2 	| 2 	| 2 	| 2 	| 2 	| 2 	| 128 	|
| hidden size 	| 256 	| 256 	| 256 	| 256 	| 256 	| 256 	| 512 	| 512 	|
| bidirectional 	| False 	| False 	| False 	| False 	| False 	| False 	| False 	| False 	|
| mask_bool 	| False 	| False 	| True 	| True 	| False 	| False 	| False 	| False 	|
| hidden_att_bool 	| False 	| False 	| False 	| False 	| False 	| True 	| False 	| False 	|
| first city fixed 	| False 	| True 	| False 	| False 	| False 	| False 	| False 	| False 	|
| C 	| None 	| None 	| None 	| 2 	| 2 	| None 	| None 	| None 	|
| normalization loss 	| True 	| True 	| True 	| True 	| True 	| True 	| True 	| True 	|

Where the indices variables in the first column means:

* **input size**: Dimension of the input nodes. The minimum dimension is two and does not involve any processing, while for larger dimensions you need to implement embedding 
* **hidden size**: Hidden size of the LSTM
* **bidirectional**: Enable bidirectional LSTM
* **mask_bool**: Masking the probabilities of the selected nodes
* **hidden_att_bool**: Use the latent state of the attention mechanism
* **first_city_fixed**: Add a fixed first node for each trip. Specifically, [0, 0] 
* **C**: Parameters that control the range of logits (Bello, et al. 2017). If its  is None, the original value is used
* **normalization loss**: Normalization of the loss function
* **Teaching forcing**: Probability of assigning the correct node to the input of the decoder in any step, regardless of the choice of attention mechanism. All results were assigned zero, so they were not tabulated. However, it should be clarified that the use of this variable overtrained the neural network, which was not able to achieve significant results in the validation set. A possible explanation for this is the learning of incorrect patterns between the previous input, the selection by the attention mechanism, and the next input of the decoder.

#### Supervised

As a first result is presented the one obtained in the resolution of a trip with 5 nodes:

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| Accuracy 	| 0.5 	| 0.485 	| 0.549 	| 0.06 	| 0 	| 0.418 	| 0.504 	| 0.498 	|
| Number of Invalid tours 	| 2451 	| 2699 	| 0 	| 0 	| 0 	| 3442 	| 2440 	| 2314 	|
|           Avg Tour Length 	| 1.998 	| 1.932 	| 2.6 	| 2.6 	| 0 	| 1.742 	| 1.999 	| * 	|
| Training time 	| 35:54.53 	| 36:00.00 	| 43:22.41 	| 42:31.77 	| 37:13.89 	| 37:10.23 	| 37:33.9 	| 41:10.55 	|

These results correspond to the evaluation of the neural network of the validation set.

With the aim of improving the results of accuracy in the predicted trips, its implemented beam search to increase the quantity of failed trips. Although this achieves an increase in the average trip size. The results are presented below:

| Configuración del modelo 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| Accuracy 	| 0.604 	| 0.606 	| 0.512 	| * 	| * 	| 0.556 	| 0.604 	| 0.586 	|
| Avg Tour Length 	| 2.602 	| 2.602 	| 2.602 	| * 	| * 	| 2.692 	| 2.602 	| 2.602 	|

In addition, the following loss curves and radius of valid trips are included:

<p align="middle">
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Loss.png" height="50%" width="40%" />
  <img src="https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/Ratio.png" height="50%" width="40%" />
</p>


Although the neural network proof learning, the results themselves are not encouraging. Cited the paper by Bello where they said:

".., we implement and train a pointer network with supervised learning, similarly to (Vinyals et al., 2015b). While our supervised data 
consists of one million optimal tours, we find that our supervised learning results are not as good as those reported in by (Vinyals et al., 2015b). We suspect that learning from optimal tours is harder for
supervised pointer networks due to subtle features that the model cannot figure out only by looking at given supervised targets"[[2]](#2).

They could not reach the original implementation. Could be that the original paper does not write about the details of the supervised training 

#### Reinforcement

The models implemented for the reinforcement training are presented below:

* **multinomial-RL**: Determine the choice of the next node sample from the obtained probabilities
* **Greedy-RL**: Determine the choice of the next node by selecting the one with the highest probability.
* **Sampling-RL**: Validation strategy by sampling the lower tour.
* **Active-Search-RL**: Active Search Validation strategy.
* **Bello's Paper**: The best result obtained from this paper for each long trip.
* **Supervised**: Results coming from the original Pointer Network paper.
* **Optimial**: Optimal length for every trip

The results obtained for the average long trip are as follows:

| Configuración del modelo 	| Multinomial-RL 	| Greedy-RL 	| Sampling-RL 	| Active-Search-RL 	| Bello's Paper 	| Supervised 	| Optimal 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| 5 	| 2.13	| 2.13	| 2.123	|  	| * 	| 2.12 	| 2.12 	|
| 10 	| 2.91 	| 3.0	| 2.882	|  	| * 	| 2.88 	| 2.87 	|
| 20 	| 3.95 	|  *	| 3.88	|  	| 3.82 	| 3.88 	| 3.82 	|

It's necessary to mention that it was trained with randomly generated trips, with Pytorch using the seed by default for training and the 666 seed for validating the model. In addition, the training of the model is sensible to the elecction of the seed, where for some maybe the model does not work or at least does not converge to the best result. Clearly, this is a direct consequence of the complexity that it could be to generate in the trips for the obtention of this itself in a random way.
El entrenamiento greedy mostro un comportamiento más inestable en los resultados que se iban obteniendo a lo largo del entrenamiento.

As a parameter analysis of the temperature parameter statement on the paper of Bello, its presented the following tables of average long:

| Temperature Parameter 	| Greedy 	| 2.0 	| 1.5 	| 1.0 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|
| 5 	| 2.124 	| 2.123 	| 2.123 	| 2.123 	|

| Temperature Parameter 	| 2.2 	| 2.0 	| 1.5 	| 1.0 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|
| 10 	| 2.904 	| 2.895 	| 2.884 	| 2.882 	|

| Temperature Parameter 	| 2.5 	| 2.0 	| 1.5 	| 1.3 	| 1.2 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| 20 	| 4.42 	| 4.057 	| 3.91 	| 3.886 	| 3.88 	|

As an example, two trips are derived: A first with 10 nodes and a second with 20 nodes.

![alt text](https://github.com/JoseVillagranE/Pointer-Networks/blob/master/Images/examples_10_20.png)

### Referencias
<a id="1">[1]</a>
O. Vinyals, M. Fortunato, and N. Jaitly, “Pointer networks,” in Proc. Adv. Neural Inf. Process. Syst., Montreal, QC, Canada, Dec. 2015, pp. 2692–2700.

<a id="2">[2]</a>
I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in Proc. Int. Conf. Learn. Represent., Toulon, France, Apr. 2017, Art. no. 09940.


 

 
