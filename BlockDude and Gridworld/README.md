# OVERVIEW

This folder presents the code for designing, testing and running experiments in the two chosen domains: BlockDude and GridWolrd.

The **BlockDude** and **GridWorld** folders are similar as their code is meant to implement the same things on the two domains. The most important functions one would need to interact with would be *TransferLearning.java and *TransferLearning_UTILS.java. The former contains functions for running ad visualizing reinforcement learning episodes over problems designed in the latter.

The **TransferLearning** folder contains all those functions that were needed to implement our transfer learning algorithm. The most of them are modifications of the original files from the library BURLAP (http://burlap.cs.brown.edu/) although the main structure of the files was not changed.

The **CurriculumLearning** folder contains the code for running curriculum learning experiments. The reinforcement learning tasks composing the curricula would be taken from the ones present in the **BlockDude** and **GridWorld** folders while the transfer learning algorithm would be the one implemented in **TransferLearning**.

The folder **myProj** contains other support files and tests.

# Create a new reinforcement learning task

In order to create a new reinforcement learning problem in the domains of GridWorld or BlockDude, start by editing the file **TransferLearning_UTILS.java*. 
This file already contains a good set of problems for each of these domains and some of them were used for designing the experiments of two peer reviewed publications.
For designing a new problem you should start by creating a state object for the environment you have chosen. 
In BlockDude you simply need to fill in the object with the standard information it requires (i.e. the map design, initial position for the agent, the exit block and any other movable block in the map). 
In GridWorld instead, the state generation is wrapped by a class (*GridWorldDomainFeatures*) which holds all the information required for creating it (i.e. the map design, initial position of the agent, the treasure, all thee fires and pits).

***NOTE***: If you are using the already implemented transfer learning algorithm, it is crucial to name/label (see strings as “fire1” in GridWorld or “block1” in BlockDude) all the features you are setting at this stage the same way across different problems to ensure successful transfer.

The next step is to set the state variables and the precision of the Tile Coding function approximator. This action takes place within the function *CreateTransferLearningTCFeatures*, which is a large *if else* statement that gathers all the setting actions for all the maps. 

Here you need to fill the map *domain*, with the range values of each variable for each dimension. This range does not need to be precise (although it is in all the already implemented problems) but it should never cover a smaller "area" than it is supposed to, otherwise you will get execution errors. Remember that we are following an agent centric implementation for improving the transferability among different problems which means that all these values have to be reported to the agent position in the map. Refer to the already implemented problems for better understanding the way to set these ranges.

The Tile Coding precision can be set immediately after having filled *domain*. Each of the elements in the array *tilingWidths* contain the size of the tile relative to the variables in *domain*. The map and the array must be sorted accordingly.

# Solve a reinforcement learning task

At the end of the previously described process it is recommended to run the newly generated task few times to be sure that the agent is performing as desired. This can be done by executing the function *PerformLearningAlgorithm* in **TransferLearning.java*. Note that this function allows you to specify the number of episodes you want the agent to be learning for as well as the number of epochs (so to be able to assess a more statistically consistent performance from the execution) although there is no variable for the number of actions per episode which has to be manually set within this function. Moreover *PerformLearningAlgorithm* can also receive a value function in input which is how the transfer learning algorithm can be used in this file.
