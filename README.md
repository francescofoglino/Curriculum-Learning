# Curriculum-Learning
Code for the experiments of the paper "Curriculum Learning for Cumulative Return Maximization" published in IJCAI19.

The code is divided into three different parts:
- Block Dude and Gridworld: this folder contains the java code relative to the experiments in the two simulated environmnets "Block Dude" and "Gridworld". All this code is based on the library BURLAP (http://burlap.cs.brown.edu/)
- Combinatorial Optimization Algorithms: this folder contains the implementation of the different combinatorial optimization algorithms against which our proposed algorithm (HTS-CR) is compared in the above mentioned paper
- MGEnv: this folder contains all the code relative to the real-world experiments. MGEnv is a simulated micro-grid domain modeled out of real data from the PecanStreet Inc. database.

The fourth folder, HTS_CR, contains the Matlab implementation for the algorithm HTS-CR. It is not meant to run online but rather to process curricula data previously collected and stored in a separate data structure, although the code can easily be adapted for online applications. Using the offline version of the algorithm was crucial to fastly assessing its performance in order to develop it to its current and best version.
