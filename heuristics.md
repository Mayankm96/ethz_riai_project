# Heuristics

## First Ideas

### Importance of Neurons
Judge how important a neuron is according to a metrics. Pruning a network is similar to judging the importance of a neuron.
Metrics can include L1/L2 mean of neuron weights, mean activations, the number of times a neuron wasn’t zero [from Pruning Network Blog Post](https://jacobgil.github.io/deeplearning/pruning-deep-learning)

--> One could rank the importance of all neuron per layer according to the L1 norm and use linear programming on the most important ones e.g. use LP on 100 most important neurons. Ensures precision but does not ensure a low computational cost. Global ranking is possible since all layers have same number of weughts with exception of first layer (has 784 weights per neuron).


### Effect of using Linear Programming vs. Box Approximation for a Given Layer.
TODO

## Open Questions
Effect of using Linear Programming vs. Box Approximation for a Given Layer, e.g. is the gain of precision and the computational cost same for usinh LP on a deep layer or on one of the first layers.

## Next Steps

- Implement testing script: Test Combination of Network, Images for a range of epsilons.

## Preliminary Results