# Course Project for RIAI 2018

The goal of the project is to design a precise and scalable automated verifier for proving the robustness of fully connected feedforward neural networks with rectified linear activations (ReLU) against adversarial attacks. We will consider adversarial attacks based on the L∞ -norm based perturbations. This type of attack tries to find a misclassified image inside a L∞ -norm based ε -ball around the original image. We will focus on leveraging both the Interval domain [3] and Linear programming [2] for designing our verifier. There exists a tradeoff between the precision of an abstract domain and its speed. For example, certification with the Interval domain is fast but the results can often be imprecise, i.e., domain fails to prove robustness when the network is robust for a given L∞ -norm based perturbation. On the other hand, Linear programming enables more precise verification than the Interval domain but it can be slow compared to the Interval domain for large networks. The challenge is thus to combine the speed of Interval domain with the precision of Linear programming to achieve a better tradeoff for the certification of neural network robustness.

### Given test Set

100 files from the MNIST testset formatted to be used by our analyzer. Each file contains 785 rows containing intervals. The first row of the file denotes the label of the image and the remaining rows describe the pixel intensity in the range [0,1].

### Given Neural Networks

10 trained fully connected feedforward networks with ReLU activation for the MNIST dataset. The number of hidden layers and the number of hidden units per layer in a network can be deduced from its name, e.g., the file “mnist_relu_3_50.txt” contains 3 hidden layers each containing 50 hidden units.

### Grading

The project will be graded based on the precision and soundness of your verifier:
- You start with 0 points.
- You receive 1 point for any verification task for which your verifier correctly outputs “verified” within the time limit
- You will be deducted 1 point if your verifier outputs “verified” when the network is not robust for the provided image and ε
- If there is a timeout on a verification task, then the result will be considered as “cannot be verified”

### Submission
Please submit your modified “analyzer.py” file along with a one page write up about your heuristics. Please make sure that your code runs and reproduces your results on the virtual machine that we provide. The deadline is 11:59 PM, December 20, 2018

## [analyzer.py](analyzer.py)

### Input:

- A fully connected feedforward neural network
- An image
- An ε value (ranging between 0.005 and 0.1)

### Output:

The verifier runs the analysis on the original image first to determine if the neural network classifies the image correctly. If the image is not classified correctly then the verifier outputs an error message and exits. Otherwise, the verifier runs the Interval analysis on the input image perturbed by the provided epsilon and outputs:
- “verified” if the verifier successfully proves that network is robust for the given image and ε 
- otherwise, it outputs “cannot be verified”

