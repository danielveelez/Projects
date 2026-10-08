This project explores the integration of parametrized quantum circuits (with PennyLane) inside the architecture of Neural Networks (achieved with the help of PyTorch).

1. Theoretical Framework

The architecture of this VQC consists of three main steps. 

First of all, in order to process data in the quantum circuit, we need to prepare it and map it into a  quantum state. For that we scale the data from 0 to $\pi$ and use AngleEmbedding to encode each feature as a rotation angle applied to a single-qubit gate. 

With our data encoded, the quantum state is processed through Strongly Entangling Layers which alternates single-qubit rotations and CNOT gates. This way the state is deeply entangled, allowing the classifier to learn highly non-linear decision boundaries.

Finally we measure the expectation value of PauliZ and we use the gradient descent method to optimize the weights of the quantum circuit. To calculate the gradients, I have used the Parameter Shift Rule. 

2. How to Run the Code

To run this project, ensure you have Python and the important libraries (PyTorch, Pennylane, Numpy, Scikit-learn) installed.

3. Results 
Training the model
Epoch [5/20], Loss: 0.7419
Epoch [10/20], Loss: 0.5327
Epoch [15/20], Loss: 0.3789
Epoch [20/20], Loss: 0.2709
Accuracy: 100.00%

 Training finished
Precision: 100.0%

*numbers may vary when you run it on your own due to random initialization. 

4. Notation

After developing this algorithm, I've noticed some people import PennyLane as qml. The choice of importing PennyLane as qp is solely base in the fact that in PennyLane's Tutorial they used the notation qp. 