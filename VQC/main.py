import torch
import torch.nn as nn
import torch.optim as optim

import pennylane as qp
from pennylane import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

## The dataset we will use is the Iris dataset. 

iris= datasets.load_iris()
data=iris.data
target=iris.target

data=data[target <= 1]
target=target[target <= 1]

data_scaled= minmax_scale(data, feature_range=(0, np.pi))

train_data, test_data, train_target, test_target= train_test_split(data_scaled, target, test_size=0.25)

train_datatensor= torch.tensor(train_data, dtype=torch.float32)
train_targettensor= torch.tensor(train_target, dtype=torch.long)
test_datatensor= torch.tensor(test_data, dtype=torch.float32)
test_targettensor= torch.tensor(test_target, dtype=torch.long)

n_qubits= 4
dev = qp.device('default.qubit', wires= n_qubits)

@qp.qnode(dev, interface='torch')

def first_layer(inputs, weights):
    qp.AngleEmbedding(inputs, wires= range(n_qubits))
    qp.StronglyEntanglingLayers(weights, wires= range(n_qubits))

    return qp.expval(qp.PauliZ(0))

class Clasifier(nn.Module):
    def __init__(self):
        super(Clasifier, self).__init__()

        weights_shape= (3, n_qubits, 3)
        self.weightsq= nn.Parameter(torch.rand(weights_shape, requires_grad=True))

        self.capa_clasica = nn.Linear(1,2)
    def forward(self, x):
        output= torch.stack([first_layer(xi, self.weightsq) for xi in x])

        output= output.unsqueeze(1).to(torch.float32)
        return self.capa_clasica(output)

modelo = Clasifier()
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.05)

print("Training the model")
epochs = 20

for epoch in range(epochs):
    optimizador.zero_grad()
    output = modelo(train_datatensor)
    loss = criterio(output, train_targettensor)
    loss.backward()
    optimizador.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
with torch.no_grad():
    test_output = modelo(test_datatensor)
    _, predicted = torch.max(test_output.data, 1)
    accuracy = (predicted == test_targettensor).sum().item() / len(test_targettensor)
    print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"\n Training finished")
print(f"Precision: {accuracy * 100:.1f}%")