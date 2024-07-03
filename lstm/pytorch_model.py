import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

input_dim = 3
hidden_dim = 4
output_dim = 3

num_samples = 1000
seq_length = 5
epochs = 10


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bidirectional = False,
                 num_layers = 1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True,
                            bidirectional=bidirectional, num_layers=self.num_layers)
        self.fnn = nn.Linear(self.D*self.hidden_dim, self.output_dim)

    def forward(self, x):
        c0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_dim)
        h0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_dim)

        x, _ = self.lstm(x, (h0, c0))
        out = self.fnn(x)
        return out
    

def train(model, dataloader):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criteria = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            output = model(X_batch)
            loss = criteria(output.transpose(1,2), y_batch)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f"Epoch: {epoch+1}, Loss: {train_loss/len(dataloader):.6f}")
    
    print("Training Compelted.")

def predict(model, x):
    with torch.no_grad():
        outputs = model(x)
        outputs = torch.argmax(outputs, dim=-1)
        print("Outputs: \n", outputs)

X = torch.randn(num_samples, hidden_dim, input_dim)
y = torch.randint(output_dim, size=(num_samples, hidden_dim))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

model = LSTM(input_dim, hidden_dim, output_dim, bidirectional=True, num_layers=2)

train(model, dataloader)
predict(model, torch.randn(2, seq_length, input_dim))







