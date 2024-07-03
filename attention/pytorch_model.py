import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, key, value):
        # x -> (batch_size, seq_length, hidden_dim)

        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        attn_scores = torch.bmm(Q, K.transpose(1,2)) # (batch_size, seq_length, seq_length)
        attn_scores = torch.softmax(attn_scores, dim = -1)

        context_vectors = torch.bmm(attn_scores, V) # (batch_size, seq_length, hidden_dim)

        return context_vectors, attn_scores


class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers = 1, bidirectional = False):
        super(LSTMAttention, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                             num_layers=num_layers, bidirectional=bidirectional)
        self.attn1 = Attention(hidden_dim*(2 if bidirectional else 1)) # for manual attention
        # self.attn1 = nn.MultiheadAttention(hidden_dim*(2 if bidirectional else 1), num_heads=1, batch_first=True) # for torch attenion class
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x, (hn, cn) = self.lstm1(x)
        context_vectors, attn_scores = self.attn1(x,x,x)
        outputs = self.fc1(context_vectors)

        return outputs
    

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


input_dim = 3
hidden_dim = 4
output_dim = 3

num_samples = 1000
seq_length = 5
epochs = 10

X = torch.randn(num_samples, hidden_dim, input_dim)
y = torch.randint(output_dim, size=(num_samples, hidden_dim))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

model = LSTMAttention(input_dim, hidden_dim, output_dim, bidirectional=False, num_layers=1)

train(model, dataloader)
predict(model, torch.randn(2, seq_length, input_dim))



