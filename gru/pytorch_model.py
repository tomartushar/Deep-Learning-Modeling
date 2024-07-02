import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


torch.manual_seed(0)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False, num_layers=1)
        self.fc = nn.Linear(1*hidden_size, output_size)

    def forward(self, x):
        '''
        Shape of x will be (num_samples, seq_length, input_features) if batch_first = False,
        Otherwise (seg_length, num_samples, input_features)
        Shape of hidden state will be (D*num_layers, seq_length, output_features),
        D=2 if bidirectional = True, otherwise 1.
        '''
        h0 = torch.zeros(1*1, x.size(0), self.hidden_size)
        out, _ = self.gru1(x, h0)
        # out = self.fc(out[:,-1,:]) # for many-to-one
        out = self.fc(out) # for many-to-many
        return out


def train(model, dataloader):

    # criteria = nn.MSELoss() # for regression task
    criteria = nn.CrossEntropyLoss() # for classfication task
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # loss = criteria(outputs, batch_y) # for many-to-one
            loss = criteria(outputs.transpose(1,2), batch_y) # for many-to-many, or loss = criteria(outputs.view(-1, output_size), batch_y.view(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch+1}, Loss: {train_loss/len(dataloader):.6f}')

    print('training complete')
    return model


def get_parameters(model):
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print()

def predict(model, x):
    with torch.no_grad():
        outputs = model(x)
        outputs = torch.argmax(outputs, dim=-1) # for classification
        print(f"Predicted output is:\n", outputs)




input_size = 3
hidden_size = 5
# output_size = 1 # for regression task
output_size = 3 # for classification task
num_epochs = 10
lr = 0.01

seq_length = 5
num_samples = 1000


X = torch.randn(num_samples, seq_length, input_size)
# y = torch.randn(num_samples, output_size) # for regression task (many-to-one)
# y = torch.randint(0, output_size, (num_samples,)) # for classificaiton task (many-to-one)
y = torch.randint(0, output_size, (num_samples, seq_length)) # for classification task (many-to-many)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32,shuffle=True)

model = GRU(input_size, hidden_size, output_size)

train(model, dataloader)
get_parameters(model)
predict(model, torch.randn(2, seq_length, input_size))

