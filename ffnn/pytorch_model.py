import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def prepare_data(num_samples = 1000, num_classes = 3):
    X = np.random.rand(1000, 20)
    y = np.random.randint(num_classes, size=(1000,))
    X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.2)
    
    return X_train, y_train, X_val, y_val


class Model(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU(32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.output(x)
        return x

def train():
    model = Model(input_dim, num_classes)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criteria(output, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader):.4f}")
        
        if val_loader != None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    output = model(X_batch)
                    loss = criteria(output, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Validation loss: {val_loss:.4f}")
            if val_loss < best_val_loss - 1e-3:
                best_val_loss = val_loss
                torch.save(model.state_dict(),'best_model.pth')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print('Early stopping')
                    break
        print("--"*10)

def predict(x):
    model = Model(input_dim, num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        output = model(x)
        predicted_class = torch.argmax(output, dim=-1)
        print(f"Predicted classes are: {predicted_class}")




X_train, X_val, y_train, y_val = prepare_data()
print(f'Training-set size: {X_train.shape}, {y_train.shape}')
print(f'Validation-set size: {X_val.shape}, {y_val.shape}')
input_dim = X_train.shape[1]
num_classes = 3
batch_size = 32
epochs = 10
patience = 5

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                           shuffle=True)
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(y_val, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=True)

train()
predict(torch.randn(2, input_dim))

