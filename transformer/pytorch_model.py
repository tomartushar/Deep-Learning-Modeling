import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                enc_layers, dec_layers, num_heads,
                dropout = 0.2):
        super(TransformerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = hidden_dim,
                                       nhead=num_heads, dropout=dropout),
            num_layers=enc_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim,
                                       nhead=num_heads, dropout=dropout),
            num_layers=dec_layers
        )

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.position_embedding = self._generate_positional_embeddings(hidden_dim)

    def forward(self, src, tgt):
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        return output

    def _generate_positional_embeddings(self, hidden_dim, max_len = 1000):
        pos_enc = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,hidden_dim,2).float()*(-torch.log(torch.tensor(10000.0))/hidden_dim))
        pos_enc[:,0::2] = torch.sin(pos*div_term)
        pos_enc[:,1::2] = torch.cos(pos*div_term)

        return pos_enc.unsqueeze(1)
    
    def encode(self, src):
        src = self.embedding(src)* torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        src += self.position_embedding[:src.size(0),:]
        memory = self.encoder(src)
        return memory
    
    def decode(self, tgt, memory):
        tgt = self.embedding(tgt)*torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        tgt + self.position_embedding[:tgt.size(0),:]
        output = self.decoder(tgt, memory)
        return self.fc_out(output)
    

def train(model, src, tgt, epochs = 10):
    criteria = nn.CrossEntropyLoss(ignore_index=0) # assuming passing index is 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    tgt_input = tgt[:,:-1]
    tgt_output = tgt[:,1:]
    src = src.transpose(0,1)
    tgt_input = tgt_input.transpose(0,1)
    tgt_output = tgt_output.transpose(0,1)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        output = model(src, tgt_input)
        output = output.view(-1, output_dim)
        tgt_output = tgt_output.reshape(-1)

        loss = criteria(output, tgt_output)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch+1}, Loss: {train_loss/src.size(0):.6f}')
    
    print("Training complete.")

def greedy_decode(model, src, max_len, start_symbol):
    src = src.transpose(0,1)
    memory = model.encode(src) # (input_seq_length, batch_size, hidden_dim)

    ys = torch.ones(1, src.size(1)).fill_(start_symbol).type_as(src.data) # (1, batch_size)

    for i in range(max_len-1):
        out = model.decode(ys, memory)
        prob = out[-1,:,:]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(0)
        ys = torch.cat([ys, next_word], dim=0)

    return ys.transpose(0,1)


input_dim = 100
hidden_dim = 512
output_dim = 100
num_heads = 8
enc_layers = 6
dec_layers = 6
start_symbol = 1 # index for <SOS>
max_len = 20


src = torch.randint(0, input_dim, (32, 10))
tgt = torch.randint(0, output_dim, (32, 20))


model = TransformerModel(input_dim,hidden_dim,output_dim,enc_layers,
                         dec_layers, num_heads)

train(model, src, tgt)

src_inference = torch.randint(0, input_dim, (2, 10))
output_inference = greedy_decode(model, src_inference,
                                max_len, start_symbol)

print(f'Output: {output_inference}')
