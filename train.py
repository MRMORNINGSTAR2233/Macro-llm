import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_eters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from charecteres to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]#encoder takes string , output a list of integers
decode = lambda l:''.join([itos[i] for i in l])#decoder takes list of integers, output a string

#train and split the data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))#only starting 90% will get trained
train_data = data[:n]
val_data = data[n:]

#loding the data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+1+block_size]for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_eters)
        for k in range(eval_eters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split]= losses.mean()
    model.train()
    return out
class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        #input size(batch,time-stamp,channels)
        #output size(batch, time-Stamp, head size)

        B,T,C = x.shape
        k = self.key(x)# (B,T,hs)
        q = self.query(x)# (B,T,hs)
        #compute attention scores
        wei = q @k.transpose(-2,-1)*k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))#(B,T,T)
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)#(B,T,hs)
        out = wei @ v #(B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()  # Fixed: Changed super.__init__() to super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,hs*nh)
        out = self.dropout(self.proj(out))  # (B,T,ne)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by communication"""

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head: the number of heads we will like 
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    """a GPT-style transformer language model"""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Fixed: Changed self.head to self.lm_head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Fixed: Changed means to mean
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Fixed: Changed arrange to arange
        x = tok_emb + pos_emb
        for block in self.blocks:  # Fixed: Added loop to apply blocks
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #id is(B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to last bloc_size tokens
            idx_cond = idx[:,-block_size:]
            #get prediction
            logits, loss = self(idx_cond)
            #sample
            logits = logits[:,-1,:] 
            #applying softmax
            probs = F.softmax(logits, dim=-1)
            #next distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next),dim = 1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
#print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters -1 :
        losses = estimate_loss()
        print(f"step{iter}: train loss{losses['train']:.4f}, val loss{losses['val']:.4f}")

        xb,yb = get_batch('train')

        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

#generate from model some text

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
