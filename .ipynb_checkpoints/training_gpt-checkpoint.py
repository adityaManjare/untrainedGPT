import torch
import torch.nn as nn
from torch.nn import functional as F
import random, mmap
import pickle
device= 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size=64
batch_size=128
learning_rate= 3e-4
max_iters = 1000
eval_iters = 100
n_layer =4
n_embd =384
n_head =4
# head_size = n_embd // n_head
dropout =0.5

chars = ""
with open('openwebtext/vocab.txt','r',encoding='utf-8') as f:
    text = f.read()
    chars=sorted(set(text))
vocab_size = len(chars)


string_to_int={ch:i for i,ch in enumerate(chars)}
int_to_string={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[string_to_int[c] for c in s] 
decode=lambda l:''.join([int_to_string[i] for i in l ])


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "openwebtext/output_train.txt" if split == 'train' else "openwebtext/output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y= get_batch(split)
            logits,loss = model.forward(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    


class head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size , bias = False)
        self.value = nn.Linear(n_embd, head_size , bias = False)
        self.query = nn.Linear(n_embd, head_size , bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout= nn.Dropout(dropout)

    def forward(self , x):
        B,T,C = x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei = q@k.transpose(-2,-1)*k.shape[-1]**-0.5
        wei =wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))

        wei = F.softmax(wei,dim =-1)
        wei =self.dropout(wei)
        out = wei@v
        return out

        
class multi_head_attention(nn.Module):
    def __init__(self , n_head , head_size):
        super().__init__()
        self.heads = nn.ModuleList([head(head_size) for _ in range (n_head)])
        self.proj = nn.Linear(head_size*n_head , n_embd)
        self.droput = nn.Dropout(dropout)
    
    def forward(self ,x ):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # shape (B,T,n_embd) = (B,T,[f1,f1,f1,f1,f2,f2,f2,f2,f3,f3,f3,f3,f4,f4,f4,f4])
        out = self.droput(self.proj(out))
        return out

class feed_forward_layer(nn.Module):
    def __init__(self , n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        ) 

    def forward(self ,x ):
        return self.net(x)

class blocks(nn.Module):
    def __init__(self , n_embd ,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = multi_head_attention(n_head , head_size)
        self.feed_forward = feed_forward_layer(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
    def forward(self , x ):
        y= self.self_attention(x)
        x=self.layer_norm_1(x+y)
        y=self.feed_forward(x)
        x=self.layer_norm_2(x+y)
        return x
                                                                                                                                                                                          
class GPTLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__() 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[blocks(n_embd,n_head=n_head)for _ in range(n_layer)]) # block == decoder

        self.layer_norm_final= nn.LayerNorm(n_embd) # final layer norm
        self.language_modelling_head= nn.Linear(n_embd, vocab_size) # to convert hidden_vector = [0.3, -1.2, 0.5, ..., 0.9]  ← shape: [n_embd] to [2.1, -0.9, 3.5, ..., -1.2] ← shape: [vocab_size]


        self.apply(self.init_weights)
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None):
        # logits = self.token_embedding_table(idx)
        tok_emb = self.token_embedding_table(idx) #(idx in bigram was of shape(B,) but here its (B,T)(similar for targets)) aslo thr output is of shape(B,T,C)
        k,input_block_size=idx.shape
        pos_emb = self.position_embedding_table(torch.arange(input_block_size,device= device)) # shape (T,C)
        # x = vector input in tansformer blocks(decoder specifically)
        x =tok_emb + pos_emb
        x= self.blocks(x)
        # the below two step is whole after multiple (all) decoder blocks
        x =self.layer_norm_final(x)
        logits = self.language_modelling_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets) 
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# the whole model is stored temporarly here 
model = GPTLanguageModel(vocab_size)
print('loading the saved parameters ')
with open('model_01.pkl','rb') as f:
    model=pickle.load(f)
print('loaded successfully')
m = model.to(device)



# the two fixes first ensure that pos_emb should not contain block_size it should be rather having input block size for that just unpack idx
# also make sure that input block size must not exceed more than the block_size for that make a new variable idx_cond and ensure idx_cond[-2] <= block_size


optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for iter in range (max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step = : {iter} ,loss {losses}')
    xb , yb = get_batch('train')
    logits,loss = model.forward(xb,yb)
    optimizer.zero_grad(set_to_none = True) # hmm this step is just cleaning the mess for backpropagation
    loss.backward() # back propagation 
    optimizer.step() # optimizing that is tuning in the opposite direction of gradient calculated by back prpagation
print(loss.item())

with open('model_01.pkl' ,'wb') as f:
    pickle.dump(model,f)
print('model saved')