#char level transformer
#imports
import re
import torch
import torch.nn as nn
from torch.nn import functional as f

#----------------------------------
#hyper-parameters
batch_size = 32
block_size = 8
max_iters = 5000 #increase number of iter bc the learning rate is lower
eval_interval = 500
learning_r = 1e-3   #self attention can not tolerate very high learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

#----------------------------------
torch.manual_seed(42)

#open file and read data
with open("poems.txt", "r" ,encoding="utf-8") as file:
    text = file.read()

#-----------------------------------
#cleans the text from line breaks
#pattern = r"\n\s*\n"
#text = re.sub(pattern, "\n", text)

#here are all the unique char that occur in the file
chars = sorted(list(set(text)))
vocab_size = len(chars)
str_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_str = {i:ch for i, ch in enumerate(chars)}
#encode string into int <=> decode string into the originl string intput
encode = lambda s : [str_to_int[c] for c in s]
decode = lambda str_out : ''.join([int_to_str[c] for c in str_out])


#---------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)

#split the data into training and validation
n = int(0.9*len(data))  #90% is training 10% is val
train_data = data[:n]
val_data = data[n:]

#---------------------------------------
def get_batch(split):
    data = train_data if split == "train" else val_data
    #random offsets between zero and the len data - block size , it is one dim tensor
    ix = torch.randint(len(data) - block_size, (batch_size, ) )
    #stack them up to become a row in a 4x8 tesnor
    #get me a random offset and read until the block size for every offset in ix
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1 ]for i in ix])

    return x, y

#----------------------------------------
#avaraging the loss over multiple batches 

#we are not calling .backward() more efficent in terms of memory use
@torch.no_grad()
def estimate_loss():
    out = {}
    #set the model to evaluation mode 
    model.eval()
    #get the loss for the split 
    for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                    x, y = get_batch(split)
                    logits, loss = model(x, y)
                    losses[k] = loss.item()
            #avg for both splits   
            out[split] = losses.mean()
    #set the model to training mode 
    model.train()
    return out 

#----------------------------------------
class Head(nn.Module):
    "one head of self attention"

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  #B = n Batch , T = block_size , C = n_embd 
        k = self.key(x)        #(B,T,16)
        q = self.query(x)      #(B,T,16)

        #compute affinaties 
        wei = torch.matmul(q, k.transpose(-2,-1)) * C**-0.5 # (B,T,16) @ (B, 16,T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #in decoder block we need it, in encoder block we do not need it such in sentiment analyss 
        wei = f.softmax(wei, dim=-1)
        #weight agg on the value 
        v = self.value(x) #v is the thing that get aggrgated for the purpouses of thing single head
        out = torch.matmul(wei,v)
        return out 
    
#----------------------------------------
#help with communication 
class MultiHeadAttention(nn.Module):
    """parallel self attention heads"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

#per token indepentently, think the data they gathered from communicating through attention
#----------------------------------------
class FeedForward(nn.Module):
    """single linear layer followed by non linear"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            )
    def forward(self, x):
        return self.net(x)
#----------------------------------------
class Block(nn.Module):
    """transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size =  n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)  #4 heads of 8 dim self attention --group conv
        self.ffw = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)              #apply one head of self attention (B,T,C)
        x = self.ffw(x)              #(B,T,C)
        return(x)    

#----------------------------------------

#bigram model --class 
#output loss and logit and can generate 

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #nn.Embedding layer map from words to vectors
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #32 dim embd  
    #encoding not just idx but also by position 
    self.position_embedding_table = nn.Embedding(block_size, n_embd) #each position from zero to block size -1 will get its own emadding 
    self.blocks = nn.Sequential(
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):

    B, T = idx.shape 
    token_embd = self.token_embedding_table(idx) #(B,T,C) #logits = self.token_embedding_table(idx) it is not going to give us token embading directly
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
    x = token_embd +  pos_embd                    #(B,T,C)  elements of the position_embedding_table tensor are repeated along the B dimension so that they have the same shape as the elements of the token_embedding_table tensor. 
    x = self.blocks(x)                            #(B,T,C)
    logits = self.lm_head(x)                      #(B,T,vocab_size)

    #to measure the quality of the prediction we are using negative log liklihood = cross entropy
    #pytorch functional cross_entropy expect B,C,T so we are reshaping our logits

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss =  f.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    #max_new_tokens the max number of new tokens that will be generated
    #idx is B,T array
    for _ in range(max_new_tokens):
      idx_croped = idx[:, -block_size:]
      #get the predictions
      logits, loss = self(idx_croped) #methods to get the predictions for the next token class MyModel
      #only the last time step
      logits = logits[:,-1,:]  # (B, C) [:-1:] crops the last time step from the logits tensor
      #get the prob using softmax
      probs = f.softmax(logits, dim=-1) #(B,C)
      #sample from the prob distribution
      idx_next = torch.multinomial(probs, num_samples=1) #(B, 1) give us one sample
      #append sampled index to the running seq
      idx = torch.cat((idx_croped, idx_next), dim=1)   #(B, T+1)

    return idx

model = BigramLanguageModel()
m = model.to(device)

#-------------------------------------------
#create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_r)

#training loop

for iter in range(max_iters): #number of steps is 100 for better results increcse the number

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch from data
    xb, yb = get_batch('train')

    #evl the loss
    logits, loss = m(xb, yb)
    #set the gradients of all the param to zero
    #setting the grad to None instead of zero is useful with Adam optimizer which can handle None
    optimizer.zero_grad(set_to_none=True)
    #backpropgate the loss through the model
    loss.backward()
    #update the param of the model
    optimizer.step()

#generating form the contex 
contex = torch.zeros((1,1), dtype=torch.long, device=device)

print(decode(m.generate(idx = contex, max_new_tokens=1000)[0].tolist()))

