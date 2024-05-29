import torch
import torch.nn as nn
from torch.nn import functional as F
from summary import Summary


# Low Rank Adaptation
# on nn.Linear layers
class LoRA_Parameterization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha):
        super().__init__()
        # initialize the A matrix with gaussian normal
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        # initialize the B matrix with 0
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)))
        # scale parameter allows changing rank without changing scale
        self.scale = alpha / rank
        
    def forward(self, original_weights):
        # move A and B matrix to device
        device = original_weights.device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)

        # W + (B*A)*scale
        return original_weights + torch.matmul(lora_B, lora_A).view(original_weights.shape) * self.scale

        

def linear_layer_parameterization(layer, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRA_Parameterization(features_in, features_out, rank=rank, alpha=lora_alpha)




# works like a linear layer but the weights are transposed.
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    
    
    
# LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, block_size, n_embd, dropout):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Conv1D(3 * n_embd, n_embd)
        # output projection
        self.c_proj = Conv1D(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


    
    
# linear layer followed by a non-linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc    = Conv1D(4 * n_embd, n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = Conv1D(n_embd, 4 * n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    
    
    

# transformer decoder block: communication followed by computation
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, block_size, n_heads, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=True)
        self.attn = MultiHeadAttention(n_heads, block_size, n_embd, dropout)
        self.ln_2 = LayerNorm(n_embd, bias=True)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
        
        
        
# main GPT2 class, decoder only Transformer
class GPT2(nn.Module):
    def __init__(self, n_embd, n_heads, n_layer, dropout=0.1, block_size=1024, vocab_size=50257):
        super().__init__()
        # parameters
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList(
                [DecoderBlock(n_embd, block_size, n_heads, dropout) for _ in range(n_layer)]
            ),
            ln_f = LayerNorm(n_embd, bias=True),
        ))
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
    
    # apply LoRA and freeze weights
    def apply_lora(self):
        # register lora for the decoder linear layers
        for decoder_block in self.transformer.h:
            linear0, linear1 = decoder_block.mlp.c_fc, decoder_block.mlp.c_proj
            nn.utils.parametrize.register_parametrization(
                linear0, 'weight', linear_layer_parameterization(linear0)
            )
            nn.utils.parametrize.register_parametrization(
                linear1, 'weight', linear_layer_parameterization(linear1)
            )
        # register lora for the final linear layer
        nn.utils.parametrize.register_parametrization(
            self.lm_head, 'weight', linear_layer_parameterization(self.lm_head)
        )

        # freeze original layers
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        
    # Return the number of parameters without position and token embeddings
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    


def main():
    device = 'cuda'
    
    # GPT2 config
    num_embed = 768
    num_heads = 12
    num_layers = 12
    
    
    # Creating model and testing output shapes 
    model = GPT2(num_embed, num_heads, num_layers)
    model = model.to(device)
    
    # apply LoRA
    model.apply_lora()
    
    Summary(model, (1024,))
    print('Total number of parameters', model.get_num_params())
    print('Number of trainable parameters', model.get_num_trainable_params())

if __name__ == "__main__": 
    main()