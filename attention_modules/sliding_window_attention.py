import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, config, window_size=3):
        super(SlidingWindowAttention, self).__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // self.num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.window_size = window_size
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        device = hidden_states.device 
        batch_size, seq_length, _ = hidden_states.size()

        q = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(-2, -1)

        matmul_qk = torch.matmul(q, k)
        scaled_attention_logits = matmul_qk / self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze().unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scaled_attention_logits += attention_mask

        arange = torch.arange(seq_length, device=device)
        mask = (arange[:, None] - arange[None, :]).abs() > self.window_size
        window_mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
        window_mask[mask] = 0 
        window_mask = window_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        scaled_attention_logits += window_mask

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

        if head_mask is not None:
            attention_weights *= head_mask

        output = torch.matmul(attention_weights, v.transpose(1, 2))
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        outputs = (output,)
        if output_attentions:
            outputs += (attention_weights,)

        if use_cache:
            outputs += ((k, v),)

        return outputs
