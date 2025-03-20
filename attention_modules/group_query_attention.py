import torch
import torch.nn as nn
import logging

class GroupQueryAttention(nn.Module):
    def __init__(self, config, num_groups=8):
        super(GroupQueryAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))
        self.num_groups = num_groups
        group_dim = config.n_embd // num_groups
        self.query = nn.Linear(config.n_embd, group_dim * num_groups)
        self.key = nn.Linear(config.n_embd, group_dim * num_groups)
        self.value = nn.Linear(config.n_embd, group_dim * num_groups)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        batch_size, seq_length, dim = hidden_states.shape
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape tensors to include group dimension
        query = query.view(batch_size, seq_length, self.num_groups, -1)
        key = key.view(batch_size, seq_length, self.num_groups, -1)
        value = value.view(batch_size, seq_length, self.num_groups, -1)
        
        logging.info(f"Query shape: {query.shape}")
        logging.info(f"Key shape: {key.shape}")
        logging.info(f"Value shape: {value.shape}")

        # Compute attention scores using einsum
        attention_scores = torch.einsum('bngh,bmgh->bnmg', query, key) / self.scale
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1).unsqueeze(-1).expand(-1, -1, -1, self.num_groups)
            attention_scores += attention_mask #* -10000.0

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.einsum('bnmg,bmgh->bngh', attention_probs, value).reshape(batch_size, seq_length, -1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs if not use_cache else outputs + (present,)
