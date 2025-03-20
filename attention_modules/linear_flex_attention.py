import torch
import torch.nn as nn

class LinearFlexAttention(nn.Module):
    def __init__(self, config):
        super(LinearFlexAttention, self).__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.scale = torch.sqrt(torch.tensor(config.n_embd, dtype=torch.float32))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        #print(" value:", value)  # Print the reshaped attention_mask values

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        attention_probs = torch.sigmoid(attention_scores)
        #print("Reshaped attention_probs before:", attention_probs)  # Print the reshaped attention_mask values


        if attention_mask is not None:
            #attention_mask = attention_mask[:, None, None, :]
            #print("Reshaped attention_mask:", attention_mask)  # Print the reshaped attention_mask values
            attention_mask = attention_mask.squeeze(1)
            attention_probs = torch.sigmoid(attention_probs * attention_mask)
            #print("Reshaped attention_probs:", attention_probs)  # Print the reshaped attention_mask values



        context_layer = torch.matmul(attention_probs, value)
        #print(" context_layer:", context_layer)  # Print the reshaped attention_mask values


        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)


        return outputs if not use_cache else outputs + (present,)
