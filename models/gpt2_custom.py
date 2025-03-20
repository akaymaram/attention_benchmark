import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class GPT2CustomAttentionModel(GPT2LMHeadModel):
    def __init__(self, config, attention_module_cls):
        super(GPT2CustomAttentionModel, self).__init__(config)
        for i, layer in enumerate(self.transformer.h):
            layer.attn = attention_module_cls(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        #print("use the orignial GPT2")
        outputs = super(GPT2CustomAttentionModel, self).forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the logits and labels
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Ensure that the number of logits matches the number of labels
            shift_logits = shift_logits[:shift_labels.size(0)]

            # Calculate the loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

            outputs = (loss,) + outputs[1:]
        return outputs
