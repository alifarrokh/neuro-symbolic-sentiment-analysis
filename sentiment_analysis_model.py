import numpy as np
from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values.
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - query (batch, q_len, d_model): tensor containing projection vector for decoder.
        - key (batch, k_len, d_model): tensor containing projection vector for encoder.
        - value (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - mask (-): tensor containing indices to be masked
    Returns: context, attn
        - context: tensor containing the context vector from attention mechanism.
        - attn: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.sqrt_dim = np.sqrt(self.hidden_size)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = nn.functional.softmax(score, -1)
        context = torch.bmm(attn, key)
        return context, attn


class HAN(nn.Module):
    """ HAN BLock """
    def __init__(self, config):
        super(HAN, self).__init__()
        self.hidden_size = config.hidden_size
        self.att = ScaledDotProductAttention(config)
        self.linear_observer = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_matrix = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(
        self,
        query: Tensor,  # (batch_size, 1, hidden_size)
        key: Tensor     # (batch_size, batch_max_len, hidden_size)
    ):
        context, att_weight = self.att(query,key)
        new_query_vec = self.dropout(self.layer_norm(self.activation(self.linear_observer(context))))
        new_key_matrix = self.dropout(self.layer_norm(self.activation(self.linear_matrix(key))))
        return new_query_vec, new_key_matrix, att_weight


class ClassificationHead(nn.Module):
    """ Classification head for RoBERTa """

    def __init__(self, config):
        super().__init__()
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.fnn1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fnn2 = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(classifier_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fnn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fnn2(x)
        return x


class RobertaForSentimentAnalysis(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        assert self.num_labels >= 2, "Invalid num_labels, this is a classification problem!"

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = ClassificationHead(config)

        self.with_han = config.with_han
        if self.with_han:
            self.query0 = nn.Parameter(torch.rand((1, config.hidden_size), dtype=torch.float32), requires_grad=True)
            self.han1 = HAN(config)
            self.han2 = HAN(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.with_han:
            batch_size = sequence_output.shape[0]
            query0 = self.query0.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, 1, hidden_size)
            query1, key1, attention_weight1 = self.han1(query0, sequence_output)
            query2, key2, attention_weight2 = self.han2(query1, key1)
            features = query2.squeeze(1)
        else:
            features = sequence_output[:, 0, :] # Take CLS (<s>) features
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
