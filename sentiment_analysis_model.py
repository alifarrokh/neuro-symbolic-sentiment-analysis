from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn, Tensor
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.sqrt_dim = np.sqrt(self.hidden_size)

    def forward(
        self,
        query: Tensor, # (batch_size, 1, hidden_size)
        key: Tensor, # (batch_size, batch_max_len, hidden_size)
        attention_mask: Tensor # (batch_size, 1, batch_max_len)
    ) -> tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        score = score * attention_mask
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
        query: Tensor, # (batch_size, 1, hidden_size)
        key: Tensor, # (batch_size, batch_max_len, hidden_size)
        attention_mask: Tensor, # (batch_size, 1, batch_max_len)
    ):
        context, att_weight = self.att(query, key, attention_mask)
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


@dataclass
class SentimentAnalysisWithAttentionOutput:
    """ A wrapper over outputs of a sentiment analysis model with attention """
    logits: torch.FloatTensor # (batch_size, num_labels)
    embeddings: torch.FloatTensor # (batch_size, sequence_length, hidden_size)
    attention_weights: torch.FloatTensor # (batch_size, sequence_length)


class RobertaForSentimentAnalysis(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Validate config params
        assert config.num_labels >= 2, "Invalid num_labels, this is a classification problem!"
        assert config.attention in ['none', 'simple', 'han']

        self.config = config
        self.num_labels = config.num_labels
        self.attention = config.attention

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = ClassificationHead(config)

        if self.attention == 'simple':
            pass
        elif self.attention == 'han':
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

        # Custom args for sentiment analysis
        return_analysis_info: bool = False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Get RoBERTa embeddings
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

        # Attention
        if self.attention == 'simple':
            pass
        elif self.attention == 'han':
            batch_size = sequence_output.shape[0]
            query0 = self.query0.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, 1, hidden_size)
            query1, key1, attention_weights1 = self.han1(query0, sequence_output, attention_mask.unsqueeze(1))
            query2, key2, attention_weights2 = self.han2(query1, key1, attention_mask.unsqueeze(1))
            features = query2.squeeze(1)
        else:
            features = sequence_output[:, 0, :] # Take CLS (<s>) features

        # Classification
        logits = self.classifier(features)

        # Sentiment analysis with attention output
        if self.attention != 'none' and return_analysis_info:
            attention_weights = None
            if self.attention == 'han':
                attention_weights = attention_weights2.squeeze(1)
            elif self.attention == 'simple':
                pass

            return SentimentAnalysisWithAttentionOutput(
                logits=logits,
                embeddings=sequence_output,
                attention_weights=attention_weights
            )

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
