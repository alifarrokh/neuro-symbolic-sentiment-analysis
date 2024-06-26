import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
)


@dataclass
class LexicalSubstitutionInputFormatter:
    """
    Takes a sentence S and its candidate words C1..Ck and converts them into the format accepted by RobertaForLexicalSubstitution.
    Input Tokenization Format: <s>sentence</s>C1</s>C2</s>..Ck</s>
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, sample):
        tokenizer = self.tokenizer
        bos = tokenizer.get_vocab()[tokenizer.bos_token]
        sep = tokenizer.get_vocab()[tokenizer.sep_token]
        eos = tokenizer.get_vocab()[tokenizer.eos_token]

        sentence = sample['sentence']
        candidates = sample['candidates']
        label = sample['label']

        # Tokenize the sentencce
        left, target_token, right = re.split('<head>|</head>', sentence)
        if len(left) > 0 and left[-1] == ' ':
            left = left[:-1]
            target_token = ' ' + target_token
        tokenized_parts = tokenizer([left, target_token, right], add_special_tokens=False)['input_ids']

        # Tokenize the candidates
        label_index = candidates.index(label)
        tokenized_candidates = tokenizer(candidates, add_special_tokens=False)['input_ids']
        tokenized_candidates_lengths = [len(c)+1 for c in tokenized_candidates]

        # Concatenate the ids
        sentence_ids = [bos] + [t for part in tokenized_parts for t in part] + [eos]
        candidate_ids = [t for candidate in tokenized_candidates for t in (candidate + [sep])]

        # Verify the manually-created sequence
        assert tokenizer(re.sub('<head>|</head>', '', sentence))['input_ids'] == sentence_ids

        # A helper function to get the index of the last token in each candidate
        candidate_last_token_index = lambda c_index: len(sentence_ids) + sum(tokenized_candidates_lengths[:c_index]) + tokenized_candidates_lengths[c_index] - 2

        item = {
            'input_ids': sentence_ids + candidate_ids,
            'attention_mask': [1] * len(sentence_ids + candidate_ids),
            'target_indices': len(tokenized_parts[0]) + len(tokenized_parts[1]),
            'label': candidate_last_token_index(label_index),
            'candidate_indices': [candidate_last_token_index(i)  for i in range(len(candidates))],
        }
        return item


@dataclass
class LexicalSubstitutionDataCollator:
    tokenizer: PreTrainedTokenizerBase
    index_padding_value = -1

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        candidate_indices = [torch.tensor(item.pop('candidate_indices'), dtype=torch.long) for item in features]
        candidate_indices = pad_sequence(candidate_indices, batch_first=True, padding_value=self.index_padding_value)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=True,
            return_tensors='pt'
        )

        batch['labels'] = batch.pop('label')
        batch['candidate_indices'] = candidate_indices
        return batch


class RobertaForLexicalSubstitution(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,

        # Additional inputs for lexical substitution
        target_indices: torch.LongTensor,       # (batch_size)
        labels: torch.LongTensor,               # (batch_size)
        candidate_indices: torch.LongTensor,    # (batch_size, max_candidates)

        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract embeddings from RoBERTa
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
        embeddings = outputs[0]
        batch_size, sequence_length, _ = embeddings.shape

        # Convert embeddings to unit vectors
        embeddings = F.normalize(embeddings, dim=-1)

        # Create the mask
        mask = torch.ones((batch_size, sequence_length, sequence_length), dtype=torch.bool)
        for item_index in range(batch_size):
            target_index = target_indices[item_index]
            for candidate_index in candidate_indices[item_index, :]:
                if candidate_index > 0:
                    mask[item_index, target_index, candidate_index] = 0

        # Compute the similarity matrix
        sim_mat = torch.bmm(embeddings, embeddings.transpose(1, 2))
        logits = sim_mat = sim_mat.masked_fill(mask, - torch.inf)
        sim_mat = sim_mat.transpose(0, 2).transpose(0, 1) # (sequence_length, sequence_length, batch_size)

        # Create labels
        cross_entropy_labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
        for item_index, target_index in enumerate(target_indices):
            cross_entropy_labels[item_index, target_index] = labels[item_index]

        # Compute the loss
        loss = F.cross_entropy(sim_mat, cross_entropy_labels.T, reduction='sum') / batch_size

        # Compute the predictions (predicted indices)
        preds = torch.tensor([logits[i, target_indices[i], :].argmax() for i in range(batch_size)], dtype=torch.long)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
