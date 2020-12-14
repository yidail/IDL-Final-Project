import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, DistilBertPreTrainedModel

from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.
    Taken <https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html#LockedDropout>

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5, batch_first=True):
        self.p = p
        super().__init__()
        self.batch_first = batch_first
    
    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        if self.batch_first == False:
            mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        else:
            mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'



class DistilBert_BiRNN(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        # self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)
        # self.lockdropout = LockedDropout(config.qa_dropout, batch_first=True)
        self.lockdropout = LockedDropout(0.2, batch_first=True)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=256, 
                            num_layers=1, bidirectional=True, batch_first=True)
        # self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        self.qa_outputs = nn.Linear(512, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        # print(hidden_states.shape)
        
        hidden_states = self.lockdropout(hidden_states)  # (bs, max_query_len, dim)
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class DistilBert_RNN(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        # self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)
        self.lockdropout = LockedDropout(config.qa_dropout, batch_first=True)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=256, 
                            num_layers=1, bidirectional=False, batch_first=True)
        # self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        self.qa_outputs = nn.Linear(256, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        # print(hidden_states.shape)
        
        hidden_states = self.lockdropout(hidden_states)  # (bs, max_query_len, dim)
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

class BERT(BertPreTrainedModel):
    '''
    ATTEMPT NO. 1
    '''
    def __init__(self, config):
        # super(BERT_CNN, self).__init__()
        super().__init__(config)
        self.num_labels = config.num_labels

        # layer1: BERT
        # pretrained_weights='bert-base-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.bert = DistilBertModel(config, add_pooling_layer=False)

        # layer2: LSTM - according to the Stanford paper: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15848021.pdf
        hidden_dim = 256 # user defined
        input_dim = config.hidden_size # warning! might need to change, 768 = sequence_output.shape[2]
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # layer3: CNN
        # WARNING! CNN dim's not yet working 
        # self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_cnn, kernel_size=3)

        # layer4: Linear Output Layer
        # use this if you want the cnn layer
        # self.qa_outputs = nn.Linear(hidden_cnn, vocab_size)
        # use this if omit the cnn layer
        self.qa_outputs = nn.Linear(2*hidden_dim, config.num_labels)
        self.init_weights()

    # WARNING! not sure if it would work for text in a batch 
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
      
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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
        # 1: encode text and send to bert layer
        # bert_output shape: (batch_size, sequence_length, hidden_size)
        # encoded_input = self.tokenizer(text, return_tensors='pt')
        # output = self.model(**encoded_input)
        # bert_output = output[0]

        # 2: lstm layer
        # input of shape (batch, seq_len, input_size)
        # output from lstm has size (batch, seq_len, 2*input_size)
        lstm_output, _ = self.lstm(sequence_output)

        # 3: cnn layer
        # cnn input (N, Cin, Hin, Win)
        # cnn input (N, Cout, Hout, Wout)
        # not sure abou the dimensions yet
        # cnn_input = lstm_output.unsqueeze(1) # did this according to the Stanford paper, need more investigation
        # cnn_output = cnn(cnn_input)

        # final linear layer
        # WARNING! I assume we output text using argmax (greedy search)
        # maybe we only need the logits to calculate F1 score
        logits = self.qa_outputs(lstm_output)
        # vocab_index = logits.argmax(dim=2)

        # found this piece of code from BERT database: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForTokenClassification
        # not sure why they added those
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # ultimately we want to return a loss
