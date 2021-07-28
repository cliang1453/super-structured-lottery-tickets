# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Head Masked BERT model. """

import math
import torch
from torch import nn
from transformers.modeling_bert import BertModel, BertEncoder, BertLayer, BertOutput, BertAttention, BertSelfAttention

class MaskHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        output = input * mask.view(1, -1, 1, 1)
        ctx.save_for_backward(output, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, mask = ctx.saved_tensors
        grad_input = grad_output * mask.view(1, -1, 1, 1)
        dot = torch.einsum("bhli,bhli->bhl", [grad_output, output])
        grad_mask = dot.abs().sum(-1).sum(0)
        return grad_input, grad_mask

class MaskFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        output = input * mask
        ctx.save_for_backward(output, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, mask = ctx.saved_tensors
        grad_input = grad_output * mask
        dot = torch.einsum("bhi,bhi->bh", [grad_output, output])
        grad_mask = dot.abs().sum(-1).sum(0, keepdim=True)
        return grad_input, grad_mask

class GradBasedMaskedBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.heads_to_mask = []
        self.heads_mask = nn.Parameter(torch.ones(self.num_attention_heads))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Mask heads
        # attention_probs has shape bsz x n_heads x N x N
        context_layer = torch.matmul(attention_probs, value_layer)
        if self.training:
            context_layer = context_layer * self.heads_mask.view(1,-1,1,1).detach()
        else:
            context_layer = MaskHead.apply(context_layer, self.heads_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

    def mask_heads(self, heads_to_mask):
        self.heads_to_mask = list(heads_to_mask)
        for head_idx in self.heads_to_mask:
            self.heads_mask.data[head_idx] = 0

    def clear_heads(self):
        self.heads_to_mask = []
        self.heads_mask.data.fill_(1)

class GradBasedMaskedBertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.ffn_mask = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states, input_tensor):
        if hidden_states is None:
            hidden_states = self.LayerNorm(input_tensor)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if self.training:
                hidden_states = hidden_states * self.ffn_mask.detach()
            else:
                hidden_states = MaskFFN.apply(hidden_states, self.ffn_mask)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class GradBasedMaskedBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = GradBasedMaskedBertSelfAttention(config)

class GradBasedMaskedBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = GradBasedMaskedBertAttention(config)
        self.output = GradBasedMaskedBertOutput(config)
        self.mask_ffn = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        if self.mask_ffn:
            intermediate_output = None
        else:
            intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs

class GradBasedMaskedBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([GradBasedMaskedBertLayer(config) for _ in range(config.num_hidden_layers)])

    def mask_ffn(self, ffn_to_mask):
        for layer_idx in ffn_to_mask:
            self.layer[layer_idx].mask_ffn = True

    def clear_ffn(self):
        for layer_idx in range(len(self.layer)):
            self.layer[layer_idx].mask_ffn = False

class GradBasedMaskedBertModel(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = GradBasedMaskedBertEncoder(config)

    def mask_heads(self, heads_to_mask):
        for layer, heads in heads_to_mask.items():
            self.encoder.layer[layer].attention.self.mask_heads(heads)

    def clear_heads(self):
        for layer in range(len(self.encoder.layer)):
            self.encoder.layer[layer].attention.self.clear_heads()

    def mask_ffn(self, ffn_to_mask):
        self.encoder.mask_ffn(ffn_to_mask) # a list of layer idx from 0-12

    def clear_ffn(self):
        self.encoder.clear_ffn()
