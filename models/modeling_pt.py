import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    rotate_half,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput
from transformers.utils import (
    logging,
    ModelOutput
)
from transformers.modeling_utils import PreTrainedModel

from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .configuration_pt import PtConfig

logger = logging.get_logger(__name__)


class RopeApplier:
    def __init__(self, cos, sin, position_ids, unsqueeze_dim=1) -> None:
        """Applies Rotary Position Embedding to the query, key and value tensors.

        Args:
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query, key and value tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        """
        self.cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        self.sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    def apply(self, qkv):
        return (qkv * self.cos) + (rotate_half(qkv) * self.sin)
    
    def apply_o(self, o):
        return (o * self.cos) - (rotate_half(o) * self.sin)


class SquaredSoftmax(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.pow(2)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class AbsNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = F.relu(hidden_states)
        hidden_states = F.normalize(hidden_states, p=1, dim=self.dim, eps=self.eps)
        return hidden_states.to(input_dtype)


class Softmax(nn.Softmax):
    # This is a workaround to allow passing the eps
    def __init__(self, dim=-1, eps=None):
        super().__init__(dim=dim)


POTENTIAL2ACT = {
    "exp": Softmax,
    "abs": AbsNormalization,
    "square": SquaredSoftmax,
}


class PtHeadSelection(nn.Module):
    """Multi-channel head selection from 'Probabilistic Transformer' paper"""
    
    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.num_channels = config.num_channels
        self.ternary_rank = config.ternary_rank
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.ternary_factor_u = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.ternary_factor_v = nn.Parameter(torch.empty(self.num_channels * self.ternary_rank, self.dim_z))
        self.dropout = nn.Dropout(config.dropout_prob_h)
        self._init_ternary()
        self._init_rope()
    
    def _init_ternary(self):
        nn.init.normal_(self.ternary_factor_u, mean=0.0, std=self.config.ternary_initializer_range)
        nn.init.normal_(self.ternary_factor_v, mean=0.0, std=self.config.ternary_initializer_range)

    def _init_rope(self):
        """we follow rope in llama"""
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.ternary_rank,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.ternary_rank,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.ternary_rank,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_heads: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, seq_len, _ = qz.size()

        qz_u = nn.functional.linear(qz, self.ternary_factor_u) * self.config.ternary_factor_scaling
        qz_v = nn.functional.linear(qz, self.ternary_factor_v) * self.config.ternary_factor_scaling

        qz_u = qz_u.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)
        qz_v = qz_v.view(bsz, seq_len, self.num_channels, self.ternary_rank).transpose(1, 2)

        cos, sin = self.rotary_emb(qz_v, seq_len=seq_len)
        rope_applier = RopeApplier(cos, sin, position_ids)
        qz_uo = rope_applier.apply_o(qz_u)
        qz_u = rope_applier.apply(qz_u)
        qz_v = rope_applier.apply(qz_v)

        message_F = torch.matmul(qz_u, qz_v.transpose(2, 3))

        if message_F.size() != (bsz, self.num_channels, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_channels, seq_len, seq_len)}, but is"
                f" {message_F.size()}"
            )

        if dependency_mask is not None:
            if dependency_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {dependency_mask.size()}"
                )
            message_F = message_F + dependency_mask # need mask diag

        # upcast attention to fp32
        qh = nn.functional.softmax(message_F / self.config.regularize_h, dim=-1, dtype=torch.float32).to(qz_u.dtype)

        qh_v1 = torch.matmul(qh, qz_v)
        qh_v2 = torch.matmul(qh.transpose(2, 3), qz_uo)

        # apply rotary position embedding to the output
        qh_v1 = rope_applier.apply_o(qh_v1)
        qh_v2 = rope_applier.apply(qh_v2)

        if qh_v1.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v1` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v1.size()}"
            )
        if qh_v2.size() != (bsz, self.num_channels, seq_len, self.ternary_rank):
            raise ValueError(
                f"`qh_v2` should be of size {(bsz, self.num_channels, seq_len, self.ternary_rank)}, but is"
                f" {qh_v2.size()}"
            )

        qh_v1 = qh_v1.transpose(1, 2).contiguous()
        qh_v2 = qh_v2.transpose(1, 2).contiguous()

        qh_v1 = qh_v1.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)
        qh_v2 = qh_v2.reshape(bsz, seq_len, self.num_channels * self.ternary_rank)

        message_G = (torch.matmul(qh_v1, self.ternary_factor_u) + torch.matmul(qh_v2, self.ternary_factor_v)) * self.config.ternary_factor_scaling

        if not output_heads:
            qh = None

        return message_G, qh


class PtTopicModeling(nn.Module):
    """Topic modeling w/ global nodes."""
    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.dim_g = config.dim_g
        self.binary_factor = nn.Parameter(torch.empty(self.dim_g, self.dim_z))
        self.act = POTENTIAL2ACT[config.potential_func_g](dim=-1, eps=config.potential_eps)
        
        self._init_binary()
        
    def _init_binary(self):
        nn.init.normal_(self.binary_factor, mean=0.0, std=self.config.binary_initializer_range)

    def forward(self, qz: torch.Tensor):
        qg = nn.functional.linear(qz, self.binary_factor) * self.config.binary_factor_scaling
        qg = self.act(qg / self.config.regularize_g)
        message_G = qg @ self.binary_factor * self.config.binary_factor_scaling
        return message_G

class PtEncoderIterator(nn.Module):
    def __init__(self, config: PtConfig):
        super().__init__()
        self.config = config
        self.dim_z = config.dim_z
        self.head_selection = (
            PtHeadSelection(config=config)
            # if not getattr(config, "_flash_attn_2_enabled", False)
            # else PtFlashHeadSelection2(config=config)
        )
        self.topic_modeling = PtTopicModeling(config)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)
    
    def forward(
        self,
        unary_potentials: torch.Tensor,
        qz: torch.Tensor,
        dependency_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_heads: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            qz (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            dependency_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_heads (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        old_qz = qz

        qz = self.norm(qz)

        # head selection
        m1, qh = self.head_selection(
            qz=qz,
            dependency_mask=dependency_mask,
            position_ids=position_ids,
            output_heads=output_heads,
        )

        # topic modeling
        m2 = self.topic_modeling(qz)

        # unary potentials
        qz = (m1 + m2 + unary_potentials) / self.config.regularize_z

        # damping
        qz = (qz + old_qz) * .5

        outputs = (qz,)

        if output_heads:
            outputs += (qh,)

        return outputs
    

@dataclass
class PtModelOutput(ModelOutput):
    """qz is un-normalized logits, qh is the normalized distribution over heads."""
    last_qz: torch.FloatTensor = None
    all_qzs: Optional[Tuple[torch.FloatTensor]] = None
    all_qhs: Optional[Tuple[torch.FloatTensor]] = None
    

class PtPreTrainedModel(PreTrainedModel):
    config_class = PtConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PtEncoderIterator"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PtModel(PtPreTrainedModel):
    config_class = PtConfig
    def __init__(self, config: PtConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.unary_factors = nn.Embedding(config.vocab_size, config.dim_z, self.padding_idx)
        self.iterator = PtEncoderIterator(config)
        self.norm = POTENTIAL2ACT[config.potential_func_z](dim=-1, eps=config.potential_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.unary_factors

    def set_input_embeddings(self, value):
        self.unary_factors = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unary_potentials: Optional[torch.FloatTensor] = None,
        output_heads: Optional[bool] = None,
        output_qzs: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PtModelOutput]:
        output_heads = output_heads if output_heads is not None else self.config.output_heads
        output_qzs = (
            output_qzs if output_qzs is not None else self.config.output_qzs
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and unary_potentials
        if input_ids is not None and unary_potentials is not None:
            raise ValueError("You cannot specify both input_ids and unary_potentials at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif unary_potentials is not None:
            batch_size, seq_length = unary_potentials.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or unary_potentials")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else unary_potentials.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if unary_potentials is None:
            unary_potentials = self.unary_factors(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            raise ValueError("Flash attention 2 is not supported in PtModel when masking diagonals")
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attn_mask_converter = AttentionMaskConverter(is_causal=False)
            attention_mask = attn_mask_converter.to_4d(
                attention_mask, seq_length, seq_length, dtype=unary_potentials.dtype
            )
            
            # mask diagonals
            diag_mask = torch.eye(seq_length, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.masked_fill(diag_mask.to(torch.bool), torch.finfo(attention_mask.dtype).min)

        # embed positions
        qz = unary_potentials

        # decoder layers
        all_qzs = () if output_qzs else None
        all_qhs = () if output_heads else None

        for idx in range(self.config.num_iterations):
            if output_qzs:
                all_qzs += (qz,)

            iter_outputs = self.iterator(
                unary_potentials,
                qz,
                dependency_mask=attention_mask,
                position_ids=position_ids,
                output_heads=output_heads,
            )

            qz = iter_outputs[0]

            if output_heads:
                all_qhs += (iter_outputs[1],)

        # add hidden states from the last decoder layer
        if output_qzs:
            all_qzs += (qz,)

        qz = self.norm(qz)

        if not return_dict:
            return tuple(v for v in [qz, all_qzs, all_qhs] if v is not None)
        return PtModelOutput(
            last_qz=qz,
            all_qzs=all_qzs,
            all_qhs=all_qhs
        )


class PtForMaskedLM(PtPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.model = PtModel(config)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dependency_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            unary_potentials=inputs_embeds,
            output_heads=output_attentions,
            output_qzs=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] * self.config.classifier_amplifier
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.all_qzs,
            attentions=outputs.all_qhs,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}