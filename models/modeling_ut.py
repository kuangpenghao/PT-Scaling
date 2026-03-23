import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from .configuration_ut import UtConfig

class UtEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            raise ValueError("You have to specify input_ids")

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UtSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, use_flash_attn=False, cu_seqlens=None, max_seqlen=None):
        dropout_p = self.dropout_prob if self.training else 0.0
        
        if use_flash_attn:
            query = self.query(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            key = self.key(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            value = self.value(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            
            context_layer = flash_attn_varlen_func(
                query, key, value, 
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, 
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, 
                dropout_p=dropout_p
            )
            
            context_layer = context_layer.view(-1, self.all_head_size)
        else:
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            context_layer = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=False
            )

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class UtSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Pre-LN 结构的改造：恒等映射直接相加，不在这里做 LN
        hidden_states = hidden_states + input_tensor
        return hidden_states


class UtAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = UtSelfAttention(config)
        self.output = UtSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, use_flash_attn=False, cu_seqlens=None, max_seqlen=None):
        # Pre-LN：先做 LayerNorm，再送入 Attention (利用 self.output 本身带有的层权重)
        normed_hidden_states = self.output.LayerNorm(hidden_states)
        self_output = self.self(
            normed_hidden_states, 
            attention_mask=attention_mask,
            use_flash_attn=use_flash_attn,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class UtIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class UtOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Pre-LN 结构的改造：恒等映射直接相加
        hidden_states = hidden_states + input_tensor
        return hidden_states


class UtLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = UtAttention(config)
        self.intermediate = UtIntermediate(config)
        self.output = UtOutput(config)

    def forward(self, hidden_states, attention_mask=None, use_flash_attn=False, cu_seqlens=None, max_seqlen=None):
        attention_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            use_flash_attn=use_flash_attn,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        # Pre-LN：先做 LayerNorm，再送入 FFN Intermediate (利用 self.output 本身带有的层权重)
        normed_attention_output = self.output.LayerNorm(attention_output)
        intermediate_output = self.intermediate(normed_attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class UtEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = UtLayer(config)
        self.timestep_embeddings = nn.Embedding(config.num_hidden_layers, config.hidden_size)
        
        # Adaptive Computation Time (ACT) components
        self.act_linear = nn.Linear(config.hidden_size, 1)
        # 修正初始化：初始化偏置为强负值，防止 “懒惰网络陷阱”
        # 使用 config 中的配置或默认为 -3.0
        act_bias_init = getattr(self.config, "act_bias_init", -3.0)
        self.act_linear.bias.data.fill_(act_bias_init)

        # 增加最后统一的全局 LayerNorm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_hidden_states=False, original_attention_mask=None, use_flash_attn=False):
        all_hidden_states = () if output_hidden_states else None

        cu_seqlens = None
        max_seqlen = None
        indices = None
        batch_size, seq_len, _ = hidden_states.shape

        if use_flash_attn and original_attention_mask is not None:
            # Unpad inputs
            hidden_states, indices, cu_seqlens, max_seqlen = unpad_input(hidden_states, original_attention_mask)[:4]

        if output_hidden_states:
            if use_flash_attn:
                padded_states = pad_input(hidden_states, indices, batch_size, seq_len)
                all_hidden_states = all_hidden_states + (padded_states,)
            else:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # ACT Initializations
        state_shape = hidden_states.shape[:-1]
        halting_probability = torch.zeros(*state_shape, 1, dtype=torch.float32, device=hidden_states.device)
        accumulated_states = torch.zeros_like(hidden_states)
        n_updates = torch.zeros(*state_shape, 1, dtype=torch.float32, device=hidden_states.device)
        still_running_mask = torch.ones(*state_shape, 1, dtype=torch.bool, device=hidden_states.device)

        # Remainders components initializations
        remainders = torch.zeros(*state_shape, 1, dtype=torch.float32, device=hidden_states.device)

        for i in range(self.config.num_hidden_layers):
            # 引入 2D 坐标时间步：这里应当传入包含位置和深度的主向量，或者直接结合 2D 编码
            # 但经典 UT 中为了兼容性，通常是 time_step Embedding 相加
            step_embeddings = self.timestep_embeddings(torch.tensor(i, device=hidden_states.device))
            current_hidden_states = hidden_states + step_embeddings

            # Calculate ACT halt probability based on the layer input
            h_i = torch.sigmoid(self.act_linear(current_hidden_states))

            # Apply shared layer
            layer_output = self.layer(
                current_hidden_states, 
                attention_mask=attention_mask,
                use_flash_attn=use_flash_attn,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

            # Determine ACT probabilities
            # 修正 UT / ACT 的严谨计算: 只有达到 1.0 的那一刻计算 remainder，提前的步长用原始的 h_i
            new_halted_mask = (halting_probability + h_i >= (1.0 - getattr(self.config, "act_epsilon", 0.01))) & still_running_mask
            
            p_i = torch.where(new_halted_mask, 1.0 - halting_probability, h_i)
            p_i = torch.where(still_running_mask, p_i, torch.zeros_like(p_i))

            # Accumulate the weighted states: ALways scale the state by p_i
            accumulated_states = accumulated_states + p_i * layer_output
            halting_probability = halting_probability + p_i
            n_updates = n_updates + still_running_mask.float()

            # Update running mask
            still_running_mask = halting_probability < (1.0 - getattr(self.config, "act_epsilon", 0.01))

            # Update hidden_states for the NEXT iteration
            # Halted tokens retain their current state; running tokens update to layer_output
            hidden_states = torch.where(still_running_mask, layer_output, hidden_states)

            if output_hidden_states:
                if use_flash_attn:
                    padded_states = pad_input(accumulated_states, indices, batch_size, seq_len)
                    all_hidden_states = all_hidden_states + (padded_states,)
                else:
                    all_hidden_states = all_hidden_states + (accumulated_states,)

            # Dynamic Halting: if all tokens in the batch have halted, stop computation
            if not still_running_mask.any():
                break

        # If forced to stop at max_steps, ensure any remaining probability is assigned to the last state
        # 根据原始ACT论文: 强行截断的最后一步，补足到 1.0 的概率，强制赋予给最后这一步的输出
        still_running_at_end = halting_probability < (1.0 - getattr(self.config, "act_epsilon", 0.01))
        remainder = torch.where(still_running_at_end, 1.0 - halting_probability, torch.zeros_like(halting_probability))
        accumulated_states = accumulated_states + remainder * layer_output
        halting_probability = halting_probability + remainder

        # Update final hidden states
        hidden_states = accumulated_states

        # Calculate Ponder Cost for the current sequence: N + remainder
        # In our implementation `n_updates` holds the number of steps each token took.
        # But ACT paper specifies cost = step_taken + remainder for the halted step.
        ponder_cost = n_updates + remainder

        # Pre-LN 结构的最后必须过一次全局的 LayerNorm 输出，防止数值爆炸
        hidden_states = self.final_layer_norm(hidden_states)

        if use_flash_attn and original_attention_mask is not None:
            hidden_states = pad_input(hidden_states, indices, batch_size, seq_len)
            ponder_cost = pad_input(ponder_cost, indices, batch_size, seq_len)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, ponder_cost] if v is not None)


class UtPreTrainedModel(PreTrainedModel):
    config_class = UtConfig
    base_model_prefix = "ut"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class UtModel(UtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = UtEmbeddings(config)
        self.encoder = UtEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Determine if we should use flash attention (if fp16/bf16 and available)
        use_flash_attn = HAS_FLASH_ATTN and self.dtype in (torch.float16, torch.bfloat16)
        
        if use_flash_attn:
            extended_attention_mask = attention_mask # Pass the pure 0/1 mask
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            original_attention_mask=attention_mask,
            use_flash_attn=use_flash_attn
        )

        sequence_output = encoder_outputs[0]
        # Ponder cost is returned as the 3rd element
        ponder_cost = encoder_outputs[2] if len(encoder_outputs) > 2 else 0.0

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # add a custom field to the output object or we can just unpack it
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            cross_attentions=ponder_cost, # hijack this field to return ponder_cost for MaskedLM to intercept
        )


class UtForMaskedLM(UtPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.ut = UtModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ut(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = outputs.last_hidden_state
            ponder_cost = outputs.cross_attentions
        else:
            sequence_output = outputs[0]
            ponder_cost = outputs[2] if len(outputs) > 2 else 0.0

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
            # Apply Ponder Penalty
            ponder_weight = getattr(self.config, "ponder_weight", 0.001)
            if ponder_weight > 0 and ponder_cost is not None and isinstance(ponder_cost, torch.Tensor):
                if attention_mask is not None:
                    active_ponder = ponder_cost.view(-1)[attention_mask.view(-1) == 1]
                    mean_ponder_cost = active_ponder.mean()
                else:
                    mean_ponder_cost = ponder_cost.mean()
                masked_lm_loss = masked_lm_loss + ponder_weight * mean_ponder_cost

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )

from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

class UtForSequenceClassification(UtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ut = UtModel(config, add_pooling_layer=True)
        
        # Classifier head
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ut(
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

        pooled_output = outputs[1] if hasattr(outputs, "__getitem__") and len(outputs) > 1 and not return_dict else None
        if return_dict:
            pooled_output = getattr(outputs, "pooler_output", None)
            
        if pooled_output is None:
            # Fallback to [CLS] if pooler is absent
            sequence_output = outputs[0] if not return_dict else outputs.last_hidden_state
            pooled_output = sequence_output[:, 0]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )
