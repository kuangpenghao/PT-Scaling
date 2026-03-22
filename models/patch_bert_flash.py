import torch
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSdpaSelfAttention, BertEncoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

_original_bert_encoder_forward = BertEncoder.forward
_original_bert_self_attention_forward = BertSelfAttention.forward
_original_bert_sdpa_self_attention_forward = BertSdpaSelfAttention.forward

def flash_encoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    use_flash_attn = HAS_FLASH_ATTN and hidden_states.dtype in (torch.float16, torch.bfloat16)
    
    if use_flash_attn and encoder_hidden_states is None and not output_attentions and not use_cache and attention_mask is not None:
        cu_seqlens = None
        max_seqlen = None
        indices = None
        batch_size, seq_len = hidden_states.shape[:2]
        
        # reconstruct 1D boolean mask from BERT's extended attention mask
        if attention_mask.dim() == 4:
            # extended mask is (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
            bool_mask = (attention_mask[:, 0, -1, :] >= -1.0).int()
            unpad_res = unpad_input(hidden_states, bool_mask)
            hidden_states = unpad_res[0]
            indices = unpad_res[1]
            cu_seqlens = unpad_res[2]
            max_seqlen = unpad_res[3]
        elif attention_mask.dim() == 2:
            bool_mask = (attention_mask >= -1.0).int()
            unpad_res = unpad_input(hidden_states, bool_mask)
            hidden_states = unpad_res[0]
            indices = unpad_res[1]
            cu_seqlens = unpad_res[2]
            max_seqlen = unpad_res[3]


        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                if indices is not None:
                    padded_states = pad_input(hidden_states, indices, batch_size, seq_len)
                    all_hidden_states = all_hidden_states + (padded_states,)
                else:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            layer_module.attention.self._flash_attn_kwargs = {
                "use_flash_attn": True,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen
            }

            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=None, 
                head_mask=layer_head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if hasattr(layer_module.attention.self, "_flash_attn_kwargs"):
                del layer_module.attention.self._flash_attn_kwargs

        if indices is not None:
            hidden_states = pad_input(hidden_states, indices, batch_size, seq_len)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None, None] if v is not None)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    return _original_bert_encoder_forward(
        self, hidden_states, attention_mask, head_mask, 
        encoder_hidden_states, encoder_attention_mask, 
        past_key_values, use_cache, output_attentions, 
        output_hidden_states, return_dict
    )

def make_flash_self_attention_forward(original_forward):
    def flash_self_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        flash_kwargs = getattr(self, "_flash_attn_kwargs", None)
        is_absolute = getattr(self, "position_embedding_type", "absolute") == "absolute"
        
        if flash_kwargs and flash_kwargs.get("use_flash_attn") and is_absolute and not output_attentions:
            cu_seqlens = flash_kwargs["cu_seqlens"]
            max_seqlen = flash_kwargs["max_seqlen"]
            
            query = self.query(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            key = self.key(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            value = self.value(hidden_states).view(-1, self.num_attention_heads, self.attention_head_size)
            
            dropout_p = self.dropout_prob if self.training else 0.0
            
            context_layer = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=dropout_p
            )
            context_layer = context_layer.view(-1, self.all_head_size)
            
            return (context_layer,)
            
        return original_forward(
            self, hidden_states, attention_mask, head_mask, 
            encoder_hidden_states, encoder_attention_mask, 
            past_key_value, output_attentions
        )
    return flash_self_attention_forward

def apply_flash_bert_patch():
    if HAS_FLASH_ATTN:
        BertEncoder.forward = flash_encoder_forward
        BertSelfAttention.forward = make_flash_self_attention_forward(_original_bert_self_attention_forward)
        BertSdpaSelfAttention.forward = make_flash_self_attention_forward(_original_bert_sdpa_self_attention_forward)
        print("Successfully applied Flash Attention 2 monkey-patch to Hugging Face BERT (both Default and SDPA)!")
    else:
        print("No flash_attn found, skipped BERT patching.")