# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PtConfig(LlamaConfig):
    model_type = "pt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_shared_kv: bool = False,
        use_shared_qo: bool = False,
        use_shared_mlp: bool = False,
        use_squared_softmax_pre_attn: bool = False,
        use_squared_softmax_post_attn: bool = False,
        use_squared_softmax_final: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_shared_kv = use_shared_kv
        self.use_shared_qo = use_shared_qo
        self.use_shared_mlp = use_shared_mlp
        self.use_squared_softmax_pre_attn = use_squared_softmax_pre_attn
        self.use_squared_softmax_post_attn = use_squared_softmax_post_attn
        self.use_squared_softmax_final = use_squared_softmax_final
