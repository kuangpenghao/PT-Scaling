from .configuration_pt import PtConfig
from .modeling_pt import PtModel, PtForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
AutoConfig.register("pt", PtConfig)
AutoModel.register(PtConfig, PtModel)
AutoModelForCausalLM.register(PtConfig, PtForCausalLM)
