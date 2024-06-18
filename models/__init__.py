from .configuration_pt import PtConfig
from .modeling_pt import PtModel, PtForMaskedLM

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
AutoConfig.register("pt", PtConfig)
AutoModel.register(PtConfig, PtModel)
AutoModelForMaskedLM.register(PtConfig, PtForMaskedLM)
