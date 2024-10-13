"""
Initialize a backward model using a pretrained forward model.
"""

import torch
from models import PtConfig, PtForMaskedLM
import argparse


def main():

    # python convert_to_backward.py --input TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T --output outputs/tiny-llama-backward-init
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to the input model")
    args = parser.parse_args()

    model = PtForMaskedLM.from_pretrained(args.input)

    for name, param in model.named_parameters():
        mean, std = param.mean().item(), param.std().item()
        print(f"{name}: mean={mean}, std={std}")

        hist = torch.histc(param, bins = 20, min = -0.1, max = 0.1)

        print(hist)



if __name__ == "__main__":
    main()