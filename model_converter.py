import torch

def load_from_standard_weights(input_file: str, device='cpu') -> dict[str, torch.Tensor]:
    original_model = torch.load(input_file, map_location=device, weights_only = False)
    
    converted = {}
    
    for name, weights in original_model.items():
        # discard attn.bias term
        if '.attn.bias' in name:
            continue
        
        converted['transformer.' + name] = weights
        
    # add the final output layer to be the same weights as wpe
    converted['lm_head.weight'] = converted['transformer.wte.weight'] 
    
    return converted