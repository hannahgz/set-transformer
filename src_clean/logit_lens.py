import torch
import torch.nn.functional as F
import random
import numpy as np
from classify import load_model_from_config
from model import GPTConfig44_Complete
from data_utils import load_tokenizer

def logit_lens_analysis(model, inputs):
    """
    Perform logit lens analysis on the model by projecting intermediate representations
    to vocabulary space using the unembedding matrix (lm_head weights).
    """
    device = inputs.device
    b, t = inputs.size()
    
    # Get position embeddings
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(inputs)
    pos_emb = model.transformer.wpe(pos)
    
    # Initial embedding
    x = tok_emb + pos_emb
    
    # Store intermediate logits
    layer_logits = []
    
    # Project input embeddings to logit space
    initial_logits = model.lm_head(model.transformer.ln_f(x))
    layer_logits.append(("embedding", initial_logits))
    
    # Get intermediate representations and project each to logit space
    for i, block in enumerate(model.transformer.h):
        # Apply attention and MLP
        x_normalized = block.ln_1(x)
        attn_output, _, _ = block.attn(x_normalized)
        x = x + attn_output
        
        # Project post-attention state to logits
        post_attn_logits = model.lm_head(model.transformer.ln_f(x))
        layer_logits.append((f"layer_{i}_attn", post_attn_logits))
        
        # Apply MLP
        x = x + block.mlp(block.ln_2(x))
        
        # Project post-MLP state to logits
        post_mlp_logits = model.lm_head(model.transformer.ln_f(x))
        layer_logits.append((f"layer_{i}_mlp", post_mlp_logits))
    
    # # Get final logits
    # final_logits = model.lm_head(model.transformer.ln_f(x))
    # layer_logits.append(("final", final_logits))
    
    return layer_logits

def analyze_predictions(layer_logits, model_config, tokenizer, tokenized_input_sequence):
    """
    Analyze how predictions evolve through the layers.
    """
    results = []
    
    for layer_name, logits in layer_logits:
        # Get predictions at each layer
        probs = F.softmax(logits, dim=-1)
        # predictions = torch.argmax(logits, dim=-1)[:, -(model_config.target_size+1):-1]
        predictions = torch.argmax(logits, dim=-1)[:, -(model_config.target_size+1):-1]
        targets = tokenized_input_sequence[:, -model_config.target_size:]

        mask = targets != model_config.padding_token 
        total_non_mask_count = mask.sum().item()
        
        matches = ((predictions == targets) | ~mask)
        accuracy = (matches.sum().item() - (model_config.target_size - total_non_mask_count)) / total_non_mask_count
            
        # Get top k predictions if we have a vocabulary
        
        top_k_values, top_k_indices = torch.topk(probs, k=5, dim=-1)
        top_k_tokens = [[idx.item() for idx in batch] for batch in top_k_indices[0]]
            
        results.append({
            'layer': layer_name,
            'accuracy': accuracy,
            'top_k': top_k_tokens,
            'logits': logits,
            'probs': probs,
            'predictions': tokenizer.decode(predictions[0].cpu().numpy()),
        })
    
    return results

# Example usage:
def run_logit_lens(model_config, input_sequence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(model_config)
    tokenizer = load_tokenizer(model_config.tokenizer_path)
    tokenized_input_sequence = torch.tensor(
        tokenizer.encode(input_sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        # Get logits at each layer
        layer_logits = logit_lens_analysis(model, tokenized_input_sequence)
        
        # Analyze how predictions evolve
        # target_indices = input_sequence[:, -model.config.target_size:]  # Assuming these are the targets
        results = analyze_predictions(layer_logits, model_config, tokenizer, tokenized_input_sequence)
        breakpoint()
        return results
    

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    input1 = [
        "E", "striped", "B", "green", "D", "two", "B", "oval", "C", "green", "D", "green", "D", "solid", "E", "two", "B", "one", "D", "oval", "E", "green", "C", "one", "A", "green", "C", "open", "A", "one", "E", "oval", "B", "striped", "C", "oval", "A", "oval", "A", "solid",
        ">", "A", "B", "C", ".", "_", "_", "_", "_"
    ]

    run_logit_lens(GPTConfig44_Complete(), input_sequence=input1)

