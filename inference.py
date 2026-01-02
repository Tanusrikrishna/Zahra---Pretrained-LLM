import model as m
import tiktoken
import torch
tokenizer=tiktoken.get_encoding("gpt2" )
vocab_size=tokenizer.n_vocab
model = m.GPR(
    dmodel=384,
    dff=64,          # head dim
    n_head=6,
    n_layer=6,
    f_dff=1536,
    max_seq=256,
    vocab_size=vocab_size,
    droprate=0.1
)

state_dict = torch.load(
    r"D:\Transformer\Zahra1\zahra\gpr2_v15.0.pth",
    map_location="cpu"   # IMPORTANT: safe loading
)
model.load_state_dict(state_dict)
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.6,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    
):
    model.eval()

   

    
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :]

     
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                logits[:, token_id] /= repetition_penalty

   
        temperature = max(temperature, 1e-6)
        logits = logits / temperature

     
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, -1e10),
                logits
            )

        probs = F.softmax(logits, dim=-1)

 
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

       
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False

            sorted_probs[sorted_indices_to_remove] = 0.0
            probs = torch.zeros_like(probs).scatter(
                1, sorted_indices, sorted_probs
            )

  
        if probs.sum() <= 0:
            probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())



print("\n\n",generate(model,   tokenizer,"in January 2013",max_new_tokens=100))