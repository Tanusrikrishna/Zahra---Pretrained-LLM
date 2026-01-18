# Zahra-Transformer Experiment

the model weight are here is the link
```link
https://drive.google.com/file/d/1WHmalqN_3StsCY9y1HN2K95YD1-4a6vU/view?usp=sharing
```

## Overview
This repository contains experiments with a custom Transformer-based language model.  
The goal is to explore model training, evaluation, and inference quality using a lightweight architecture.

## Model Configuration
The model was trained with the following hyperparameters:
---
```json
{
  "dmodel": 384,
  "dff": 64,
  "n_head": 6,
  "n_layer": 6,
  "f_dff": 1536,
  "max_seq": 256,
  "droprate": 0.1
}
```

## Training & Evaluation
Training was conducted on a custom dataset with tokenized text sequences.

Evaluation metric: Perplexity

Current model perplexity: 157.951 .

Inference Results
Example inference output shows grammatically valid but semantically incoherent text:

inference
"...superman, and the latter was to be one of the most popular culture. 
The first book by the United States, in a review for the film's development, was released on June 13, 2011..."
This reflects the high perplexity score — the model generates plausible sentences but lacks coherence.

Usage
Python
Clone the repo:

```bash
git clone https://github.com/Yasar26951/zahra.git
cd zahra
pip install -r requirements.txt
python inference.py --prompt "Your text here"
```

---
