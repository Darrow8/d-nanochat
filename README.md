# D-NanoChat

Notes from videos, articles, etc: [Notes](Notes.md) 

Plan of building nanochat: [Plan](Plan.md)

init python venv: `source .venv/bin/activate`

## Day 8

I have written out AdamW and researched how it is an improvement over SGD. I am now working on the gpt.py file. 

I have decided to directly copy-paste common.py instead of write it out because it is just boring logging and filesystem management code rather than anything important.

I have written out and researched Muon optimizer. 

## Day 7

It looks like the advice given from chatgpt was not great on making a plan because the plan they have involves making files that don't exist. Instead I've asked about the repo [DeepWiki](https://deepwiki.com/karpathy/nanochat). 

## Day 6

Finished rust tokenizer. Wrote my own token decoder in rust: 

input_text: The unexamined life is not worth living
encoded: [354, 346, 110, 101, 120, 465, 262, 304, 340, 102, 101, 314, 326, 263, 275, 480, 340, 118, 293]
decoded: The unexamined life is not worth living



## Day 5

Finished watching Karpathy video on GPT tokenizer and took notes on it. 

## Day 4

Continued writing out rustbpe code and watching Karpathy video on GPT tokenizer. 

## Day 3 

Continued writing out rustbpe code and learned about how tokenizers work. 

## Day 2 

Wrote out a lot more of the rustbpe code. Got into the tokenizer implementation. 

## Day 1

The ambition of this project is to replicate [nanochat](https://github.com/karpathy/nanochat) without using AI, as Andrej Karpathy recommended in an [interview](https://www.dwarkesh.com/p/andrej-karpathy) with Dwarkesh Patel. 

I may also add some additional specs if I find interesting improvements along the way. 

I have experience in machine learning and implementing software projects, but I think this is going to be challenge as I have little experience with rust.  

I asked ChatGPT to generate a set of milestones for implementing nanochat which are saved in PLAN.md. 

Ok looks like I've installed the requisite packages and `uv sync` works. 

I am starting by working through the rustbpe code. 

