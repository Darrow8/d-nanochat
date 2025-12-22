AI generated plan to implement nanochat. 

#### Phase 1: Foundation & "Tiny" Verification*Before scaling up, prove the logic works on a CPU or single GPU.*

* **Milestone 1: Project Skeleton & Tools**
* Same as before: Set up `uv`, `torch`, and the folder structure.


* **Milestone 2: The Rust Tokenizer (The Hard Part First)**
* **Task:** Implement BPE in Rust.
* **Crucial Detail:** Ensure your tokenizer produces the *exact same* token IDs as GPT-4/o200k_base for the same text. If your tokenization drifts, your benchmarks will be invalid.
* **Verification:** Write a test that compares your output against the `tiktoken` library.


* **Milestone 3: Model Architecture & The "Sanity Check"**
* **Task:** Implement the Llama-style Transformer (RMSNorm, RoPE, SwiGLU).
* **New Verification (Critical):** Create a script `verify_model.py`.
* Create a generic random batch of data `(x, y)`.
* Run a training loop on *just this one batch* for 100 steps.
* **Success Criteria:** The loss must drop to near zero (0.01 or lower). If it hovers around 10.0, your architecture is broken (likely a broadcasting error in RoPE or Attention masking).





#### Phase 2: The Training Pipeline*Now we prepare for scale.*

* **Milestone 4: High-Performance Dataloader**
* **Task:** Download FineWeb shards and create a `DataLoader`.
* **Specifics:** Implement a background thread (using `threading` or `multiprocessing`) that pre-fetches data from disk into pinned memory (`pin_memory=True`) while the GPU is computing.
* **Verification:** Measure tokens/second throughput. It must exceed your GPU's consumption rate, or training will be bottlenecked by disk I/O.


* **Milestone 5: The "Hybrid" Optimizer (Muon + AdamW)**
* **Task:** This is unique to `nanochat`. You need to split your model parameters into two groups:
1. **2D weights** (projections, embeddings) -> Use the **Muon** optimizer (Newton-step based).
2. **1D weights** (biases, norms) -> Use **AdamW**.


* *Note:* If this is too hard initially, start with pure AdamW. You will converge slower, but you will still converge.



#### Phase 3: Distributed Training & Scaling*This is where you replicate `speedrun.sh`.*

* **Milestone 6: Gradient Accumulation**
* **Task:** Decouple your "micro-batch size" (what fits on your GPU) from the "global batch size" (what the math requires).
* **Implementation:** Accumulate gradients for N steps before calling `optimizer.step()`. This allows you to replicate the 8xH100 dynamics on a single GPU (it will just take 8x longer).


* **Milestone 7: Base Pretraining (The Run)**
* **Task:** Run the training.
* **Verification:** Monitor the "Cross Entropy Loss." For a GPT-2 sized model, a loss starting at ~10.0 and dropping to ~3.0 is expected. Anything >4.0 after training means something is wrong.



#### Phase 4: Alignment & Serving*Turning the raw brain into a product.*

* **Milestone 8: Mid-Training (The Format Shift)**
* **Task:** Fine-tune on "Identity" data.
* **Why:** Base models just autocomplete. Mid-training teaches the model *when* to stop generating (EOS tokens) and how to format a dialogue.


* **Milestone 9: The Web UI & KV Cache**
* **Task:** Build the UI.
* **Optimization:** Implement **KV Caching**. Without this, generating a 500-word response will take minutes instead of seconds, because the model re-computes the entire history for every new word.



###Summary ChecklistIf you can answer "Yes" to these, you have successfully replicated the project:

1. [ ] Does my tokenizer match `tiktoken` exactly?
2. [ ] Can my model overfit a single batch to zero loss?
3. [ ] Do I have Gradient Accumulation to simulate large batch sizes?
4. [ ] Do I have a KV Cache implementation for inference?