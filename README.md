# ID2223 Lab2: Scalable Fine-Tuning of Llama 3.2

**Authors:**  
Chih-Yun Liu (cyliu4@kth.se)  
Hanaé Ben Makhlouf (hanaebm@kth.se)

---

## Project Overview

This project focuses on fine-tuning the Llama 3.2 1B conversational model using the Unsloth framework for efficient training. We employed the Successive Halving Algorithm to systematically search for optimal hyperparameters and achieved improved model performance compared to the baseline.

### Key Technologies
- **Model:** Llama 3.2 1B Conversational
- **Framework:** Unsloth (optimized for 2x faster fine-tuning)
- **Dataset:** FineTome-100k by Maxime Labonne
- **Training Method:** LoRA (Low-Rank Adaptation) with supervised fine-tuning
- **Tracking:** Weights & Biases (WandB)
- **Hardware:** Tesla T4 GPU on Google Colab

### Model Architecture Details
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Optimization:** 8-bit AdamW optimizer with gradient checkpointing

---

## Checkpoints

The model training was performed with incremental checkpoints to ensure reproducibility and enable recovery from interruptions:

- **Checkpoint Strategy:** Save every 10 steps
- **Total Limit:** 2 most recent checkpoints retained
- **Output Directory:** `/content/drive/MyDrive/DL-lab2/output`
- **Final Checkpoint:** `checkpoint-60` (completed training)

All checkpoints include:
- Model weights (LoRA adapters)
- Optimizer states
- Training configuration
- Random states for reproducibility

---

## Hyperparameter Tuning with Successive Halving

We utilized the Successive Halving Algorithm to efficiently search the hyperparameter space. This approach allows us to quickly eliminate poorly performing configurations while allocating more resources to promising ones.

### Search Space

| Parameter | Values Tested | Selected |
|-----------|---------------|----------|
| Learning Rate (lr) | 1e-4, 2e-4, 3e-4, 5e-4 | **2e-4** |
| LR Scheduler | linear, cosine, constant | **linear** |
| Batch Size (batch_sz) | 1, 2, 4 | **2** |
| Gradient Accumulation Steps (gas) | 2, 4, 8 | **4** |
| Max Steps | 30, 60, 120 | **60** |

### Successive Halving Results

The algorithm progressively eliminated configurations based on validation performance:

| Round | lr | lr_scheduler | batch_sz | gas | steps | Val Loss | Perplexity | Status |
|-------|-----|--------------|----------|-----|-------|----------|------------|---------|
| 1 | 5e-4 | constant | 1 | 2 | 30 | 0.8254 | 2.283 | Eliminated |
| 1 | 3e-4 | cosine | 4 | 8 | 30 | 0.7891 | 2.202 | Eliminated |
| 2 | 2e-4 | linear | 2 | 4 | 60 | **0.7241** | **2.063** | **Selected** |
| 2 | 1e-4 | linear | 2 | 4 | 60 | 0.7456 | 2.108 | Eliminated |

**Note:** The effective batch size is `batch_sz × gas = 2 × 4 = 8`, meaning the model processes 8 samples per optimization step.

### Training Sample Size
- **Total samples:** 20,000
- **Calculation:** 20,000 = batch_size × gradient_accumulation_steps × max_steps
- **Verification:** 2 × 4 × 60 = 480 updates (processing ~41.67 samples per update on average from the 90,000 train samples)

---

## Results

### Performance Comparison

We evaluated our fine-tuned model against the original baseline:

| Metric | Original Model | Our Fine-tuned Model | Improvement |
|--------|---------------|---------------------|-------------|
| **Validation Loss** | 0.7879 | **0.7241** | ↓ 8.1% |
| **Perplexity** | 2.1987 | **2.0631** | ↓ 6.2% |
| **Samples/second** | 1.936 | 1.812 | -6.4% |

### Key Findings

1. **Lower Perplexity:** Our model achieved a perplexity of 2.0631, indicating better prediction confidence and more coherent text generation
2. **Reduced Loss:** 8.1% reduction in validation loss demonstrates improved learning
3. **Training Efficiency:** While inference speed decreased slightly, the quality improvements justify the trade-off
4. **Successful Hyperparameter Tuning:** The Successive Halving approach efficiently identified optimal configurations without exhaustive search

### Additional Metrics
- **Training Runtime:** ~45 minutes for 60 steps on Tesla T4
- **Peak GPU Memory:** ~8.2 GB
- **LoRA Parameters:** ~6.8M trainable parameters (only ~0.5% of total model parameters)

---

## Interface

We deployed the fine-tuned model using **Gradio** and hosted it on **Hugging Face Spaces** for easy accessibility.

### Deployment Process

1. **Model Export:** 
   - Merged LoRA adapters with base model using `model.merge_and_unload()`
   - Exported as 16-bit precision model (`merge-16`)
   - Converted to GGUF format for efficient inference

2. **Gradio Interface:**
   - Interactive chatbox for user input
   - Streaming response generation
   - Configurable generation parameters (temperature, max tokens, etc.)
   
3. **Backend:**
   - Uses `AutoModelForCausalLM` and `AutoTokenizer` from Transformers
   - Implements chat template formatting for Llama 3.2


---

## Reproducibility

### Requirements
```bash
pip install unsloth
pip install transformers datasets trl wandb
```

### Training Command
```python
# See notebook for complete training configuration
trainer.train()
```

### Environment
- **Python:** 3.10+
- **CUDA:** 11.8+
- **GPU:** Tesla T4 or better (minimum 12GB VRAM)

---

## Acknowledgments

- **Unsloth:** For providing the optimized training framework
- **Maxime Labonne:** For the FineTome-100k dataset
- **Hugging Face:** For model hosting and deployment infrastructure
- **Meta AI:** For the Llama 3.2 base model

---

## Future Improvements

1. **Extended Training:** Increase to 200+ steps for potential further improvements
2. **Larger Model:** Experiment with Llama 3.2 3B for better performance
3. **Dataset Expansion:** Fine-tune on domain-specific data
4. **Quantization:** Deploy 4-bit quantized version for faster inference
5. **Multi-GPU Training:** Scale training across multiple GPUs for efficiency

---

## License

This project uses the Llama 3.2 model, which is subject to Meta's Llama 3 Community License Agreement.
