# Phase 10 — Fine-Tuning Experiment

## Objective

Evaluate whether QLoRA fine-tuning improves the behavior of the financial reasoning LLM.

## Base Model

Qwen/Qwen2.5-1.5B-Instruct

## Fine-Tuning Method

- QLoRA
- 4-bit NF4 quantization
- LoRA adapters
- FP16
- Gradient checkpointing
- 8-bit optimizer
- Maximum sequence length: 512
- Batch size: 1
- Gradient accumulation: 8
- 3 epochs

## Hardware

NVIDIA GeForce RTX 3050 Laptop GPU
4 GB VRAM

## Dataset

Behavioral dataset containing 41 examples:

- Training: 32
- Validation: 4
- Test: 5

The dataset focuses on stable reasoning behavior rather than live financial knowledge.

Live financial information remains handled by:

- deterministic analytics
- RAG
- MCP

## Training Result

Final training loss: 1.9869

Final validation loss: 1.9576

Validation loss decreased during training:

2.5562 → 2.1395 → 1.9576

## Holdout Evaluation

The same 5 unseen test cases were evaluated on both the base and fine-tuned models.

### Results

1. Operating profit percentage change:
   - Base: correct
   - Fine-tuned: correct

2. Absolute revenue change:
   - Base: correct
   - Fine-tuned: correct

3. Revenue growth:
   - Both models returned the percentage but omitted the absolute change.

4. Revenue growth:
   - Both models returned the percentage but omitted the absolute change.

5. Unsupported future prediction:
   - Both models made an unsupported prediction.

## Decision

The fine-tuned adapter was NOT adopted as the production reasoning model.

The evaluation did not demonstrate a meaningful improvement over the base model on the unseen holdout set.

This experiment demonstrates an evaluation-driven approach to fine-tuning rather than fine-tuning for its own sake.

## Production Decision

The production architecture continues to use:

- deterministic financial analytics for calculations
- RAG for document-grounded information
- MCP for external market-data tools
- LLM reasoning for explanation
- input and output guardrails
- observability and evaluation

The QLoRA adapter is retained as an experimental artifact.

## Key Engineering Conclusion

Fine-tuning financial knowledge is not appropriate for changing financial data.

The system therefore separates:

Stable model behavior:
    Fine-tuning / prompting

Changing financial knowledge:
    SQL / analytics / RAG / MCP

Final financial claims:
    Evidence + deterministic computation + output verification
