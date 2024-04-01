# Description

This repository contains notebooks that finetune and test three models: LLamM-2-7B, Mistral-7B, and Phi-2-2.7B, on the `tatsu-lab/alpaca` dataset using the Transformers and PEFT libraries.

## Environment Setup

Before starting, ensure that Python 3.11 is installed. Then, install the required libraries using the following command:

```bash
pip install datasets transformers peft trl tqdm
```

## Usage

1. **Load the Model and Dataset:**

   The base models are:
   - LLaMa-2 (7B) [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf)
   - Phi-2 (2.7B) [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
   - Mistral (7B) [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

3. **Run the Notebook:**
   - Open the notebook in Jupyter or another compatible environment and execute the cells sequentially.

4. **Fine-tune the Model:**
   - Follow the instructions in the notebook to fine-tune the model. This includes setting training arguments and initializing the trainer.
   - After initial fine-tuning, the model will be saved locally for further use or evaluation.

5. **Evaluate the Model:**
   - Evaluate the fine-tuned model on the test dataset to assess its performance. Metrics used include perplexity, BLEU score, ROUGE-L score, and BERTScore.

Additionally, the notebooks explore four different hyperparameter configurations for model generation, applied to each of the three models. These configurations vary across three parameters: `top_k`, `num_beams`, and `temperature`, resulting in a total of 12 configurations.

Hyperparameter values tested:

- `top_k`: [10, 25, 40, 75]
- `num_beams`: [2, 4, 6, 8]
- `temperature`: [0.25, 0.5, 0.7, 1.0]

For each configuration, only one parameter is changed, while the others are set to their default values (`top_k=50`, `num_beams=1`, `temperature=0.8`).

### Task 2: Llama2 vs Mistral vs Phi-2

When comparing the three models, Phi2, Llama2, and Mistral, it's clear that Phi2 outperforms the others based on the metrics. Phi2 has the highest scores in BLEU, Rouge-L, and BERTScore, indicating better performance in understanding and generating text. Llama2 and Mistral have significantly lower scores, suggesting they might not be as effective in these tasks.

In terms of which metrics are more appropriate for comparison to human evaluation, BLEU and Rouge-L focus on the similarity of words between the model's output and a reference text, but they might not fully capture the meaning. BERTScore, on the other hand, attempts to consider the semantics, making it potentially closer to human judgment. Perplexity measures how well the model predicts text, which is different from the other metrics. Therefore, relying on multiple metrics is often the best approach to get a comprehensive understanding of a model's performance.

Generated using settings: top_k = 50, num_beams = 1, and temperature = 0.8.

| Model Name                            | BLEU     | ROUGE-L   | BERTScore | Perplexity | Human Evaluation |
|---------------------------------------|----------|-----------|-----------|------------|------------------|
| Phi-2-fine-tuned                      | 0.387053 | 0.580461  | 0.929016  | 21.4520    | 0.94             |
| Llama-2-7b-hf-fine-tuned              | 0.120098 | 0.303915  | 0.859611  | 16.8683    | 0.90             |
| Mistral-7B-v0.1-fine-tuned            | 0.106375 | 0.288294  | 0.883203  | 15.6959    | 0.95             |

### Task 3: Hyperparameter Testing

Changing the hyperparameters of the Phi-2 and LLaMA-2 models can affect how well the model performs on different metrics.

1. **top_k**: Increasing `top_k` generally leads to more diverse text generation. In Phi-2, a higher `top_k` leads to higher BLEU and Rouge-L scores, indicating better quality and coherence in generated text. However, in LLaMA2, the impact is less consistent, with some increases in `top_k` resulting in lower scores.

2. **beam_size**: Increasing `beam_size` improves the quality of generated text, as seen in higher BLEU and Rouge-L scores for both models. This is because a larger beam size allows the model to explore more candidate sequences, leading to better overall text generation.

3. **temperature**: Adjusting the `temperature` affects the randomness of text generation. A lower temperature (e.g., 0.25) tends to produce more deterministic and less diverse text, leading to higher BLEU and Rouge-L scores, indicating better quality. However, a very low temperature can also lead to overly repetitive text, which might not be desirable in some contexts.

Overall, the choice of hyperparameters depends on the desired balance between diversity, coherence, and quality in the generated text.

### References

https://www.datacamp.com/tutorial/fine-tuning-llama-2

https://www.kaggle.com/code/kingabzpro/fine-tuning-phi-2
