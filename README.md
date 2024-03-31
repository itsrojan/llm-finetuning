# Description

This repository contains notebooks that finetune and test three models: Llama2-7B, Mistral-7B, and Phi-2-2.7B, on the `tatsu-lab/alpaca` dataset using the Transformers and PEFT libraries.

## Environment Setup

Before starting, ensure that Python 3.11 is installed. Then, install the required libraries using the following command:

```bash
pip install datasets transformers peft trl tqdm
```

## Usage

1. **Load the Model and Dataset:**
   - The base models are `NousResearch/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`, and `microsoft/phi-2`.

2. **Run the Notebook:**
   - Open the notebook in Jupyter or another compatible environment and execute the cells sequentially.

3. **Fine-tune the Model:**
   - Follow the instructions in the notebook to fine-tune the model. This includes setting training arguments and initializing the trainer.
   - After initial fine-tuning, the model will be saved locally for further use or evaluation.

4. **Evaluate the Model:**
   - Evaluate the fine-tuned model on the test dataset to assess its performance. Metrics used include perplexity, BLEU score, ROUGE-L score, and BERTScore.

Additionally, the notebooks explore four different hyperparameter configurations for model generation, applied to each of the three models. These configurations vary across three parameters: `top_k`, `num_beams`, and `temperature`, resulting in a total of 12 configurations.

Hyperparameter values tested:

- `top_k`: [10, 25, 40, 75]
- `num_beams`: [2, 4, 6, 8]
- `temperature`: [0.25, 0.5, 0.7, 1.0]

For each configuration, only one parameter is changed, while the others are set to their default values (`top_k=50`, `num_beams=1`, `temperature=0.8`).

---

# Task 2: Llama2 vs Mistral vs Phi-2

When comparing the three models, Phi2, Llama2, and Mistral, it's clear that Phi2 outperforms the others based on the metrics. Phi2 has the highest scores in BLEU, Rouge-L, and BERTScore, indicating better performance in understanding and generating text. Llama2 and Mistral have significantly lower scores, suggesting they might not be as effective in these tasks.

In terms of which metrics are more appropriate for comparison to human evaluation, BLEU and Rouge-L focus on the similarity of words between the model's output and a reference text, but they might not fully capture the meaning. BERTScore, on the other hand, attempts to consider the semantics, making it potentially closer to human judgment. Perplexity measures how well the model predicts text, which is different from the other metrics. Therefore, relying on multiple metrics is often the best approach to get a comprehensive understanding of a model's performance.

# Task 3: Hyperparameter Testing

Changing the hyperparameters of the Phi-2 model can affect how well the model performs on different metrics.

1. **top_k**: As top_k increases from 10 to 75, the BLEU and Rouge-L scores go up and down a bit, but they don't have a clear trend. However, the BERTScore and Perplexity change more noticeably. For example, when top_k is 75, the BERTScore goes down, and Perplexity goes up a lot, which might mean the text is less similar to the reference and more uncertain.

2. **beam_size**: Changing the beam_size from 2 to 8 shows that a beam_size of 4 gives the best BLEU (42.57) and Rouge-L (0.60) scores. This means that using a beam_size of 4 might help the model make better choices when generating text. But, when the beam_size gets too big, like 8, the scores don't get much better, and sometimes they get worse.

3. **temperature**: Adjusting the temperature from 0.25 to 1.0 shows that lower temperatures (like 0.25 and 0.5) give better BLEU and Rouge-L scores than higher temperatures (like 1.0). This might mean that lower temperatures help the model be more sure about its choices and make text that's closer to the reference.

It's important to find the right balance to get the best results.

# Useful Resources Used

https://www.datacamp.com/tutorial/fine-tuning-llama-2
https://www.kaggle.com/code/kingabzpro/fine-tuning-phi-2
