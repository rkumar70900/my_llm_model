import requests
import numpy as np

url = "http://127.0.0.1:8033/v1/chat/completions"

data = {
    "model": "Qwen3-4B-GGUF.Q4_K_M",
    "messages": [
        {"role": "user", "content": "Tell me a joke about llamas"}
    ],
    "logprobs": 1
}

response = requests.post(url, json=data)
logprobs_example = response.json()["choices"][0]["logprobs"]['content']
# print(response.json()["choices"][0]["message"]["content"])

prompt_texts = [
    "Write a short fantasy story about a dragon and a wizard.",
    "Explain quantum computing in simple terms.",
]

def compute_perplexity_llama_format(logprobs_list):
    """
    logprobs_list: list of dicts like {'id': ..., 'token': ..., 'logprob': ...}
    """
    # Extract the per-token log probabilities
    token_logprobs = [tok['logprob'] for tok in logprobs_list if 'logprob' in tok]
    
    if not token_logprobs:
        return None
    
    mean_logprob = np.mean(token_logprobs)
    perplexity = np.exp(-mean_logprob)
    return perplexity

# Example usage
# logprobs_example = [
#     {'id': 151667, 'token': '<think>', 'logprob': -3.3856e-05},
#     {'id': 151668, 'token': 'The', 'logprob': -0.0023},
#     {'id': 151669, 'token': 'dragon', 'logprob': -0.0045}
# ]

perplexity = compute_perplexity_llama_format(logprobs_example)
print("Perplexity:", perplexity)
