import numpy as np
import math
from collections import Counter

def calculate_shannon_entropy(text: str) -> float:
    """
    Calculates the Shannon entropy of the characters in the text.
    This serves as a proxy for 'entropy-related measures' without 
    requiring the original LLM's token logits.
    """
    if not text:
        return 0.0
    
    # Calculate word-level entropy as a proxy for structural unpredictability
    words = text.split()
    if not words:
        return 0.0
        
    counts = Counter(words)
    total_words = len(words)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total_words
        entropy -= p * math.log2(p)
        
    return entropy


def extract_advanced_features(step_dict: dict) -> list[float]:
    """
    Extracts additional advanced features from a reasoning step:
      1. Lexical Entropy (Proxy for prediction confidence / repetitiveness)
      2. Repetition Ratio (High repetition often correlates with hallucination/loops)
      3. Relative position measure (Derived from token counts and length constraints)
    
    Returns:
        List of float features to be appended to the standard embeddings.
    """
    text_added = step_dict.get("text_added", "")
    cumulative_text = step_dict.get("cumulative_text", "")
    num_tokens = step_dict.get("num_tokens", 0)
    step_idx = step_dict.get("step_index", 0)
    
    # 1. Entropy measure on the newly added text
    entropy = calculate_shannon_entropy(text_added)
    
    # 2. Repetition Ratio (unique words / total words in cumulative text)
    words = cumulative_text.split()
    if len(words) > 0:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
    else:
        repetition_ratio = 1.0
        
    # 3. Step density (tokens per step)
    # How verbose was this specific step?
    words_in_step = len(text_added.split())
    step_density = words_in_step / max(1, num_tokens)
    
    return [float(entropy), float(repetition_ratio), float(step_density)]
