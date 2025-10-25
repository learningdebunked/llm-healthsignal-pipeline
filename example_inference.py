"""
Example: Enhanced GPT-2 inference with structured prompts
Demonstrates prompt engineering from Section IV.A of the paper
"""

from inference import explain_with_llm, generate_prompt_based_response

# Example 1: Classification interpretation
classification = {"prediction": "Atrial Fibrillation", "confidence": 0.92}
interpretation = explain_with_llm(classification, signal_type="ECG")
print("Interpretation:", interpretation)

# Example 2: Query with context
query = "What are treatment options?"
response = generate_prompt_based_response(query, classification_result=classification)
print("\nResponse:", response)
