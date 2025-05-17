# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="bczhou/TinyLLaVA-1.5B")
result = pipe("do you know how to use this model?")
print(result)