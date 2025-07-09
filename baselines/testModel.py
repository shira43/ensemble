# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./falcon-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./falcon-7b", trust_remote_code=True)