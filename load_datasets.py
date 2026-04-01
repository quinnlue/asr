from datasets import load_dataset
from huggingface_hub import login

login()

ds = load_dataset("quinnlue/asr")
realclass = load_dataset("quinnlue/realclass")
rirs = load_dataset("quinnlue/rirs")

