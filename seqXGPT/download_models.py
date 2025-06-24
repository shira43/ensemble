from transformers import AutoModelForCausalLM, AutoTokenizer


         #    "EleutherAI/gpt-neo-2.7B",
        #     "EleutherAI/gpt-j-6B",
       #      "meta-llama/Llama-2-7b-hf"
       #        "gpt2-xl"

for name in ["meta-llama/Llama-3.1-8B"]:
    AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir=f"pretrained/{name.split('/')[-1]}",
        use_auth_token=True
    )
    AutoTokenizer.from_pretrained(
        name,
        cache_dir=f"pretrained/{name.split('/')[-1]}"
    )

