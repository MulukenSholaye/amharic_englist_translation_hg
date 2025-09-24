from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

text = "ይህ በጣም ጥሩ ነው።"  # Amharic
inputs = tokenizer(text, return_tensors="pt", src_lang="amh")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng"])
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translation)  # "This is very good."