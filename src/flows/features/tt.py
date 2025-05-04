from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# model_name = 'dicta-il/dictalm2.0'
model_name = 'dicta-il/dictalm2.0-instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda', load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
עבר: הלכתי
עתיד: אלך

עבר: שמרתי
עתיד: אשמור

עבר: שמעתי
עתיד: אשמע

עבר: הבנתי
עתיד:
"""

encoded = tokenizer(prompt.strip(), return_tensors='pt').to(model.device)
print(tokenizer.batch_decode(model.generate(**encoded, do_sample=False, max_new_tokens=4)))
# ['<s> עבר: הלכתי\nעתיד: אלך\n\nעבר: שמרתי\nעתיד: אשמור\n\nעבר: שמעתי\nעתיד: אשמע\n\nעבר: הבנתי\nעתיד: אבין\n\n']