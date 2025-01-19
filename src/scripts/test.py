from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained("dicta-il/dictalm2.0-instruct", torch_dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

messages = [
    {"role": "user", "content": """
    "הנאשם החביא את נשקו ברכבו"
    האם הנשק ברכב? ענה בכן או לא בלבד!
    """}
]

#     {"role": "user", "content": """
#     "4.	מעובדות האישום השני בכתב האישום המתוקן עולה כי במועד ובדרך שאינה ידועה במדויק למאשימה, השיג הנאשם רובה מסוג "תבור" (להלן הנשק). הנאשם החביא את הנשק בשכונה הסמוכה למקום מגוריו. "
# איפה הנאשם החזיק את הנשק? בחר באחת מהאפשרויות הבאו 
# [בבית, ברכב, על גופו,מוסלק-מוסתר,סמוך לבית]    """}
# ]


# <s> [INST] 
#     "4.	מעובדות האישום השני בכתב האישום המתוקן עולה כי במועד ובדרך שאינה ידועה במדויק למאשימה, השיג הנאשם רובה מסוג "תבור" (להלן הנשק). הנאשם החביא את הנשק בשכונה הסמוכה למקום מגוריו. "
# איפה הנאשם החזיק את הנשק? בחר באחת מהאפשרויות הבאו 
# [בבית, ברכב, על גופו,מוסלק-מוסתר,סמוך לבית]     [/INST]
# מוסלק-מוסתר, סמוך לבית</s>
# encoded = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

# <s> [INST] 
#     "4.	מעובדות האישום השני בכתב האישום המתוקן עולה כי במועד ובדרך שאינה ידועה במדויק למאשימה, השיג הנאשם רובה מסוג "תבור" (להלן הנשק). הנאשם החביא את הנשק בשכונה הסמוכה למקום מגוריו. "
# איפה הנאשם החזיק את הנשק]     [/INST]
# הנאשם החזיק את הנשק בשכונה הסמוכה למקום מגוריו.</s>

generated_ids = model.generate(encoded, max_new_tokens=50)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])