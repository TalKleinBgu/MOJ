from collections import defaultdict
import pandas as pd
import re
data = [
    "[400 גרם קוקאין]",
    "[קוקאין-400 גרם]",
    "[סם-3.4 טון, סם-5.5 טון]",
    "[87.25 קילוגרם קנאביס, 5 גרם קנאביס]",
    "[קנאביס-5.5 טון, קנאביס-3.4 טון]",
    "[קנאביס-2 ק\"ג]",
    "[5 טונות קנאביס]",
    "[קוקאין-120 טבליות]",
    "[MDMA-200 מ\"ל]",
    "[הרואין-13 חבילות]",
    "[קנאביס-5 שתילים]",
    "[1.085 ק\"ג אינדזול קרבוקסאמיד, 6.55 גרם קנבוס]",
    "[קנבואידים סינטטיים-לא צויין]",
    "[קנבואיד סינטטי-19.643 בולים, קוקאין-0.4181 גרם]",
    "[קנאביס-3.4 טון]",
    "[סם-3.4 טון, סם-5.5 טון]",
    "[קנאביס-87.25 קילוגרם, קנאביס-5 גרם]",
    "[קנאביס-5.5 טון, קנאביס-3.4 טון]"
]
path = '/home/tak/MOJ/results/evaluations/drugs/features_extraction/final_table_4.csv'
df = pd.read_csv(path)

def normalize_commas_in_numbers(text):
    # תופס מספרים עם פסיק בין ספרות (למשל: 1,033.6) והופך ל־ 1033.6
    return re.sub(r'(?<=\d),(?=\d)', '', text)

def normalize_quantity(num, unit):
    unit = unit.strip().lower()
    if unit in ["גרם", "ג'", "g"]:
        return num / 1000, "קילוגרם"
    elif unit in ["קילוגרם", "ק\"ג", "קג", "קג.", "ק''ג"]:
        return num, "קילוגרם"
    elif "טון" in unit or 'טונות' in unit:  # כולל טון, טונות
        return num * 1000, "קילוגרם"
    elif unit in ["ליטר"]:
        return num, "ליטר"
    elif unit in ["מ\"ל", "מיליליטר", "מל"]:
        return num / 1000, "ליטר"
    else:
        return None, unit  # can't be normalized
for entry in df['amount_type']:
    entry = entry.strip("[]")
    entry = normalize_commas_in_numbers(entry)

    parts = entry.split(",")
    weight_totals = defaultdict(float)
    other_units = defaultdict(lambda: defaultdict(float))
    for part in parts:
        part = part.strip().strip("[]")

        if "-" in part:
            drug, amount_str = part.split("-", 1)
        else:
            tokens = part.strip().split()
            if len(tokens) < 3:
                continue
            amount_str = " ".join(tokens[:2])
            drug = " ".join(tokens[2:])
            if not drug:
                amount_str = tokens[0]
                drug = tokens[1]

        drug = drug.strip()
        amount_str = amount_str.replace('"', '').replace("'", '').strip()
        tokens = amount_str.split()
        if len(tokens) < 2:
            continue
        try:
            num = float(tokens[0].replace(",", "").replace("٫", "."))  # handle 19.643
            unit = tokens[1]
            result = normalize_quantity(num, unit)
            if result[0] is not None:
                normalized_val, normalized_unit = result
                weight_totals[(drug, normalized_unit)] += normalized_val
            else:
                other_units[drug][unit] += num
        except Exception as e:
            print(f"❌ Skipped '{part}' due to: {e}")

    # ✅ Print Results
    print("\n💠 סיכום לפי משקל / נפח:")
    for (drug, unit), total in weight_totals.items():
        print(f"{drug}: {total:.3f} {unit}")

    print("\n📦 סיכום לפי יחידות אחרות:")
    for drug, units in other_units.items():
        for unit, count in units.items():
            print(f"{drug}: {count} {unit}")
