import csv
import itertools

attributes = {
    "age": [25, 45],
    "gender": ["Male", "Female"],
    "ethnicity": ["African American", "Hispanic", "Caucasian"],
}

def save_personas_csv(attrs, path="personas.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "gender", "ethnicity", "persona_text"])
        for age, gender, ethnicity in itertools.product(attrs["age"], attrs["gender"], attrs["ethnicity"]):
            persona_text = f"A {age}-year-old {ethnicity} {gender}"
            writer.writerow([age, gender, ethnicity, persona_text])

save_personas_csv(attributes, "../datasets/personas/cose_personas.csv")
