import json
import os

# Attribute values in desired order
attributes = {
    "age": [15, 35, 65],
    "education": ["no formal education", "high school education", "higher education"],
    "gender": ["male", "female"],
    "loneliness": ["not lonely", "somewhat lonely"],
    "political_view": ["left-wing", "right-wing", "centrist"],
    "race": ["white", "black", "asian"],
    "religion": ["Christian", "Muslim", "Jewish", "Atheist", "Hindu"]
}

# Create output directory
output_dir = "../datasets/personas_&_questions"
os.makedirs(output_dir, exist_ok=True)

# Prepare output path

output_file = os.path.join(output_dir, "personas.jsonl")

# Generate personas and write to file
with open(output_file, "w") as f:
    for attr, values in attributes.items():
        for val in values:
            persona = {
                attr: val,
                "id": str(val).replace(" ", "_").lower()
            }
            f.write(json.dumps(persona) + "\n")

print(f"All single-attribute personas written to {output_dir}")
