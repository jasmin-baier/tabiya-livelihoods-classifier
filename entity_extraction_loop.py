import csv
from inference.linker import EntityLinker

# Initialize the entity linker
pipeline = EntityLinker(k=5)

# Sample dataset (Replace with actual data)
text_snippets = [
    {"id": 1, "text": "We are looking for a Head Chef who can plan menus."},
    {"id": 2, "text": "Software engineers must have strong Python skills."},
    {"id": 3, "text": "Looking for a marketing specialist with SEO experience."}
]

# Open CSV file for writing results
with open("extracted_entities.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Text ID", "Paragraph", "Extracted Entities"])  # Header

    # Process each text snippet
    for snippet in text_snippets:
        extracted = pipeline(snippet["text"])
        writer.writerow([snippet["id"], snippet["text"], extracted])

print("Extraction completed! Results saved to `extracted_entities.csv` 🎉")