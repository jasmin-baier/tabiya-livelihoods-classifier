import ast

# Assume 'data_string' is your input from the previous example

try:
    # 1. Parse the string
    parsed_data = ast.literal_eval(data_string)
    all_retrieved_lists = [item['retrieved'] for item in parsed_data]

    # 2. Flatten the list of lists into a single list
    all_skills_flat = [skill for sublist in all_retrieved_lists for skill in sublist]

    # 3. Normalize and Deduplicate in one step
    # - skill.lower().strip() cleans the string
    # - Using a set {} automatically handles duplicates
    # - Convert the set back to a list to finish
    unique_cleaned_skills = list({skill.lower().strip() for skill in all_skills_flat})

    # Sort the list alphabetically for clean presentation
    unique_cleaned_skills.sort()

    print(f"Found {len(unique_cleaned_skills)} unique skills.")
    print("Here are the first 10:")
    print(unique_cleaned_skills[:10])


except (ValueError, SyntaxError) as e:
    print(f"Error processing string: {e}")