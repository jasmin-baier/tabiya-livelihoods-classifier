# TODO: harambee_id is missing for now since it is all ajira people
# TODO: play with this file and opportunity database to decide threshold for match

import json
from pathlib import Path
from collections import defaultdict

def restructure_skills_data(input_path, output_path):
    """
    Reads a JSON file with skills data, groups it by conversation_id,
    and writes the restructured data to a new JSON file.

    This version is updated to be more robust against inconsistencies in the
    source data, such as fields being single strings instead of lists.

    Args:
        input_path (Path): The path to the input JSON file.
        output_path (Path): The path where the output JSON file will be saved.
    """
    try:
        # Load the source JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Use defaultdict to easily group skills by conversation_id
        grouped_skills = defaultdict(list)

        # Iterate over each record in the source data
        for record in data:
            conversation_id = record.get("conversation_id")
            if not conversation_id:
                # If there's no conversation_id, we can't group the data, so we skip this record.
                continue

            # --- Process UUIDs ---
            # This block handles cases where 'uuid' might be a single string or a list of strings.
            uuids = record.get("origin_uuid")
            uuid_list = []
            if isinstance(uuids, list):
                uuid_list = uuids
            elif isinstance(uuids, str) and uuids:
                uuid_list = [uuids]  # Treat a single string as a list with one item
            else:
                # If no valid UUIDs are found, skip to the next record.
                continue

            # --- Process Labels ---
            # This block handles various possible structures for the 'preferred_label'.
            preferred_label_obj = record.get("preferred_label")
            primary_label = None
            if isinstance(preferred_label_obj, dict):
                # Handles the case where the label is in a dictionary, e.g., {"en": ["label1", "label2"]}
                labels = preferred_label_obj.get("en")
                if isinstance(labels, list) and labels:
                    primary_label = labels[0] # Take the first label from the list
                elif isinstance(labels, str) and labels:
                    primary_label = labels
            elif isinstance(preferred_label_obj, str) and preferred_label_obj:
                # Handles the case where the label is just a simple string.
                primary_label = preferred_label_obj

            if not primary_label:
                # If we couldn't find a label, we can't create a useful skill entry.
                continue

            # --- Combine and Group ---
            # Create a skill entry for each UUID found and add it to the group.
            for single_uuid in uuid_list:
                if single_uuid:  # Ensure the uuid is not an empty string
                    skill_info = {
                        "preferred_label": primary_label,
                        "uuid": single_uuid
                    }
                    grouped_skills[conversation_id].append(skill_info)

        # Restructure the grouped data into the desired final format
        output_data = []
        for conversation_id, skills in grouped_skills.items():
            if skills:  # Only include users who have one or more skills
                output_data.append({
                    "compass_id": conversation_id,
                    "skills": skills
                })

        # Write the restructured data to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully restructured data and saved to {output_path}")
        if not output_data:
            print("Warning: The output file is empty. This may be due to the input file having no processable records.")


    except FileNotFoundError:
        print(f"Error: The file at {input_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {input_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Define the base directory and data directory
    base_dir = Path.home() / "OneDrive - Nexus365/Documents/PhD - Oxford BSG/Paper writing projects/Ongoing/Compass"
    data_dir = base_dir / "data"

    # Define the input and output file paths
    jobseekerskills_json_path = data_dir / "pre_study/ajira-discovered_skills.json"
    output_json_path = data_dir / "pre_study/pilot_jobseeker_database.json"

    # Create the output directory if it doesn't exist to prevent errors
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the restructuring function
    restructure_skills_data(jobseekerskills_json_path, output_json_path)
