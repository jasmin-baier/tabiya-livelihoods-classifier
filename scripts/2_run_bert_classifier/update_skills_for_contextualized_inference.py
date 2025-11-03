"""
This script takes in the skills.csv in the format of Tabiya/ESCO taxonomies and transforms it into the format needed by the BERT classifier during inference
This should be run only whenever the taxonomy is updated
"""

# TODO this file currently cannot be used, because BERT classifier expects 13897 skills exactly AND may even be sensitive to order
# TODO Talk to Apostolos if/how we can make this more robust to taxonomy changes; i.e. whether whole BERT would need to be re-trained


import csv
import os

def transform_skills_csv(input_file, output_file):
    """
    Reads the input CSV, extracts the last UUID from UUIDHISTORY and the PREFERREDLABEL,
    and writes them to a new CSV file.

    Args:
        input_file (str): Path to the source CSV file (e.g., skills.csv from taxonomy)
        output_file (str): Path to the destination CSV file (e.g., skills.csv in inference files)
    """
    
    print(f"Starting transformation...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    try:
        with open(input_file, mode='r', encoding='utf-8', newline='') as infile:
            # Use csv.reader to handle quoted fields correctly
            reader = csv.reader(infile)
            
            try:
                # Read the header row to find column indices
                header = next(reader)
                
                # Find the index for 'UUIDHISTORY' and 'PREFERREDLABEL'
                try:
                    uuid_history_index = header.index("UUIDHISTORY")
                    label_index = header.index("PREFERREDLABEL")
                except ValueError as e:
                    print(f"Error: Missing required column in input file - {e}")
                    print("Please ensure 'UUIDHISTORY' and 'PREFERREDLABEL' columns exist.")
                    return

                # Open the output file for writing
                with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    
                    # Write the new header
                    writer.writerow(['uuid', 'skills'])
                    
                    processed_count = 0
                    skipped_count = 0
                    
                    # Process each row in the input file
                    for row in reader:
                        # Ensure row has enough columns
                        if len(row) > uuid_history_index and len(row) > label_index:
                            skill_label = row[label_index]
                            uuid_history_str = row[uuid_history_index]
                            
                            # Split the UUIDHISTORY string by newline, strip whitespace,
                            # and filter out any empty strings.
                            uuids = [u.strip() for u in uuid_history_str.split('\n') if u.strip()]
                            
                            if uuids:
                                # Get the last UUID from the filtered list
                                latest_uuid = uuids[-1]
                                writer.writerow([latest_uuid, skill_label])
                                processed_count += 1
                            else:
                                # Log if a row is skipped due to no valid UUID
                                # print(f"Warning: Skipping row with empty/invalid UUIDHISTORY for skill: {skill_label}")
                                skipped_count += 1
                        else:
                            # Log if row is malformed
                            # print(f"Warning: Skipping malformed row: {row}")
                            skipped_count += 1
            
            except StopIteration:
                print("Error: The input file is empty.")
                return

        print("\nTransformation complete.")
        print(f"Processed and wrote {processed_count} rows.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} rows due to missing UUIDs or malformed data.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define file paths based on the user's description
    input_path = os.path.join('.', 'taxonomy', 'skills.csv')
    output_path = os.path.join('.', 'inference', 'files', 'skills.csv')
    
    # Run the transformation
    transform_skills_csv(input_path, output_path)
