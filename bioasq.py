# dataset_parsers.py
import json
import os
from typing import List, Tuple, Optional

# These questions were fetched from https://participants-area.bioasq.org/datasets/
def load_bioasq_yesno_questions(bioasq_dir_path: str = "BIOASQ") -> List[Tuple[str, str]]:
    """
    Loads 'yesno' type questions and their exact answers from JSON files
    within subdirectories of the specified BioASQ directory.

    The expected structure is:
    BIOASQ_DIR/
        ├── subdir1/
        │   ├── file1.json
        │   └── file2.json
        ├── subdir2/
        │   └── file3.json
        ... (up to 5 subdirectories or as found)

    Each JSON file is expected to have a "questions" key, which is a list
    of question objects. Each question object should have:
    - "body": The question text.
    - "type": The type of question (e.g., "yesno", "factoid", "list").
    - "exact_answer": The ground truth answer (e.g., "yes", "no").

    Args:
        bioasq_dir_path: The file path to the main BIOASQ directory.

    Returns:
        A list of tuples, where each tuple is (question_body, exact_answer).
        The exact_answer will be normalized to lowercase 'yes' or 'no'.
        Returns an empty list if the directory doesn't exist or no suitable
        questions are found.
    """
    yes_no_questions: List[Tuple[str, str]] = []

    if not os.path.isdir(bioasq_dir_path):
        print(f"Error: Directory not found at {bioasq_dir_path}")
        return yes_no_questions

    print(f"Starting to load BioASQ yes/no questions from: {bioasq_dir_path}")
    subdirectories_processed = 0

    # Iterate over items in the main BioASQ directory
    for subdir_name in os.listdir(bioasq_dir_path):
        subdir_path = os.path.join(bioasq_dir_path, subdir_name)
        if os.path.isdir(subdir_path):
            subdirectories_processed += 1
            print(f"  Processing subdirectory: {subdir_name}")
            files_in_subdir_processed = 0
            for filename in os.listdir(subdir_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        questions_list = data.get("questions")
                        if not isinstance(questions_list, list):
                            # print(f"    Warning: 'questions' key not found or not a list in {filename}. Skipping.")
                            continue
                        
                        files_in_subdir_processed +=1
                        questions_found_in_file = 0
                        for question_obj in questions_list:
                            if not isinstance(question_obj, dict):
                                # print(f"    Warning: Question object is not a dictionary in {filename}. Skipping entry.")
                                continue

                            q_type = question_obj.get("type")
                            q_body = question_obj.get("body")
                            q_exact_answer = question_obj.get("exact_answer")

                            if q_type == "yesno":
                                if isinstance(q_body, str) and isinstance(q_exact_answer, str):
                                    normalized_answer = q_exact_answer.strip().lower()
                                    if normalized_answer in ["yes", "no"]:
                                        yes_no_questions.append((q_body, normalized_answer))
                                        questions_found_in_file +=1
                                    # else:
                                        # print(f"      Info: 'yesno' question with unexpected exact_answer '{q_exact_answer}' in {filename}. Question: {q_body[:50]}...")
                                # else:
                                    # print(f"      Warning: 'yesno' question with missing body or exact_answer in {filename}. Skipping.")
                        # if questions_found_in_file > 0:
                            # print(f"    Found {questions_found_in_file} 'yesno' questions in {filename}")

                    except json.JSONDecodeError:
                        print(f"    Error: Could not decode JSON from {file_path}. Skipping.")
                    except Exception as e:
                        print(f"    Error processing file {file_path}: {e}")
            # if files_in_subdir_processed > 0:
                # print(f"  Finished processing {files_in_subdir_processed} JSON files in {subdir_name}.")
            # else:
                # print(f"  No JSON files found or processed in {subdir_name}.")

    print(f"Finished processing BioASQ directory. Found {len(yes_no_questions)} 'yesno' questions in total from {subdirectories_processed} subdirectories.")
    return yes_no_questions

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy BIOASQ directory structure for testing
    print("Creating dummy BIOASQ directory for testing...")
    bioasq_main_dir = "BIOASQ"
    
    loaded_questions = load_bioasq_yesno_questions(bioasq_main_dir)
    
    if loaded_questions:
        print(f"\nSuccessfully loaded {len(loaded_questions)} yes/no questions:")
        for i, (q, a) in enumerate(loaded_questions[:5]): # Print first 5
            print(f"  {i+1}. Q: {q} - A: {a}")
        if len(loaded_questions) > 5:
            print(f"  ... and {len(loaded_questions) - 5} more.")
    else:
        print("No yes/no questions were loaded. Check the path and file structure.")
