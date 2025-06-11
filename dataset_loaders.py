import time
import traceback
from datasets import load_dataset
import json
import random
import gzip
import time
import traceback
from chunking import chunk_with_metadata # Import the chunking function
from typing import Optional

GOOGLE_NQ_TRAIN_FILEPATH = "v1.0-simplified_simplified-nq-train.jsonl.gz"
GOOGLE_NQ_DEV_FILEPATH = "v1.0-simplified_simplified-nq-train.jsonl.gz"

def load_nq_dataset(file_path):
    """
    Load the Natural Questions dataset from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file containing the NQ dataset
        
    Returns:
        list: List of examples from the dataset
    """
    examples = []
    try:
        print(f"Loading dataset from {file_path}...")
        num_examples = 0
        with gzip.open(file_path, "rt", encoding='utf-8') as f:
            print(f"Reading file: {file_path}")
            for line in f:
                example = json.loads(line.strip())

                # Only include entries with 'long_answer' key
                if ('annotations' not in example or
                     'long_answer' not in example['annotations'][0] or 
                     'start_token' not in example['annotations'][0]['long_answer'] or
                     example['annotations'][0]['long_answer']['start_token'] < 0):
                    continue

                num_examples += 1
                if num_examples % 10000 == 0:
                    print(f"Loaded {num_examples} examples so far...")
                examples.append(example)
        print(f"Successfully loaded {len(examples)} examples from dataset.")
        return examples
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File contains invalid JSON format.")
        return []
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return []

# --- Data Loading Function (Unchanged) ---
def load_wikipedia_data(num_original_docs_to_load):
    """
    Loads the specified number of rows from a Hugging Face dataset stream.
    Extracts the 'text' field.
    """
    DATASET_PATH = "wikipedia"  # Use the standard Wikipedia dataset
    DATASET_NAME = "20220301.en"  # Explicitly get English Wikipedia

    print(f"Attempting to load and chunk first {num_original_docs_to_load:,} original documents from '{DATASET_PATH}'...")
    chunked_texts = []
    original_texts_count = 0

    try:
        dataset_stream = load_dataset(DATASET_PATH, DATASET_NAME, language="en", streaming=True, split='train')
        start_time = time.time()
        processed_original_docs = 0
        limited_stream = dataset_stream.take(num_original_docs_to_load)

        for row in limited_stream:
            if row and 'text' in row and isinstance(row['text'], str):
                content = row['text'].strip()
                if content:
                    #chunked_texts.append(content)
                    original_texts_count += 1
                    # Apply chunking to this text
                    chunks = chunk_with_metadata(
                        content, 
                        source=f"{DATASET_PATH}/{DATASET_NAME}:doc_id_{processed_original_docs}",
                    )
                    # Extract just the text for simplicity
                    chunked_texts.extend([chunk["text"] for chunk in chunks])

            processed_original_docs += 1
            if processed_original_docs % (num_original_docs_to_load // 20 or 1) == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {processed_original_docs: ,}/{num_original_docs_to_load:,} original docs... ({elapsed:.1f}s elapsed)")

        end_time = time.time()
        print(f"Processed {original_texts_count:,} original documents.")
        print(f"Resulted in {len(chunked_texts):,} chunks.")
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")
        
        return chunked_texts
    except Exception as e:
        print(f"Error loading dataset '{DATASET_NAME}': {e}")
        return None

# --- Data Loading Function (Google NQ) ---
def load_googlenq_data(num_original_docs_to_load, batch_size):
    """
    Loads the specified number of rows from a local Google NQ dataset file.
    Extracts Question, Long Answer, and Short Answers and formats them
    as a single text document per example. Does NOT perform further chunking.
    """

    def extract_answer_text(doc_tokens, start_token, end_token):
        """Helper to extract a text span from tokenized document text."""
        return " ".join(doc_tokens[start_token:end_token])

    print(f"Attempting to load and process first {num_original_docs_to_load:,} Q&A sets from '{GOOGLE_NQ_TRAIN_FILEPATH}'...")
    original_texts_count = 0
    current_batch = []

    try:
        start_time = time.time()
        processed_original_docs = 0

        with gzip.open(GOOGLE_NQ_TRAIN_FILEPATH, "rt", encoding='utf-8') as f:
            for line in f:
                if processed_original_docs >= num_original_docs_to_load:
                    print(f"Reached limit of {num_original_docs_to_load:,} documents.")
                    break # Stop when we reach the limit

                example = json.loads(line.strip())

                # Filter for entries with valid long answers
                if ('annotations' not in example or
                    not example['annotations'] or
                    'long_answer' not in example['annotations'][0] or
                    'start_token' not in example['annotations'][0]['long_answer'] or
                    example['annotations'][0]['long_answer']['start_token'] < 0):
                    continue

                # --- Extract Q&A components ---
                question = example.get("question_text", "N/A")
                doc_text = example.get("document_text", "")
                # We need tokens to use start/end indices. Split doc_text as per NQ example.
                # Note: NQ 'document_tokens' might be more precise if available.
                tokens = doc_text.split()

                annotation = example.get("annotations", [{}])[0]

                # Extract long answer
                long_answer_data = annotation.get("long_answer", {})
                long_start = long_answer_data.get("start_token", -1)
                long_end = long_answer_data.get("end_token", -1)
                long_ans_text = extract_answer_text(tokens, long_start, long_end) if long_start >= 0 and long_end > long_start else "N/A"

                # Extract short answers
                short_answers_data = annotation.get("short_answers", [])
                short_ans_texts = []
                for sa in short_answers_data:
                    sa_start = sa.get("start_token", -1)
                    sa_end = sa.get("end_token", -1)
                    if sa_start >= 0 and sa_end >= 0 and sa_end > sa_start:
                        short_ans_texts.append(extract_answer_text(tokens, sa_start, sa_end))

                # --- Format into a single document ---
                short_ans_str = ", ".join(short_ans_texts) if short_ans_texts else "N/A"
                #formatted_qa = f"Question: {question}\nLong Answer: {long_ans_text}\nShort Answers: {short_ans_str}"
                formatted_qa = f"Question: {question}\nShort Answers: {short_ans_str}"

                current_batch.append(formatted_qa) # <--- Add to current batch
                original_texts_count += 1
                processed_original_docs += 1

                if len(current_batch) >= batch_size: # <--- Check if batch is full
                    yield current_batch
                    current_batch = [] # <--- Reset batch

                # Provide progress updates
                if processed_original_docs % (num_original_docs_to_load // 20 or 1) == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {processed_original_docs: ,}/{num_original_docs_to_load:,} Q&A sets... ({elapsed:.1f}s elapsed)")

        # --- Yield any remaining items in the last batch ---
        if current_batch: # <--- Check for and yield the last incomplete batch
            yield current_batch

        end_time = time.time()
        print(f"Finished processing. Total Q&A sets processed: {original_texts_count:,}.")
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    except FileNotFoundError:
        print(f"Error: NQ File '{GOOGLE_NQ_TRAIN_FILEPATH}' not found. Please check the path.")
    except json.JSONDecodeError as e:
        print(f"Error: NQ File contains invalid JSON format near doc {processed_original_docs}. Error: {e}")
    except Exception as e:
        print(f"Error loading NQ dataset '{GOOGLE_NQ_TRAIN_FILEPATH}': {e}")
        traceback.print_exc()

def get_googlenq_dev_questions(n_questions: int):
    """
    Loads a specified number of questions from a Google NQ dataset file.

    Args:
        file_path (str): The full path to the .jsonl.gz dataset file.
        num_questions_to_get (int): The desired number of questions to retrieve.

    Returns:
        list: A list of question strings. Returns an empty list if
              the file is not found or an error occurs.
    """
    questions = []
    print(f"Attempting to load {n_questions} questions from '{GOOGLE_NQ_DEV_FILEPATH}'...")

    try:
        with gzip.open(GOOGLE_NQ_DEV_FILEPATH, "rt", encoding='utf-8') as f:
            for line in f:
                # Stop when we have collected the desired number of questions
                if len(questions) >= n_questions:
                    break

                try:
                    # Load the JSON object from the line
                    example = json.loads(line.strip())

                    # Extract the question text
                    question_text = example.get("question_text")

                    # Add the question if it exists
                    if question_text:
                        questions.append(question_text)

                except json.JSONDecodeError:
                    print(f"Warning: Skipping a line due to JSON decoding error.")
                    continue # Move to the next line if one is malformed

        print(f"Successfully loaded {len(questions)} questions.")
        return questions

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

def load_pubmed_data(docs_to_load: Optional[int] = None, batch_size=500, skip_n_docs: int = 0):
    """
    Loads the MedRAG/pubmed dataset from Hugging Face using streaming.
    Skips a specified number of documents before starting to load.
    If docs_to_load is None, it attempts to load all documents.
    It extracts title and abstract, and yields them in batches.
    Prints progress, including total count if available.

    Args:
        batch_size (int): The size of each batch to yield.
        docs_to_load (int, optional): The target number of documents to load.
                                                    If None or < 0, attempts to load all.
                                                    Defaults to None.

    Yields:
        list: A list of strings, where each string is a formatted document
              (Title + Abstract). Returns an empty list if an error occurs.
    """
    DATASET_PATH = "MedRAG/pubmed"
    DATASET_SPLIT = 'train'

    load_all = (docs_to_load is None or docs_to_load < 0)

    # Determine how many docs we *intend* to process for progress reporting
    if load_all:
        docs_to_process_display = "all available"
        limit_for_progress = None # We don't know the limit
    else:
        docs_to_process_display = f"{docs_to_load:,}"
        limit_for_progress = docs_to_load

    print(f"\nAttempting to load {docs_to_process_display} documents from '{DATASET_PATH}'...")

    current_batch = []
    processed_count = 0
    total_length = 0
    start_time = time.time()

    try:
        # Load the dataset stream
        dataset_stream = load_dataset(DATASET_PATH, streaming=True, split=DATASET_SPLIT)

        # Apply the .skip() method if necessary
        if skip_n_docs > 0:
            print(f"  Skipping {skip_n_docs:,} documents...")
            dataset_stream = dataset_stream.skip(skip_n_docs)
            print("  Skipping complete.")

        # Apply the .take() method to the (potentially skipped) stream
        processing_stream = dataset_stream if load_all else dataset_stream.take(docs_to_load)

        # Iterate through the dataset stream
        for example in processing_stream:
            processed_count += 1

            title = example.get("title", "").strip()
            abstract = example.get("content", "").strip()

            if title or abstract:
                document_text = f"Title: {title}\nAbstract: {abstract}"
                total_length += len(document_text)
                current_batch.append(document_text)
            else:
                continue # Skip empty documents

            # Yield a batch when it's full
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

            # --- Print Progress ---
            print_interval = 10000 # Print every 10000 if total is unknown
            should_print = False

            if limit_for_progress:
                # Calculate interval for ~50 updates or use a fallback
                progress_interval = limit_for_progress // 50 or 10000
                if processed_count % progress_interval == 0 or processed_count == limit_for_progress:
                    should_print = True
            elif processed_count % print_interval == 0:
                should_print = True

            if should_print:
                elapsed = time.time() - start_time
                progress_msg = f"  Fetched {processed_count:,}"
                if limit_for_progress:
                    progress_msg += f"/{limit_for_progress:,}"
                progress_msg += f" documents... ({elapsed:.1f}s elapsed)"
                print(progress_msg)

        # Yield the last remaining batch if it's not empty
        if current_batch:
            yield current_batch

        end_time = time.time()
        print(f"\nFinished loading. Fetched a total of {processed_count:,} documents in {end_time - start_time:.2f} seconds.")
        avg_length = total_length / processed_count if processed_count > 0 else 0
        print(f"Average document length: {avg_length:.2f} characters.")

    except Exception as e:
        print(f"\nAn error occurred while loading/processing the PubMed dataset: {e}")
        traceback.print_exc()
        yield []