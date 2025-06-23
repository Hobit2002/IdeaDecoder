import os
import torch
import openai
import numpy as np
from tqdm import tqdm 
from transformers import AutoModel, AutoTokenizer
import ufal.morphodita
from unidecode import unidecode
import json
from client_whisper_transcribe import WhisperSpecialistDecoder, model as whisper_model, tokenizer as whisper_tokenizer

# Configuration and File PathsQUERY_FILE = "vocabularies/query.txt"
ALL_CZECH_WORDS_FILE = "vocabularies/all_czech_words.txt" 
MOST_COMMON_CZECH_WORDS_FILE = "vocabularies/most_common_czech_words.txt"
OPENAI_API_KEY = open(os.path.join(os.getcwd(),"openai_api_key.txt")).read()
client = openai.OpenAI(api_key=OPENAI_API_KEY)
# Load the Czech MorphoDiTa lemmatizer model
tagger = ufal.morphodita.Tagger.load("vocabularies/czech-morfflex2.0-pdtc1.0-220710-pos_only.tagger")


# Similarity threshold
SIMILARITY_THRESHOLD = 0.81

# Placeholder for GPT API Call
def get_words_from_gpt(topic_description: str, num_words: int = 500):

    prompt = f"The user wants {num_words} Czech words on the following topic: {topic_description}. Generate them. Return only a comma separated list of words, do not return anything else."

    response = client.chat.completions.create( 
        model="gpt-4o-mini",  # Using GPT-4 model for conversation
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.replace(",","").split()

    # Get also a name

    prompt = f"Summarize the following topic in two words (come up with a 2-word headline for it): {topic_description}. Return just the summarization, write nothing else"

    response = client.chat.completions.create( 
        model="gpt-4o-mini",  # Using GPT-4 model for conversation
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    headline = response.choices[0].message.content
    
    return text, headline

# Embedding Generation Placeholder
def get_word_embedding(word: str, tokenizer, model) -> np.ndarray:
    """
    Computes the embedding for a single word using a pre-trained model.
    Adjust this function based on your chosen embedding model (FastText, Word2Vec, mBERT, etc.).
    """
    if not word: # Handle empty strings
        return np.zeros(tokenizer.model_max_length) # Or some other default

    # For transformer models (like mBERT, XLM-R)
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad(): # Use torch.no_grad() or tf.GradientTape for inference
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


# Utility Functions
def load_words_from_file(filepath: str) -> list[str]:
    """Loads words from a text file, one word per line, lowercases and deduplicates."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        words = f.read().split(",")
    # Deduplicate
    return list(set(words))

# Lemmatize Czech
def lemmatize_czech(word: str) -> str:
    """
    Takes a Czech word as input and returns its lemma.

    Args:
        word (str): The input word in Czech.

    Returns:
        str: The lemmatized form of the word.
    """
    if not tagger:
        raise RuntimeError("Tagger model failed to load.")

    # Create a tokenizer and tagger
    tokenizer = tagger.newTokenizer()
    if not tokenizer:
        raise RuntimeError("Failed to create tokenizer.")

    # Tokenize input
    tokenizer.setText(word)
    forms = ufal.morphodita.Forms()
    lemmas = ufal.morphodita.TaggedLemmas()
    tokens = ufal.morphodita.TokenRanges()

    while tokenizer.nextSentence(forms,tokens):
        tagger.tag(forms, lemmas)

    # Return the first lemma (assuming one word input)
    return lemmas[0].lemma if lemmas else word 


def main():
    # 1. Load the coarse description
    try:
        with open('whisper_client.log', 'r', encoding='utf-8') as f:
            log_lines = f.readlines()[::-1]
            for line in log_lines:
                if "create_vocabulary:" in line:
                    simplified_line = line.replace("create_vocabulary:","&")
                    topic_description = simplified_line[simplified_line.index("&") + 1:]
                    break
    except FileNotFoundError:
        print(f"Error: Log file not found. Please create it.")
        return

    if not topic_description:
        print("Query file is empty. Please provide a topic description.")
        return

    print(f"Topic Description: '{topic_description}'")

    # 2. Ask GPT to generate a list of relevant words
    print("\nStep 2: Generating field-specific words using GPT...")
    field_specific_words_raw, headline = get_words_from_gpt(topic_description, num_words=500)
    # Basic cleaning: remove empty strings, convert to lowercase, deduplicate
    field_specific_words = list(set([w.strip().lower() for w in field_specific_words_raw if w.strip()]))
    print(f"Generated {len(field_specific_words)} unique field-specific words.")
    # print("Sample GPT words:", field_specific_words[:10])

    # 3. Load common Czech words and all Czech words
    print("\nStep 3: Loading common and all Czech words...")
    all_czech_words = load_words_from_file(ALL_CZECH_WORDS_FILE)
    # Shuffle to get words from all over the alphabet even testing only a sample
    np.random.shuffle(all_czech_words) 
    if not all_czech_words:
        print("No words loaded from all_czech_words.txt. Exiting.")
        return
    print(f"Loaded {len(all_czech_words)} unique words from '{ALL_CZECH_WORDS_FILE}'.")

    most_common_czech_words = load_words_from_file(MOST_COMMON_CZECH_WORDS_FILE)
    if not most_common_czech_words:
        print("No words loaded from most_common_czech_words.txt.")
    print(f"Loaded {len(most_common_czech_words)} unique words from '{MOST_COMMON_CZECH_WORDS_FILE}'.")

    # Combine GPT words with common words, ensuring no duplicates.
    # We will filter against 'all_czech_words' later.
    initial_candidate_words = list(set(field_specific_words + most_common_czech_words))
    lemmatized_initial_candidate_words = [lemmatize_czech(word) for word in initial_candidate_words]
    print(f"Initial candidate list for embedding: {len(initial_candidate_words)} words.")

    # 4. Prepare Embeddings for all words
    print("\nStep 4: Computing/Loading embeddings for all words. This can take a while...")

    # Example for Transformer (mBERT/XLM-R):
    tokenizer = AutoTokenizer.from_pretrained("Seznam/simcse-small-e-czech")
    model = AutoModel.from_pretrained("Seznam/simcse-small-e-czech")
    model.eval() # Set model to evaluation mode
    print("Loaded Seznam/simcse-small-e-czech model.")

    # Collect embeddings for all words in `initial_candidate_words`
    word_embeddings = {}
    for word in tqdm(initial_candidate_words, desc="Generating embeddings"):
        word_embeddings[word] = get_word_embedding(word, tokenizer, model)

    field_specific_embeddings = {word: word_embeddings[word] for word in field_specific_words if word in word_embeddings}
    
    # 5. Compute similarity and select words
    print("\nStep 5: Computing similarities and selecting vocabulary...")
    selected_vocabulary = set()

    # Add all field-specific words directly (as they are the core)
    selected_vocabulary.update(field_specific_words)
    print(f"Initially selected {len(selected_vocabulary)} words from GPT list.")

    # Convert field-specific embeddings to a matrix for efficient similarity search
    field_vectors = np.array(list(field_specific_embeddings.values()))
    field_norms = np.linalg.norm(field_vectors,2,axis=1)

    for czech_word in tqdm(all_czech_words[:488], desc="Comparing all Czech words"):
        if lemmatize_czech(czech_word) in lemmatized_initial_candidate_words: selected_vocabulary.add(czech_word)
        else:
            czech_embedding = get_word_embedding(czech_word, tokenizer, model)
            czech_embedding_norm = np.linalg.norm(czech_embedding)

            # Compute similarity to all field-specific words
            dot_products = field_vectors @ czech_embedding 
            cosine_similarities = dot_products / (field_norms * czech_embedding_norm)

            max_similarity = np.max(cosine_similarities)

            if max_similarity >= SIMILARITY_THRESHOLD:
                selected_vocabulary.add(czech_word)

    print(f"\nFinal selected vocabulary size: {len(selected_vocabulary)} words.")

    # Save the selected vocabulary to a file
    vocab_id = f"voc_{unidecode(field_specific_words[0])}{np.random.randint(0,100)}" 
    output_filename = f"vocabularies/{vocab_id}.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for word in sorted(list(selected_vocabulary)):
            f.write(word + '\n')
    print(f"Selected vocabulary saved to '{output_filename}'")

    # Prepare the decoder
    WhisperSpecialistDecoder(vocab_id, whisper_tokenizer, whisper_model)

    # Update vocabulary list
    voc_list = json.load(open("vocabularies/vocabularies.json"))
    voc_list.append([headline, vocab_id])
    open("vocabularies/vocabularies.json","w").write(json.dumps(voc_list))


if __name__ == "__main__":
    main()