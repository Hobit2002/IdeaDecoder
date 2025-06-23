import csv, torch, whisper, os, json
from tqdm import tqdm

class WhisperSpecialistDecoder:

    def __init__(self, vocabulary_source, tokenizer, model, penalty_strength = 0.3):
        self.tokenizer = tokenizer
        self.model = model
        self.transcriptions = []
        self.first_token_dict = {}
        self.next_token_dict = {}
        self.probs_when_unrelated = []
        self.penalty_strength = penalty_strength
        
        # Load CSV and process transcriptions
        self._load_vocab(vocabulary_source)
        self._build_token_dict(first=True)
        self._build_token_dict(first=False)
    
    def _load_vocab(self, vocabulary_source):
        word_list = open(f"vocabularies/{vocabulary_source}.txt", encoding="utf-8").read().replace("\n",",").split(",")
        word_list = list(set([word.capitalize() for word in tqdm(word_list) if word]))[1:]
        self.transcriptions = word_list
        self.vocabulary_source = vocabulary_source
    
    def _build_token_dict(self, first = True):
        """Encodes transcriptions and builds a nested token dictionary."""
        cache_name = f"vocabularies/whisper_specialist_decoder_{self.vocabulary_source}{'_first' if first else ''}.json"
        if os.path.exists(cache_name):
            token_dict = json.load(open(cache_name))
            print("Loading from saved")
            if first: self.first_token_dict = token_dict
            else: self.next_token_dict = token_dict
        else:
            for text in self.transcriptions if len(self.transcriptions) < 200 else tqdm(self.transcriptions):
                if not first: text = " " + text.lower()
                tokens = self.tokenizer.encode(f"{text}<|endoftext|>", allowed_special = "all")
                current_dict = self.first_token_dict if first else self.next_token_dict 
                for token in tokens:
                    if token in current_dict.keys():
                        current_dict = current_dict[token]
                    else:
                        current_dict[token] = {}
                        current_dict = current_dict[token]
            # Save
            with open(cache_name, "w") as f:
                f.write(json.dumps(self.first_token_dict if first else self.next_token_dict))

    # OPTIONAL: add your own decoding strategies
    def decode_beam(self, audio_segment, beam_size=5, n_results = 3, previous_transcription = "", guided = True):
        """
        Performs beam search decoding with masked logits, batching all beam hypotheses at once.
        
        Args:
            audio_segment: Input audio waveform.
            beam_size: Number of hypotheses to maintain.

        Returns:
            Best decoded transcription.
        """
        assert beam_size >= n_results, "N results to be returned cannot exceed the beam size"
        with torch.inference_mode():
            # Prepare input
            audio = whisper.pad_or_trim(audio_segment)
            mel = whisper.log_mel_spectrogram(audio, n_mels=80)
            encoder_output = self.model.encoder(mel.unsqueeze(0) if len(mel.shape) == 2 else mel)

            # Initialize search
            initial_tokens = self.tokenizer.encode(f"<|startoftranscript|><|cs|><|transcribe|><|notimestamps|>{previous_transcription}", allowed_special="all")
            beam = [(initial_tokens, 0.0, self.next_token_dict if previous_transcription else self.first_token_dict)]  # (tokens, log_prob, current_dict)

            # Beam search loop
            for e in range(25):  # Max length of sequence
                new_beam = []
                # Remove finished hypotheses
                removed_hypotheses = []
                for b, el_tuple in enumerate(beam):
                    tokens, _, _ = el_tuple
                    if tokens[-1] == self.tokenizer.eot:
                        new_beam.append(el_tuple)
                        removed_hypotheses.append(b)
                for h_idx in removed_hypotheses[::-1]: del beam[h_idx] 
                if not len(beam):
                    beam = new_beam
                    break  # Stop early if all hypotheses are finished

                # Prepare batched inputs
                token_batch = [tokens for tokens, _, _ in beam]
                current_dicts = [current_dict for _, _, current_dict in beam]
                encoded_txt_tokens = torch.tensor(token_batch).long().to(encoder_output.device)

                # Decode batch
                logits = self.model.decoder(encoded_txt_tokens, encoder_output)[:, -1, :]  # Shape: (beam_size, vocab_size)

                if guided:
                    # Apply individual masks for each beam
                    vocab_size = logits.shape[-1]
                    masks = torch.full((len(beam), vocab_size), float('-inf'), device=logits.device)

                    for i, current_dict in enumerate(current_dicts):
                        allowed_indices = [int(cd) for cd in list(current_dict.keys())]
                        masks[i, allowed_indices] = 0  # Allow only valid tokens

                    logits = logits + masks  # Mask invalid tokens

                log_probs = torch.log_softmax(logits, dim=-1)  # Compute log probabilities

                # Get top-k candidates for each beam hypothesis
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                # Expand beam
                for i in range(len(beam)):
                    for j in range(beam_size):
                        new_token = topk_indices[i, j].item()
                        new_log_prob = beam[i][1] + topk_log_probs[i, j].item()
                        if len(beam[i][2].keys()) and isinstance(list(beam[i][2].keys())[0],str): new_token = str(new_token)
                        new_dict = beam[i][2].get(new_token, {})  # Navigate token tree
                        new_token = int(new_token)
                        new_beam.append((beam[i][0] + [new_token], new_log_prob, new_dict))

                # Keep top beam_size hypotheses
                beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]

            # Return best sequence
            best_tokens = [(beam_el[0],beam_el[1]) for beam_el in beam[:n_results+1]]  # Best scoring sequence
            return [(self.tokenizer.decode(tokens[len(initial_tokens):-1]).split()[0] if self.tokenizer.decode(tokens[len(initial_tokens):-1]).split() else '', log_prob) for tokens,log_prob in best_tokens]
