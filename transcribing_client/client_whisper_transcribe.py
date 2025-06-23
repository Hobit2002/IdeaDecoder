import whisper
from whisper_client_tools import log_message
import time
import numpy as np
import torch
from specialist_decoder import WhisperSpecialistDecoder
from tqdm import tqdm

AUDIO_CUMULATOR = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = -0.6
VOCABULARY = "Default"

# Load the Whisper model
model = whisper.load_model("base")
checkpoint = torch.load(f"base_finetuned.ckpt",  map_location=DEVICE)
options = whisper.DecodingOptions(language="cs", without_timestamps=True, fp16=False)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k,v in state_dict.items():
    if "model." in k:new_state_dict[k.replace("model.","")] = v
model.load_state_dict(new_state_dict)
tokenizer = whisper.tokenizer.get_tokenizer(True, language="cs", task = options.task) 
# Prepare specialist deocder
SPECIALIZED_DECODERS = {}


def change_vocabulary(vocabulary):
    global VOCABULARY
    VOCABULARY = vocabulary

def default_transcribe(audio_inp):
    log_message(f"Transcribing {len(AUDIO_CUMULATOR + audio_inp)} words")
    start_time = time.time()
    valid_audio = []
    for counter, audio in AUDIO_CUMULATOR + audio_inp:
        valid_audio.append(audio)
    if not valid_audio: return None

    audio_merged = np.concatenate(valid_audio)

    if len(audio_merged.shape) > 1:
        audio_merged = np.mean(audio_merged, axis=1)

    audio = whisper.pad_or_trim(audio_merged)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    result = whisper.decode(model, mel, options)

    log_message("Vanilla transcription time:", time.time() - start_time)
    logprob = result.avg_logprob
    print("Raw transcription:",result.text,"......",logprob)
    if logprob > THRESHOLD:
        text = result.text.replace("Zaměříme","").strip()
        if text == "Z": text = ""
        elif len(AUDIO_CUMULATOR): text = " ".join(text.split()[1:])
        AUDIO_CUMULATOR.clear() 
        if text.strip():
            AUDIO_CUMULATOR.extend(audio_inp)
            return text

def custom_transcribe(audio_inp, vocabulary):
    audio_inp = audio_inp[0]
    if len(audio_inp) != 2:
        print("Received: ", audio_inp," => skipping")
        return None 
    _, audio = audio_inp
    start_time = time.time()
    if vocabulary not in SPECIALIZED_DECODERS.keys():
        SPECIALIZED_DECODERS[vocabulary] = WhisperSpecialistDecoder(vocabulary.replace(".txt",""),tokenizer, model)
    
    specialized_decoder = SPECIALIZED_DECODERS[vocabulary]
    res_list = specialized_decoder.decode_beam(audio,2,1)[0] 
    word,logprob = res_list

    log_message("Vanilla transcription time:", time.time() - start_time)
    print("Raw transcription:",word,"......",logprob)
    if logprob > THRESHOLD:
        return word

def transcribe_audio(audio_inp):
    if VOCABULARY == "Default": return default_transcribe(audio_inp)
    else: return custom_transcribe(audio_inp, VOCABULARY)