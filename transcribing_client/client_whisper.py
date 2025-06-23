import asyncio
import websockets
import whisper, torch
import numpy as np
import soundfile as sf
import json, time
import pickle, librosa, torch
from torch.nn.functional import softmax, sigmoid
from concurrent.futures import ThreadPoolExecutor
from speech_detection.cp_resnet import get_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import aiohttp
from client_whisper_transcribe import transcribe_audio, change_vocabulary
from whisper_client_tools import log_message, load_wave, compute_mel

idx = 0
speaker_id = None
TO_TRANSCRIBE = asyncio.Queue()
WORD_CUMULATOR = []
LAST_NON_SPEECH = False
SAMPLE_RATE = 16000
OVERLAP_TRIM = int(0.75 * SAMPLE_RATE)
PRE_SPEECH_PAD = int(1.3 * SAMPLE_RATE)
POST_SPEECH_PAD = SAMPLE_RATE
PREDICTION_THRESHOLD = 0.3
MAX_CUMULATED = 4
BLOCKED_COUNTERS = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load speech detection models
speech_detection_model = get_model(n_classes=1, in_channels=1)
state_dict = torch.load("speech_detection_model_2.pth", map_location=DEVICE)
speech_detection_model.load_state_dict(state_dict)
speech_detection_model.eval()

executor = ThreadPoolExecutor(max_workers=1)

# Audio Processing
async def process_and_filter(pcm_bytes, sample_rate=48000):
    global idx, LAST_NON_SPEECH, WORD_CUMULATOR

    # Extract counter (first 4 bytes)
    counter = int.from_bytes(pcm_bytes[:4], byteorder='little')

    # Extract and normalize audio
    raw_pcm = pcm_bytes[4:]
    # int16_array = np.frombuffer(raw_pcm, dtype=np.int16)
    # float32_array = int16_array.astype(np.float32) / 327678.0
    float32_array = np.frombuffer(pcm_bytes[4:], dtype=np.float32)
    
    # print(1e10 * (abs(float32_array) % 1e-10))

    # Resample to 16 kHz
    #plt.imsave(f"mels/{counter}_44.png", compute_mel(float32_array,sample_rate)[0,0])
    float32_array_resampled = librosa.resample(float32_array, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    #float32_array_resampled = float32_array

    mel_spec = compute_mel(float32_array_resampled)
    plt.imsave(f"mels/{counter}.png", mel_spec[0,0])
    sf.write(f"tmp_wav/{counter}.wav", float32_array_resampled, samplerate=SAMPLE_RATE)
    prediction = sigmoid(speech_detection_model(mel_spec))[0].item() #speaker_recognition_model.predict_proba(mel_spec.reshape(1, -1))[0][1]
    log_message(f"[#{counter}] Jasmi's speech probability:", prediction)
    idx += 1

    # Helper constants for trimming

    if prediction >= PREDICTION_THRESHOLD:
        # It's speech — append to cumulator
        WORD_CUMULATOR.append((counter, float32_array_resampled))

        # Enqueue for transcription
        # await TO_TRANSCRIBE.put((counter, float32_array_resampled))
        await TO_TRANSCRIBE.put((counter, WORD_CUMULATOR.copy()))
        WORD_CUMULATOR.clear()

# Transcription (runs separately, doesn't block socket)


# WebSocket Listener
async def websocket_listener(websocket):
    global speaker_id
    while True:
        msg = await websocket.recv()
        if isinstance(msg, bytes):
            await process_and_filter(msg)
        elif isinstance(msg, str):
            try:
                msg_object = json.loads(msg)
                if msg_object.get("action") == "speaker_id":
                    speaker_id = msg_object["speaker_id"]
                elif msg_object.get("action") == "log":
                    log_message(msg_object["msg"])
                elif msg_object.get("action") == "select_vocabulary":
                    log_message("Selecting vocabulary: ",msg_object["msg"])
                    change_vocabulary(msg_object["msg"])
            except Exception as e:
                log_message("Error parsing message:", e)

# Transcription loop
async def transcription_loop(websocket):
    loop = asyncio.get_running_loop()
    
    while True:
        counter, audio = await TO_TRANSCRIBE.get()
        if counter in BLOCKED_COUNTERS: continue

        # Offload the CPU-bound task to a thread
        audio = load_wave(f"tmp_wav/{counter}.wav")
        text = await loop.run_in_executor(executor, transcribe_audio, [(counter,audio)])
        if text:
            log_message(f"[#{counter}] Transcription:", text)

            await websocket.send(json.dumps({
                "action": "transcription",
                "text": text,
                "speaker_id": speaker_id,
                "counter": counter
            }))
        else:
            BLOCKED_COUNTERS.append(counter)

async def wait_for_server(max_retries=4, wait_seconds=30):
    for attempt in range(1, max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://jasmiapp.onrender.com/") as response:
                    log_message(f"Server responded with status: {response.status}")
                    if response.status == 200:
                        return True
        except Exception as e:
            log_message(f"Attempt {attempt} failed to reach server: {e}")

        if attempt < max_retries:
            log_message(f"Retrying in {wait_seconds} seconds...")
            await asyncio.sleep(wait_seconds)
    log_message("Failed to reach server after multiple attempts.")
    return False

async def keep_server_alive():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://jasmiapp.onrender.com/") as response:
                    log_message(f"Keep-alive ping status: {response.status}")
        except Exception as e:
            log_message("Keep-alive ping failed:", e)
        await asyncio.sleep(60 * 5)  # Every 5 minutes

async def main():
    if not await wait_for_server():
        return  # Abort if server unreachable

    uri = "wss://jasmiapp.onrender.com"
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
        log_message("Connected to server")
        await websocket.send(json.dumps({
            "action": "register_client",
            "role": "transcriber"
        }))

        await asyncio.gather(
            websocket_listener(websocket),
            transcription_loop(websocket),
            keep_server_alive()
        )

asyncio.run(main())



