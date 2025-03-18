import asyncio
import websockets
import json
import pyaudio

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

# Queue to hold audio data
audio_queue = asyncio.Queue()

# Global event loop for use in callback
main_event_loop = None


def audio_callback(in_data, frame_count, time_info, status):
    """Callback function for non-blocking audio streaming."""
    try:
        # Use the global event loop explicitly
        if main_event_loop is not None:
            asyncio.run_coroutine_threadsafe(audio_queue.put(in_data), main_event_loop)
    except RuntimeError as e:
        print(f"Error in callback: {e}")
    return (None, pyaudio.paContinue)


async def record_and_send(ws):
    try:
        while True:
            audio_data = await audio_queue.get()
            if ws.open:
                await ws.send(audio_data)
    except asyncio.CancelledError:
        print("Audio recording stopped.")
    except Exception as e:
        print(f"Error while recording: {e}")


async def receive_messages(ws):
    try:
        async for message in ws:
            print("Received message:", message)
            try:
                res_json = json.loads(message)
                if res_json.get("code") == 0:

                    print("Transcription:", res_json.get("data", "No speech recognized"))

            except json.JSONDecodeError:
                print("Failed to parse response:", message)
    except Exception as e:
        print(f"Error while receiving: {e}")


async def start_recording(lang="auto", sv=0):
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()  # Save the main event loop
    #替换为你的启动地址，默认无证书启动ws:,有证书启动改为wss:
    url = f"ws://127.0.0.1:8888/ws/transcribe?lang={lang}&sv={sv}"
    print(f"Connecting to {url}...")

    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"设备 {i}: {info['name']}, 最大输入通道数: {info['maxInputChannels']},采样率RATE: {info['defaultSampleRate']}")
    index = int(input("设备id:"))
    audio_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=index,
        stream_callback=audio_callback,
    )

    try:
        async with websockets.connect(url) as ws:
            print("WebSocket connection established.")
            record_task = asyncio.create_task(record_and_send(ws))
            receive_task = asyncio.create_task(receive_messages(ws))

            await asyncio.gather(record_task, receive_task)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Cleaning up resources...")
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()


if __name__ == "__main__":
    lang = input("Enter language code (default: auto): ") or "auto"
    sv = input("Enable speaker verification? (1 for Yes, 0 for No): ") or "0"

    # Run asyncio event loop in the main thread
    asyncio.run(start_recording(lang=lang, sv=int(sv)))
