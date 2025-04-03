import pyaudio

MIC_DEVICE_INDEX = 1  # Índice de tu micrófono MacBook Air (verificado)

p = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("🎙️ Probando PyAudio, captura por 3 segundos...")
frames = []

for i in range(0, int(RATE / CHUNK * 3)):
    data = stream.read(CHUNK)
    frames.append(data)
    print(".", end="", flush=True)

print("\n✅ Captura completada exitosamente.")

stream.stop_stream()
stream.close()
p.terminate()
