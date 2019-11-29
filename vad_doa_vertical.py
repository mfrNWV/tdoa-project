
import maestro
import sys
import webrtcvad
import numpy as np
from mic_array_vertical import MicArray
from pixel_ring import pixel_ring

servo = maestro.Controller()
servo.setAccel(0,50)
servo.setSpeed(0,80)

RATE = 48000
CHANNELS = 4
VAD_FRAMES = 10     # ms
DOA_FRAMES = 100    # ms

def main():
    vad = webrtcvad.Vad(3)

    servodir = 0
    speech_count = 0
    chunks = []
    #doa_chunks = int(DOA_FRAMES / VAD_FRAMES)
    doa_chunks = 128

    try:
        with MicArray(RATE, CHANNELS, RATE * VAD_FRAMES /1000)  as mic:
            # chunck size com doa_chuncks para o caso sem vad
            for chunk in mic.read_chunks():
                # Use single channel audio to detect voice activity
                if vad.is_speech(chunk[0::CHANNELS].tobytes(), RATE):
                    speech_count += 1

                chunks.append(chunk)
                if len(chunks) == doa_chunks:
                    if speech_count > (doa_chunks / 2):
                        frames = np.concatenate(chunks)
                        horizontal, vertical = mic.get_direction(frames)
                        servodir = int(6273 - 45.5*horizontal)
                        servo.setTarget(0,servodir)
                        print('\n H:{h} V:{v}'.format(h = int(horizontal), v = int(vertical)))

                    speech_count = 0
                    chunks = []

    except KeyboardInterrupt:
        pass
        
    pixel_ring.off()


if __name__ == '__main__':
    main()
