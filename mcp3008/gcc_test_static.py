import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from subprocess import run
from time import sleep
from pickle import dump

_NMIC = 3
_FS = 25e3
_VSOUND = 343

def pick_data(data):
    with open('test.pickle', 'wb') as outf:
        dump(data, outf)

def run_capture():
    run(['sudo', './rawMCP3008'])

def get_capture_data(output_name):
    with open(output_name) as inpf:
        temp = inpf.readlines()

    temp = np.array([tt.split() for tt in temp]).astype(int)
    temp = temp - temp.mean(axis=0, keepdims=True)
    temp = temp / np.max(np.abs(temp), axis=0)
    temp[:,2] = np.negative(temp[:,2])

    return temp

def get_k_max(dist):
    return int(np.floor(dist * _FS/_VSOUND))

def cross_corr(sig_1, sig_2, k_max):
    cc = np.correlate(sig_1, sig_2, 'same')
    n = (cc.shape[0])//2

    cc = np.concatenate((cc[n-k_max:n], cc[n:n+k_max+1]))
    shift = np.argmax(np.abs(cc)) - k_max

    return [shift]


def gcc_phat(sig_1, sig_2, k_max):
    n = len(sig_1) + len(sig_2)

    sig_1_fft = np.fft.rfft(sig_1, n=n)
    sig_2_fft = np.fft.rfft(sig_2, n=n)

    res = sig_1_fft * np.conj(sig_2_fft)

    gcc_phat = res / np.abs(res)

    cross_corr = np.fft.irfft(gcc_phat, n=n)
    cross_corr = np.concatenate((cross_corr[-k_max:], cross_corr[:k_max+1]))

    shift = np.argmax(np.abs(cross_corr)) - k_max

    return [shift]

distances = {(0, 1): 8e-2, (2, 0): 29e-2, (2, 1): 37e-2}
dist_values = [[(0,1), 8e-2], [(2,0), 29e-2], [(2,1), 37e-2]]

pow_noise = [0.004106, 0.001621, 0.003282]

input_sounds = ['ruido_branco', 'musica', 'voz']
for input_sound in input_sounds:
    input_sound_name = ' '.join(input_sound.split('_'))
    output_name = f'output_{input_sound}_40graus.txt'
    data = get_capture_data(output_name)

    for power in range(9, 15):
        window = 2**power
        full_data = []
        for i in range(0, data.shape[0] - window, window):
            aux_data = []
            pass_loop = False
            for mics, dist in sorted(distances.items()):
                k_max = get_k_max(dist)
                mic_left, mic_right = mics
                if (sum(data[i:i+window,mic_left]**2)/window <= pow_noise[mic_left]) or (sum(data[i:i+window,mic_right]**2)/window <= pow_noise[mic_right]):
                    pass_loop = True
                    continue
                result_data = cross_corr(data[i:i+window,mic_left], data[i:i+window,mic_right], k_max)
                angle = np.arcsin(_VSOUND/dist * result_data[-1]/_FS)*180/np.pi
                aux_data.append(angle)
            if not pass_loop:
                full_data.append(aux_data.copy())

        full_data = np.array(full_data)
        fig, axs = plt.subplots(1, 3)
        for i in range(len(distances)):
            values = dist_values[i]
            axs[i].hist(full_data[:, i], bins=100)
            axs[i].set_title(f'{round(values[1]*100)}cm')
        fig.suptitle(f'GCC-PHAT em {input_sound_name} com Janela de {window} amostras')
        fig.savefig(f'C:/Users/blazz/OneDrive/Desktop/pi/original/{output_name}_{power}_2.png')
        plt.close(fig)
