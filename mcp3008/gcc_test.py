import numpy as np
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

def get_capture_data():
    with open(f'test.txt') as inpf:
        temp = inpf.readlines()

    temp = np.array([tt.split() for tt in temp]).astype(int)
    temp = temp - temp.mean(axis=0, keepdims=True)
    temp = temp / np.max(np.abs(temp), axis=0)

    return temp

def get_k_max(dist):
    return int(np.floor(dist * _FS/_VSOUND))

def gcc_phat(sig_1, sig_2, k_max):
    n = len(sig_1) + len(sig_2)

    sig_1_fft = np.fft.rfft(sig_1, n=n)
    sig_2_fft = np.fft.rfft(sig_2, n=n)

    res = sig_1_fft * np.conj(sig_2_fft)

    gcc_phat = res / np.abs(res)

    cross_corr = np.fft.irfft(res / np.abs(res), n=n)
    cross_corr = np.concatenate((cross_corr[-k_max:], cross_corr[:k_max+1]))

    shift = np.argmax(np.abs(cross_corr)) - k_max

    return sig_1_fft, sig_2_fft, res, gcc_phat, cross_corr, shift

dist = 41e-2
k_max = get_k_max(dist)
full_data = []

while True:
    run_capture()
    data = get_capture_data()
    result_data = gcc_phat(data[:,2], data[:,1], k_max)
    full_data.append([data, *result_data])
    angle = np.arcsin(_VSOUND/dist * result_data[-1]/_FS)*180/np.pi
    print(f'{angle:.2f}º')

