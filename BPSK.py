import numpy as np
import matplotlib.pyplot as plt

# массив данных из сигнала 0 или 1
numb = [1, 1, 0, 0, 1, 0, 0, 1, 1]

# манчестерский код
manchester_time = np.arange(0, 9, 0.5)
box = [(- 0.5, 0.5) if symbol == 1
       else (0.5, -0.5)
       for symbol in numb]

manchester_code = [symbol_2
                   for symbol_1 in box
                   for symbol_2 in symbol_1]

plt.subplot(2, 1, 1)
plt.plot(manchester_time, manchester_code, drawstyle='steps-post')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1, 1)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('Манчестерский код', size=25)
plt.grid()

# BPSK signal
plt.subplot(2, 1, 2)
time_BPSK = np.arange(0.0, 9, 0.01)
signal_BPSK = []

w = 2 * np.pi * 5
for i in range(len(manchester_code)):
    time_tmp = time_BPSK[(i * 50):((i + 1) * 50)]
    if manchester_code[i] == 0:
        signal_BPSK_tmp = (manchester_code[i] * np.ones(50)) * np.cos(w * time_tmp + np.pi/4 * time_tmp)
    else:
        signal_BPSK_tmp = (manchester_code[i] * np.ones(50)) * np.cos(w * time_tmp + 5 * np.pi/4 * time_tmp)
    signal_BPSK.append(signal_BPSK_tmp)
print(len(np.array(signal_BPSK).flatten()))
time_B = np.arange(0.0, 9, 0.01)
plt.plot(time_B, np.array(signal_BPSK).flatten(), 'r')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1, 1)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('BPSK сигнал', size=25)
plt.grid()

plt.tight_layout()
plt.show()
