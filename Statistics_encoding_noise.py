import numpy as np
from matplotlib import pyplot as plt
from numpy import cos, pi
from numpy.typing import NDArray

# массив данных из сигнала 0 или 1
rng = np.random.default_rng()
byte = rng.choice([0, 1], size=1_000_000)


def miller(numb: NDArray[int]) -> NDArray[int]:
    count = 0
    miller_cod = []
    while True:

        if len(miller_cod) == 0:
            if numb[count] == 1:
                miller_cod.append([-1, 1])
            elif numb[count] == 0:
                miller_cod.append([1, 1])

        elif len(miller_cod) != 0 and numb[count] == 1:
            if miller_cod[count - 1] == [-1, 1] or miller_cod[count - 1] == [-1, -1]:
                miller_cod.append([1, -1])
            elif miller_cod[count - 1] == [1, 1] or miller_cod[count - 1] == [1, -1]:
                miller_cod.append([-1, 1])

        elif len(miller_cod) != 0 and numb[count] == 0:
            if miller_cod[count - 1] == [-1, -1] or miller_cod[count - 1] == [1, -1] \
                    or miller_cod[count - 1] == [-1, 1]:
                miller_cod.append([1, 1])
            else:
                miller_cod.append([-1, -1])

        count += 1
        if count == len(numb):
            break
    return np.array(miller_cod).flatten()


cod: NDArray[int] = miller(numb=byte)


def I_and_Q(cod_end: NDArray[int]) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    return cod_end[0::2].astype(np.float_), cod_end[1::2].astype(np.float_)


func_I, func_Q = I_and_Q(cod_end=cod)


def QPSK(bytes_I: NDArray[int] = list, bytes_Q: NDArray[int] = list) -> NDArray[np.float_]:
    QPSK_manipulation = []
    freq = 10
    A = 1
    time_QPSK = np.arange(0, bytes_I.size, 0.01)
    for index in range(bytes_I.size):
        time = time_QPSK[(index * 100):((index + 1) * 100)]

        if bytes_I[index] == 1 and bytes_Q[index] == 1:
            QPSK_manipulation.append(np.hypot(A, A) * cos(2 * pi * freq * time + (pi / 4)))

        elif bytes_I[index] == -1 and bytes_Q[index] == 1:
            QPSK_manipulation.append(np.hypot(A, A) * cos(2 * pi * freq * time + (3 * pi / 4)))

        elif bytes_I[index] == -1 and bytes_Q[index] == -1:
            QPSK_manipulation.append(np.hypot(A, A) * cos(2 * pi * freq * time + (5 * pi / 4)))

        else:
            QPSK_manipulation.append(np.hypot(A, A) * cos(2 * pi * freq * time + (7 * pi / 4)))
    return np.array(QPSK_manipulation)


def demodulation(signal: NDArray[float]) -> NDArray[float]:
    freq = 10
    time = np.linspace(0., 1., num=100, endpoint=False)

    phase = np.asarray([np.arctan2(
        -np.sum(np.array(signal).flatten()[i:i + time.size] * sin(2 * pi * freq * time))
        ,
        np.sum(np.array(signal).flatten()[i:i + time.size] * cos(2 * pi * freq * time))
    )
        for i in range(0, signal.size, time.size)])
    return phase


demodulation(signal=signal)


def main():
    noises = np.linspace(0.1, 10.0, num=101)
    signal = QPSK(bytes_I=func_I, bytes_Q=func_Q)
    signal_noise = float()
    signal_to_noise = np.empty(noises.shape)
    for noise_index, noise in enumerate(noises):
        signal_noise += np.random.normal(scale=noise, size=signal.size // 2)
        signal_to_noise[noise_index] = np.std(signal) / np.std(signal_noise)

    np.savetxt('result_demodulation.txt', np.column_stack((noises, signal_to_noise)), fmt='%g', delimiter='\t')
    plt.plot(noises, signal_to_noise)
    plt.xlabel('Шум, B', size=25)
    plt.xticks(fontsize=17)
    plt.ylabel('Сигнал/шум', size=25)
    plt.yticks(fontsize=17)
    plt.title('Отношение сигнал/шум к шуму', size=25)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
