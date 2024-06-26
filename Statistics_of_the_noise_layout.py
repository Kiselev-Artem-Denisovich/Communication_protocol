from collections import deque
from typing import Final, Literal

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


freq: Final[float] = 10.0
omega: Final[float] = 2.0 * np.pi * freq

time = np.arange(0.0, 1.0, 0.1 / freq)
phase = omega * time

sin_phase = np.sin(phase)
cos_phase = np.cos(phase)


def miller(bits: NDArray[int]) -> NDArray[int]:
    miller_code: deque[tuple[Literal[1, -1], Literal[1, -1]]] = deque()

    for count, bit in enumerate(bits):
        if count == 0:
            if bits[count] == 1:
                miller_code.append((-1, 1))
            else:  # 0
                miller_code.append((1, 1))
        else:
            if bits[count] == 1:
                if miller_code[-1][0] == -1:
                    miller_code.append((1, -1))
                else:  # 1
                    miller_code.append((-1, 1))

            else:  # 0
                if miller_code[-1] != (1, 1):
                    miller_code.append((1, 1))
                else:
                    miller_code.append((-1, -1))

    return np.concatenate(miller_code)


def I_and_Q(code: NDArray[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return code[0::2].astype(np.float64), code[1::2].astype(np.float64)


def QPSK(bytes_I: NDArray[int], bytes_Q: NDArray[int]) -> NDArray[np.float64]:
    QPSK_manipulation = []
    for i, q in zip(bytes_I, bytes_Q, strict=True):
        if i == 1 and q == 1:
            QPSK_manipulation.append(np.cos(phase + 0.25 * np.pi))

        elif i == -1 and q == 1:
            QPSK_manipulation.append(np.cos(phase + 0.75 * np.pi))

        elif i == -1 and q == -1:
            QPSK_manipulation.append(np.cos(phase - 0.75 * np.pi))

        elif i == 1 and q == -1:
            QPSK_manipulation.append(np.cos(phase - 0.25 * np.pi))

        else:
            raise ValueError
    return np.concatenate(QPSK_manipulation)


def demodulation(signal: NDArray[float]) -> NDArray[float]:
    return np.asarray(
        [
            np.arctan2(
                -np.sum(s * sin_phase),
                np.sum(s * cos_phase),
            )
            for s in np.reshape(signal, (-1, time.size))
        ]
    )


def restored_signal(phase: NDArray[float]) -> NDArray[int]:
    box: deque[tuple[Literal[0, 1], Literal[0, 1]]] = deque()
    # (00, 10, 11, 01)
    # (0, 1, 0, 1)
    # ((1, 1), (1, -1), (-1, -1), (-1, 1))
    for func_11, func_10, func_00, func_01 in zip(
        np.abs(phase + 0.75 * np.pi),
        np.abs(phase + 0.25 * np.pi),
        np.abs(phase - 0.25 * np.pi),
        np.abs(phase - 0.75 * np.pi),
    ):
        d = {func_11: (1, 1), func_01: (0, 1), func_10: (1, 0), func_00: (0, 0)}
        box.append(min(d.items())[1])
    return np.concatenate(box)


def decoder(code: NDArray[int]) -> NDArray[int]:
    length: int = code.shape[0]
    assert length % 2 == 0
    bits = np.zeros(length // 2, dtype=int)
    counts = 0
    pos = 0
    while counts < length:

        if code[counts] != code[counts + 1]:
            bits[pos] = 1

        counts += 2
        pos += 1

    return bits


def main() -> None:
    from datetime import datetime

    # массив данных из сигнала 0 или 1
    rng = np.random.default_rng()
    initial_data = rng.choice([0, 1], size=1_000_000)

    noise_levels = np.linspace(0.1, 10.0, num=100)
    encoded_initial_data: NDArray[int] = miller(bits=initial_data)
    func_I, func_Q = I_and_Q(code=encoded_initial_data)
    signal = QPSK(bytes_I=func_I, bytes_Q=func_Q)

    signal_std = np.std(signal)
    standard_noise = rng.standard_normal(size=signal.shape)
    standard_noise_std = np.std(standard_noise)

    errors = np.empty(noise_levels.shape)
    snr = np.empty(noise_levels.shape)

    for noise_index, noise in enumerate(noise_levels):
        signal_noise = standard_noise * noise
        snr[noise_index] = signal_std / (standard_noise_std * noise)
        demodulation_signal = demodulation(signal=signal_noise + signal)
        digital_signal = restored_signal(phase=demodulation_signal)
        decoder_end = decoder(digital_signal)
        err = (
            np.count_nonzero(initial_data != decoder_end)
            + abs(initial_data.size - decoder_end.size)
        ) / max(initial_data.size, decoder_end.size)
        errors[noise_index] = err
        print(datetime.now(), f"{noise:g}", err)

    np.savetxt(
        "result_maket.txt",
        np.column_stack((noise_levels, snr, errors)),
        fmt="%g",
        delimiter="\t",
    )
    plt.loglog(snr, errors)
    plt.xlabel("Отношение сигнала к шуму", size=25)
    plt.xticks(fontsize=17)
    plt.ylabel("Вероятность ошибки, %", size=25)
    plt.yticks(fontsize=17)
    plt.title("Отношение вероятность ошибки к отношению сигнал/шум", size=25)
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
