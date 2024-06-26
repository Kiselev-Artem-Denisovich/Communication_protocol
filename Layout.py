import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# массив данных из сигнала 0 или 1
numb = [1, 1, 0, 0, 1, 0, 0, 1, 1]
time = np.arange(0, 9, 1)


def miller(count=0):
    miller_cod = []
    while True:

        if len(miller_cod) == 0:
            if numb[count] == 1:
                miller_cod.append([0, 1])
            elif numb[count] == 0:
                miller_cod.append([0, -1])

        elif len(miller_cod) != 0 and numb[count] == 1:
            if miller_cod[count - 1] == [0, 1] or miller_cod[count - 1] == [1, 1]:
                miller_cod.append([1, 0])
            elif miller_cod[count - 1] == [0, 0] or miller_cod[count - 1] == [1, 0]:
                miller_cod.append([0, 1])

        elif len(miller_cod) != 0 and numb[count] == 0:
            if miller_cod[count - 1] == [0, 0] or miller_cod[count - 1] == [1, 0] \
                    or miller_cod[count - 1] == [0, 1]:
                miller_cod.append([1, 1])
            else:
                miller_cod.append([0, 0])

        count += 1
        if count == len(numb):
            break
    return miller_cod


cod = miller()


def I_and_Q(cod_end):
    return cod_end[0::2], cod_end[1::2]


func_I, func_Q = I_and_Q(cod_end=np.array(cod).flatten())


def I_Q_QPSK(bytes_I=list, bytes_Q=list):
    I, Q, QPSK_manipul = [], [], []
    f = 10
    cos = np.cos
    sin = np.sin
    pi = np.pi
    num = np.ones(100)
    A = 1
    time_QPSK = np.arange(0, 9, 0.01)
    for index in range(len(bytes_I)):
        time = time_QPSK[(index * 100):((index + 1) * 100)]

        if bytes_I[index] == 0 and bytes_Q[index] == 0:
            I.append((A * num) * cos(2 * pi * f * time + pi / 4))
            Q.append((A * num) * sin(2 * pi * f * time + pi / 4))
            QPSK_manipul.append(((2 * (A ** 2)) ** 0.5) * cos(2 * pi * f * time + pi / 4))

        elif bytes_I[index] == 0 and bytes_Q[index] == 1:
            I.append((-A * num) * cos(2 * pi * f * time + 3 * pi / 4))
            Q.append((A * num) * sin(2 * pi * f * time + 3 * pi / 4))
            QPSK_manipul.append(((2 * (A ** 2)) ** 0.5) * cos(2 * pi * f * time + 3 * pi / 4))

        elif bytes_I[index] == 1 and bytes_Q[index] == 1:
            I.append((-A * num) * cos(2 * pi * f * time + (5 * pi / 4)))
            Q.append((-A * num) * sin(2 * pi * f * time + (5 * pi / 4)))
            QPSK_manipul.append(((2 * (A ** 2)) ** 0.5) * cos(2 * pi * f * time + 5 * pi / 4))

        else:
            I.append((A * num) * cos(2 * pi * f * time + (7 * pi) / 4))
            Q.append((-A * num) * sin(2 * pi * f * time + (7 * pi / 4)))
            QPSK_manipul.append(((2 * (A ** 2)) ** 0.5) * cos(2 * pi * f * time + 7 * pi / 4))
    return I, Q, QPSK_manipul


signal_I, signal_Q, signal_QPSK = I_Q_QPSK(bytes_I=func_I, bytes_Q=func_Q)
time_I_Q_QPSK = np.arange(0, 9, 0.01)

# I сигнал
plt.subplot(3, 1, 1)
plt.plot(time_I_Q_QPSK, np.array(signal_I).flatten())
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1.5, 1.5)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('I сигнал', size=25)
plt.grid()

# Q сигнал
plt.subplot(3, 1, 2)
plt.plot(time_I_Q_QPSK, np.array(signal_Q).flatten())
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1.5, 1.5)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('Q сигнал', size=25)
plt.grid()

# QPSK сигнал
plt.subplot(3, 1, 3)
plt.plot(time_I_Q_QPSK, np.array(signal_QPSK).flatten(), 'r')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-2, 2)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('QPSK сигнал', size=25)
plt.grid()

plt.tight_layout()
plt.show()


freq = 10
t = np.linspace(0., 1., num=100, endpoint=False)

phase = np.asarray([np.arctan2(
    -np.sum(np.array(signal_QPSK).flatten()[i:i + t.size] * np.sin(2 * np.pi * freq * t))
    ,
    np.sum(np.array(signal_QPSK).flatten()[i:i + t.size] * np.cos(2 * np.pi * freq * t))
)
    for i in range(0, len(np.array(signal_QPSK).flatten()), t.size)])

plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, 9, num=np.array(signal_QPSK).flatten().size), np.array(signal_QPSK).flatten())
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-2, 2)
plt.yticks(fontsize=15)
plt.ylabel('A, В', size=25)
plt.title('QPSK сигнал', size=25)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(phase / (np.pi / 4), 'r', drawstyle='steps-post')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(- np.pi, np.pi)
plt.yticks(fontsize=15)
plt.ylabel('Фаза', size=25)
plt.title('Фазовая огибающая', size=25)
plt.grid()

plt.tight_layout()
plt.show()

time_decod = np.arange(0, 9, 0.5)
box = []
# (00, 10, 11, 01)
# (0, 1, 0, 1)
# ((1, 1), (1, -1), (-1, -1), (-1, 1))
for index in range(len(phase)):
    func_11 = np.hypot((phase[index] + 2.5), (phase[index]) + 2.5)
    func_10 = np.hypot((phase[index] + 0.8), (phase[index]) + 0.8)
    func_00 = np.hypot((phase[index] - 0.8), (phase[index] - 0.8))
    func_01 = np.hypot((phase[index] - 2.5), (phase[index] - 2.5))
    d = {func_11: (1, 1), func_01: (0, 1), func_10: (1, 0), func_00: (0, 0)}
    box.extend(min(d.items())[1])

plt.subplot(2, 1, 1)
plt.plot(phase / (np.pi / 4), 'r', drawstyle='steps-post')
plt.plot(np.linspace(0, 9, num=np.array(signal_QPSK).flatten().size), np.array(signal_QPSK).flatten())
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(- np.pi, np.pi)
plt.yticks(fontsize=15)
plt.ylabel('Фаза', size=25)
plt.title('Фазовая огибающая и QPSK сигнал', size=25)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_decod, box, 'r', drawstyle='steps-post')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1., 1.5)
plt.yticks(fontsize=15)
plt.ylabel('A, B', size=25)
plt.title('Восстановленный цифровой сигнал', size=25)
plt.grid()

plt.tight_layout()
plt.show()

time_cod = np.arange(0, 9, 1)


def decoder(cod_end: NDArray[int]) -> NDArray[int]:
    length: int = cod_end.shape[0]
    assert length % 2 == 0
    byt = np.zeros(length // 2, dtype=int)
    counts = 0
    pos = 0
    while counts < length:
        print(cod_end[cod])
        if counts == 0:
            if cod_end[counts] != cod_end[counts + 1]:
                byt[pos] = 1  # elif cod_end[counts:counts + 2] == [0, 0]:  #     byt[pos] = 0

        else:
            if cod_end[counts] != cod_end[counts + 1]:
                byt[pos] = 1  # else:  #     byt[pos] = 0

        counts += 2
        pos += 1

    return byt


cod = decoder(cod_end=np.asarray(box))
print(cod)
plt.subplot(2, 1, 1)
plt.plot(time_decod, box, drawstyle='steps-post')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1., 1.5)
plt.yticks(fontsize=15)
plt.ylabel('A, B', size=25)
plt.title('Восстановленный цифровой сигнал', size=25)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_cod, cod, 'r', drawstyle='steps-post')
plt.xlim(0, 8)
plt.xticks(fontsize=15)
plt.xlabel('t', size=25)
plt.ylim(-1., 1.5)
plt.yticks(fontsize=15)
plt.ylabel('A, B', size=25)
plt.title('Бит информация', size=25)
plt.grid()

plt.tight_layout()
plt.show()
