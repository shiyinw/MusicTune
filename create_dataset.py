import os
import csv
import pickle
import librosa.display
import time
from multiprocessing import Pool

notes = [207.657, 220.005, 233.087, 246.947, 261.632, 277.189, 293.672, 311.135, 329.636, 349.237, 370.003, 392.005, 415.315, 440.010, 466.175, 493.895, 523.264, 554.379, 587.344, 622.269]

def naive_tune(f0):
    _f0 = f0.copy()
    for i in range(len(_f0)):
        if _f0[i] > 0:
            nearest_note = 0
            for note in notes:
                if abs(_f0[i] - note) < abs(_f0[i] - nearest_note):
                    nearest_note = note
            _f0[i] = nearest_note
    for i in range(len(_f0)):
        if _f0[i] > 0 and _f0[i] == _f0[min(i + 10, len(_f0)) - 1]:
            for j in range(i, min(i + 10, len(_f0))):
                _f0[j] = _f0[i]
    for rounds in range(20):
        tmp_f0 = _f0.copy()
        for i in range(1, len(_f0) - 1):
            _f0[i] = (tmp_f0[i - 1] + tmp_f0[i] + tmp_f0[i + 1]) / 3
    for i in range(len(_f0)):
        if _f0[i] > 0:
            _f0[i] = (_f0[i] * 4 + f0[i] * 6) / 10
    return _f0


def run(filelist):
    if (str(filelist[0]) + ".pickle" in os.listdir("./pitch_pickles")):
        print(str(filelist[0]) + ".pickle", "exists")
        return
    smule_pitch = []
    cnt = 0
    start_time = time.time()
    for file in filelist[1]:
        cnt += 1
        if (cnt % 10 == 0):
            print(filelist[0], cnt, time.time() - start_time)
        if (file[-4:] == ".csv"):

            # pitch data from https://ccrma.stanford.edu/damp/
            with open("./pitch/" + file, "r") as f:
                f0 = []
                spamreader = csv.reader(f, delimiter=' ', quotechar='|')
                for row in spamreader:
                    if row[0] != 'F0_sma':
                        f0.append(float(row[0]))
                while (len(f0) > 2 and f0[0] == 0):
                    f0 = f0[1:]
                while (len(f0) > 2 and f0[-1] == 0):
                    f0 = f0[:-1]
                # print(file, len(f0))
                while (len(f0) > 3000):
                    _f0 = naive_tune(f0[:3000])
                    _f0 = np.array(_f0)
                    smule_pitch.append([np.array(f0[:3000]), _f0])
                    # print(file, _f0.shape)
                    f0 = f0[:3000]

    with open("./pitch_pickles/" + str(filelist[0]) + ".pickle", "wb") as f:
        pickle.dump(smule_pitch, f)


files = os.listdir("./pitch")
iterfile = []
for i in range(343):
    iterfile.append([i, files[i * 100:(i + 1) * 100]])

iterfile.append([343, files[34300:]])

if __name__ == '__main__':
    pool = Pool(20)  # Create a multiprocessing Pool
    pool.map(run, iterfile)

#             plt.figure(1)
#             plt.subplot(211)
#             plt.title("raw input")
#             plt.xlabel("")
#             plt.axis("off")
#             librosa.display.waveplot(f0, 1)
#             plt.subplot(212)
#             plt.axis("off")
#             plt.title("naive tuned output")
#             librosa.display.waveplot(_f0, 1)
#             plt.savefig("naive_tune.png")