import matplotlib.pyplot as plt
import numpy as np
import os

def EMA(data, alpha=0.1):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def plot(output_dir, title, alpha, *data):
    for d in data:
        plt.plot(np.array(d))
        if alpha > 0:
            plt.plot(np.array(EMA(d, alpha)))
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()