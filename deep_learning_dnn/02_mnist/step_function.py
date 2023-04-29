# 2강. 활성화 함수
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    '''0보다 작으면 0, 크면 1로 출력'''
    return np.array(x > 0, dtype=int)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
