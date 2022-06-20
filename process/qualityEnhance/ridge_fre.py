import numpy as np
import math
import scipy

ridge_freq_windsze = 5
min_wave_length = 5
max_wave_length = 15
ridge_freq_blksze = 38

def frequest(blkim, blkor):
    rows, cols = np.shape(blkim)
    cosorient = np.mean(np.cos(2 * blkor))
    sinorient = np.mean(np.sin(2 * blkor))
    orient = math.atan2(sinorient, cosorient) / 2
    rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                    mode='nearest')
    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]
    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, ridge_freq_windsze, structure=np.ones(ridge_freq_windsze))
    temp = np.abs(dilation - proj)
    peak_thresh = 2
    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)
    rows_maxind, cols_maxind = np.shape(maxind)
    if (cols_maxind < 2):
        return(np.zeros(blkim.shape))
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= min_wave_length and waveLength <= max_wave_length:
            return(1 / np.double(waveLength) * np.ones(blkim.shape))
        else:
            return(np.zeros(blkim.shape))

#uoc luong tan so duong van
def ridge_freq(norimg, mask, orientimg):
    rows, cols = norimg.shape
    freq = np.zeros((rows, cols))
    # tinh tan so tren moi block
    for r in range(0, rows - ridge_freq_blksze, ridge_freq_blksze):
        for c in range(0, cols - ridge_freq_blksze, ridge_freq_blksze):
            blkim = norimg[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze]
            blkor = orientimg[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze]

            freq[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze] = frequest(blkim, blkor)

    _freq = freq * mask
    freq_1d = np.reshape(_freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    _mean_freq = np.mean(non_zero_elems_in_freq)

    _freq = _mean_freq * mask   
    return _freq