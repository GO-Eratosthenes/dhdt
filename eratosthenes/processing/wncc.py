# weighted normalize cross-correlation

import numpy as np
from scipy import ndimage 


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id



# main testing
from PIL import Image

img = Image.open('lena.jpg')
I = np.array(img)
I1 = np.double(I[:,:,0])/256

x = np.linspace(0,I1.shape[1]-1,I1.shape[1])
y = np.linspace(0,I1.shape[0]-1,I1.shape[0])
X, Y = np.meshgrid(x,y)

d = (+0.15,-0.25) # artificial displacement
ts = 2**7

# bi-linear interpolation
I2 = bilinear_interpolate(I1, X+d[0], Y+d[1])

# padfield
I1sub = I1[ts:-ts,ts:-ts]
I2sub = I2[ts:-ts,ts:-ts]

I1f = np.fft.fft2(I1sub)
I2f = np.fft.fft2(I2sub)
M1f = np.fft.fft2(M1sub)
M2f = np.fft.fft2(M2sub)

fF1F2 = np.fft.ifft2( I1f*np.conj(I2f) )
fM1M2 = np.fft.ifft2( M1f*np.conj(M2f) )
fM1F2 = np.fft.ifft2( M1f*np.conj(I2f) )
fF1M2 = np.fft.ifft2( I1f*np.conj(M2f) )

ff1M2 = np.fft.ifft2( np.fft.fft2(I1sub**2)*np.conj(M2f) )
fM1f2 = np.fft.ifft2( M1f*np.fft.fft2( np.flipud(I2sub**2) ) )


NCC_num = fF1F2 - \
    (np.divide(
        np.multiply( fF1M1, fM1F2 ), fM1M2 ))
NCC_den = np.multiply( \
                      np.sqrt(ff1M2 - np.divide( fF1M2**2, fM1M2) ),
                      np.sqrt(fM1f2 - np.divide( fM1F2**2, fM1M2) ))
NCC = np.divide(NCC_num, NCC_den)
