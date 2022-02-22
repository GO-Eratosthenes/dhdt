from PIL import Image, ImageDraw

import numpy as np

import matplotlib.pyplot as plt

from ..preprocessing.image_transforms import mat_to_gray

def make_image(data, outputname, size=(1, 1), dpi=80, cmap='hot'):
    """ create a matplotlib figure and exprot this view

    Parameters
    ----------
    data : np.array
        intensity array
    outputname : string
        name of the output file
    size : list
        size of the canvas in inches
    dpi : integer
        dots per inch, that is the resolution of the figure
    cmap : string
        name of the colormap to be used
    """
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect='equal', cmap=cmap)
    if outputname[-4:]=='.png':
        plt.savefig(outputname, dpi=dpi, transparent = True)
    else:
        plt.savefig(outputname, dpi=dpi)
    return

def resize_image(data, size, method='nearest'):
    img = Image.fromarray(data)
    # PIL has a reversed order i.r.t. Python
    size = size[::-1]
    if method in ('nearest'):
        new_img = img.resize(size, resample=Image.NEAREST)
    elif method in ('linear'):
        new_img = img.resize(size, resample=Image.BILINEAR)
    elif method in ('cubic'):
        new_img = img.resize(size, resample=Image.BICUBIC)
    else:
        new_img = img.resize(size, resample=Image.LANCZOS)
    new_img = np.asarray(new_img)
    return new_img

def output_image(data, outputname, cmap='bone', compress=95):

    # test if no-data is present
    OUT = np.isnan(data)
    if np.any(OUT):
        data[OUT] = np.median(data[~OUT]) # create placeholder

    if data.dtype == np.bool8: data = np.uint8(data.astype('float')*255)
    if data.ndim==2:
        m,n = data.shape[0],data.shape[1]
        col_map = plt.cm.get_cmap(cmap, 256)
        col_arr = (col_map(range(256))*255).astype(np.uint8)
    if not (data.dtype == np.uint8):
        data = np.uint8(np.round(mat_to_gray(data)* 255))

    if data.ndim==2:
        rgb = np.zeros((m,n,3), dtype=np.uint8)
        rgb = np.take(col_arr[:,0:3], data, axis=0, out=rgb)
    else:
        rgb = data

    if np.any(OUT):
        if outputname[-4:] == '.jpg': # whiten the NaN values
            rgb[np.tile(OUT[:,:,np.newaxis], (1,1,3))] = 255
        else:
            rgb = np.dstack((rgb, np.uint8(np.invert(OUT).astype('float')*255)[:,:,np.newaxis]))
    img = Image.fromarray(rgb)
    img.save(outputname, quality=compress)
    return

def output_mask(data, outputname, color=np.array([255, 0, 64])):

    col_arr = np.stack((np.array([255,255,255,0]),
                        np.hstack((color,255))))

    data = np.uint8(data.astype('float'))
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    rgba = np.take(col_arr, data, axis=0, out=rgba)

    img = Image.fromarray(rgba)
    img.save(outputname)
    return

def output_draw_lines_overlay(i_1,j_1, i_2,j_2, im_size, outputname,
                              color=np.array([255, 0, 64]), interval=4):
    """ saves a transparent image of a list of line coordinates

    Parameters
    ----------
    i_1 : np.array, size=(,m), {float,integer}
        row location of the first point of the line
    j_1 : np.array, size=(,m), {float,integer}
        collumn location of the first point of the line
    i_2 : np.array, size=(,m), {float,integer}
        row location of the second point of the line
    j_2 : np.array, size=(,m), {float,integer}
        collumn location of the second point of the line
    im_size : tuple, size=2
        size of the image extent
    outputname : string, ending='.png'
        name of the output file
    color : np.array, size=(,3), range=0...255
        color of the lines to be drawn
    interval : integer
        interval of the lines to be used

    """
    # PIL has a reversed order i.r.t. Python
    im_size = im_size[::-1]
    img = Image.new("RGBA", im_size) # create transparent image

    draw = ImageDraw.Draw(img)
    for k in np.arange(0,len(i_1),interval):
        draw.line((j_1[k], i_1[k], j_2[k], i_2[k]),
                  fill=(color[0],color[2],color[2],255), width=1)

    img.save(outputname)
    return
