import os
import numpy as np

from PIL import Image, ImageDraw
from skimage.color import hsv2rgb
from scipy import ndimage

import matplotlib.pyplot as plt

from dhdt.generic.mapping_tools import map2pix
from dhdt.generic.handler_im import bilinear_interpolation
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.postprocessing.photohypsometric_tools import read_conn_to_df
from dhdt.postprocessing.solar_tools import make_shading


def make_image(data, outputname, size=(1, 1), dpi=80, cmap='hot'):
    """ create a matplotlib figure and export this view

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
    if outputname[-4:] == '.png':
        plt.savefig(outputname, dpi=dpi, transparent=True)
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
        data[OUT] = np.median(data[~OUT])  # create placeholder

    if data.dtype == np.bool8: data = np.uint8(data.astype('float') * 255)
    if data.ndim == 2:
        m, n = data.shape[0], data.shape[1]
        col_map = plt.cm.get_cmap(cmap, 256)
        col_arr = (col_map(range(256)) * 255).astype(np.uint8)
    if not (data.dtype == np.uint8):
        data = np.uint8(np.round(mat_to_gray(data) * 255))

    if data.ndim == 2:
        rgb = np.zeros((m, n, 3), dtype=np.uint8)
        rgb = np.take(col_arr[:, 0:3], data, axis=0, out=rgb)
    else:
        rgb = data

    if np.any(OUT):
        if outputname[-4:] == '.jpg':  # whiten the NaN values
            rgb[np.tile(np.atleast_3d(OUT), (1, 1, 3))] = 255
        else:
            out = np.atleast_3d(np.uint8(np.invert(OUT).astype('float') * 255))
            rgb = np.dstack((rgb, out))
    img = Image.fromarray(rgb)
    img.save(outputname, quality=compress)
    return


def output_mask(data, outputname, color=np.array([255, 0, 64]), alpha=255):

    col_arr = np.stack((np.array([255, 255, 255, 0]), np.hstack(
        (color, alpha))))

    data = np.uint8(data.astype('float'))
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    rgba = np.take(col_arr, data, axis=0, out=rgba)

    img = Image.fromarray(rgba)
    img.save(outputname)
    return


def output_draw_lines_overlay(i_1,
                              j_1,
                              i_2,
                              j_2,
                              im_size,
                              outputname,
                              color=np.array([255, 0, 64]),
                              interval=4,
                              pen=1):
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

    See Also
    --------
    output_cast_lines_from_conn_txt
    """
    # PIL has a reversed order i.r.t. Python
    im_size = im_size[::-1]
    img = Image.new("RGBA", im_size)  # create transparent image

    draw = ImageDraw.Draw(img)
    for k in np.arange(0, len(i_1), interval):
        draw.line((j_1[k], i_1[k], j_2[k], i_2[k]),
                  fill=(color[0], color[2], color[2], 255),
                  width=pen)

    img.save(outputname)
    return


def output_draw_color_lines_overlay(i_1,
                                    j_1,
                                    i_2,
                                    j_2,
                                    val,
                                    im_size,
                                    outputname,
                                    cmap='gray',
                                    vmin=-100,
                                    vmax=+100,
                                    interval=4,
                                    pen=1):
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

    See Also
    --------
    output_cast_lines_from_conn_txt
    """
    # PIL has a reversed order i.r.t. Python
    im_size = im_size[::-1]
    img = Image.new("RGBA", im_size)  # create transparent image

    col_map = plt.cm.get_cmap(cmap, 256)
    col_arr = (col_map(range(256)) * 255).astype(np.uint8)

    val = np.uint8(np.round(mat_to_gray(val, vmin=vmin, vmax=vmax) * 255))

    draw = ImageDraw.Draw(img)
    for k in np.arange(0, len(i_1), interval):
        draw.line((j_1[k], i_1[k], j_2[k], i_2[k]),
                  fill=(col_arr[val[k], 0], col_arr[val[k],
                                                    1], col_arr[val[k],
                                                                2], 255),
                  width=pen)
    img.save(outputname)
    return


def output_glacier_map_background(Z,
                                  RGI,
                                  outputname,
                                  compress=95,
                                  az=45,
                                  zn=45):
    # make typical shading image, or depedent on the image
    Shd = make_shading(Z, az, zn, spac=10)

    # transform to binary class
    RGI = RGI != 0
    # make composite
    I_rgb = hsv2rgb(
        np.dstack((RGI.astype(float) * .55, RGI.astype(float) * .25,
                   (Shd * .2) + .8)))
    I_rgb = np.uint8(I_rgb * 2**8)

    img = Image.fromarray(I_rgb)
    img.save(outputname, quality=compress)
    return


def output_cast_lines_from_conn_txt(conn_dir,
                                    geoTransform,
                                    RGI=None,
                                    outputname='conn.png',
                                    pen=1,
                                    step=2,
                                    inputname='conn.txt',
                                    color=np.array([255, 128, 0])):
    if RGI is None:
        assert len(geoTransform) > 6, ('image dimensions should be given')
        m, n = geoTransform[-2:]
    else:
        RGI = RGI.astype(bool)
        m, n = RGI.shape

    file_path = os.path.join(conn_dir, inputname)
    assert os.path.isfile(file_path), ('please provide correct whereabouts')

    cast_df = read_conn_to_df(file_path)

    # transform to pixel coordinates
    if 'casted_X_refine' in cast_df.columns:
        shdw_i, shdw_j = map2pix(geoTransform,
                                 cast_df['casted_X_refine'].to_numpy(),
                                 cast_df['casted_Y_refine'].to_numpy())
    else:
        shdw_i, shdw_j = map2pix(geoTransform, cast_df['casted_X'].to_numpy(),
                                 cast_df['casted_Y'].to_numpy())

    # make selection if shadow is on glacier
    if RGI is not None:
        IN = ndimage.map_coordinates(RGI, [shdw_i, shdw_j],
                                     order=1,
                                     mode='mirror') > .5
        cast_df = cast_df[IN]
        shdw_i, shdw_j = shdw_i[IN], shdw_j[IN]

    # transform to pixel coordinates
    ridg_i, ridg_j = map2pix(geoTransform, cast_df['caster_X'].to_numpy(),
                             cast_df['caster_Y'].to_numpy())

    output_draw_lines_overlay(ridg_i,
                              ridg_j,
                              shdw_i,
                              shdw_j, (m, n),
                              os.path.join(conn_dir, outputname),
                              color=color,
                              interval=step,
                              pen=pen)
    return
