# general libraries
import numpy as np
from scipy import fftpack, ndimage

from ..generic.unit_check import are_two_arrays_equal
from ..generic.handler_im import get_grad_filters
from .matching_tools import \
    reposition_templates_from_center, make_templates_same_size, \
    get_integer_peak_location, get_data_and_mask
from .matching_tools_frequency_filters import \
    raised_cosine, thresh_masking, normalize_power_spectrum, gaussian_mask, \
    make_template_float
from .matching_tools_harmonic_functions import create_complex_fftpack_DCT

def upsample_dft(Q, up_m=0, up_n=0, upsampling=1,
                 i_offset=0, j_offset=0):
    (m,n) = Q.shape
    if up_m==0:
        up_m = m.copy()
    if up_n==0:
        up_n = n.copy()

    kernel_collumn = np.exp((1j*2*np.pi/(n*upsampling)) *\
                            ( np.fft.fftshift(np.arange(n) - \
                                              (n//2))[:,np.newaxis] )*\
                            ( np.arange(up_n) - j_offset ))
    kernel_row = np.exp((1j*2*np.pi/(m*upsampling)) *\
                        ( np.arange(up_m)[:,np.newaxis] - i_offset )*\
                        ( np.fft.fftshift(np.arange(m) - (m//2)) ))
    Q_up = np.matmul(kernel_row, np.matmul(Q,kernel_collumn))
    return Q_up

def pad_dft(Q, m_new, n_new):
    assert type(Q)==np.ndarray, ("please provide an array")
    (m,n) = Q.shape

    Q_ij = np.fft.fftshift(Q) # in normal configuration
    center_old = np.array([m//2, n//2])

    Q_new = np.zeros((m_new, n_new), dtype=np.complex64)
    center_new =  np.array([m_new//2, n_new//2])
    center_offset = center_new - center_old

    # fill the old data in the new array
    Q_new[np.maximum(center_offset[0], 0):np.minimum(center_offset[0]+m, m_new),\
          np.maximum(center_offset[1], 0):np.minimum(center_offset[1]+n, n_new)]\
        = \
        Q_ij[np.maximum(-center_offset[0], 0):\
             np.minimum(-center_offset[0]+m_new, m),\
             np.maximum(-center_offset[1], 0):\
             np.minimum(-center_offset[1]+n_new, n)]

    Q_new = (np.fft.fftshift(Q_new)*m_new*n_new)/(m*n) # scaling
    return Q_new

# frequency/spectrum matching functions
def cosi_corr(I1, I2, beta1=.35, beta2=.50, m=1e-4):
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    mt,nt = I1.shape[0], I1.shape[1] # dimensions of the template

    W1 = raised_cosine(np.zeros((mt,nt)), beta1)
    W2 = raised_cosine(np.zeros((mt,nt)), beta2)

    if I1.size==I2.size: # if templates are same size, no refinement is done
        tries = [0]
    else:
        tries = [0, 1]

    di,dj, m0 = 0,0,np.array([0, 0])
    for trying in tries: # implement refinement step to have more overlap
        if I1.ndim==3: # multi-spectral frequency stacking
            bands = I1.shape[2]
            I1sub,I2sub = reposition_templates_from_center(I1,I2,di,dj)

            for i in range(bands): # loop through all bands
                I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
                S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd)

                if i == 0:
                    Q = (W1*S1)*np.conj((W2*S2))
                else:
                    Q_b = (W1*S1)*np.conj((W2*S2))
                    Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        else:
            I1sub,I2sub = reposition_templates_from_center(I1,I2,di,dj)
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)

            Q = (W1*S1)*np.conj((W2*S2))

        # transform back to spatial domain
        C = np.real(np.fft.fftshift(np.fft.ifft2(Q)))
        ddi, ddj,_,_ = get_integer_peak_location(C)
        m_int = np.round(np.array([ddi, ddj])).astype(int)
        if np.amax(abs(np.array([ddi, ddj])))<.5:
            break
        else:
            di,dj = m_int[0], m_int[1]

    m0[0] += di
    m0[1] += dj

    WS = thresh_masking(S1, m)
    Qn = normalize_power_spectrum(Q)

    return Qn, WS, m0

def cosine_corr(I1, I2):
    """ match two imagery through discrete cosine transformation, see [Li04].

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}, dtype=float
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22].
    I2 : numpy.array, size=(m,n), ndim={2,3}, dtype=float
        array with intensities

    Returns
    -------
    C : numpy.array, size=(m,n), dtype=real
        displacement surface

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.create_complex_DCT
    dhdt.processing.matching_tools_frequency_correlators.sign_only_corr

    References
    ----------
    .. [Li04] Li, et al. "DCT-based phase correlation motion estimation",
              IEEE international conference on image processing, vol. 1, 2004.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _cosine_corr_core(I1,I2):
        C1 = create_complex_fftpack_DCT(I1)
        C2 = create_complex_fftpack_DCT(I2)
#            C1 = create_complex_DCT(I1bnd, Cc, Cs)
#            C2 = create_complex_DCT(I2bnd, Cc, Cs)
        Q = (C1) * np.conj(C2)
        Qn = normalize_power_spectrum(Q)
        return Q

#    # construct cosine and sine basis matrices
#    Cc, Cs = get_cosine_matrix(I1), get_sine_matrix(I1)

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _cosine_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _cosine_corr_core(I1sub[...,i], I2sub[...,i])
        if i ==0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def masked_cosine_corr(I1, I2, M1, M2): #todo
    """

    Parameters
    ----------
    I1 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    I2 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    M1 : numpy.array, size=(m,n), ndim=2, dtype={bool,float}
        array with mask, hence False's are neglected
    M2 : numpy.array, size=(m,n), ndim=2, dtype={bool,float}
        array with mask, hence False's are neglected


    """
    assert type(I1) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert (type(M1)==np.ndarray) or (M1 is None), ('please provide an array')
    assert (type(M2)==np.ndarray) or (M2 is None), ('please provide an array')

    # make compatible with masekd array
    I1,M1, I2,M2 = get_data_and_mask(I1, M1), get_data_and_mask(I2, M2)

    # construct cosine and sine basis matrices
    Cc, Cs = get_cosine_matrix(I1), get_sine_matrix(I1)

    # look at how many frequencies can be estimated with this data
    (m,n) = M1.shape
    X1 = np.ones((m,n), dtype=bool)
    min_span = int(np.floor(np.sqrt(min(np.sum(M1), np.sum(M2)))))
    X1[min_span:,:] = False
    X1[:,min_span:] = False

    y = (I1[M1].astype(dtype=float)/255)-.5

    # build matrix
    Ccc = np.kron(Cc,Cc)
    # shrink size
    Ccc = Ccc[M1.flatten(),:] # remove rows, as these are missing
    Ccc = Ccc[:,X1.flatten()] # remove collumns, since these can't be estimated
    Icc = np.linalg.lstsq(Ccc, y, rcond=None)[0]
    Icc = np.reshape(Icc, (min_span, min_span))

    iCC = Ccc.T*y

    np.reshape(Ccc.T*y, (min_span, min_span))


    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)

        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            I1sub, I2sub = make_template_float(I1sub), \
                           make_template_float(I2sub)

            C1 = create_complex_DCT(I1sub, Cc, Cs)
            C2 = create_complex_DCT(I2sub, Cc, Cs)
            if i == 0:
                Q = C1*np.conj(C2)
            else:
                Q_b = (C1)*np.conj(C2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q

    else:
        I1sub,I2sub = make_templates_same_size(I1,I2)
        I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)

        C1 = create_complex_DCT(I1sub, Cc, Cs)
        C2 = create_complex_DCT(I2sub, Cc, Cs)
        Q = (C1)*np.conj(C2)
    return Q

def phase_only_corr(I1, I2):
    r""" match two imagery through phase only correlation, based upon [HG84]_
    and [KJ91]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.phase_corr
    dhdt.processing.matching_tools_frequency_correlators.symmetric_phase_corr
    dhdt.processing.matching_tools_frequency_correlators.amplitude_comp_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{W} = 1 / \|\mathbf{S}_2 \|
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 [\mathbf{W}\mathbf{S}_2]^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star`
    a complex conjugate operation

    Schematically this looks like:

        .. code-block:: text

          I2  ┌----┐ S2
           ---┤ F  ├--┬------------┐
              └----┘  | ┌-------┐W |
                      └-┤ 1/|x| ├--×
                        └-------┘  | Q12 ┌------┐  C12
                                   ×-----┤ F^-1 ├---
          I1  ┌----┐ S1            |     └------┘
           ---┤ F  ├---------------┘
              └----┘

    If multiple bands are given, cross-spectral stacking is applied,see [AL22]_.

    References
    ----------
    .. [HG84] Horner & Gianino, "Phase-only matched filtering", Applied optics,
              vol. 23(6) pp.812--816, 1984.
    .. [KJ91] Kumar & Juday, "Design of phase-only, binary phase-only, and
              complex ternary matched filters with increased signal-to-noise
              ratios for colored noise", Optics letters, vol.16(13)
              pp.1025--1027, 1991.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair
    >>> from dhdt.processing.matching_tools import get_integer_peak_location

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_only_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _phase_only_corr_core(I1, I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        W2 = np.divide(1, np.abs(I2),
                       out=np.zeros_like(I2), where=I2!=0)
        Q = S1 * np.conj((W2 * S2))
        return Q

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _phase_only_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _phase_only_corr_core(I1sub[:,:,i], I2sub[:,:,i])
        if i ==0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def projected_phase_corr(I1, I2, M1=np.array(()), M2=np.array(())):
    """ match two imagery through separated phase correlation

    Parameters
    ----------
    I1 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    I2 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    M1 : numpy.array, size=(m,n), ndim=2, dtype={bool,float}
        array with mask, hence False's are neglected
    M2 : numpy.array, size=(m,n), ndim=2, dtype={bool,float}
        array with mask, hence False's are neglected

    Returns
    -------
    C : numpy.array, size=(m,n), dtype=real
        displacement surface

    Notes
    -----
    Schematically this looks like:

        .. code-block:: text

          I1  ┌----┐ I1x  ┌----┐ S1x
           -┬-┤ Σx ├------┤ F  ├------┐ Q12x ┌------┐ C12x
            | └----┘      └----┘      ×------┤ F^-1 ├------┐
            | ┌----┐ I2x  ┌----┐ S2x  |      └------┘      |
            |┌┤ Σx ├------┤ F  ├------┘                    |  ┌---┐ C12
            ||└----┘      └----┘                           ×--┤ √ ├----
            ||┌----┐ I1y  ┌----┐ S1y                       |  └---┘
            └┼┤ Σy ├------┤ F  ├------┐ Q12y ┌------┐ C12y |
             |└----┘      └----┘      ×------┤ F^-1 ├------┘
          I2 |┌----┐ I2y  ┌----┐ S2y  |      └------┘
           --┴┤ Σy ├------┤ F  ├------┘
              └----┘      └----┘

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.phase_corr

    References
    ----------
    .. [Zh10] Zhang et al. "An efficient subpixel image registration based on
              the phase-only correlations of image projections", IEEE
              proceedings of the 10th international symposium on communications
              and information technologies, 2010.
    """
    assert type(I1) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert (type(M1)==np.ndarray) or (M1 is None), ('please provide an array')
    assert (type(M2)==np.ndarray) or (M2 is None), ('please provide an array')

    # make compatible with masekd array
    I1,M1, I2,M2 = get_data_and_mask(I1, M1), get_data_and_mask(I2, M2)

    I1sub,I2sub = make_templates_same_size(I1,I2)
    I1sub,I2sub = make_template_float(I1sub), make_template_float(I2sub)
    M1sub,M2sub = make_templates_same_size(M1,M2)

    def project_spectrum(I, M, axis=0):
        if axis==1 : I,M = I.T, M.T
        # projection
        I_p = np.sum(I*M, axis=1)
        # windowing
        I_w = I_p*np.hamming(I_p.size)
        # Fourier transform
        S = np.fft.fft(I_w)
        if axis==1:
            S = S.T
        return S

    def phase_corr_1d(S1, S2):
        # normalize power spectrum
        Q12 = S1*np.conj(S2)
        return Q12

    S1_m = project_spectrum(I1sub, M1sub, axis=0)
    S2_m = project_spectrum(I2sub, M2sub, axis=0)
    Q12_m = phase_corr_1d(S1_m, S2_m)
    C_m = np.fft.fftshift(np.real(np.fft.ifft(Q12_m)))

    S1_n = project_spectrum(I1sub, M1sub, axis=1)
    S2_n = project_spectrum(I2sub, M2sub, axis=1)
    Q12_n = phase_corr_1d(S1_n, S2_n)
    C_n = np.fft.fftshift(np.real(np.fft.ifft(Q12_n)))

    C = np.sqrt(np.outer(C_m, C_n))
    return C

def sign_only_corr(I1, I2): # to do
    """ match two imagery through phase only correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    C : numpy.array, size=(m,n), real
        displacement surface

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.cosine_corr

    References
    ----------
    .. [IK07] Ito & Kiya, "DCT sign-only correlation with application to image
              matching and the relationship with phase-only correlation",
              IEEE international conference on acoustics, speech and signal
              processing, vol. 1, 2007.
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim==3) or (I2.ndim==3): # multi-spectral frequency stacking
        bands = I1.shape[2]
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]

            C1 = np.sign(fftpack.dctn(I1bnd, type=2)),
            C2 = np.sign(fftpack.dctn(I2bnd, type=2))
            if i == 0:
                Q = C1*np.conj(C2)
            else:
                Q_b = (C1)*np.conj(C2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
    else: # single pair processing
        C1,C2 = fftpack.dctn(I1sub, type=2), fftpack.dctn(I2sub, type=2)
#        C1,C2 = np.multiply(C1,1/C1), np.multiply(C2,1/C2)
        C1,C2 = np.sign(C1), np.sign(C2)
        Q = (C1)*np.conj(C2)
        C = fftpack.idctn(Q,type=1)

        C_cc = fftpack.idct(fftpack.idct(Q, axis=1, type=1), axis=0, type=1)
        C_sc = fftpack.idst(fftpack.idct(Q, axis=1, type=1), axis=0, type=1)
        C_cs = fftpack.idct(fftpack.idst(Q, axis=1, type=1), axis=0, type=1)
        C_ss = fftpack.idst(fftpack.idst(Q, axis=1, type=1), axis=0, type=1)

#        iC1 = fft.idctn(C1,2)
#        import matplotlib.pyplot as plt
#        plt.imshow(iC1), plt.show()
    return C

def symmetric_phase_corr(I1, I2):
    r""" match two imagery through symmetric phase only correlation (SPOF)
    also known as Smoothed Coherence Transform (SCOT)

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [3].
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1], \quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{W} = 1 / \sqrt{||\mathbf{S}_1||||\mathbf{S}_2||}
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 [\mathbf{W}\mathbf{S}_2]^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star`
    a complex conjugate operation

    Schematically this looks like:

        .. code-block:: text

          I2  ┌----┐ S2
           ---┤ F  ├---┬-------------------┐
              └----┘   | ┌--------------┐W |
                       ├-┤ 1/(|x||x|)^½ ├--×
                       | └--------------┘  | Q12 ┌------┐  C12
                       |                   ×-----┤ F^-1 ├---
          I1  ┌----┐ S1|                   |     └------┘
           ---┤ F  ├---┴-------------------┘
              └----┘

    If multiple bands are given, cross-spectral stacking is applied, see
    [AL22]_.

    References
    ----------
    .. [NP93] Nikias & Petropoulou. "Higher order spectral analysis: a nonlinear
              signal processing framework", Prentice hall. pp.313-322, 1993.
    .. [We05] Wernet. "Symmetric phase only filtering: a new paradigm for DPIV
              data processing", Measurement science and technology, vol.16
              pp.601-618, 2005.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair
    >>> from dhdt.processing.matching_tools import get_integer_peak_location

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = symmetric_phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    are_two_arrays_equal(I1, I2)

    def _symmetric_phase_corr_core(I1, I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        denom = np.sqrt(abs(S1)) * np.sqrt(abs(S2))
        W2 = np.divide(1, denom, where=denom!=0)
        Q = S1 * np.conj((W2 * S2))
        return Q

    I1sub,I2sub = make_templates_same_size(I1,I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _symmetric_phase_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _symmetric_phase_corr_core(I1sub[...,i], I2sub[...,i])
        if i ==0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def amplitude_comp_corr(I1, I2, F_0=0.04):
    """ match two imagery through amplitude compensated phase correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities
    F_0 : float, default=4e-2
        cut-off intensity in respect to maximum

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    References
    ----------
    .. [Mu88] Mu et al. "Amplitude-compensated matched filtering", Applied
              optics, vol. 27(16) pp. 3461-3463, 1988.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = amplitude_comp_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    are_two_arrays_equal(I1, I2)

    def _amplitude_comp_corr_core(I1,I2,F_0):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        s_0 = F_0 * np.amax(abs(S2))

        W = np.divide(1, abs(I2),
                      out=np.zeros_like(I2), where=I2!=0)
        A = np.divide(s_0, abs(I2)**2,
                      out=np.zeros_like(I2), where=I2!=0)
        W[abs(S2) > s_0] = A[abs(S2) > s_0]
        Q = (S1) * np.conj((W * S2))
        return Q

    I1sub, I2sub = make_templates_same_size(I1,I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _amplitude_comp_corr_core(I1sub, I2sub, F_0)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _amplitude_comp_corr_core(I1sub[...,i], I2sub[...,i],F_0)
        if i ==0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def robust_corr(I1, I2):
    """ match two imagery through fast robust correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim=2
        array with intensities
    I2 : numpy.array, size=(m,n), ndim=2
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    References
    ----------
    .. [Fi05] Fitch et al. "Fast robust correlation", IEEE transactions on image
              processing vol. 14(8) pp. 1063-1073, 2005.
    .. [Es07] Essannouni et al. "Adjustable SAD matching algorithm using
              frequency domain" Journal of real-time image processing, vol.1
              pp.257-265, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = robust_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    I1sub,I2sub = make_templates_same_size(I1,I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)

    p_steps = 10**np.arange(0,1,.5)
    for idx, p in enumerate(p_steps):
        I1p = 1/p**(1/3) * np.exp(1j*(2*p -1)*I1sub)
        I2p = 1/p**(1/3) * np.exp(1j*(2*p -1)*I2sub)

        S1p, S2p = np.fft.fft2(I1p), np.fft.fft2(I2p)
        if idx==0:
            Q = (S1p)*np.conj(S2p)
        else:
            Q += (S1p)*np.conj(S2p)
    return Q

def gradient_corr(I1, I2):
    r""" match two imagery through gradient correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.normalized_gradient_corr
    dhdt.processing.matching_tools_frequency_correlators.phase_corr
    dhdt.processing.matching_tools_frequency_correlators.orientation_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{G}_1, \mathbf{G}_2 =
              \partial_{x}\mathbf{I}_1 + i \partial_{y}\mathbf{I}_1, \quad
              \partial_{x}\mathbf{I}_2 + i \partial_{y}\mathbf{I}_2
    .. math:: \mathbf{S}_1, \mathbf{S}_2 =
              \mathcal{F}[\mathbf{G}_1], \quad
              \mathcal{F}[\mathbf{G}_2]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 \mathbf{S}_2^{\star}

    where :math:`\partial_{x}` is the spatial derivate in horizontal direction,
    :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation.

    Schematically this looks like:

        .. code-block:: text

                    ∂x I1
          I1  ┌----┬-----┬----┐ S1
           ---┤ ∂  |     | F  ├------┐
              └----┴-----┴----┘      |
                    ∂y I1            |
                                     | Q12 ┌------┐  C12
                    ∂x I2            ×-----┤ F^-1 ├---
          I2  ┌----┬-----┬----┐ S2   |     └------┘
           ---┤ ∂  |     | F  ├------┘
              └----┴-----┴----┘
                    ∂y I2

    References
    ----------
    .. [AV03] Argyriou & Vlachos. "Estimation of sub-pixel motion using gradient
              cross-correlation", Electronics letters, vol.39(13) pp.980--982,
              2003.
    .. [Tz10] Tzimiropoulos et al. "Subpixel registration with gradient
              correlation" IEEE transactions on image processing, vol.20(6)
              pp.1761--1767, 2010.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = gradient_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _gradient_corr_core(I1,I2,H_x,H_y):
        G1 = ndimage.convolve(I1, H_x) + 1j * ndimage.convolve(I1, H_y)
        G2 = ndimage.convolve(I2, H_x) + 1j * ndimage.convolve(I2, H_y)
        S1, S2 = np.fft.fft2(G1), np.fft.fft2(G2)
        Q = (S1)*np.conj(S2)
        return Q

    H_x = get_grad_filters(ftype='kroon', tsize=3, order=1)[0]
    H_y = np.transpose(H_x)

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _gradient_corr_core(I1sub, I2sub, H_x, H_y)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _gradient_corr_core(I1sub[...,i], I2sub[...,i], H_x, H_y)
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def normalized_gradient_corr(I1, I2):
    r""" match two imagery through gradient correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.gradient_corr
    dhdt.processing.matching_tools_frequency_correlators.phase_corr
    dhdt.processing.matching_tools_frequency_correlators.windrose_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{G}_1, \mathbf{G}_2 =
              \partial_{x}\mathbf{I}_1 + i \partial_{y}\mathbf{I}_1, \quad
              \partial_{x}\mathbf{I}_2 + i \partial_{y}\mathbf{I}_2
    .. math:: \mathbf{S}_1, \mathbf{S}_2 =
              \mathcal{F}[\dot{\mathbf{G}}_1], \quad
              \mathcal{F}[\dot{\mathbf{G}}_2]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 \mathbf{S}_2^{\star}

    where :math:`\partial_{x}` is the spatial derivate in horizontal direction,
    :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation and :math:`\dot{x}` represents a statistical
    normalization through the sampled mean and standard deviation.

    Schematically this looks like:

        .. code-block:: text

                    ∂x I1
          I1  ┌----┬-----┬----┐ S1 ┌------┐
           ---┤ ∂  |     | F  ├----┤ norm ├--┐
              └----┴-----┴----┘    └------┘  |
                    ∂y I1                    |
                                             | Q12 ┌------┐  C12
                    ∂x I2                    ×-----┤ F^-1 ├---
          I2  ┌----┬-----┬----┐ S2 ┌------┐  |     └------┘
           ---┤ ∂  |     | F  ├----┤ norm ├--┘
              └----┴-----┴----┘    └------┘
                    ∂y I2

    References
    ----------
    .. [Tz10] Tzimiropoulos et al. "Subpixel registration with gradient
              correlation" IEEE transactions on image processing, vol.20(6)
              pp.1761--1767, 2010.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = normalized_gradient_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _normalized_gradient_corr_core(I1,I2,H_x,H_y):
        G1 = ndimage.convolve(I1, H_x) + 1j * ndimage.convolve(I1, H_y)
        G2 = ndimage.convolve(I2, H_x) + 1j * ndimage.convolve(I2, H_y)

        G1 -= np.mean(G1)
        G2 -= np.mean(G2)
        G1 /= np.max(G1)
        G2 /= np.max(G2)

        S1, S2 = np.fft.fft2(G1), np.fft.fft2(G2)
        Q = (S1)*np.conj(S2)
        return Q

    H_x = get_grad_filters(ftype='kroon', tsize=3, order=1)[0]
    H_y = np.transpose(H_x)

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _normalized_gradient_corr_core(I1sub, I2sub, H_x, H_y)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _normalized_gradient_corr_core(I1sub[...,i], I2sub[...,i],
                                             H_x, H_y)
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def orientation_corr(I1, I2):
    r""" match two imagery through orientation correlation, developed by [Fi02]_
    while demonstrated its benefits over glaciated terrain by [HK12]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.phase_corr
    dhdt.processing.matching_tools_frequency_correlators.windrose_corr
    dhdt.processing.matching_tools_spatial_correlators.cosine_similarity

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{G}_1, \mathbf{G}_2 =
              \partial_{x}\mathbf{I}_1 + i \partial_{y}\mathbf{I}_1,\quad
              \partial_{x}\mathbf{I}_2 + i \partial_{y}\mathbf{I}_2
    .. math:: \hat{\mathbf{G}_1}, \hat{\mathbf{G}_2} =
              \mathbf{G}_1 / \| \mathbf{G}_1\|,\quad
              \mathbf{G}_2 / \| \mathbf{G}_2\|
    .. math:: \mathbf{S}_1, \mathbf{S}_2 =
              \mathcal{F}[\hat{\mathbf{G}_1}],\quad
              \mathcal{F}[\hat{\mathbf{G}_2}]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 \mathbf{S}_2^{\star}

    where :math:`\partial_{x}` is the spatial derivate in horizontal direction,
    :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation and :math:`\hat{x}` is a normalizing operator.

    Schematically this looks like:

        .. code-block:: text

                 ∂x I1┌-------┐
          I1  ┌----┬--┤ 1/|x| ├--┬----┐ S1
           →--┤ ∂  |  ├-------┤  | F  ├--→-┐
              └----┴--┤ 1/|x| ├--┴----┘    |
                 ∂y I1└-------┘            |
                                           | Q12 ┌------┐  C12
                                           ×--→--┤ F^-1 ├---
                 ∂x I2┌-------┐            |     └------┘
          I2  ┌----┬--┤ 1/|x| ├--┬----┐ S2 |
           →--┤ ∂  |  ├-------┤  | F  ├--→-┘
              └----┴--┤ 1/|x| ├--┴----┘
                 ∂y I2└-------┘

    References
    ----------
    .. [Fi02] Fitch et al. "Orientation correlation", Proceeding of the Britisch
              machine vison conference, pp. 1--10, 2002.
    .. [HK12] Heid & Kääb. "Evaluation of existing image matching methods for
              deriving glacier surface displacements globally from optical
              satellite imagery", Remote sensing of environment, vol.118
              pp.339-355, 2012.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = orientation_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _orientation_corr_core(I1, I2, H_x, H_y):
        G1 = ndimage.convolve(I1, H_x) + 1j * ndimage.convolve(I1, H_y)
        G2 = ndimage.convolve(I2, H_x) + 1j * ndimage.convolve(I2, H_y)
        G1, G2 = normalize_power_spectrum(G1), normalize_power_spectrum(G2)

        S1, S2 = np.fft.fft2(G1), np.fft.fft2(G2)
        Q = (S1) * np.conj(S2)
        return Q

    H_x = get_grad_filters(ftype='kroon', tsize=3, order=1)[0]
    H_y = np.transpose(H_x)

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _orientation_corr_core(I1sub, I2sub, H_x, H_y)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _orientation_corr_core(I1sub[...,i], I2sub[...,i], H_x, H_y)
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def windrose_corr(I1, I2):
    """ match two imagery through windrose phase correlation, see [KJ91]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.binary_orientation_corr
    dhdt.processing.matching_tools_frequency_correlators.orientation_corr
    dhdt.processing.matching_tools_frequency_correlators.phase_only_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{Q}_{12} = sign(\mathbf{S}_1) sign(\mathbf{S}_2)^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star`
    a complex conjugate operation


    References
    ----------
    .. [KJ91] Kumar & Juday, "Design of phase-only, binary phase-only, and complex
              ternary matched filters with increased signal-to-noise ratios for
              colored noise", Optics letters, vol. 16(13) pp. 1025--1027, 1991.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.


    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = windrose_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _windrose_corr_core(I1,I2):
        S1, S2 = np.sign(np.fft.fft2(I1)), np.sign(np.fft.fft2(I2))
        Q = S1 * np.conj(S2)
        return Q

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _windrose_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _windrose_corr_core(I1sub[...,i], I2sub[...,i])
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def phase_corr(I1, I2):
    """ match two imagery through phase correlation

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.orientation_corr
    dhdt.processing.matching_tools_frequency_correlators.cross_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \hat{\mathbf{S}}_1, \hat{\mathbf{S}}_2 =
              \mathbf{S}_1 / \| \mathbf{S}_1\|, \quad
              \mathbf{S}_2 / \| \mathbf{S}_2\|
    .. math:: \mathbf{Q}_{12} = \hat{\mathbf{S}}_1 {\hat{\mathbf{S}}_2}^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation and :math:`\hat{x}` is a normalizing operator.

    Schematically this looks like:

        .. code-block:: text

          I1  ┌----┐ S1 ┌-------┐
           ---┤ F  ├----┤ 1/|x| ├--┐
              └----┘    └-------┘  | Q12 ┌------┐  C12
                                   ×-----┤ F^-1 ├---
          I2  ┌----┐ S2 ┌-------┐  |     └------┘
           ---┤ F  ├----┤ 1/|x| ├--┘
              └----┘    └-------┘

    References
    ----------
    .. [KH75] Kuglin & Hines. "The phase correlation image alignment method",
              proceedings of the IEEE international conference on cybernetics
              and society, pp. 163-165, 1975.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _phase_corr_core(I1,I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        Q = S1 * np.conj(S2)
        Q = normalize_power_spectrum(Q)
        return Q

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _phase_corr_core(I1sub, I2sub)

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _phase_corr_core(I1sub[...,i], I2sub[...,i])
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def gaussian_transformed_phase_corr(I1, I2):
    """ match two imagery through Gaussian transformed phase correlation, see
    [Ec08]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.phase_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 {\mathbf{S}_2}^{\star}
    .. math:: \mathbf{Q}_{12} / \| \mathbf{Q}_{12}\|
    .. math:: \mathbf{Q}_{12} = \mathbf{W} \mathbf{Q}_{12}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation.

    Schematically this looks like:

        .. code-block:: text

          I1  ┌----┐ S1                  ┌----┐
           ---┤ F  ├-----┐               | W  |
              └----┘     | Q12 ┌-------┐ └-┬--┘ ┌------┐ C12
                         ×-----┤ 1/|x| ├---┴----┤ F^-1 ├----
          I2  ┌----┐ S2  |     └-------┘        └------┘
           ---┤ F  ├-----┘
              └----┘


    References
    ----------
    .. [Ec08] Eckstein et al. "Phase correlation processing for DPIV
           measurements", Experiments in fluids, vol.45 pp.485-500, 2008.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = gaussian_transformed_phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _gaussian_transformed_phase_corr_core(I1,I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        Q = normalize_power_spectrum(S1 * np.conj(S2))
        M = gaussian_mask(S1)
        Q = np.multiply(M, Q)
        return Q

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _gaussian_transformed_phase_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _gaussian_transformed_phase_corr_core(I1sub[...,i],
                                                    I2sub[...,i])
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def upsampled_cross_corr(S1, S2, upsampling=2):
    """ apply cros correlation, and upsample the correlation peak

    Parameters
    ----------
    S1 : numpy.array, size=(m,n), dtype=complex, ndim=2
        array with intensities
    S2 : numpy.array, size=(m,n), dtype=complex, ndim=2
        array with intensities

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.pad_dft
    dhdt.processing.matching_tools_frequency_correlators.upsample_dft

    References
    ----------
    .. [GS08] Guizar-Sicairo, et al. "Efficient subpixel image registration
              algorithms", Applied optics, vol. 33 pp.156--158, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> di,dj = upsampled_cross_corr(im1, im2)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(S1)==np.ndarray, ('please provide an array')
    assert type(S2)==np.ndarray, ('please provide an array')

    (m,n) = S1.shape
    S1,S2 = pad_dft(S1, 2*m, 2*n), pad_dft(S2, 2*m, 2*n)
    Q = normalize_power_spectrum(S1)*np.conj(normalize_power_spectrum(S2))
    C = np.real(np.fft.ifft2(Q))

    ij = np.unravel_index(np.argmax(C), C.shape, order='F')
    di, dj = ij[::-1]

    # transform to shifted fourier coordinate frame (being twice as big)
    i_F = np.fft.fftshift(np.arange(-np.fix(m),m))
    j_F = np.fft.fftshift(np.arange(-np.fix(n),n))

    i_offset, j_offset = i_F[di]/2, j_F[dj]/2

    if upsampling >2:
        i_shift = 1 + np.round(i_offset*upsampling)/upsampling
        j_shift = 1 + np.round(j_offset*upsampling)/upsampling
        F_shift = np.fix(np.ceil(1.5*upsampling)/2)

        CC = np.conj(upsample_dft(
            Q, up_m=np.ceil(upsampling*1.5), up_n=np.ceil(upsampling*1.5),
            upsampling=upsampling, i_offset=F_shift-(i_shift*upsampling),
            j_offset=F_shift-(j_shift*upsampling)))
        ij = np.unravel_index(np.argmax(CC), CC.shape, order='F')
        ddi, ddj = ij[::-1]
        ddi -= (F_shift )
        ddj -= (F_shift )

        i_offset += ddi/upsampling
        j_offset += ddj/upsampling
    return i_offset,j_offset

def cross_corr(I1, I2):
    """ match two imagery through cross correlation in FFT

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.phase_corr
    dhdt.processing.matching_tools_frequency_correlators.upsampled_cross_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 {\mathbf{S}_2}^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a
    complex conjugate operation.

    Schematically this looks like:

        .. code-block:: text

          I1  ┌----┐ S1
           ---┤ F  ├-----┐
              └----┘     | Q12 ┌------┐  C12
                         ×-----┤ F^-1 ├---
          I2  ┌----┐ S2  |     └------┘
           ---┤ F  ├-----┘
              └----┘

    References
    ----------
    .. [HK12] Heid & Kääb. "Evaluation of existing image matching methods for
           deriving glacier surface displacements globally from optical
           satellite imagery", Remote sensing of environment, vol.118
           pp.339-355, 2012.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = cross_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _cross_corr_core(I1,I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        Q = S1 * np.conj(S2)
        return Q

    I1sub, I2sub = make_templates_same_size(I1, I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _cross_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _cross_corr_core(I1sub[...,i], I2sub[...,i])
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def binary_orientation_corr(I1, I2):
    """ match two imagery through binary phase only correlation, see [KJ91]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities, if multiple bands are given the cross-
        correlation spectra are stacked, see [AL22]_.
    I2 : numpy.array, size=(m,n), ndim={2,3}
        array with intensities

    Returns
    -------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.orientation_corr
    dhdt.processing.matching_tools_frequency_correlators.phase_only_corr

    Notes
    -----
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1],\quad
              \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{W} = sign(\mathfrak{Re} [ \mathbf{S}_2 ] )
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 [\mathbf{W}\mathbf{S}_2]^{\star}

    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star`
    a complex conjugate operation

    References
    ----------
    .. [KJ91] Kumar & Juday, "Design of phase-only, binary phase-only, and
              complex ternary matched filters with increased signal-to-noise
              ratios for colored noise", Optics letters, vol.16(13)
              pp.1025-1027, 1991.
    .. [AL22] Altena & Leinss, "Improved surface displacement estimation through
              stacking cross-correlation spectra from multi-channel imagery",
              Science of Remote Sensing, vol.6 pp.100070, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = binary_orientation_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1)==np.ndarray, ('please provide an array')
    assert type(I2)==np.ndarray, ('please provide an array')

    def _binary_orientation_corr_core(I1,I2):
        S1, S2 = np.fft.fft2(I1), np.fft.fft2(I2)
        W = np.sign(np.real(S2))
        Q = S1 * np.conj(W * S2)
        return Q

    I1sub, I2sub = make_templates_same_size(I1,I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    if (I1.ndim!=3) and (I2.ndim!=3): # single pair processing
        Q = _binary_orientation_corr_core(I1sub, I2sub)
        return Q

    # multi-spectral frequency stacking
    bands = I1.shape[2]
    for i in range(bands): # loop through all bands
        Q_b = _binary_orientation_corr_core(I1sub[...,i], I2sub[...,i])
        if i == 0:
            Q = Q_b.copy()/bands
        else:
            Q += Q_b/bands
    return Q

def masked_corr(I1, I2, M1=np.array(()), M2=np.array(())):
    """ match two imagery through masked normalized cross-correlation in FFT,
    see [Pa11]_.

    Parameters
    ----------
    I1 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    I2 : {numpy.array, numpy.ma}, size=(m,n), ndim=2
        array with intensities
    M1 : numpy.array, size=(m,n)
        array with mask, hence False's are neglected
    M2 : numpy.array, size=(m,n)
        array with mask, hence False's are neglected

    Returns
    -------
    NCC : numpy.array, size=(m,n)
        correlation surface

    See Also
    --------
    dhdt.processing.matching_tools_spatial_correlators.normalized_cross_corr
    dhdt.processing.matching_tools_spatial_correlators.weighted_normalized_cross_corr

    References
    ----------
    .. [Pa11] Padfield. "Masked object registration in the Fourier domain",
              IEEE transactions on image processing, vol. 21(5) pp. 2706-2718,
              2011.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> msk1,msk2 = np.ones_like(im1), np.ones_like(im2)
    >>> Q = masked_corr(im1, im2, msk1, msk2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    """
    assert type(I1) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.array), \
        ("please provide an array")
    assert (type(M1)==np.ndarray) or (M1 is None), ('please provide an array')
    assert (type(M2)==np.ndarray) or (M2 is None), ('please provide an array')

    # init
    I1,M1, I2,M2 = get_data_and_mask(I1, M1), get_data_and_mask(I2, M2)

    I1sub, I2sub = make_templates_same_size(I1,I2)
    I1sub, I2sub = make_template_float(I1sub), make_template_float(I2sub)
    M1sub, M2sub = make_templates_same_size(M1,M2)

    # preparation
    I1f, I2f = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
    M1f, M2f = np.fft.fft2(M1sub), np.fft.fft2(M2sub)

    fF1F2 = np.fft.ifft2( I1f*np.conj(I2f) )
    fM1M2 = np.fft.ifft2( M1f*np.conj(M2f) )
    fM1F2 = np.fft.ifft2( M1f*np.conj(I2f) )
    fF1M2 = np.fft.ifft2( I1f*np.conj(M2f) )

    ff1M2 = np.fft.ifft2( np.fft.fft2(I1sub**2)*np.conj(M2f) )
    fM1f2 = np.fft.ifft2( M1f*np.fft.fft2( np.flipud(I2sub**2) ) )


    NCC_num = fF1F2 - \
        (np.divide(np.multiply( fF1M2, fM1F2 ), fM1M2,
                   out=np.zeros_like(fM1M2), where=fM1M2!=0))
    NCC_den_den = np.divide(fF1M2**2, fM1M2,
                            out=np.zeros_like(fM1M2), where=fM1M2!=0)
    NCC_den = np.multiply(np.sqrt(ff1M2 - NCC_den_den ),
                          np.sqrt(fM1f2 - NCC_den_den ))
    NCC = np.divide(NCC_num, NCC_den,
                    out=np.zeros_like(NCC_den),
                    where=np.abs(NCC_den)!=0 )
    return np.real(NCC)
