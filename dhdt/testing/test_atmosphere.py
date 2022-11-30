import numpy as np

from ..generic.unit_conversion import celsius2kelvin
from ..preprocessing.atmospheric_geometry import \
    get_sat_vapor_press, get_water_vapor_enhancement, \
    refractive_index_visible, refractive_index_broadband

# testing functions, based on literature values
def test_water_vapour_fraction():
    """ test if the calculations comply with the original work, using values
    of Table2 in [1]

    References
    ----------
    .. [1] Giacomo, "Equation for the determination of the density of moist air"
           Metrologica, vol.18, pp.33-40, 1982.
    """
    t = np.arange(0, 27.1, 0.1)
    svp_tilde = np.array([611.2, 615.7, 620.2, 624.7, 629.2, 633.8, 638.4,
                          643.0, 647.7, 652.4, 657.1, 661.8, 666.6, 671.4,
                          676.2, 681.1, 686.0, 691.0, 695.9, 700.9, 705.9,
                          711.0, 716.1, 721.2, 726.4, 731.6, 736.8, 742.1,
                          747.3, 752.7, 758.0, 763.4, 768.8, 774.3, 779.8,
                          785.3, 790.9, 796.5, 802.1, 807.8, 813.5, 819.2,
                          825.0, 830.8, 836.6, 842.5, 848.4, 854.4, 860.4,
                          866.4, 872.5, 878.6, 884.7, 890.9, 897.1, 903.4,
                          909.7, 916.0, 922.4, 928.8, 935.2, 941.7, 948.2,
                          954.8, 961.4, 968.1, 974.8, 981.5, 988.3, 995.1,
                          1001.9, 1008.8, 1015.8, 1022.7, 1029.8, 1036.8,
                          1043.9, 1051.1, 1058.3, 1065.5, 1072.8, 1080.1,
                          1087.5, 1094.9, 1102.4, 1109.9, 1117.4, 1125.0,
                          1132.6, 1140.3, 1148.1, 1155.8, 1163.7, 1171.5,
                          1179.4, 1187.4, 1195.4, 1203.5, 1211.6, 1219.7,
                          1227.9, 1236.2, 1244.5, 1252.8, 1261.2, 1269.7,
                          1278.2, 1286.7, 1295.3, 1304.0, 1312.7, 1321.4,
                          1330.2, 1339.1, 1348.0, 1356.9, 1366.0, 1375.0,
                          1384.1, 1393.3, 1402.5, 1411.8, 1421.1, 1430.5,
                          1439.9, 1449.4, 1459.0, 1468.6, 1478.2, 1488.0,
                          1497.7, 1507.5, 1517.4, 1527.4, 1537.4, 1547.4,
                          1557.5, 1567.7, 1577.9, 1588.2, 1598.6, 1609.0,
                          1619.4, 1630.0, 1640.5, 1651.2, 1661.9, 1672.6,
                          1683.5, 1694.4, 1705.3, 1716.3, 1727.4, 1738.5,
                          1749.8, 1761.0, 1772.3, 1783.7, 1795.2, 1806.7,
                          1818.3, 1829.9, 1841.7, 1853.4, 1865.3, 1877.2,
                          1889.2, 1901.2, 1913.3, 1925.5, 1937.8, 1950.1,
                          1962.5, 1974.9, 1987.5, 2000.1, 2012.7, 2025.5,
                          2038.3, 2051.1, 2064.1, 2077.1, 2090.2, 2103.4,
                          2116.6, 2129.9, 2143.3, 2156.8, 2170.3, 2183.9,
                          2197.6, 2211.3, 2225.2, 2239.1, 2253.0, 2267.1,
                          2281.2, 2295.4, 2309.7, 2324.1, 2338.5, 2353.1,
                          2367.7, 2382.4, 2397.1, 2412.0, 2426.9, 2441.9,
                          2456.9, 2472.1, 2487.4, 2502.7, 2518.1, 2533.6,
                          2549.2, 2564.8, 2580.6, 2596.4, 2612.3, 2628.3,
                          2644.4, 2660.6, 2676.8, 2693.2, 2709.6, 2726.1,
                          2742.8, 2759.4, 2776.2, 2793.1, 2810.1, 2827.1,
                          2844.3, 2861.5, 2878.8, 2896.2, 2913.7, 2931.3,
                          2949.0, 2966.8, 2984.7, 3002.7, 3020.7, 3038.9,
                          3057.2, 3075.5, 3094.0, 3112.5, 3131.2, 3149.9,
                          3168.7, 3187.7, 3206.7, 3225.8, 3245.1, 3264.4,
                          3283.8, 3303.4, 3323.0, 3342.8, 3362.6, 3382.5,
                          3402.6, 3422.7, 3443.0, 3463.3, 3483.8, 3504.4,
                          3525.0, 3545.8, 3566.7])
    T = celsius2kelvin(t)
    svp = get_sat_vapor_press(T)
    assert np.all(np.isclose(svp, svp_tilde, atol=1E2))
    return

def test_water_vapor_enhancement():
    """ test if the calculations comply with the original work, using values
    of Table3 in [1]

    References
    ----------
    .. [1] Giacomo, "Equation for the determination of the density of moist air"
           Metrologica, vol.18, pp.33-40, 1982.
    """
    f_hat = np.array([[1.0024, 1.0025, 1.0025, 1.0026, 1.0028, 1.0029, 1.0031],
                    [1.0026, 1.0026, 1.0027, 1.0028, 1.0029, 1.0031, 1.0032],
                    [1.0028, 1.0028, 1.0029, 1.0029, 1.0031, 1.0032, 1.0034],
                    [1.0029, 1.0030, 1.0030, 1.0031, 1.0032, 1.0034, 1.0035],
                    [1.0031, 1.0031, 1.0032, 1.0033, 1.0034, 1.0035, 1.0037],
                    [1.0033, 1.0033, 1.0033, 1.0034, 1.0035, 1.0036, 1.0038],
                    [1.0035, 1.0035, 1.0035, 1.0036, 1.0037, 1.0038, 1.0039],
                    [1.0036, 1.0036, 1.0037, 1.0037, 1.0038, 1.0039, 1.0041],
                    [1.0038, 1.0038, 1.0038, 1.0039, 1.0040, 1.0041, 1.0042],
                    [1.0040, 1.0040, 1.0040, 1.0040, 1.0041, 1.0042, 1.0044],
                    [1.0042, 1.0041, 1.0041, 1.0042, 1.0042, 1.0044, 1.0045]])
    p, t = np.mgrid[60E3:115E3:5E3, 0:35:5]
    f = get_water_vapor_enhancement(t.ravel(), p.ravel())
    assert np.all(np.isclose(f, f_hat.ravel(), atol=1E-3))
    return

def test_refraction_calculation():
    """ follow the examples in the paper [1], to see if they are correct

    References
    ----------
    .. [1] Birch and Jones, "Correction to the updated Edlen equation for the
       refractive index of air", Metrologica, vol.31(4) pp.315-316, 1994.
    .. [2] Ciddor, "Refractive index of air: new equations for the visible and
       near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """

    # from Table 3 in [1], or Table 1 in [2]
    sigma = 633.                                # central wavelength in [nm]
    sigma /= 1E3                                # convert to µm
    T = np.array([20., 20., 20., 10., 30.])     # temperature in [Celsius]
    P = np.array([80., 100., 120., 100., 100.]) # atm. pressure in [kiloPascal]
    P *= 1E3
    n_0 = refractive_index_visible(sigma, T, P)

    n_tilde = np.array([21458.0, 26824.4, 32191.6, 27774.7, 25937.2])
    n_tilde *= 1E-8 # convert to refraction
    n_tilde += 1
    assert np.all(np.isclose(n_0, n_tilde, atol=1E-8))

    frac_Hum = np.array([0., 0., 0., 0., 0.])
    n_0 = refractive_index_broadband(sigma, T, P, frac_Hum, LorentzLorenz=False) #todo
    assert np.all(np.isclose(n_0, n_tilde, atol=1E-8))

    # from Table.4 in [1], for ambient air
    T = np.array([19.526, 19.517, 19.173, 19.173, 19.188,
                  19.189, 19.532, 19.534, 19.534])
                                            # temperature in [Celsius]
    P = np.array([102094.8, 102096.8, 102993.0, 103006.0, 102918.8,
                  102927.8, 103603.2, 103596.2, 103599.2])
                                            # atmospheric pressure in [Pa]
    p_w = np.array([1065., 1065., 641., 642., 706., 708., 986., 962., 951.])
                                            # vapor pressure of water in [Pa]
    CO2 = np.array([510., 510., 450., 440., 450., 440., 600., 600., 610.])
                                            # carbondioxide concentration, [ppm]
    n_0 = refractive_index_visible(sigma, T, P, p_w, CO2=CO2)
    n_0 = refractive_index_broadband(sigma, T, P, p_w, CO2=CO2)

    n_tilde = np.array([27392.3, 27394.0, 27683.4, 27686.9, 27659.1,
                      27661.4, 27802.1, 27800.3, 27801.8])  # measured refraction
    n_tilde *= 1E-8
    n_tilde += 1
    assert np.all(np.isclose(n_0, n_tilde, atol=1E-8))
    return
