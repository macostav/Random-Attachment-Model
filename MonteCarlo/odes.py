def odes(t,y):
    """
    Return odes for the system.

    :param t: time
    :param y: state vector
    :return: odes
    """
    n = y

    dn = (N_A - n) * (N_B - n) - np.exp(energy) * n

    return [dn]