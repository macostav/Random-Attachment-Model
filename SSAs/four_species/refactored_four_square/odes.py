from rates import rates

def odes(t,y):
    """
    Return odes for the system.

    :param t: time
    :param y: state vector
    :return: odes
    """
    A, B, C, D, AB, AC, BD, CD, ABC, ABD, ACD, BCD, ABCD = y
    km = rates

    dA = (km["k2"]*AB + km["k4"]*AC + km["k12"]*ABD + km["k18"]*ACD + km["k32"]*ABCD
            - (km["k1"]*B + km["k3"]*C + km["k11"]*BD + km["k17"]*CD + km["k31"]*BCD)*A)

    dB = (km["k2"]*AB + km["k6"]*BD + km["k16"]*BCD + km["k22"]*ABC + km["k34"]*ABCD
            - (km["k1"]*A + km["k5"]*D + km["k15"]*CD + km["k21"]*AC + km["k33"]*ACD)*B)

    dC = (km["k4"]*AC + km["k8"]*CD + km["k14"]*BCD + km["k24"]*ABC + km["k30"]*ABCD
            - (km["k3"]*A + km["k7"]*D + km["k13"]*BD + km["k23"]*AB + km["k29"]*ABD)*C)

    dD = (km["k6"]*BD + km["k8"]*CD + km["k10"]*ABD + km["k20"]*ACD + km["k36"]*ABCD
            - (km["k5"]*B + km["k7"]*C + km["k9"]*AB + km["k19"]*AC + km["k35"]*ABC)*D)

    dAB = (km["k1"]*A*B + km["k10"]*ABD + km["k24"]*ABC + km["k26"]*ABCD
            - (km["k2"] + km["k9"]*D + km["k23"]*C + km["k25"]*CD)*AB)

    dAC = (km["k3"]*A*C + km["k20"]*ACD + km["k22"]*ABC + km["k28"]*ABCD
            - (km["k4"] + km["k19"]*D + km["k21"]*B + km["k27"]*BD)*AC)

    dBD = (km["k5"]*B*D + km["k12"]*ABD + km["k14"]*BCD + km["k28"]*ABCD
            - (km["k6"] + km["k11"]*A + km["k13"]*C + km["k27"]*AC)*BD)

    dCD = (km["k7"]*C*D + km["k16"]*BCD + km["k18"]*ACD + km["k26"]*ABCD
            - (km["k8"] + km["k15"]*B + km["k17"]*A + km["k25"]*AB)*CD)

    dABC = (km["k21"]*B*AC + km["k23"]*C*AB + km["k36"]*ABCD
                - (km["k22"] + km["k24"] + km["k35"]*D)*ABC)

    dABD = (km["k9"]*D*AB + km["k11"]*A*BD + km["k30"]*ABCD
                - (km["k10"] + km["k12"] + km["k29"]*C)*ABD)

    dACD = (km["k17"]*A*CD + km["k19"]*D*AC + km["k34"]*ABCD
                - (km["k18"] + km["k20"] + km["k33"]*B)*ACD)

    dBCD = (km["k13"]*C*BD + km["k15"]*B*CD + km["k32"]*ABCD
                - (km["k14"] + km["k16"] + km["k31"]*A)*BCD)

    dABCD = (km["k25"]*AB*CD + km["k27"]*AC*BD + km["k29"]*C*ABD + km["k31"]*A*BCD
                + km["k33"]*B*ACD + km["k35"]*D*ABC
                - (km["k26"] + km["k28"] + km["k30"] + km["k32"] + km["k34"] + km["k36"])*ABCD)

    return [dA, dB, dC, dD, dAB, dAC, dBD, dCD, dABC, dABD, dACD, dBCD, dABCD]