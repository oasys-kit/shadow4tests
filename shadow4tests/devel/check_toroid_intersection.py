

def s4(r_maj=10.0, r_min=1.0, P1=0, P2=0, P3=0, V1=0, V2=1, V3=0):
    A = r_maj ** 2 - r_min ** 2
    B = -(r_maj ** 2 + r_min ** 2)

    AA = P1 * V1 ** 3 + P2 * V2 ** 3 + P3 * V3 ** 3 + \
         V1 * V2 ** 2 * P1 + V1 ** 2 * V2 * P2 + \
         V1 * V3 ** 2 * P1 + V1 ** 2 * V3 * P3 + \
         V2 * V3 ** 2 * P2 + V2 ** 2 * V3 * P3
    AA = 4 * AA

    BB = 3 * P1 ** 2 * V1 ** 2 + 3 * P2 ** 2 * V2 ** 2 + \
         3 * P3 ** 2 * V3 ** 2 + \
         V2 ** 2 * P1 ** 2 + V1 ** 2 * P2 ** 2 + \
         V3 ** 2 * P1 ** 2 + V1 ** 2 * P3 ** 2 + \
         V3 ** 2 * P2 ** 2 + V2 ** 2 * P3 ** 2 + \
         A * V1 ** 2 + B * V2 ** 2 + B * V3 ** 2 + \
         4 * V1 * V2 * P1 * P2 + \
         4 * V1 * V3 * P1 * P3 + \
         4 * V2 * V3 * P2 * P3
    BB = 2 * BB

    CC = P1 ** 3 * V1 + P2 ** 3 * V2 + P3 ** 3 * V3 + \
         P2 * P1 ** 2 * V2 + P1 * P2 ** 2 * V1 + \
         P3 * P1 ** 2 * V3 + P1 * P3 ** 2 * V1 + \
         P3 * P2 ** 2 * V3 + P2 * P3 ** 2 * V2 + \
         A * V1 * P1 + B * V2 * P2 + B * V3 * P3
    CC = 4 * CC

    DD = P1 ** 4 + P2 ** 4 + P3 ** 4 + \
         2 * P1 ** 2 * P2 ** 2 + 2 * P1 ** 2 * P3 ** 2 + \
         2 * P2 ** 2 * P3 ** 2 + \
         2 * A * P1 ** 2 + 2 * B * P2 ** 2 + 2 * B * P3 ** 2 + \
         A ** 2

    return 1, AA, BB, CC, DD

def m4(r_maj=10.0, r_min=1.0, P1=0, P2=0, P3=0, V1=0, V2=1, V3=0):
    A = r_maj ** 2 - r_min ** 2

    PdotV = P1 * V1 + P2 * V2 + P3 * V3
    PdotP = P1**2 + P2**2 + P3**2

    AA = 4 * PdotV

    BB = 4 * PdotV**2 + 2 * (A + PdotP) - 4 * r_maj**2 * (V2**2 + V3**2)

    CC = 4 * PdotV * (A + PdotP) - 8 * r_maj**2 * (P2*V2 + P3*V3)

    DD = (A + PdotP)**2 - 4 * r_maj**2 * (P2**2 + P3**2)

    return 1, AA, BB, CC, DD

if __name__ == "__main__":
    from numpy import sqrt
    # print(s4(r_maj=10.0, r_min=9.0, P1=0, P2=1, P3=1, V1=1/sqrt(3), V2=1/sqrt(3), V3=1/sqrt(3)))
    # print(m4(r_maj=10.0, r_min=9.0, P1=0, P2=1, P3=1, V1=1/sqrt(3), V2=1/sqrt(3), V3=1/sqrt(3)))

    print(s4(r_maj=520.859, r_min=0.006347, P1=-4.08432e-07, P2=-9.99994, P3=-520.83, V1=4.65873e-06, V2=0.999994, V3=-0.00349504))
    print(m4(r_maj=520.859, r_min=0.006347, P1=-4.08432e-07, P2=-9.99994, P3=-520.83, V1=4.65873e-06, V2=0.999994, V3=-0.00349504))