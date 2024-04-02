import numpy

import numpy as np
def rotate_and_translate_coefficients(coe_list,R_M,T):
    axx, ayy, azz, axy, ayz, axz, ax, ay, az, a0 = coe_list
    A2 = np.array([[axx,axy/2,axz/2],
    [axy/2,ayy,ayz/2],
    [axz/2,ayz/2,azz]])
    A1 = np.array([ax,ay,az])
    A0 = a0
    B2 = np.dot(R_M, np.dot(A2,R_M.T)) # first equation 6.29
    B1 = np.dot(R_M, A1) - 2 * np.dot(B2,T) # 2nd equation 6.29
    B0 = A0 + np.dot(T.T, (np.dot(B2, T) - \
    np.dot(R_M, A1))) # 3rd equation 6.29
    return [ B2[0,0], B2[1,1], B2[2,2], B2[0,1] + B2[1,0], B2[1,2] + B2[2,1], B2[0,2] + B2[2,0], B1[0], B1[1], B1[2], B0]

# factory parameters (input)

Theta = 0.001
# rotation matrix
R_M = np.array([[1,0,0],
[0,np.cos(Theta),-np.sin(Theta)],
[0,np.sin(Theta),np.cos(Theta)]])
# translation vector
T = np.array([0,0,0])


oe2CCC = [0.0, 0.045836362869578186, 5846.4462174678365, 0.0, 32.04741572531343, 0.0, 0.0, 0.0, -31.291997277978908,
          0.0]
oe2DDD = rotate_and_translate_coefficients(oe2CCC,R_M,T)
print("coeffs in centered frame: ", oe2CCC)
print("coeffs in local frame: ", oe2DDD)

oe4CCC = [0.0, -46.20118343205854, -1847960.8414621525, 0.0, -27444.927571197237, 0.0, 0.0, 1.1510792319313623e-11,
          2985.178365734318, 0.0]

oe4DDD = rotate_and_translate_coefficients(oe4CCC,R_M,T)
print("coeffs in centered frame: ", oe4CCC)
print("coeffs in local frame: ", oe4DDD)



