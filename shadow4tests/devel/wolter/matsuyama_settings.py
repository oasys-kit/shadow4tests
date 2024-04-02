import numpy

# https://drive.google.com/drive/folders/1o3kykSIT4NPAEuF6oHEg-an5-9WGhT6K


#
# ellipse
#
# a=22.75
# b = 58.46e-3
# theta = 2.8e-3
# #
# q1 = a + numpy.sqrt(a**2 - (b / numpy.sin(theta)) **2)
# q2 = a - numpy.sqrt(a**2 - (b / numpy.sin(theta)) **2)
#
#
# p1 = 2 * a - q1
# p2 = 2 * a - q2
#
# print("p: ", p1, p2)
# print("q: ", q1, q2)



# p = 45.15 / numpy.cos(theta)
# q = 215e-3
# p=p2
# q=q2
# # theta = 2.8e-3
#
# a = 0.5 * (p + q)
# b = numpy.sqrt(p * q) * numpy.sin(theta)
#
# print("a,b", a,b)
#
theta = 2.8e-3
p = 45.17
a = 22.75


q = 2 * a - p
print("p, q: ", p, q)

b = numpy.sqrt(p * q) * numpy.sin(theta)
print("a, b: ", a, b)

q1 = a + numpy.sqrt(a**2 - (b / numpy.sin(theta)) **2)
q2 = a - numpy.sqrt(a**2 - (b / numpy.sin(theta)) **2)


# p1 = 2 * a - q1
# p2 = 2 * a - q2
#
# print("p: ", p1, p2)
# print("q: ", q1, q2)

# a = 0.5 * (p + q)
# b = numpy.sqrt(p * q) * numpy.sin(theta)
#
# print("a,b", a,b)