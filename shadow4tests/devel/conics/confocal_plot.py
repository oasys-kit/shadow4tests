import numpy
from srxraylib.plot.gol import plot


x = numpy.linspace(-5, 5, 100)

a = 4
b = 2
c = numpy.sqrt(a**2 - b**2)

A = 2
B = numpy.sqrt(c**2 - A**2)
print(">>>>> B: ", B)
# B = 3

y1e =   b * numpy.sqrt(1 - (x/a)**2)
y2e = - b * numpy.sqrt(1 - (x/a)**2)

y1h =   B * numpy.sqrt((x/A)**2 - 1)
y2h = - B * numpy.sqrt((x/A)**2 - 1)

P = 0.5
y1p =   x**2 / (4 * P)

plot(x, y1e, x, y2e,
    x, y1h, x, y2h,
    x, y1p,
     [-c,-c], [-.1, .1],
    [c,c], [-.1, .1],
     color=['orange','orange','blue','blue','green','black','green'],
     xrange=[-5,5], yrange=[-5,5],
    )

print("Ellipse a, b: ", a, b)
print("Hyperbola A, B: ", A, B)
print("c:", c)
print("Parabola P", P)
