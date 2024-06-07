import matplotlib
import matplotlib.pyplot as plt
import numpy

y = []
with open("logs.txt", "r") as f:
    for line in f:
        y.append(float(line))

matplotlib.use('agg')

plt.plot(y)

z = numpy.polyfit(numpy.arange(len(y)), y, 1)
p = numpy.poly1d(z)
plt.plot(numpy.arange(len(y)), p(numpy.arange(len(y))))

# z = numpy.polyfit(numpy.arange(len(y)), y, 2)
# p = numpy.poly1d(z)
# plt.plot(numpy.arange(len(y)), p(numpy.arange(len(y))))

plt.plot([0, len(y)], [sum(y) / len(y), sum(y) / len(y)])

plt.savefig("graph.png")
# plt.show()
