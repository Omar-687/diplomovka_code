import matplotlib.pyplot as plt


# TODO:treba to vykreslovat nazivo tiez dat taku moznost



def draw_graph(array, x_label, y_label, title):
    plt.plot(range(len(array)), array, marker='o')  # x-values are indices, y-values are array values
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()