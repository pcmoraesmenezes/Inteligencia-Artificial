import numpy as np
import matplotlib.pyplot as plt

def rbf_gaussian(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)

def plot_similarity():
    x1s = np.linspace(-4.5, 4.5, 20).reshape(-1, 1)
    x2s = rbf_gaussian(x1s, -2, 1)
    x3s = rbf_gaussian(x1s, 1, 0.3)

    XK = np.c_[rbf_gaussian(x1s, -2, 1), rbf_gaussian(x1s, 1, 0.3)]
    
    # Fix: Create a label array matching the number of rows in XK
    yk = np.random.choice([0, 1], size=XK.shape[0])

    plt.figure(figsize=(11, 4))

    # First plot
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")
    plt.plot(x1s, x2s, "g--", label="Gaussian (x2)")
    plt.plot(x1s, x3s, "b:", label="Gaussian (x3)")
    plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"Similarity", fontsize=14)
    plt.annotate(r'$\mathbf{x}$',
                 xy=(x1s[3], x2s[3]),
                 xytext=(-0.5, 0.5),
                 textcoords='offset points',
                 ha='center',
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=14,
                 )
    plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=20)
    plt.text(1, 0.9, "$x_3$", ha="center", fontsize=20)
    plt.axis([-4.5, 4.5, -0.1, 1.1])
    plt.legend()

    # Second plot
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.scatter(XK[yk == 1, 0], XK[yk == 1, 1], s=150, alpha=0.5, c="red", label="Class 1")
    plt.plot(XK[yk == 1, 0], XK[yk == 1, 1], "bs", label="Positive class")
    plt.plot(XK[yk == 0, 0], XK[yk == 0, 1], "g^", label="Negative class")
    plt.xlabel(r"$x_2$", fontsize=20)
    plt.ylabel(r"$x_3$", fontsize=20, rotation=0)
    plt.annotate(r'$\phi\left(\mathbf{x}\right)$',
                 xy=(XK[3, 0], XK[3, 1]),
                 xytext=(-0.5, 0.5),
                 textcoords='offset points',
                 ha='center',
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=14,
                 )
    plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_similarity()
