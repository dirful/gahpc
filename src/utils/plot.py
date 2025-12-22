import matplotlib.pyplot as plt

def draw_loss(d_losses, g_losses):
    plt.plot(d_losses)
    plt.plot(g_losses)
    plt.title("GAN Loss")
    plt.show()
