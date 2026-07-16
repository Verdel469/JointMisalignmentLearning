import matplotlib.pyplot as plt
def set_latex_font(font_size = 10):

    plt.rc('font',  size = font_size)

    plt.rc('text', usetex = True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

    plt.rc('font', family='serif')
