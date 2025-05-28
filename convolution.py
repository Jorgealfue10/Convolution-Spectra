import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve


def lorentzian(x, a, x0):
    return (1 / np.pi) * ((0.5 * a) / ((x - x0) ** 2 + (0.5 * a) ** 2))


def extract_label(filepath, mode):
    if mode == "Temp":
        label = "T = "
        for char in filepath:
            if char == "/":
                break
            elif char != "T":
                label += char
        return f"{label} K"
    else:
        label = ""
        inside = False
        for char in filepath:
            if char == "/":
                inside = True
                continue
            if inside:
                if char == ".":
                    break
                label += "->" if char == "_" else char
        return label


def process_spectrum(data, amp, stddev):
    emin, emax = np.min(data[:, 2]), np.max(data[:, 2])
    npoints = int((emax + 500) - (emin - 500))
    x = np.zeros((data.shape[0], npoints))
    y = np.zeros((data.shape[0], npoints))
    shifted = data[:, 2] - emin + 100

    for i, x0 in enumerate(shifted):
        max_intensity = lorentzian(x0, amp, x0)
        for j in range(npoints):
            x[i, j] = j if abs(j - x0) >= 0.1 else x0
            y[i, j] = lorentzian(x[i, j], amp, x0) / max_intensity * data[i, 4]

    x += emin - 100
    max_y = np.max(y, axis=0)
    kernel = Gaussian1DKernel(stddev=stddev)
    convoluted = convolve(max_y, kernel, normalize_kernel=True, boundary='extend')
    return x[1], convoluted / np.max(convoluted) * np.max(data[:, 4])


def plot_spectra(files, rots, label_mode, amp, output, stddev, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    for filepath in files:
        data = np.loadtxt(filepath)
        x_vals, y_vals = process_spectrum(data, amp, stddev)

        ax.plot(x_vals, y_vals, alpha=1)
        label = extract_label(filepath, label_mode)
        ax.fill_between(x_vals, 0, y_vals, alpha=0.2, label=label)

        if xlim:
            ax.set_xlim(*xlim)

        if rots:
            ax.vlines(data[:, 2], 0, data[:, 4], color="black")
            for i in range(len(data)):
                if data[i, 4] > 0.1e-20 and data[i, 0] != data[i, 1]:
                    ax.annotate(f"{int(data[i, 1])}$\rightarrow${int(data[i, 0])}",
                                xy=(data[i, 2] - 8, data[i, 4] + 0.8e-22),
                                rotation=90, fontsize=10)

    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_xlabel(r"$\nu$ ($cm^{-1}$)", fontsize=30)
    ax.set_ylabel(r"$I$ (cm/mol)", fontsize=30)
    ax.legend(fontsize=30, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{output}.png", dpi=300, transparent=True)


def main():
    parser = argparse.ArgumentParser(
        description="Procesa y grafica espectros con convolución Gaussiana + picos Lorentzianos."
    )
    parser.add_argument("files", nargs="+", help="Lista de archivos de datos")
    parser.add_argument("--rots", action="store_true", help="Mostrar transiciones rotacionales")
    parser.add_argument("--label", choices=["Temp", "Trans"], default="Trans", help="Tipo de etiquetas en las curvas")
    parser.add_argument("--amp", type=float, required=True, help="Ancho del pico Lorentziano")
    parser.add_argument("--output", required=True, help="Nombre base del archivo PNG de salida")
    parser.add_argument("--stddev", type=float, default=1.5, help="Desviación estándar del suavizado Gaussiano")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=('XMIN', 'XMAX'),
                        help="Límites del eje x (min max)")

    args = parser.parse_args()

    plot_spectra(
        files=args.files,
        rots=args.rots,
        label_mode=args.label,
        amp=args.amp,
        output=args.output,
        stddev=args.stddev,
        xlim=tuple(args.xlim) if args.xlim else None
    )


if __name__ == "__main__":
    main()
