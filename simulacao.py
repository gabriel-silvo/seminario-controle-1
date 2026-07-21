import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---- Os tres sistemas do Problema 20: (rotulo, num, den) ----
sistemas = [
    ("a", [16], [1, 3, 16]),
    ("b", [0.04], [1, 0.02, 0.04]),
    ("c", [1.05e7], [1, 1.6e3, 1.05e7]),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (rot, num, den) in zip(axes, sistemas):
    # ---- Previsao teorica (formulas de 2a ordem) ----
    wn = np.sqrt(den[2])
    zeta = den[1] / (2 * wn)
    wd = wn * np.sqrt(1 - zeta**2)
    Tp = np.pi / wd
    OS = 100 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    Ts = 4 / (zeta * wn)
    Tr = (
        1.76 * zeta**3 - 0.417 * zeta**2 + 1.039 * zeta + 1
    ) / wn  # aprox. de Tr (Nise)

    print(f"--- Sistema ({rot}) ---")
    print(f"  wn = {wn:.4g} rad/s | zeta = {zeta:.4g}")
    print(f"  Teoria : Tp={Tp:.4g}s  %OS={OS:.2f}%  Ts={Ts:.4g}s  Tr={Tr:.4g}s")

    # ---- Simulacao da resposta ao degrau ----
    t = np.linspace(0, 1.5 * Ts, 4000)
    t, y = signal.step(signal.TransferFunction(num, den), T=t)

    yf = y[-1]
    i_pk = np.argmax(y)
    OS_sim = (y[i_pk] - yf) / yf * 100
    Tp_sim = t[i_pk]
    fora = np.where(np.abs(y - yf) > 0.02 * yf)[0]
    Ts_sim = t[fora[-1] + 1] if len(fora) else 0
    print(f"  Simul. : Tp={Tp_sim:.4g}s  %OS={OS_sim:.2f}%  Ts={Ts_sim:.4g}s")

    # ---- Grafico ----
    ax.plot(t, y, color="#175E7A", lw=2)
    ax.axhline(yf, ls="--", color="gray", lw=0.8)
    ax.plot(Tp_sim, y[i_pk], "o", color="#F08A24", ms=7)
    ax.set_title(f"Sistema ({rot})  |  zeta={zeta:.3g}, wn={wn:.4g}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("c(t)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./problema20-tres-respostas.png", dpi=150, facecolor="white")
print("\ngrafico salvo")
