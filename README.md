import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 参数设定
λ = 2.0
γ = 1.0

def f(e):
    """净效用函数 f(e) = λ√e - ½γ e²"""
    return λ * np.sqrt(e) - 0.5 * γ * e**2

def f_prime(e):
    """一阶导数，用于求最大值点"""
    return λ * 0.5 / np.sqrt(e) - γ * e

# 求最大值点 e*
e_opt = fsolve(f_prime, 1.0)[0]
f_max = f(e_opt)

# 生成 e 值（从略大于0到3）
e = np.linspace(0.01, 3, 500)
y = f(e)

# 绘图
plt.figure(figsize=(8,5))
plt.plot(e, y, 'b-', linewidth=2.5,
         label=r'$f(e)=\lambda \sqrt{e} - \frac{1}{2}\gamma e^2$')
plt.scatter(e_opt, f_max, color='red', zorder=5,
            label=f'唯一最大值点\n$e^*={e_opt:.3f}$, $f(e^*)={f_max:.3f}$')
plt.axvline(e_opt, color='gray', linestyle='--', linewidth=0.8)

# 画一条弦（展示凹性）
e1, e2 = 0.5, 2.5
y1, y2 = f(e1), f(e2)
plt.plot([e1, e2], [y1, y2], 'r--', linewidth=1,
         label='任意弦（全部在曲线下方）')

plt.xlabel(r'$e$ (Prosumption 努力)', fontsize=12)
plt.ylabel(r'$f(e)$ (净效用)', fontsize=12)
plt.title('严格凹函数：唯一内点最大值', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"最优努力水平 e* = {e_opt:.4f}")
print(f"最大净效用 f(e*) = {f_max:.4f}")
