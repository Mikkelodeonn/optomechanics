import numpy as np
import matplotlib.pyplot as plt

## 20250226 452um cavity -> plot these for the appendix

lws = [78.367, 72.957, 68.413, 55.656, 56.452, 54.841, 63.918, 57.093]
lw_errs = [1.578, 1.492, 1.757, 2.118, 2.124, 1.818, 1.971, 1.970]

meas_num = [1,2,3,4,5,6,7,8]

plt.figure(figsize=(10,7))
plt.errorbar(meas_num, lws, lw_errs, fmt="o", capsize=3, color="royalblue")
plt.xlabel("measurement #", fontsize=24)
plt.ylabel("HWHM [pm]", fontsize=24)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
#plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)
#plt.subplots_adjust(bottom=0.3)
plt.show()