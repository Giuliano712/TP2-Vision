import matplotlib.pyplot as plt
import numpy as np

A = np.random.rand(500,500)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7, 7),
                        gridspec_kw=dict(width_ratios=[4,1,1]))
axs[0, 0].set_title('A')
axs[0, 0].imshow(A)
axs[0, 1].set_title('B')
axs[0, 1].imshow(A)
axs[1, 0].set_title('C')
axs[1, 0].imshow(A)
axs[1, 1].set_title('D')
axs[1, 1].imshow(A)
axs[0, 2].set_title('E')
axs[0, 2].imshow(A)
axs[1, 2].set_title('F')
axs[1, 2].imshow(A)
plt.show()