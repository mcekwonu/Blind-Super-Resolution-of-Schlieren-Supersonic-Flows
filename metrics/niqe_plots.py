import os
import pandas as pd
import matplotlib.pyplot as plt


source_dir = "/media/mce/DrMCE_Crucial/JOV_2023/niqe_score.csv"
df = pd.read_csv(source_dir)

plt.figure(1)
plt.plot(df["NPR"], df["ESRGAN_HKNE_x4"], "r-*", label="ESRGAN_HKNE_x4")
plt.plot(df["NPR"], df["ESRGAN_VKNE_x4"], "g-o", label="ESRGAN_VKNE_x4")
plt.plot(df["NPR"], df["RESRGAN_HKNE_x4"], "b-+", label="RealESRGAN_HKNE_x4")
plt.plot(df["NPR"], df["RESRGAN_VKNE_x4"], "m-^", label="RealESRGAN_VKNE_x4")
plt.ylabel("NIQE score", fontsize=14)
plt.xlabel("NPR", fontsize=14)
plt.legend(loc="best", frameon=False)
plt.tight_layout()
plt.savefig("niqe_x4_plots.png", dpi=600)
plt.show()
# plt.pause(5)
plt.close()

plt.figure(2)
plt.plot(df["NPR"], df["ESRGAN_HKNE_x8"], "r-*", label="ESRGAN_HKNE_x8")
plt.plot(df["NPR"], df["ESRGAN_VKNE_x8"], "g-o", label="ESRGAN_VKNE_x8")
plt.plot(df["NPR"], df["RESRGAN_HKNE_x8"], "b-+", label="RealESRGAN_HKNE_x8")
plt.plot(df["NPR"], df["RESRGAN_VKNE_x8"], "m-^", label="RealESRGAN_VKNE_x8")
plt.ylabel("NIQE score", fontsize=14)
plt.xlabel("NPR", fontsize=14)
plt.legend(loc="best", frameon=False)
plt.tight_layout()
plt.savefig("niqe_x8_plots.png", dpi=600)
plt.show()