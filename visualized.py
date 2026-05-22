import os
import sys

# Respect env vars for model selection — coarse_to_fine by default (RTX 4060 profile)
os.environ.setdefault("BME_MODEL", "coarse_to_fine")
os.environ.setdefault("BME_ROI_SIZE", "64")
os.environ.setdefault("BME_FEATURE_SIZE", "12")
os.environ.setdefault("BME_DUAL_2D_FEATURE_SIZE", "12")
os.environ.setdefault("BME_CTF_CONTEXT_SLICES", "3")

from visaulize import main

if __name__ == "__main__":
    main()
