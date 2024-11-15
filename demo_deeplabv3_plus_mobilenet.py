# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from qai_hub_models.models._shared.deeplab.demo import deeplabv3_demo
from qai_hub_models.models._shared.deeplab.model import NUM_CLASSES
from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DeepLabV3PlusMobilenetQuantizable,
)

# Removed CachedWebModelAsset import since it's no longer needed

# Relative path
INPUT_IMAGE_LOCAL_PATH = "images/image11.jpg"

# Convert to absolute path
INPUT_IMAGE_ABSOLUTE_PATH = os.path.abspath(INPUT_IMAGE_LOCAL_PATH)

print("Absolute Path:", INPUT_IMAGE_ABSOLUTE_PATH)

# Ensure the local image exists
if not os.path.exists(INPUT_IMAGE_ABSOLUTE_PATH):
    raise FileNotFoundError(
        f"The image file was not found at {INPUT_IMAGE_ABSOLUTE_PATH}"
    )

# Use the absolute path directly
INPUT_IMAGE_ADDRESS = INPUT_IMAGE_ABSOLUTE_PATH


def main(is_test: bool = False):
    deeplabv3_demo(
        DeepLabV3PlusMobilenetQuantizable,
        MODEL_ID,
        INPUT_IMAGE_ADDRESS,
        NUM_CLASSES,
        is_test,
    )


if __name__ == "__main__":
    main()
