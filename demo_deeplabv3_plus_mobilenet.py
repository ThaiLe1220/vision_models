# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.deeplab.demo import deeplabv3_demo
from qai_hub_models.models._shared.deeplab.model import NUM_CLASSES
from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DeepLabV3PlusMobilenetQuantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE_LOCAL_PATH = "/home/ubuntu/Desktop/eugene/qualcomm_model/images/image11.jpg"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, INPUT_IMAGE_LOCAL_PATH
)


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
