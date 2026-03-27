# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

from datasets import Image as HFImage
from datasets import load_dataset

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class SafeDocs(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "SafeDocs.Split",
        root: Optional[str] = None,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        dataset_name = root or "albertklorer/safedocs"
        super().__init__(
            root=dataset_name,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self._split = split
        self._cache_dir = extra
        self._dataset = load_dataset(dataset_name, split=split.value, cache_dir=extra)
        self._dataset = self._dataset.cast_column("image", HFImage(decode=False))

    @property
    def split(self) -> "SafeDocs.Split":
        return self._split

    def get_image_relpath(self, index: int) -> str:
        sample = self._dataset[index]
        return f"{sample['foldername']}/{sample['filename']}#page={sample['page_number']}"

    def get_image_data(self, index: int) -> bytes:
        image = self._dataset[index]["image"]
        image_bytes = image.get("bytes")
        if image_bytes is not None:
            return bytes(image_bytes)

        image_path = image.get("path")
        if image_path is None:
            raise RuntimeError(f"SafeDocs sample {index} has no image bytes or cached path")

        with open(Path(image_path), mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> int:
        # SSL pretraining discards the target, so a constant placeholder is enough.
        return 0

    def __len__(self) -> int:
        return len(self._dataset)
