# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .checkpointer import (
    CheckpointRetentionPolicy,
    cleanup_checkpoint,
    find_all_checkpoints,
    find_latest_checkpoint,
    init_fsdp_backbone_from_hf_hub,
    init_fsdp_model_from_checkpoint,
    init_model_from_checkpoint_for_evals,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    register_dont_save_hooks,
    resolve_hf_hub_repo_id,
    resolve_hf_hub_token,
    save_checkpoint,
    upload_checkpoint_to_hf_hub,
)
