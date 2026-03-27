# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Suggested file structure:

output_dir/
|-- ckpt/
|   |-- 0/
|   |-- 99/
|   |-- 199/
|   |-- 199_keep/
|   |-- 299/
|   `-- ...
`-- eval/
    `-- 0/
    `-- 99/
        `-- ckpt/

Distributed checkpointer docs:
- https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- https://pytorch.org/docs/stable/distributed.checkpoint.html
"""

import logging
import json
import os
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Set

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger("dinov3")


class CheckpointRetentionPolicy(Enum):
    ALL = "all"  # keep all checkpoints
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"
    NONE = "none"  # do not keep any checkpoints

    @property
    def keep_filters(self) -> Set[str]:
        """Files that match these patterns are not deleted by cleanup"""
        if self == CheckpointRetentionPolicy.LAST:
            return set(["final"])
        if self == CheckpointRetentionPolicy.BEST:
            return set(["best"])
        if self == CheckpointRetentionPolicy.LAST_AND_BEST:
            return set(["final", "best"])
        if self == CheckpointRetentionPolicy.ALL:
            return set()
        return set()

    @property
    def max_to_keep(self) -> int | None:
        """
        maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and not `save`. `None` for keep all
        """
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1


def save_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    iteration: int | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
):
    """Save a plain/DDP/FSDP/FSDP2 model, its optimizer, an integer iteration and other stateful objects."""
    rank = torch.distributed.get_rank(group=process_group)

    # Rank 0 checks if the checkpoint directory exists, but all ranks need to know if if exists,
    # so they can raise an error when overwrite is False. If overwrite is True, rank 0 will delete it
    # and other ranks wait for the deletion to finish.
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is not None:
        src_rank = torch.distributed.get_global_rank(group=process_group, group_rank=0)
    torch.distributed.broadcast_object_list(ckpt_dir_exists, src=src_rank, group=process_group)
    ckpt_dir_exists = ckpt_dir_exists[0]
    if ckpt_dir_exists:
        if overwrite:
            if rank == 0:
                if ckpt_dir.is_dir():
                    shutil.rmtree(ckpt_dir)
                else:
                    ckpt_dir.unlink()
                logger.info(f"Deleted: {ckpt_dir}")
            torch.distributed.barrier(group=process_group)
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    # Rank 0 creates a temporary directory for the checkpoint and broadcasts the name to all ranks.
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = [tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=ckpt_dir.name) if rank == 0 else None]
    torch.distributed.broadcast_object_list(ckpt_dir_tmp, src=src_rank, group=process_group)
    ckpt_dir_tmp = Path(ckpt_dir_tmp[0])

    to_save = {"iteration": iteration}
    to_save["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_save["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_save.update(others)
    dcp.save(
        to_save,
        storage_writer=dcpfs.FileSystemWriter(ckpt_dir_tmp),
        process_group=process_group,
    )

    # Rank 0 renames the temporary directory to the final checkpoint directory. All ranks wait for the rename.
    if rank == 0:
        ckpt_dir_tmp.rename(ckpt_dir)
    torch.distributed.barrier()

    logger.info(f"Saved: {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: str | Path,  # output_dir/ckpt/199
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict_loading: bool = True,
    process_group: dist.ProcessGroup = None,
    **others: Stateful,
) -> int | None:
    """
    Load a plain/DDP/FSDP/FSDP2 model, its optimizer, an integer iteration and other stateful objects.
    Can you take a checkpoint saved on N ranks and load it on M ranks? Sure you can!
    Activation checkpointing and torch-compile can also be different between save and load, no problem.
    """
    ckpt_dir = Path(ckpt_dir)
    to_load = {"iteration": None}
    to_load["model"] = dcpsd.get_model_state_dict(model)
    if optimizer is not None:
        to_load["optimizer"] = dcpsd.get_optimizer_state_dict(model, optimizer)
    to_load.update(others)
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(ckpt_dir),
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=not strict_loading),
        process_group=process_group,
    )
    iteration = to_load["iteration"]
    dcpsd.set_model_state_dict(model, to_load["model"])
    if optimizer is not None:
        dcpsd.set_optimizer_state_dict(model, optimizer, to_load["optimizer"])
    logger.info(f"Loaded: {ckpt_dir}")
    return iteration


def register_dont_save_hooks(module: torch.nn.Module, dont_save: Sequence[str]):
    """
    Registers save/load state dict hooks such that the weights in `dont_save` are not persisted in the checkpoint.

    Typical use case: a classification model composed of a frozen backbone and a trainable head.
    If the frozen backbone is loaded from torch hub, it does't make sense to save a copy of it in each checkpoint.
    """

    def state_dict_post_hook(module, state_dict, prefix, local_metadata):
        # Remove frozen weights so they won't get saved.
        # If this module is not the top-level module, its weights will have a prefix in the state dict.
        nonlocal _dont_save
        for k in _dont_save:
            del state_dict[prefix + k]

    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This pre hook exists only to pass the prefix to the post hook when loading the state dict.
        nonlocal _prefix
        assert _prefix is None
        _prefix = prefix

    def load_state_dict_post_hook(module, incompatible_keys):
        # Remove the frozen weights from the missing keys so they don't raise an error.
        nonlocal _prefix
        assert _prefix is not None
        to_remove = []
        for missing_key in incompatible_keys.missing_keys:
            k = missing_key.removeprefix(_prefix)
            k = k.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing
            if k in _dont_save:
                to_remove.append(missing_key)
        for r in to_remove:
            incompatible_keys.missing_keys.remove(r)
        _prefix = None

    _dont_save = set(name.replace("_checkpoint_wrapped_module.", "") for name in dont_save)
    _prefix = None
    module.register_state_dict_post_hook(state_dict_post_hook)
    module.register_load_state_dict_pre_hook(load_state_dict_pre_hook)
    module.register_load_state_dict_post_hook(load_state_dict_post_hook)


def find_all_checkpoints(ckpt_dir: Path | str) -> list[Path]:
    """Find all checkpoints in a directory, i.e. subdirs with integer name. Sorted from first to last."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and _is_int(p.name)]
    checkpoints.sort(key=lambda p: int(p.name))
    return checkpoints


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    """Find the latest checkpoint in a directory, i.e. the subdir with the highest integer name."""
    checkpoints = find_all_checkpoints(ckpt_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None):
    """In a directory with integer-named subdirs, keep only the n subdirs with the highest number."""
    if n is None:
        return
    checkpoints = find_all_checkpoints(ckpt_dir)
    for ckpt_dir in checkpoints[:-n]:
        try:
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted: {ckpt_dir}")
        except Exception:
            logger.exception(f"Failed to delete: {ckpt_dir}")


def keep_checkpoint_copy(src: Path | str):
    """Copy a file/directory next to itself with a _keep suffix. Files are hardlinked."""
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    subprocess.check_output(["cp", "--recursive", "--link", src, dst])
    logger.info(f"Copied: {src} -> {dst}")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# Initialize a FSDP2 model from DCP or PyTorch standard checkpoint
def init_fsdp_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    skip_load_keys: List[str] | None = None,
    keys_not_sharded: List[str] | None = None,
    process_group: dist.ProcessGroup = None,
):
    if not Path(checkpoint_path).is_dir():  # PyTorch standard checkpoint
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        chkpt = torch.load(checkpoint_path, map_location="cpu")["teacher"]
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        if process_group is None:
            world_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        else:
            world_mesh = DeviceMesh.from_group(process_group, "cuda")
        chkpt = {
            key: (
                torch.distributed.tensor.distribute_tensor(tensor, world_mesh, src_data_rank=None)
                if not any(key_not_sharded in key for key_not_sharded in keys_not_sharded)
                else tensor
            )
            for key, tensor in chkpt.items()
        }
        model.load_state_dict(
            {
                key: tensor
                for key, tensor in chkpt.items()
                if not any(skip_load_key in key for skip_load_key in skip_load_keys)
            }
        )
    else:  # DCP checkpoint
        load_checkpoint(ckpt_dir=checkpoint_path, model=model, process_group=process_group)


def resolve_hf_hub_token(token_env: str | None) -> str:
    env_names = []
    if token_env:
        env_names.append(token_env)
    for fallback in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if fallback not in env_names:
            env_names.append(fallback)

    for env_name in env_names:
        token = os.environ.get(env_name)
        if token:
            return token
    raise RuntimeError(
        "No Hugging Face token found. Set one of these environment variables: "
        + ", ".join(env_names)
    )


def resolve_hf_hub_repo_id(repo_id: str | None, repo_id_env: str | None = None) -> str:
    if repo_id:
        return repo_id
    if repo_id_env:
        env_repo_id = os.environ.get(repo_id_env)
        if env_repo_id:
            return env_repo_id
    raise RuntimeError("No Hugging Face repo id configured. Set `hf_hub.repo_id` or the configured repo-id env var.")


def _get_world_mesh(process_group: dist.ProcessGroup | None):
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    if process_group is None:
        return init_device_mesh(
            "cuda",
            mesh_shape=(dist.get_world_size(),),
            mesh_dim_names=("dp",),
        )
    return DeviceMesh.from_group(process_group, "cuda")


def _convert_hf_dinov3_vit_backbone_state_dict(
    *,
    model_state_dict: dict[str, torch.Tensor],
    repo_id: str,
    revision: str,
    token: str,
) -> dict[str, torch.Tensor]:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, token=token)
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision, token=token)

    with open(config_path) as f:
        hf_config = json.load(f)

    converted: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt") as f:
        converted["cls_token"] = f.get_tensor("embeddings.cls_token")
        converted["mask_token"] = f.get_tensor("embeddings.mask_token").squeeze(0)
        converted["patch_embed.proj.weight"] = f.get_tensor("embeddings.patch_embeddings.weight")
        converted["patch_embed.proj.bias"] = f.get_tensor("embeddings.patch_embeddings.bias")

        if "embeddings.register_tokens" in f.keys():
            converted["storage_tokens"] = f.get_tensor("embeddings.register_tokens")

        num_hidden_layers = int(hf_config["num_hidden_layers"])
        for i in range(num_hidden_layers):
            hf_prefix = f"layer.{i}"
            local_prefix = f"blocks.{i}"

            q_weight = f.get_tensor(f"{hf_prefix}.attention.q_proj.weight")
            k_weight = f.get_tensor(f"{hf_prefix}.attention.k_proj.weight")
            v_weight = f.get_tensor(f"{hf_prefix}.attention.v_proj.weight")
            converted[f"{local_prefix}.attn.qkv.weight"] = torch.cat((q_weight, k_weight, v_weight), dim=0)

            q_bias = f.get_tensor(f"{hf_prefix}.attention.q_proj.bias")
            v_bias = f.get_tensor(f"{hf_prefix}.attention.v_proj.bias")
            k_bias = torch.zeros_like(q_bias)
            converted[f"{local_prefix}.attn.qkv.bias"] = torch.cat((q_bias, k_bias, v_bias), dim=0)

            converted[f"{local_prefix}.attn.proj.weight"] = f.get_tensor(f"{hf_prefix}.attention.o_proj.weight")
            converted[f"{local_prefix}.attn.proj.bias"] = f.get_tensor(f"{hf_prefix}.attention.o_proj.bias")
            converted[f"{local_prefix}.ls1.gamma"] = f.get_tensor(f"{hf_prefix}.layer_scale1.lambda1")
            converted[f"{local_prefix}.ls2.gamma"] = f.get_tensor(f"{hf_prefix}.layer_scale2.lambda1")
            converted[f"{local_prefix}.norm1.weight"] = f.get_tensor(f"{hf_prefix}.norm1.weight")
            converted[f"{local_prefix}.norm1.bias"] = f.get_tensor(f"{hf_prefix}.norm1.bias")
            converted[f"{local_prefix}.norm2.weight"] = f.get_tensor(f"{hf_prefix}.norm2.weight")
            converted[f"{local_prefix}.norm2.bias"] = f.get_tensor(f"{hf_prefix}.norm2.bias")
            converted[f"{local_prefix}.mlp.fc1.weight"] = f.get_tensor(f"{hf_prefix}.mlp.up_proj.weight")
            converted[f"{local_prefix}.mlp.fc1.bias"] = f.get_tensor(f"{hf_prefix}.mlp.up_proj.bias")
            converted[f"{local_prefix}.mlp.fc2.weight"] = f.get_tensor(f"{hf_prefix}.mlp.down_proj.weight")
            converted[f"{local_prefix}.mlp.fc2.bias"] = f.get_tensor(f"{hf_prefix}.mlp.down_proj.bias")

        converted["norm.weight"] = f.get_tensor("norm.weight")
        converted["norm.bias"] = f.get_tensor("norm.bias")

    for key, tensor in converted.items():
        if key not in model_state_dict:
            raise RuntimeError(
                f"HF weight {key} does not exist in the local backbone. "
                "Check that the local config matches the HF model architecture."
            )
        if tuple(model_state_dict[key].shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: local {tuple(model_state_dict[key].shape)} vs HF {tuple(tensor.shape)}. "
                "Check `student.arch`, `student.n_storage_tokens`, and RoPE/register-token settings."
            )
    return converted


def init_fsdp_backbone_from_hf_hub(
    model: torch.nn.Module,
    *,
    repo_id: str,
    revision: str = "main",
    token_env: str | None = "HF_TOKEN",
    process_group: dist.ProcessGroup = None,
):
    token = resolve_hf_hub_token(token_env)
    state_dict = dcpsd.get_model_state_dict(model)
    converted = _convert_hf_dinov3_vit_backbone_state_dict(
        model_state_dict=state_dict,
        repo_id=repo_id,
        revision=revision,
        token=token,
    )
    world_mesh = _get_world_mesh(process_group)
    for key, tensor in converted.items():
        if isinstance(state_dict[key], DTensor):
            state_dict[key] = torch.distributed.tensor.distribute_tensor(tensor, world_mesh, src_data_rank=None)
        else:
            state_dict[key] = tensor
    dcpsd.set_model_state_dict(model, state_dict)
    logger.info("Loaded Hugging Face pretrained backbone from %s@%s", repo_id, revision)


def upload_checkpoint_to_hf_hub(
    local_path: str | Path,
    *,
    repo_id: str | None,
    repo_id_env: str | None = None,
    path_in_repo: str,
    token_env: str | None = "HF_TOKEN",
    create_repo: bool = True,
    private_repo: bool = False,
    commit_message: str | None = None,
):
    token = resolve_hf_hub_token(token_env)
    resolved_repo_id = resolve_hf_hub_repo_id(repo_id, repo_id_env)
    local_path = Path(local_path)
    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id=resolved_repo_id, repo_type="model", private=private_repo, exist_ok=True)

    commit_message = commit_message or f"Upload {local_path.name}"
    if local_path.is_dir():
        api.upload_folder(
            repo_id=resolved_repo_id,
            repo_type="model",
            folder_path=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
    else:
        api.upload_file(
            repo_id=resolved_repo_id,
            repo_type="model",
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
    logger.info("Uploaded %s to hf://%s/%s", local_path, resolved_repo_id, path_in_repo)


# Initialize a standard non distributed PyTorch model from PyTorch standard checkpoint for evals
def init_model_from_checkpoint_for_evals(
    model: torch.nn.Module, pretrained_weights: str | Path, checkpoint_key: str = None
):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def cleanup_checkpoint(ckpt_dir: str, checkpoint_retention_policy: CheckpointRetentionPolicy):
    """
    ckpt_dir is the directory containing each individual checkpoint directories (either at iteration, best (validation performance) or final)
    |-- ckpt_dir/
    |   |-- 0/
    |       |--checkpoint.pth  or dcp_sharded_checkpoint_dir
    |   |-- 99/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 199/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- best/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- 299/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    |   |-- final/
            |--checkpoint.pth or dcp_sharded_checkpoint_dir
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    checkpoint_filters = checkpoint_retention_policy.keep_filters
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir()]
    for checkpoint in checkpoints:
        if checkpoint in checkpoint_filters:
            continue
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted: {checkpoint}")
        except Exception:
            logger.exception(f"Failed to delete: {checkpoint}")
