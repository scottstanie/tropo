import functools
from pathlib import Path

import click
import numpy as np
from RAiDER.models.model_levels import LEVELS_137_HEIGHTS

from opera_tropo.config import pge_runconfig

click.option = functools.partial(click.option, show_default=True)


def create_config(
    input_file: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "./runconfig.yaml",
    height_levels: list[float] = LEVELS_137_HEIGHTS,
    n_workers: int = 4,
    n_threads: int = 2,
    worker_memory: int = 8,
    block_shape: tuple[int, int] = (128, 128),
    log_file: str = "tropo_run.log",
) -> None:
    """Generate and save a run configuration file for tropospheric processing."""
    config_path = Path(config_path).resolve()

    if config_path.suffix != ".yaml":
        raise ValueError("config_path must be a YAML file (e.g., runconfig.yaml)")

    runconfig = pge_runconfig.RunConfig()
    runconfig.input_file.input_file_path = Path(input_file).resolve()
    runconfig.output_options.output_heights = height_levels
    runconfig.product_path_group.scratch_path = Path(output_dir).resolve()

    runconfig.worker_settings.n_workers = n_workers
    runconfig.worker_settings.threads_per_worker = n_threads
    runconfig.worker_settings.max_memory = worker_memory
    runconfig.worker_settings.block_shape = block_shape
    runconfig.log_file = Path(output_dir).resolve() / log_file

    runconfig.to_yaml(config_path)
    print(f"Configuration saved to {config_path}")


@click.command("config")
@click.option(
    "--config-file",
    "-c",
    type=Path,
    default=Path.cwd() / "runconfig.yaml",
    help="Path to output config file",
)
@click.option(
    "--tropo-input",
    "-input",
    type=Path,
    required=True,
    help="Path to input file for tropospheric processing",
)
@click.option(
    "--tropo-output",
    "-out",
    type=Path,
    required=True,
    help="Directory for output files",
)
@click.option(
    "--height-levels",
    "-hlevels",
    type=list,
    default=np.flipud(LEVELS_137_HEIGHTS).tolist(),
    help="List of height levels for output",
)
@click.option(
    "--worker-settings",
    type=(int, int, int),
    default=(4, 2, 8),
    help="Worker settings: (n_workers, n_threads, worker_memory)",
)
@click.option(
    "--chunks",
    type=(int, int),
    default=(128, 128),
    help="Block shape for worker processing",
)
@click.option("--log", type=str, default="run_tropo.log", help="Log filename")
def run_create_config(
    config_file: Path,
    tropo_input: Path,
    tropo_output: Path,
    height_levels: list[float],
    worker_settings: tuple[int, int, int],
    chunks: tuple[int, int],
    log: str,
) -> None:
    """CLI wrapper to create a tropospheric runconfig file."""
    create_config(
        input_file=tropo_input,
        output_dir=tropo_output,
        config_path=config_file,
        height_levels=height_levels,
        n_workers=worker_settings[0],
        n_threads=worker_settings[1],
        worker_memory=worker_settings[2],
        block_shape=chunks,
        log_file=log,
    )
