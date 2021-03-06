import json
import pathlib
import subprocess
from datetime import datetime
from typing import Optional, Dict

import git
import hjson
import rospkg
from colorama import Fore


def make_unique_trial_subdirectory_name(*names):
    stamp = "{:%Y_%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}" + len(names) * '_{}'
    return format_string.format(stamp, sha, *names)


def create_or_load_trial(group_name: Optional[pathlib.Path] = None,
                         trial_path: Optional[pathlib.Path] = None,
                         params: Optional[Dict] = None,
                         trials_directory: Optional[pathlib.Path] = None,
                         write_summary: Optional[bool] = True):
    if trial_path is not None:
        # Resume and warn for errors
        if group_name is not None:
            msg = "Ignoring group_name {} and resuming from trial_path {}".format(group_name, trial_path)
            print(Fore.YELLOW + msg + Fore.RESET)
        if params is not None:
            print(Fore.YELLOW + "Ignoring params, loading existing ones" + Fore.RESET)
        return load_trial(trial_path)
    elif group_name is not None:
        return create_trial(group_name, params, trials_directory)
    else:
        return create_trial('tmp', params, trials_directory)


def load_trial(trial_path):
    trial_path = pathlib.Path(trial_path)
    if not trial_path.is_absolute():
        r = rospkg.RosPack()
        trial_path = pathlib.Path(r.get_path('realsense_prediction')) / "trials" / trial_path
    if not trial_path.is_dir():
        raise ValueError("Cannot load, the path {} is not an existing directory".format(trial_path))

    params = load_params(trial_path)
    return trial_path, params


def load_params(directory: pathlib.Path):
    possible_names = ['hparams.json', 'hparams.hjson', 'params.json', 'params.hjson']
    for n in possible_names:
        filename = directory / n
        if filename.is_file():
            params = load_hjson(filename)
            return params
    raise RuntimeError(f"no params file in {directory.as_posix()}")


def load_json_or_hjson(directory: pathlib.Path, prefix: str):
    params_filename = directory / f'{prefix}.json'
    if params_filename.is_file():
        params = load_hjson(params_filename)
    else:
        params_filename = directory / f'{prefix}.hjson'
        if params_filename.exists():
            params = load_hjson(params_filename)
        else:
            either_or_name = directory / f'{prefix}.(h)json'
            raise RuntimeError(f"no such file {either_or_name.as_posix()}")
    return params


def load_hjson(path: pathlib.Path):
    with path.open("r") as file:
        data = hjson.load(file)
    return data


def get_trial_path(group_name, trials_directory=None):
    if trials_directory is None:
        r = rospkg.RosPack()
        base_path = pathlib.Path(r.get_path('realsense_prediction'))
        trials_directory = base_path / 'trials'
    trials_directory.mkdir(parents=True, exist_ok=True)

    # make subdirectory
    unique_trial_subdirectory_name = make_unique_trial_subdirectory_name()
    trial_path = trials_directory / group_name / unique_trial_subdirectory_name

    return trial_path


def create_trial(group_name, params, trials_directory=None):
    trial_path = get_trial_path(group_name, trials_directory=trials_directory)
    trial_path.mkdir(parents=True, exist_ok=False)

    # save params
    params_filename = trial_path / 'params.json'
    with params_filename.open("w") as params_file:
        json.dump(params, params_file, indent=2)
    with (trial_path / 'readme.txt').open("w") as f:
        f.write(f'Trials for group {group_name}\n'
                f'Original filepath during training was {trial_path}')

    return trial_path, params


def _write_summary(full_trial_directory, group_name, unique_trial_subdirectory_name):
    with (full_trial_directory / 'readme.txt').open("w") as f:
        f.write(datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
        f.write("\nTrial trial_nickname: {}/{}\n".format(group_name, unique_trial_subdirectory_name))

        f.write("git show --summary:\n")
        f.write(subprocess.check_output(['git', 'show', '--summary']))
        f.write("git status:\n")
        f.write(subprocess.check_output(['git', 'status']))
        f.write("git diff:\n")
        f.write(subprocess.check_output(['git', 'diff']))


def get_default_params():
    # r = rospkg.RosPack()
    # shape_completion_training_path = pathlib.Path(r.get_path('shape_completion_training'))
    # default_params_filename = shape_completion_training_path / 'default_params.json'
    # with default_params_filename.open('r') as default_params_file:
    #     return json.load(default_params_file)
    return {}
