import yaml
from typing import Callable, Dict, Tuple


def open_yaml(path: str) -> Dict:
    '''Try to open yaml file, given as camera configuration file.
    Return dictionary in case of succsses or an exeption'''
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data


def get_config(path: Tuple[str, str]) -> Tuple[Callable, Callable]:
    '''Creates instances of Config class, based on dict created from
    yaml file.
    '''
    camera = open_yaml(path[0])
    segmentation = open_yaml(path[1])

    camera_config = type('ConfigClass', (), camera)
    segmentation_config = type('ConfigClass', (), segmentation)

    return (camera_config, segmentation_config)
