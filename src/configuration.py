import os
from typing import Any, Dict

import yaml


def load_config(config_path) -> Dict[str, Any]:
    class Loader(yaml.SafeLoader):
        def __init__(self, stream):
            super().__init__(stream)
            self._root = os.path.split(stream.name)[0]

        def include(self, node):
            filename = os.path.join(self._root, self.construct_scalar(node))
            with open(filename, 'r') as f:
                return yaml.load(f, Loader)

    Loader.add_constructor("!include", Loader.include)

    with open(config_path, "r") as f:
        return yaml.load(f, Loader)
