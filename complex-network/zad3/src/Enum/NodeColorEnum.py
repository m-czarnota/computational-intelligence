from enum import Enum


class NodeColorEnum(Enum):
    SEED = [0.42, 0.96, 0.53]  # green
    INFECTED = [0.94, 0.47, 0.47]  # red
    NO_INFECTED = [0.47, 0.77, 0.94]  # blue
