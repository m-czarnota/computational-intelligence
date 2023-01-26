from enum import Enum


class MinimizeMethod(Enum):
    L_BFGS_B = 'L-BFGS-B'
    Newton_CG = 'Newton-CG'