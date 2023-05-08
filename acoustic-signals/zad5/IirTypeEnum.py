from enum import Enum


class IirTypeFilter(Enum):
    BUTTERWORTH = 'butter'
    CHEBYSHEV_1 = 'cheby1'
    CHEBYSHEV_2 = 'cheby2'
    ELLIPTIC = 'ellip'
    BESSEL = 'bessel'
