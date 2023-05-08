from enum import Enum


class FilterTypeEnum(Enum):
    BANDPASS = 'bandpass'
    LOWPASS = 'lowpass'
    HIGHPASS = 'highpass'
    BANDSTOP = 'bandstop'
