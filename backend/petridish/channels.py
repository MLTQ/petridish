"""Stable semantic indices for the 16-channel cell tensor."""

from enum import IntEnum


class Channel(IntEnum):
    ALIVE = 0
    ACTIVATION = 1
    PHASE_SIN = 2
    PHASE_COS = 3
    MEMORY_0 = 4
    MEMORY_1 = 5
    MEMORY_2 = 6
    MEMORY_3 = 7
    ENERGY = 8
    AXON_GROWTH = 9
    DENDRITE_GROWTH = 10
    REWARD_TRACE = 11
    POSITION_X = 12
    POSITION_Y = 13
    SENSOR_ID = 14
    MOTOR_ID = 15
