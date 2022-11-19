from enum import Enum
import networkx as nx


class NxGraphAnimatorPosLayoutEnum(Enum):
    SPRING = nx.spring_layout
    CIRCULAR = nx.circular_layout
    PLANAR = nx.planar_layout
    SHELL = nx.shell_layout

    @classmethod
    def get_nx_layout(cls, value):
        if value == cls.SPRING:
            return nx.spring_layout
        if value == cls.CIRCULAR:
            return nx.circular_layout
        if value == cls.PLANAR:
            return nx.planar_layout
        if value == cls.SHELL:
            return nx.shell_layout
