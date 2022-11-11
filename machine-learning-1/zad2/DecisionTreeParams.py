from DirtMetricEnum import DirtMetricEnum


class DecisionTreeParams:
    def __init__(self, depth: int = 10, number_of_nodes: int = None,
                 threshold_value: int = None, dirt_metric: DirtMetricEnum = DirtMetricEnum.INFO_GAIN):
        self.depth = depth
        self.number_of_nodes = number_of_nodes
        self.threshold_value = threshold_value
        self.dirt_metric = dirt_metric

    def set_params(self, depth: int = 10, number_of_nodes: int = None,
                   threshold_value: int = None, dirt_metric: DirtMetricEnum = DirtMetricEnum.INFO_GAIN):
        self.depth = depth
        self.number_of_nodes = number_of_nodes
        self.threshold_value = threshold_value
        self.dirt_metric = dirt_metric

    def get_params_as_dict(self) -> dict:
        return {
            'depth': self.depth,
            'number_of_nodes': self.number_of_nodes,
            'threshold_value': self.threshold_value,
            'dirt_metric': self.dirt_metric,
        }
