class NodeDto:
    def __init__(self, index: int, is_infected: bool = False, infected_by: int = None):
        self.index = index
        self.is_infected = is_infected
        self.infected_by = infected_by
