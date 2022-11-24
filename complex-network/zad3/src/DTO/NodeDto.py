class NodeDto:
    def __init__(self, index: int, is_infected: bool = False, infected_by: int = None, infect_prob: float = None):
        self.index = index
        self.is_infected = is_infected
        self.infected_by = infected_by
        self.infect_prob = infect_prob

    def __str__(self):
        infected_info = f'infected by: {self.infected_by}, infect prob: {self.infect_prob}' if self.is_infected else ''

        return f'NodeDto {self.index}, infected: {self.is_infected}. {infected_info}'
