

class Project:
    def __init__(self, project_directory):
        self.project_directory = project_directory
        self.levels = {}

    def read(self):
        for level in self.levels:
            level.write()

    def write(self):
        for level in self.levels:
            level.write()


class ProjectLevel:
    def __init__(self, level=None, children=None, name=None, observations=None):
        if children is None:
            children = {}
        if observations is None:
            observations = {}
        self.level = level
        self.children = children
        self.name = name
        self.observations = observations

    def _add_child(self, child):
        if child.name in self.children.keys():
            raise NameError
        self.children[child.name] = child
        return child

    def _add_observation(self, observation):
        if observation.name in self.observations.keys():
            raise NameError
        self.observations[observation.name] = observation
        return observation

    def print_observations(self):
        print(f"========== Observation Summary for {self.level} {self.name} ==========")
        for idx, observation in enumerate(self.observations.values()):
            print("")
            print(f"Observation {idx}")
            observation.print_info()
        print("=============================================================")

    def _print_children(self, recursive=False):
        for child in self.children.values():
            level = child.level
            name = child.name
            print(f"({self.level}) {self.name}: ({level}) {name}")
            if recursive:
                child._print_children()

