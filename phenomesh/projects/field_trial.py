from .base_project import Project
from .base_project import ProjectLevel


class FieldTrial(Project):
    def __init__(self, project_directory, field_name):
        super().__init__(project_directory=project_directory)
        self.field = Field(name=field_name)


class FieldLevel(ProjectLevel):
    def __init__(self, level, name):
        super().__init__(level=level, name=name)

    def add_observation(self, observation):
        try:
            self._add_observation(observation)
        except NameError:
            print(f"Not adding observation {observation.name}. Observation {observation.name} already exists!")
        return observation


class Field(FieldLevel):
    def __init__(self, name):
        super().__init__(level="Field", name=name)
        """
        The field containing plots and plants
        """
        self.plots = self.children

    def add_plot(self, name: str):
        plot = Plot(name=name)
        try:
            self._add_child(plot)
        except NameError:
            print(f"Not adding plot {plot.name}. Plot {plot.name} already exists!")
        return plot

    def print_plots(self):
        self._print_children()


class Plot(FieldLevel):
    def __init__(self, name):
        super().__init__(level="Plot", name=name)
        """
        A plot within a field
        """
        self.plants = self.children

    def add_plant(self, name: str):
        plant = Plant(name=name)
        self._add_child(plant)

    def print_plants(self):
        self._print_children()


class Plant(FieldLevel):
    def __init__(self, name):
        super().__init__(level="Plant", name=name)
        """
        A plant within a plot
        """
        pass
