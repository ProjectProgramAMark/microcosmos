import math

from experiments.swim_sweep.swimmers.worm import Worm


class FastWorm(Worm):
    """Worm at 3x rotation speed — stress-tests fluid coupling at higher velocities."""

    swimmer_name = 'fast_worm'
    plot_color = 'darkorange'

    def setup(self):
        super().setup()
        self.rotation_speed = self.rotation_speed * 3.0
