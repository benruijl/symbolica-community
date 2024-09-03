from symbolica_community.symbolica_community import Expression


ELEMENTARY_CHARGE: float = 1.602176634e-19
"""Elementary charge in Coulombs"""

GRAVITATIONAL_CONSTANT: float = 6.67430e-11
"""Gravitational constant in m^3 kg^-1 s^-2"""

PLANCK_CONSTANT: float = 6.62607015e-34
"""Planck constant in Joule seconds"""


def get_velocity(position: Expression, time_variable: Expression, time: float) -> Expression:
    """Get the velocity of an object from its position evolution in `time_variable` at `time`"""
    return position.derivative(time_variable).replace_all(time_variable, time)
