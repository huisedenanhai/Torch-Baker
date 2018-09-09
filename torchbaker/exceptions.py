class BadRecipe(Exception):
    pass


class NotPreparedRecipe(BadRecipe):
    pass


class ArgumentError(Exception):
    pass


class CheckpointError(Exception):
    pass
