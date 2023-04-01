class EnvironmentVariableMissing(Exception):
    pass


class MissingAuthorData(Exception):
    pass


class InvalidAuthorData(Exception):
    pass


class ThermoLibraryLoadError(Exception):
    pass


class KineticsLibraryLoadError(Exception):
    pass


class CreateSpeciesError(Exception):
    pass


class CreateSourceError(Exception):
    pass


class DOIError(Exception):
    pass
