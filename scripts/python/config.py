import configparser


class Configuration:
    """
    This class manage the configuration parameter contain in the config file (.cfg) for the project parameter.
    """

    def __init__(self, configFilename):

        self.parser = None
        self.modeldict= None
        self.pttdict = None
        self.sigdict = None
        self.datasetdict = None
        self.uNetdict = None
        self.parse_cfg(configFilename)

    def parse_cfg(self, configFilename):
        """ parses the given configuration file for loading the test's parameters.

        Args:
            configFilename: configuation file (.cfg) name of path .

        """
        self.parser = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'))
        self.parser.optionxform = str
        if not self.parser.read(configFilename):
            raise FileNotFoundError(configFilename)

        # load paramas
        self.modeldict = dict(self.parser['RFModel'].items())
        self.pttdict = dict(self.parser['PTT'].items())
        self.sigdict = dict(self.parser['Sig'].items())
        self.datasetdict = dict(self.parser['DATASET'].items())
        self.uNetdict = dict(self.parser['UnetParameter'].items())

    def get_array(self, name):
        """Extracts and converts the 'array' parameter."""
        array_str = self.uNetdict.get(name)
        if array_str:
            # Convert the string to a list of integers, stripping spaces
            return [int(x.strip()) for x in array_str.split(',')]
        else:
            raise KeyError("Array not found in RFModel section")

    def get_boolean(self, section, key):
        """ Get a boolean value from the configuration file.

        Args:
            section: The section in the .cfg file (e.g., 'RFModel').
            key: The key for which to get the boolean value (e.g., 'debug').

        Returns:
            The boolean value (True or False).
        """
        if section in self.parser and key in self.parser[section]:
            return self.parser.getboolean(section, key)
        else:
            raise KeyError(f"Key '{key}' not found in section '{section}'")