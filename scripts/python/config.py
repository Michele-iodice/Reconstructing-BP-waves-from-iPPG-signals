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
