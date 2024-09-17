from dotenv import load_dotenv
from pathlib import Path
import logging
import os
import argparse

"""
 This script is made to manage the environment of a research project that can be run from different environments using the same codebase.
 For example, if the code is stored in a network-shared repository but the model/dataset folders are different between the different environments.
 It also contains some utility functions.

 Usage in your script:

 from envutils import ENV, load_env, get_tiff_paths, get_argparser


 if __name__ == "__main__":
    # This loads
    parser = get_argparser(description="Train a N2V model on the given dataset.")
    parser.add_argument('dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    args = parser.parse_args()
 
    # This loads the provided .env file
   load_env(args.env)
 
"""

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# Global vars
ENV = {"DATASET_FOLDER": None, "OUTPUT_FOLDER": None, "MODELS_FOLDER": None}

### ENV MANAGEMENT AND ARGPARSE
def load_env(dot_env:str | Path, env_dict: dict=ENV, parser_args: dict | argparse.Namespace | None = None) -> dict:
    """
       Loads a .env file and validates the required environment variables, 
       allowing to keep track of environment variables used from your script and to manage multiple environments (e.g., multiple machines.)
       Envvars are loaded both in os.environ and in the ENV dictionary, you should use ENV dict since it is already validated (i.e., every envvar is not None).
       By default, ENV dictionary be defined as global variable if not provided a custom one.

       Args:
            - dot_env:
                A path to a .env file to load for this environment.
            - env_dict [default: ENV]
                A ENV dictionary containing the environment variables loaded by the ENV file.
                This defaults to envutils.ENV, but a different one can be provided and managed externally by the user script.
            - parser_args [dict or argparse.Namespace, Optional]:
                A dictionary of externally provided arguments. If provided, the check of ENV variables will also include this dictionary.
                This is used to allow overriding the ENV variables using script arguments.


    """
    is_args_provided = parser_args is not None

    if is_args_provided and type(parser_args) is argparse.Namespace:
        # Convert to dict
        parser_args = vars(parser_args)

    # Load the ENV file as environment variables
    if not load_dotenv(dot_env):
        raise FileNotFoundError(f"{dot_env} file not found. Please create a .env file or provide one using -e <env_filepath>. Has to contain the following envvars: {list(env_dict.keys())}")
    log.debug(f"{dot_env} environment loaded.")

    for k in env_dict:
        # Store environment variables into env_dict
        env_dict[k] = os.getenv(k)

        is_var_defined_in_dict = env_dict[k] is not None
        is_var_defined_in_args = is_args_provided and (k in parser_args) and parser_args[k] is not None
        if (not is_var_defined_in_dict) and (not is_var_defined_in_args):
            raise ValueError(f"Environment Variable {k} has not been set in {dot_env} nor passed as script parameter.")
        else:
            if is_var_defined_in_args:
                log.debug(f"Overriding {k}: {env_dict[k]} from script arguments.")
                env_dict[k] = parser_args[k]
            log.debug(f"{k}: {env_dict[k]} (from {'args' if is_var_defined_in_args else 'env file'})")

    return env_dict


def get_tiff_paths(dataset_subfolder: str, extension: str=".tif", env=ENV):
    """
        Returns all the paths for the given dataset subfolder.
        The root of the dataset is fetched from the environment loaded using load_env (i.e., ENV["DATASET_FOLDER"]).

        Args:
            - dataaset_subfolder [str]:
                The dataset subfoler to search .tif file in.
            - extension [str] (default: '.tif')
                Extension to use when looking for tiff files, as they can be either named .tif or .tiff.
            - env [dict] (default: ENV)
                A dictionary containing the environment variables, in particular "DATASET_FOLDER" is required.
                By default, this function uses the environment that is previously loaded using the load_env folder. 
    """
    
    DATASET_SUBFOLDER = os.path.join(env['DATASET_FOLDER'], dataset_subfolder)
    return sorted(Path(DATASET_SUBFOLDER).glob(f'*{extension}'))

def add_default_arguments(parser:argparse.ArgumentParser, env_dict: dict=ENV):
    """
        Add common arguments to this parser, such as the .env filepath argument and overrides for every varible found in the .env file.
    """
    parser.add_argument('-e', '--env', type=str, default='.env', help="Path to an .env file containing required environment variables for the script.")
    parser.add_argument('--level', type=str, default=logging.WARNING, help="Logging level.")

    for envvar in env_dict.keys():
        parser.add_argument(f"--{envvar}", type=str, default=None, help=f"Override for the {envvar} environment variable. By default it is read from the provided .env file or from environment variables.")


def get_argparser(description: str, add_default_args: bool=True) -> argparse.ArgumentParser:
    """
        Returns an Arg parser with given description, followed by required environment variables. It also optionally adds common flags used in this project.
    """
    parser = argparse.ArgumentParser(description=f"{description}. REQUIRED ENV VARS: {list(ENV.keys())}")
    if add_default_args:
        add_default_arguments(parser)
    return parser