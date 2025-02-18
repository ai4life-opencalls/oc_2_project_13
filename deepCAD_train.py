from argparse import ArgumentParser
import yaml
from deepcad.train_collection import training_class


argparser = ArgumentParser(description="Train a DeepCAD model on the given Dataset")
argparser.add_argument("config", type=str, help="Path to the YAML configuration file")
args = argparser.parse_args()

if __name__ == "__main__":
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        tc = training_class(params_dict=config["train_params_dict"])
        tc.run()