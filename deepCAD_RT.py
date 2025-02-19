from argparse import ArgumentParser
import yaml
from deepcad.train_collection import training_class
from deepcad.test_collection import testing_class

argparser = ArgumentParser(description="Train or Predict a DeepCAD model on the given Dataset")
argparser.add_argument("mode", type=str, help="Mode of operation: train or predict")
argparser.add_argument("config", type=str, help="Path to the YAML configuration file")
args = argparser.parse_args()

if __name__ == "__main__":

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        if args.mode == "train":
            tc = training_class(params_dict=config["train_params_dict"])
        elif args.mode == "predict":
            tc = testing_class(params_dict=config["test_params_dict"])
        else:
            raise ValueError("Mode must be either 'train' or 'predict'")
        tc.run()