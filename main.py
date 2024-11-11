import argparse
from train import train
from evaluate import evaluate_model
from plot import plot_training_history
from model import ToxicityPredictionModel
import pandas as pd
from tensorflow.keras.models import load_model
from loss_functions import masked_loss

def main(args):
    if args.mode == 'train':
        history, model = train()
        plot_training_history(history=history)

        if args.save_model:
            model.save(args.model_path)
    
    elif args.mode == 'evaluate':
        evaluate_model(args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Toxicity Prediction using Deep Learning")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help="Mode: train or evaluate")
    parser.add_argument('--model_path', type=str, default='toxicity_model.keras', help="Path to save/load model")
    parser.add_argument('--save_model', action='store_true', help="Save the model after training")
    args = parser.parse_args()

    main(args)