# Make imports
import argparse

import predict_helper

parser = argparse.ArgumentParser(description="Neural Network Settings for prediction")
parser.add_argument('image', type=str, nargs="+")
parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth')
parser.add_argument('--top_k', type=int, default=2)
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()
    
model = predict_helper.load_checkpoint(args.checkpoint)

for images in args.image:
    predict_helper.image_predict(images, model, topk=args.top_k, gpu=args.gpu)
