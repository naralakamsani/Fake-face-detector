# Make imports
import predict_helper
import argparse

parser = argparse.ArgumentParser(description="Neural Network Settings for prediction")
parser.add_argument('image', type=str)
parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()
    
model = predict_helper.load_checkpoint(args.checkpoint)

predict_helper.image_predict(args.image, model, topk=args.top_k, gpu=args.gpu)
