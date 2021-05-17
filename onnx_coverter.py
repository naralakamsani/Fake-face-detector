import torch
import predict_helper

#Load the model from checkpoint
model = predict_helper.load_checkpoint("checkpoint.pth")

#Create a dummy input
model.to('cpu')
image = predict_helper.process_image("img4.jpg")
dummy_input = image.unsqueeze_(0).to('cpu').float()

#export model to onnx
torch.onnx.export(model,
                  dummy_input,
                  "model.onnx")
