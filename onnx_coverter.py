import torch
import predict_helper

model = predict_helper.load_checkpoint("checkpoint.pth")

model.to('cpu')
image = predict_helper.process_image("img4.jpg")
dummy_input = image.unsqueeze_(0).to('cpu').float()

torch.onnx.export(model,
                  dummy_input,
                  "model.onnx")