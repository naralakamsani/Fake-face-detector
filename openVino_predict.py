import sys
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
from predict_helper import process_image

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml or .onnx file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=2, type=int)

    return parser


log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
args = build_argparser().parse_args()

# Plugin initialization for specified device and load extensions library if specified
log.info("Creating Inference Engine")
ie = IECore()
if args.cpu_extension and 'CPU' in args.device:
    ie.add_extension(args.cpu_extension, "CPU")

# Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
model = args.model
log.info(f"Loading network:\n\t{model}")
net = ie.read_network(model=model)

assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
assert len(net.outputs) == 1, "Sample supports only single output topologies"

log.info("Preparing input blobs")
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
net.batch_size = len(args.input)

# Read and pre-process input images
n, c, h, w = net.input_info[input_blob].input_data.shape
images = np.ndarray(shape=(n, c, h, w))
for i in range(n):
    image = process_image(args.input[i])
    image = image.unsqueeze_(0)
    image = image.to("cpu").float()
    images[i] = image

log.info("Batch size is {}".format(n))

# Loading model to the plugin
log.info("Loading model to the plugin")
exec_net = ie.load_network(network=net, device_name=args.device)

# Start sync inference
log.info("Starting inference in synchronous mode")
res = exec_net.infer(inputs={input_blob: images})

# Processing output blob
log.info("Processing output blob")
res = res[out_blob]
log.info("Top {} results: ".format(args.number_top))
if args.labels:
    with open(args.labels, 'r') as f:
        labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
else:
    labels_map = [None]
classid_str = "class"
probability_str = "probability"
for i, probs in enumerate(res):
    probs = np.squeeze(probs)
    top_ind = np.argsort(probs)[-args.number_top:][::-1]
    print(f"\nImage Path:\n\t{args.input[i]}")
    print(classid_str, probability_str)
    print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))

    probability_sum = 0
    for i in range(len(probs)):
        probability_sum = probability_sum + probs[i]

    for id in top_ind:
        det_label = labels_map[id] if labels_map else "{}".format(id)
        label_length = len(det_label)
        space_num_before = (len(classid_str) - label_length) // 2
        space_num_after = len(classid_str) - (space_num_before + label_length) + 2
        space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
        print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                      ' ' * space_num_after, ' ' * space_num_before_prob,
                                      1-probs[id]/probability_sum))