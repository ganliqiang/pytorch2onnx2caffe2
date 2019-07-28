#-*-coding=utf-8-8-

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from onnx_caffe2.backend import Caffe2Backend
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("psenet.onnx")

onnx_model_path="psenet.onnx"
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
model_name = onnx_model_path.replace('.onnx','')
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph, device="CUDA")
with open(model_name + "_init.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open(model_name + "_predict.pb", "wb") as f:
    f.write(predict_net.SerializeToString())
with open(model_name + "_predict.txt","wb") as f:
    f.write(str(predict_net))

