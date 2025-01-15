# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import os.path as osp
import math
import cv2
import PIL

from typing import List, Dict
import onnx
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed, fix_output_shapes
from iree.compiler.tools import compile_str
from iree.compiler.tools.import_onnx.importer_externalization_overrides import *
from insightface.model_zoo.arcface_onnx import *
from insightface.model_zoo.retinaface import *
#from insightface.model_zoo.scrfd import *
from insightface.model_zoo.landmark import *
from insightface.model_zoo.attribute import Attribute as MZAttribute
from insightface.model_zoo.inswapper import INSwapper
from insightface.app.face_analysis import FaceAnalysis

import numpy as np


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class NodeArg:
    def __init__(self, name, type, shape):
        self.name = name
        self.type = type
        self.shape = shape

    @staticmethod
    def from_value_info_proto(name, value_info : onnx.ValueInfoProto):
        tt = value_info.type.tensor_type
        elem_type = tt.elem_type
        type_str = ""
        for key, value in onnx.TensorProto.DataType.items():
            if value == elem_type:
                type_str += f'tensor({key.lower()})'
                break
        if type_str == "":
            raise ValueError(f"could not find tensor proto elem type corresponding to int {elem_type}")
        dims = []
        for d in tt.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)
            elif d.dim_value:
                dims.append(d.dim_value)
            else:
                raise KeyError(f"expected {d} to have either dim_param or dim_value")
        return NodeArg(name, type_str, dims)

class IREEInferenceSession:
    def __init__(self, model_file: str, *, dim_param_dict: Dict[str, int] | None = None, target_backends = ["llvm-cpu"],  extra_compile_args : List[str] = [], device : str = "local-task"):
        self.model_file=model_file
        self.dim_param_dict = dim_param_dict
        self.device = device
        self.mlir_module = self.import_model()
        compile_kwargs = {"target_backends" : target_backends, "extra_args" : extra_compile_args}
        self.vmfb = self.compile(**compile_kwargs)

    def load_model(self):
        # pre-process onnx model:
        if not osp.isfile(self.model_file):
            raise FileNotFoundError(f'No model present. Path provided: {self.model_file}')
        print("loading onnx model")
        model = onnx.load(self.model_file)
        print("model loaded")
        # make dim params static
        print("fixing dim params")
        for p, v in self.dim_param_dict.items():
            make_dim_param_fixed(model.graph, p, v)
        fix_output_shapes(model)
        print("dim_params fixed")
        # update the opset version if necessary
        print("updating opset version")
        is_updated = model.opset_import[0].version >= 21
        updated_model = model if is_updated else onnx.version_converter.convert_version(model, 21)
        print("opset version updated")
        print("performing shape inference")
        return onnx.shape_inference.infer_shapes(updated_model, data_prop=True)

    def import_model(self):
        inferred_model = self.load_model()
        # save the updated model
        print("importing to mlir")
        model_info = onnx_importer.ModelInfo(inferred_model)
        self.declared_input_map = model_info.main_graph.declared_input_map
        self.output_map = model_info.main_graph.output_map
        m = model_info.create_module()
        imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m.operation)
        imp.import_all()
        self.func_name = str(m.body.operations[0].name).strip('"')
        print(f'func_name = {self.func_name}')
        return m
    
    def compile(self, **compile_kwargs):
        return compile_str(str(self.mlir_module), **compile_kwargs)
    
    def get_inputs(self):
        input_list = []
        for key, value in self.declared_input_map.items():
            input_list.append(NodeArg.from_value_info_proto(key, value))
        return input_list

    def get_outputs(self):
        output_list = []
        for key, value in self.output_map.items():
            output_list.append(NodeArg.from_value_info_proto(key, value))
        return output_list

    def run(self, output_names, input_dict):
        import iree.runtime as ireert
        config = ireert.Config(self.device)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, self.vmfb)
        ctx.add_vm_module(vm_module)
        inputs_cfg = self.get_inputs()
        input_list = []
        for input_info in inputs_cfg:
            input_list.append(input_dict[input_info.name])         
        print(f'running model {self.model_file}')

        device_arrays = ctx.modules.module[self.func_name](*input_list)

        def get_numpy_output(d, expected_shape):
            array = d if isinstance(d, np.ndarray) else d.to_host()
            print(f'output array shape = {array.shape}')
            print(f'expected = {expected_shape}')
            return array.reshape(expected_shape)

        if not isinstance(device_arrays, tuple):
            device_arrays = tuple(device_arrays, )
        
        output_cfg = self.get_outputs()
        np_arrays = []
        for (idx, d) in enumerate(device_arrays):
            np_arrays.append(get_numpy_output(d, output_cfg[idx].shape))
        return tuple(np_arrays)

class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        session = IREEInferenceSession(self.onnx_file, **kwargs)
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        if len(outputs)>=5:
            return RetinaFace(model_file=self.onnx_file, session=session)
        elif input_shape[2]==192 and input_shape[3]==192:
            return Landmark(model_file=self.onnx_file, session=session)
        elif input_shape[2]==96 and input_shape[3]==96:
            return MZAttribute(model_file=self.onnx_file, session=session)
        elif len(inputs)==2 and input_shape[2]==128 and input_shape[3]==128:
            return INSwapper(model_file=self.onnx_file, session=session)
        elif input_shape[2]==input_shape[3] and input_shape[2]>=112 and input_shape[2]%16==0:
            return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            #raise RuntimeError('error on model routing')
            return None

class IREEFaceAnalysis(FaceAnalysis):
    def __init__(self, name="antelopev2", root='~/.cache/shark/', allowed_modules=None, **kwargs):
        self.models = {}
        _root = os.path.expanduser(root)
        dir_path = os.path.join(_root, "insightface_onnx", name)
        onnx_files = glob.glob(osp.join(dir_path, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            mr = ModelRouter(onnx_file)
            model = mr.get_model(**kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']