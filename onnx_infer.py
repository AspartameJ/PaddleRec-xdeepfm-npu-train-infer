# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import os
import sys
from importlib import import_module
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from paddle.io import DataLoader
import argparse
import onnxruntime as rt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_file", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--reader_file", type=str)
    parser.add_argument("--batchsize", type=int)
    args = parser.parse_args()
    return args


def create_data_loader(args):
    data_dir = args.data_dir
    reader_path, reader_file = os.path.split(args.reader_file)
    reader_file, extension = os.path.splitext(reader_file)
    batchsize = args.batchsize
    place = paddle.set_device('cpu')
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sys.path.append(reader_path)
    #sys.path.append(os.path.abspath("."))
    reader_class = import_module(reader_file)
    config = {"runner.inference": True}
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader


def main(args):
    sess = rt.InferenceSession(args.onnx_file)

    test_dataloader = create_data_loader(args)

    input_names = [input_node.name for input_node in sess.get_inputs()]
    output_names = [sess.get_outputs()[0].name]

    for batch_id, batch_data in enumerate(test_dataloader):
        name_data_pair = dict(zip(input_names, batch_data))

        input_names_dict = {}
        for name in input_names:
            input_names_dict[name] = name_data_pair[name].numpy()
        pred_onnx = sess.run(output_names, input_names_dict)
        results = []
        results_type = []
        for name in output_names:
            results_type.append(type(pred_onnx))
            results.append(pred_onnx[0])
        for clicked in results:
            if clicked > 0.5:
                print([1])
            else:
                print([0])


if __name__ == '__main__':
    args = parse_args()
    main(args)
