#!/usr/bin/env python3
"""Generate simple ONNX models for testing the Zig ONNX runtime bindings."""

import subprocess
import sys

def ensure_packages():
    """Install required packages if not present."""
    try:
        import onnx
    except ImportError:
        print("Installing onnx package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "-q"])

ensure_packages()

import onnx
from onnx import helper, TensorProto


def create_add_model():
    """Create a simple model that adds two tensors: C = A + B"""
    # Define inputs
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])

    # Define output
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 3])

    # Create Add node
    add_node = helper.make_node(
        'Add',
        inputs=['A', 'B'],
        outputs=['C'],
        name='add_node'
    )

    # Create graph
    graph = helper.make_graph(
        [add_node],
        'add_graph',
        [A, B],
        [C]
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    return model


def create_identity_model():
    """Create an identity model: Y = X"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

    identity_node = helper.make_node(
        'Identity',
        inputs=['X'],
        outputs=['Y'],
        name='identity_node'
    )

    graph = helper.make_graph(
        [identity_node],
        'identity_graph',
        [X],
        [Y]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    return model


def create_relu_model():
    """Create a ReLU activation model: Y = ReLU(X)"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8])

    relu_node = helper.make_node(
        'Relu',
        inputs=['X'],
        outputs=['Y'],
        name='relu_node'
    )

    graph = helper.make_graph(
        [relu_node],
        'relu_graph',
        [X],
        [Y]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    return model


def create_matmul_model():
    """Create a MatMul model: Y = A @ B (batch matrix multiply)"""
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])

    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['Y'],
        name='matmul_node'
    )

    graph = helper.make_graph(
        [matmul_node],
        'matmul_graph',
        [A, B],
        [Y]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    return model


def main():
    import os

    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'test')
    os.makedirs(models_dir, exist_ok=True)

    # Generate models
    models = [
        ('add.onnx', create_add_model()),
        ('identity.onnx', create_identity_model()),
        ('relu.onnx', create_relu_model()),
        ('matmul.onnx', create_matmul_model()),
    ]

    for filename, model in models:
        filepath = os.path.join(models_dir, filename)
        onnx.checker.check_model(model)
        onnx.save(model, filepath)
        print(f"Created: {filepath}")

    print("\nTest models generated successfully!")
    print("\nExpected test values:")
    print("- add.onnx: C = A + B where A, B are [2,3] tensors")
    print("- identity.onnx: Y = X where X is [1,4] tensor")
    print("- relu.onnx: Y = max(0, X) where X is [1,8] tensor")
    print("- matmul.onnx: Y = A @ B where A is [2,3], B is [3,4]")


if __name__ == '__main__':
    main()
