"""init file for Mixed Precision Training"""
from .decorators import auto_fp16
from .utils import cast_tensor_type

__all__ = [
    'auto_fp16', 'cast_tensor_type'
]
