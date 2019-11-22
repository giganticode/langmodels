from dataclasses import dataclass

from typing import Dict, Union, TypeVar


T = TypeVar('T')
TreeLikeDict = Dict[str, Union['TreeLikeDict', str]]


def patch_config(a: T, patch: Dict[str, str]) -> T:
    """
    >>> @dataclass(frozen=True)
    ... class C(object):
    ...     d: float = 3.0

    >>> @dataclass(frozen=True)
    ... class A(object):
    ...     a: int = 1
    ...     b: str = '2'
    ...     c: C = C()
    ...     d: bool = True

    >>> patch_config(A(), {})
    A(a=1, b='2', c=C(d=3.0), d=True)

    >>> patch_config(A(), {'c.d': '91', 'b': '89', 'd': 'False'})
    A(a=1, b='89', c=C(d=91.0), d=False)
    """
    tree_like_dct: TreeLikeDict = {}
    for path_to_param, new_value in patch.items():
        cur = tree_like_dct
        attrs = path_to_param.split(".")
        for attr in attrs[:-1]:
            if attr not in cur:
                cur[attr] = {}
            cur = cur[attr]
        cur[attrs[-1]] = new_value
    return patch_object(a, tree_like_dct)


def patch_object(a: T, params_to_be_patched: TreeLikeDict) -> T:
    """
    >>> @dataclass(frozen=True)
    ... class C(object):
    ...     d: float = 3.0

    >>> @dataclass(frozen=True)
    ... class A(object):
    ...     a: int = 1
    ...     b: str = '2'
    ...     c: C = C()
    ...     d: bool = True

    >>> patch_object(A(), {})
    A(a=1, b='2', c=C(d=3.0), d=True)

    >>> patch_object(A(), {'c': {'d': '91.0'}, 'b': '89', 'd': 'False'})
    A(a=1, b='89', c=C(d=91.0), d=False)
    """
    patched_attributes = {}
    for k, v in params_to_be_patched.items():
        attr_to_patch = getattr(a, k)
        if isinstance(v, dict):
            patched_attributes[k] = patch_object(attr_to_patch, v)
        else:
            tp = type(attr_to_patch)
            patched_attributes[k] = tp(v) if tp != bool else v.lower() == 'true'
    a_type = type(a)
    return a_type(**patched_attributes)
