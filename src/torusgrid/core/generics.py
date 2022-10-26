from typing import TYPE_CHECKING, TypeVar, Type, Generic

T = TypeVar('T')

if TYPE_CHECKING:
    def generic(cls: Type[T]) -> Type[T]: 
        """
        Remove generic class inheritance and add trivial subscript
        such that A[T] = A.
        """
        return cls
else: 
    def generic(cls: type) -> type:
        """
        Remove generic class inheritance and add trivial subscript
        such that A[T] = A.
        """

        cls_ = add_trivial_subscript(remove_generic_base(cls))
        
        return cls_


def remove_generic_base(cls: type) -> type:
    """
    Remove Generic[T] base
    """
    metatype = type(cls)
    
    old_bases = cls.__bases__
    new_bases = list(cls.__bases__)

    for i, base in enumerate(old_bases):
        if base is Generic:
            del new_bases[i]

    if len(new_bases) == 0: new_bases = [object]

    cls_ = metatype(cls.__name__, tuple(new_bases), dict(cls.__dict__)) # type: ignore

    cls_.__name__ = cls.__name__
    cls_.__qualname__ = cls.__qualname__

    return cls_


def add_trivial_subscript(cls: type) -> type:
    """
    Add trivial class subscript
    """
    class cls_(cls): # type: ignore
        def __class_getitem__(cls, _):
            return cls
    cls_.__name__ = cls.__name__
    cls_.__qualname__ = cls.__qualname__
    return cls_

