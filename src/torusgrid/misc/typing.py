from typing import TYPE_CHECKING, TypeVar, Type, Generic

T = TypeVar('T')

if TYPE_CHECKING:
    def generic(cls: Type[T]) -> Type[T]: return cls
else: 
    def generic(cls: type) -> type:
        '''
        Remove generic class inheritance and add trivial subscript
        such that A[T] = A.
        '''
        meta = type(cls) 
        class meta_(meta):
            def __getitem__(self, key): return self

        meta_.__name__ = meta.__name__
        meta_.__qualname__ = meta.__qualname__

        old_bases = cls.__bases__
        new_bases = list(cls.__bases__)

        for i, base in enumerate(old_bases):
            if base is Generic:
                del new_bases[i]

        if len(new_bases) == 0: new_bases = [object]

        cls_ = meta_(cls.__name__, tuple(new_bases), dict(cls.__dict__))
        # class cls_(cls, metaclass=meta_):
        #     ...
        cls_.__name__ = cls.__name__
        cls_.__qualname__ = cls.__qualname__

        return cls_

