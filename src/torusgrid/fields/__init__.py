'''
Field module. Fields are grids with length scales.
'''

from ._base import Field
from ._complex import ComplexField
from ._real import RealField

from ._lowdim import ComplexField2D, RealField2D
from ._util import load_field, import_field, FieldOperationError




