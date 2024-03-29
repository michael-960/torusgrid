from __future__ import annotations
from typing import Any, List, Literal, Tuple, TypeVar, Union, overload, Optional

import numpy as np
from matplotlib import pyplot as plt

from .fields import ComplexField2D, RealField2D
from .grids import ComplexGrid

import warnings

warnings.warn('The field_util module is no longer maintained. Use torusgrid.transforms instead', DeprecationWarning)


def plot(fields: ComplexField2D|List[ComplexField2D],
        cmap: str='jet', show: bool=True, vlim: Tuple[float, float]=(-1., 1.), 
        colorbar: bool=True, ncols=4, fig_dims=(4, 4)):

    if not type(fields) in [tuple, list]:
        assert isinstance(fields, ComplexField2D)
        fields = [fields]

    assert isinstance(fields, list)
    nrows = (len(fields)-1) // ncols + 1

    if len(fields) < ncols:
        ncols = len(fields)

    fig = plt.figure(figsize=(fig_dims[0]*ncols, fig_dims[1]*nrows))
    cms = []
    axs = []
    for i, field in enumerate(fields):
        ax = plt.subplot(nrows, ncols, i+1)
        cm = ax.pcolormesh(field.X, field.Y, np.real(field.psi), cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='nearest')
        if colorbar:
            plt.colorbar(cm, ax=ax, orientation='horizontal')

        ax.set_aspect('equal', adjustable='box')
        cms.append(cm)
        cms.append(ax)

    if show:
       plt.show() 
    else:
       return fig, axs

@overload
def set_size(field: RealField2D, Lx: float, Ly: float) -> RealField2D: ...
@overload
def set_size(field: ComplexField2D, Lx: float, Ly: float) -> ComplexField2D: ...
@overload
def set_size(field: ComplexField2D, Lx: float, Ly: float, in_place: Literal[True]) -> None: ...
@overload
def set_size(field: RealField2D, Lx: float, Ly: float, in_place: Literal[False]) -> RealField2D: ...
@overload
def set_size(field: ComplexField2D, Lx: float, Ly: float, in_place: Literal[False]) -> ComplexField2D: ...

def set_size(field: ComplexField2D, Lx: float, Ly: float, in_place: bool=False) -> Union[None, ComplexField2D, RealField2D]:
    if in_place:
        field.set_size(Lx, Ly)
        return
    else:
        field1 = field.copy()
        field1.set_size(Lx, Ly)
        return field1


@overload
def change_resolution(field: RealField2D, Nx: int, Ny: int) -> RealField2D: ...
@overload
def change_resolution(field: ComplexField2D, Nx: int, Ny: int) -> ComplexField2D: ...
@overload
def change_resolution(field: ComplexField2D, Nx: int, Ny: int, in_place: Literal[True]) -> None: ...
@overload
def change_resolution(field: RealField2D, Nx: int, Ny: int, in_place: Literal[False]) -> RealField2D: ...
@overload
def change_resolution(field: ComplexField2D, Nx: int, Ny: int, in_place: Literal[False]) -> ComplexField2D: ...

def change_resolution(field: ComplexField2D, Nx: int, Ny: int, in_place: bool=False) -> Union[None, ComplexField2D, RealField2D]:
    psi1 = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            psi1[i,j] = field.psi[int(i*field.Nx/Nx),int(j*field.Ny/Ny)]

    if in_place:
        field.set_dimensions(field.Lx, field.Ly, Nx, Ny)
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        field1.set_dimensions(field.Lx, field.Ly, Nx, Ny)
        field1.set_psi(psi1)
        return field1


@overload
def extend(field: RealField2D, Mx: int, My: int) -> RealField2D: ...
@overload
def extend(field: ComplexField2D, Mx: int, My: int) -> ComplexField2D: ...
@overload
def extend(field: ComplexField2D, Mx: int, My: int, in_place: Literal[True]) -> None: ...
@overload
def extend(field: RealField2D, Mx: int, My: int, in_place: Literal[False]) -> RealField2D: ...
@overload
def extend(field: ComplexField2D, Mx: int, My: int, in_place: Literal[False]) -> ComplexField2D: ...

def extend(field: ComplexField2D, Mx: int, My: int, in_place: bool=False) -> None | ComplexField2D | RealField2D:
    Nx1 = field.Nx * Mx
    Ny1 = field.Ny * My

    psi1 = np.zeros((Nx1, Ny1))

    for i in range(Nx1):
        for j in range(Ny1):
            psi1[i,j] = field.psi[i%field.Nx,j%field.Ny]
    
    if in_place:
        field.set_dimensions(field.Lx*Mx, field.Ly*My, Nx1, Ny1)
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        field1.set_dimensions(field.Lx*Mx, field.Ly*My, Nx1, Ny1)
        field1.set_psi(psi1)
        return field1


@overload
def flip(field: RealField2D, axis: str) -> RealField2D: ...
@overload
def flip(field: ComplexField2D, axis: str) -> ComplexField2D: ...
@overload
def flip(field: ComplexField2D, axis: str, in_place: Literal[True]) -> None: ...
@overload
def flip(field: RealField2D, axis: str, in_place: Literal[False]) -> RealField2D: ...
@overload
def flip(field: ComplexField2D, axis: str, in_place: Literal[False]) -> ComplexField2D: ...

def flip(field: ComplexField2D, axis: str, in_place: bool=False) -> None | RealField2D | ComplexField2D:
    if axis not in ['X', 'Y']:
        raise ValueError(f'\'{axis}\' is not a valid axis for flipping') 

    psi1 = field.psi.copy()
    if axis == 'X':
        psi1 = psi1[:,::-1]
    if axis == 'Y':
        psi1 = psi1[::-1,:]

    if in_place:
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        field1.set_psi(psi1)
        return field1

@overload
def transpose(field: RealField2D) -> RealField2D: ...
@overload
def transpose(field: ComplexField2D) -> ComplexField2D: ...
@overload
def transpose(field: ComplexField2D, in_place: Literal[True]) -> None: ...
@overload
def transpose(field: RealField2D, in_place: Literal[False]) -> RealField2D: ...
@overload
def transpose(field: ComplexField2D, in_place: Literal[False]) -> ComplexField2D: ...

def transpose(field: ComplexField2D, in_place=False) -> None | RealField2D | ComplexField2D:
    psi1 = np.transpose(field.psi)
    if in_place:
        field.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field.set_psi(psi1)

    else:
        field1 = field.copy()
        field1.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field1.set_psi(psi1)
        return field1

@overload
def rotate(field: RealField2D, angle: str) -> RealField2D: ...
@overload
def rotate(field: ComplexField2D, angle: str) -> ComplexField2D: ...
@overload
def rotate(field: ComplexField2D, angle: str, in_place: Literal[True]) -> None: ...
@overload
def rotate(field: RealField2D, angle: str, in_place: Literal[False]) -> RealField2D: ...
@overload
def rotate(field: ComplexField2D, angle: str, in_place: Literal[False]) -> ComplexField2D: ...

def rotate(field: ComplexField2D, angle: str, in_place=False) -> None | RealField2D | ComplexField2D:
    if not angle in ['90', '180', '270']:
        raise ValueError(f'\'{angle}\' is not a valid angle for rotation')

    psi1 = field.psi.copy()
    if angle == '90':
        psi1 = np.transpose(psi1)[::-1,:]
    if angle == '180':
        psi1 = psi1[::-1,::-1]
    if angle == '270':
        psi1 = np.transpose(psi1)[:,::-1]

    if in_place:
        if angle in ['90', '270']:
            field.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        if angle in ['90', '270']:
            field1.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field1.set_psi(psi1)
        return field1

@overload
def crop(field: RealField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float) -> RealField2D: ...
@overload
def crop(field: ComplexField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float) -> ComplexField2D: ...
@overload
def crop(field: ComplexField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float, in_place: Literal[True]) -> None: ...
@overload
def crop(field: RealField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float, in_place: Literal[False]) -> RealField2D: ...
@overload
def crop(field: ComplexField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float, in_place: Literal[False]) -> ComplexField2D: ...

def crop(field: ComplexField2D, ratio_x1: float, ratio_x2: float, ratio_y1: float, ratio_y2: float, in_place=False):
    Nx2 = int(field.Nx * (ratio_x2 - ratio_x1))
    Ny2 = int(field.Ny * (ratio_y2 - ratio_y1))
    first_x = int(field.Nx * ratio_x1)
    first_y = int(field.Nx * ratio_y1)
    Lx2 = field.Lx*(ratio_x2-ratio_x1)
    Ly2 = field.Ly*(ratio_y2-ratio_y1)

    psi2 = field.psi[first_x:first_x+Nx2, first_y:first_y+Ny2]

    if in_place:
        field.set_dimensions(Lx2, Ly2, Nx2, Ny2)
        field.set_psi(psi2)
        return
    else:
        field1 = field.copy()
        field1.set_dimensions(Lx2, Ly2, Nx2, Ny2)
        field1.set_psi(psi2)
        return field1


def concat(field_a: ComplexField2D, field_b: ComplexField2D, 
        direction='horizontal', in_place=False):
    if not direction in ['horizontal', 'vertical']:
        raise ValueError(f'\'{direction}\' is not a valid direction')

    psi1 = None 
    if direction == 'horizontal':
        if field_a.Ny != field_b.Ny or not np.isclose(field_a.Ly, field_b.Ly, rtol=1e-9, atol=1e-10):
            raise FieldOperationError(f'could not horizontally concat two fields with different vertical dimensions:'+\
                    'Ly1={field_a.Ly}, Ny1={field_a.Ny}, Ly2={field_b.Ly}, Ny2={field_b.Ny}')

        psi1 = np.zeros(shape=(field_a.Nx+field_b.Nx, field_a.Ny)) 
        psi1[:field_a.Nx,:] = field_a.psi[:,:]
        psi1[field_a.Nx:,:] = field_b.psi[:,:]

        if in_place:
            field_a.set_dimensions(field_a.Lx+field_b.Lx, field_a.Ly, field_a.Nx+field_b.Nx, field_a.Ny)
            field_a.set_psi(psi1)
            return
        else:
            field1 = field_a.copy()
            field1.set_dimensions(field_a.Lx+field_b.Lx, field_a.Ly, field_a.Nx+field_b.Nx, field_a.Ny)
            field1.set_psi(psi1)
            return field1


    if direction == 'vertical':
        if field_a.Ny != field_b.Ny or not np.isclose(field_a.Ly, field_b.Ly, rtol=1e-9, atol=1e-10):
            raise FieldOperationError(f'could not vertically concat two fields with different horizontal dimensions:'+\
                    'Lx1={field_a.Lx}, Nx1={field_a.Nx}, Lx2={field_b.Lx}, Nx2={field_b.Nx}')

        psi1 = np.zeros(shape=(field_a.Nx, field_a.Ny+field_b.Ny)) 
        psi1[:,:field_a.Ny] = field_a.psi[:,:]
        psi1[:,field_a.Ny:] = field_b.psi[:,:]

        if in_place:
            field_a.set_dimensions(field_a.Lx, field_a.Ly+field_b.Ly, field_a.Nx, field_a.Ny+field_b.Ny)
            field_a.set_psi(psi1)
            return
        else:
            field1 = field_a.copy()
            field_a.set_dimensions(field_a.Lx, field_a.Ly+field_b.Ly, field_a.Nx, field_a.Ny+field_b.Ny)
            field_a.set_psi(psi1)
            return field_a


def insert(
        field_large: ComplexField2D, field_small: ComplexField2D, 
        position='center', in_place=False, position_format='ratio',
        periodic_boundary=False
    ):
    '''
    '''
    valid = False
    if position in ['left', 'right', 'top', 'bottom', 'center']:
        valid = True
    if type(position) is tuple:
        if len(position) == 2:
            valid = True
    if not valid:
        raise ValueError(f'\'{position}\' is not a valid insertion position')

    if not position_format in ['ratio', 'points', 'length']:
        raise ValueError(f'\'{position_format}\' is not a valid position format')

    if not (field_small.Lx <= field_large.Lx and field_small.Ly <= field_small.Ly):
        raise FieldOperationError('the first model must have smaller dimensions than the second one')

    box_Nx, box_Ny = int(field_small.Lx / field_large.Lx * field_large.Nx), int(field_small.Ly / field_large.Ly * field_large.Ny)
    psi_small_modified = change_resolution(field_small, box_Nx, box_Ny)
   
    box_X, box_Y = int(field_large.Nx/2 - box_Nx/2), int(field_large.Ny/2 - box_Ny/2)

    if position == 'left':
        box_X = 0

    if position == 'right':
        box_X = field_large.Nx - box_Nx

    if position == 'top':
        box_Y = 0
        
    if position == 'bottom':
        box_Y = field_large.Ny - box_Ny

    if type(position) is tuple:
        if position_format == 'ratio':
            box_X = int(position[0] * field_large.Nx)
            box_Y = int(position[1] * field_large.Ny)

        if position_format == 'points':
            box_X = int(position[0])
            box_Y = int(position[1])

        if position_format == 'length':
            box_X = int(position[0]/field_large.Lx * field_large.Nx)
            box_Y = int(position[1]/field_large.Ly * field_large.Ny)

        if (not periodic_boundary) and (box_X + box_Nx > field_large.Nx or box_Y + box_Ny > field_large.Ny):
            raise FieldOperationError('insertion out of boundary with periodic_boundary=False')

    psi1 = field_large.psi.copy()

    psi1[box_X:(box_X+box_Nx)%field_large.Nx,box_Y:(box_Y+box_Ny)%field_large.Ny] = psi_small_modified[:,:]

    if in_place:
        field_large.set_psi(psi1)
        return


    else:
        field1 = field_large.copy()
        field1.set_psi(psi1)
        return field1

@overload 
def interface(field_a: RealField2D, field_b: RealField2D, width:
        float, left=0.25, right=0.75) -> RealField2D: ...
@overload
def interface(field_a: ComplexField2D, field_b: ComplexField2D, width: float,
        left=0.25, right=0.75) -> ComplexField2D: ...
@overload
def interface(field_a: ComplexField2D, field_b: ComplexField2D, width: float,
        left: float=0.25, right: float=0.75, in_place: Literal[True]=True) -> None:
    ...
@overload
def interface(field_a: RealField2D, field_b: RealField2D, width: float,
        left: float=0.25, right: float=0.75, in_place: Literal[False]=False) -> RealField2D: ...
@overload
def interface(field_a: ComplexField2D, field_b: ComplexField2D, width: float,
        left: float=0.25, right: float=0.75, in_place: Literal[False]=False) -> ComplexField2D: ...

def interface(field_a: ComplexField2D, field_b: ComplexField2D, width: float,
        left: float=0.25, right: float=0.75, in_place: bool=False) -> None | ComplexField2D | RealField2D:
    try:
        assert np.isclose(field_a.Lx, field_b.Lx, rtol=1e-6, atol=1e-6)
        assert np.isclose(field_a.Ly, field_b.Ly, rtol=1e-6, atol=1e-6)
        assert field_a.Nx == field_b.Nx
        assert field_a.Ny == field_b.Ny

    except AssertionError as e:
        raise ValueError(f'two fields with different dimensions can not be combined using interface()')


    X, Y = field_a.X, field_a.Y

    xa = field_a.Lx * left
    xb = field_a.Lx * right

    bump = (1+np.tanh((X-xa)/width))/2 * (1+np.tanh((-X+xb)/width))/2

    psi1 = field_a.psi * bump + field_b.psi * (1-bump)

    if in_place:
        field_a.set_psi(psi1)
        return

    else:
        field1 = field_a.copy()
        field1.set_psi(psi1)
        return field1

T = TypeVar('T', bound=ComplexGrid)



@overload
def liquefy(field: T, psi0=None, *, in_place: Literal[False]=False) -> T: ...

@overload
def liquefy(field: ComplexGrid, psi0=None, *, in_place: Literal[True]) -> None: ...

@overload
def liquefy(field: T, psi0=None, *, in_place: bool=False) -> None|T: ...

def liquefy(
    field, psi0=None, *,
    in_place: bool=False):

    if psi0 is None:
        psi0 = np.mean(field.psi)
    
    if in_place:
        field.set_psi(psi0)
        return

    else:
        field1 = field.copy()
        field1.set_psi(psi0)
        return field1



class FieldOperationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


