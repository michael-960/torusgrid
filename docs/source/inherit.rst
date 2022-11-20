Inheritance
============

Following the `DRY principle <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_,
:code:`torusgrid` uses multiple inheritance to ensure that the same
functionality is implemented only once. So, for example, :code:`RealGrid1D` has
`RealGrid` and `Grid1D` as base classes, where the latter is an abstract base
class for 1D grid interfaces. The complete inheritance diagram is shown below.


.. inheritance-diagram:: torusgrid.Grid1D torusgrid.Grid2D 
                        torusgrid.ComplexGrid torusgrid.ComplexGrid1D torusgrid.ComplexGrid2D
                        torusgrid.RealGrid torusgrid.RealGrid1D torusgrid.RealGrid2D
                        torusgrid.Field 
                        torusgrid.Field1D torusgrid.Field2D 
                        torusgrid.ComplexField torusgrid.ComplexField1D torusgrid.ComplexField2D
                        torusgrid.RealField torusgrid.RealField1D torusgrid.RealField2D
   :top-classes: torusgrid.Grid
   :parts: 1

