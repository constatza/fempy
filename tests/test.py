# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pyximport
pyximport.install()


import numpy as np
import matplotlib.pyplot as plt
import smartplot as splt

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver
from fem.systems import LinearSystem

from fem.core.loads import TimeDependentLoad
from fem.core.entities import DOFtype
from fem.core.providers import ElementMaterialOnlyStiffnessProvider, RayleighDampingMatrixProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4
import mathematics.linalg as linalg
from fem.problems import ProblemStructuralDynamic



a = np.ones(10)
dot = linalg.dot(a,a)

import sys
print(sys.path)
