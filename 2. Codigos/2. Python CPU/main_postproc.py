# MAIN SCRIPT

"""
  Academic 2D HDG code for solving the Poisson equation with Dirichlet 
  boundary conditions.
 
  First version by Sonia Fernandez-Mendez LaCaN UPC-BarcelonaTech 2016
  Second version by Ruben Sevilla, Swansea University 2017
 
  www.lacan.upc.edu
 
  Main data variables:
   X: nodal coordinates
   T: mesh connectivitity matrix
   F: faces (here sides) for each element 
     (Faces are numbered so that interior faces are first)
   elemInfo: element type
   infoFaces.intFaces: [elem1 face1 elem2 face2 rotation] for each face
     (Each face is assumed to have the orientation given by the 
     first element, it is flipped when seen from the second element)
   infoFaces.extFaces: [elem1 face1] for each exterior face
   referenceElement: integration points and shape functions (volume and
     boundary sides)
 
"""

### Libraries required ###
#########################################################

import numpy as np
import matplotlib.pyplot as plt

import time
from scipy import sparse  

### Initialization of variables ###
#########################################################

mu = 1

### Load computational mesh (from .mat files) ###
#########################################################

import importlib
import mesh.mesh_matrices_mat
importlib.reload(mesh.mesh_matrices_mat)  # Force recharge of the module
from mesh.mesh_matrices_mat import load_mesh

matrix_number = 7 # User-defined matrix number
degree = 4 # User-defined degree


#from mesh.mesh_converter import generate_python_mesh_file
#directory = r'C:\Users\Usuario\Desktop\TFM\TASK 1\HDG_Poisson_Python\T1_HDG_Poisson\mesh'
#generated_files = generate_python_mesh_file(directory)
#print("Archivos NPZ generados:", generated_files)

X, T, Tb_Dirichlet = load_mesh(matrix_number, degree) # Load the matrices and initialize them as variables

"""if X is not None: #To compare with Matlab
    print(f"\n Loaded file: mesh{matrix_number}_P{degree}.mat")
    print("\nX (Nodal Coordinates):\n", X)
    print("\nT (Mesh Connectivity):\n", T)
    print("\nTb_Dirichlet (Dirichlet Boundary Conditions):\n", Tb_Dirichlet)"""

nOfElements = T.shape[0]

import mesh.mesh_plot
importlib.reload(mesh.mesh_plot)  # Force recharge of the module
from mesh.mesh_plot import plot_mesh

plot_mesh(X, T,option='', nodesNum='all')  # Mesh representation in a 2D plot

### HDG preprocess ###
#########################################################

import hdgCommon.hdg_preprocess
from hdgCommon.hdg_preprocess import hdg_preprocess

t0 = time.perf_counter()
F, infoFaces = hdg_preprocess(T)
ms_preprocess = 1000.0 * (time.perf_counter() - t0)
nOfFaces = np.max(F)+1  # Find the total number of faces

"""print("\nF:\n", F)
print("\ninfoFaces:\n", infoFaces)
print("Total Number of Faces:", nOfFaces)"""

### Viscosity parameter ###
#########################################################

muElem = mu * np.ones(nOfElements)

### Stabilization parameter ###
#########################################################

tau = np.ones((nOfElements,3))

### Computation ###
#########################################################

import referenceElement.createReferenceElementTri
from referenceElement.createReferenceElementTri import create_reference_element_tri

referenceElement = create_reference_element_tri(degree)


import pprint
pprint.pprint(referenceElement)

print("\nLoop in elements...\n")

from hdgPoisson.hdgMatrixPoisson_postproc import hdgMatrixPoisson

t0 = time.perf_counter()
K, f, QQ, UU, Qf, Uf, KKe_cells, All_cells = hdgMatrixPoisson(
    muElem, X, T, F, referenceElement, infoFaces, tau
)
ms_assembly = 1000.0 * (time.perf_counter() - t0)
print("Assembly time (ms):", ms_assembly)

"""print("===== GLOBAL MATRIX K (disperse) =====")
print(K)
print("Shape:", K.shape)
print("Number of non null elements:", K.nnz)

print("\n===== GLOBAL SOURCE VERCTOR f =====")
print(f)
print("Shape:", f.shape)

print("\n===== LOCAL MATRIXES QQ =====")
for i, QQe in enumerate(QQ):
    print(f"Element {i}:")
    print(QQe.toarray())

print("\n===== LOCAL MATRIXES UU =====")
for i, UUe in enumerate(UU):
    print(f"Element {i}:")
    print(UUe.toarray())

print("\n===== LOCAL SOURCE VERCTORS Qf =====")
for i, Qfe in enumerate(Qf):
    print(f"Element {i}:")
    print(Qfe.flatten())

print("\n===== GLOBAL SOURCE VERCTORS Uf =====")
for i, Ufe in enumerate(Uf):
    print(f"Element {i}:")
    print(Ufe.flatten())"""

### Dirichlet BC ###
# Dirichlet face nodal coordinates

nOfFaceNodes = degree + 1

nOfInteriorFaces = infoFaces["intFaces"].shape[0]
nOfExteriorFaces = infoFaces["extFaces"].shape[0]

import poisson.analyticalPoisson
from poisson.analyticalPoisson import analytical_poisson

import hdgCommon.computeProjectionFaces
from hdgCommon.computeProjectionFaces import compute_projection_faces

from scipy.sparse.linalg import spsolve, factorized

t0 = time.perf_counter()
uDirichlet = compute_projection_faces(analytical_poisson, infoFaces["extFaces"], X, T, referenceElement)
"""
print("\nDirichlet boundary condition (uDirichlet):\n", uDirichlet)
"""
dofDirichlet = np.arange(nOfInteriorFaces * nOfFaceNodes, 
                         nOfInteriorFaces * nOfFaceNodes + nOfExteriorFaces * nOfFaceNodes)
dofUnknown = np.arange(nOfInteriorFaces * nOfFaceNodes)
# The previous dofs are 0-indexed

# Extract submatrixes and source vector (CSR compatible)
K_ud = K[dofUnknown[:, None], dofDirichlet]
f_reduced = f[dofUnknown] - K_ud @ uDirichlet
K_reduced = K[dofUnknown[:, None], dofUnknown]
ms_dirichlet_reduce = 1000.0 * (time.perf_counter() - t0)
print("Dirichlet reduction time (ms):", ms_dirichlet_reduce)
A = K_reduced.tocsc()
b = np.asarray(f_reduced).ravel()

### Face solution ###

from numpy.linalg import solve
t0 = time.perf_counter()
lambda_vals = spsolve(A, b, permc_spec='MMD_AT_PLUS_A')
ms_solve_global = 1000.0 * (time.perf_counter() - t0)
print("ms_solve_global:", ms_solve_global)

"""print("\nLambda values (solution for the unknowns):\n", lambda_vals)"""  

# complete solution (lambda + uDirichlet)
uhat = np.zeros(K.shape[0])
uhat[dofUnknown] = lambda_vals
uhat[dofDirichlet] = uDirichlet

"""print("\nComplete solution (uhat):\n", uhat)"""

### Elemental solution ###

import hdgPoisson.computeElementsSolution
from hdgPoisson.computeElementsSolution import compute_elements_solution

print("Calculating element by element solution...")
t0 = time.perf_counter()
u, q = compute_elements_solution(uhat, UU, QQ, Uf, Qf, F)
ms_elem_solve = 1000.0 * (time.perf_counter() - t0)
print("Elemental solution time (ms):", ms_elem_solve)

"""print("\nu:\n", u)
print("q:\n", q)"""


from Utilities.plotDiscontinousSolution import plot_discontinuous_solution

fig, ax, tpc = plot_discontinuous_solution(X, T, u, referenceElement, n_deg_ref=20)

plt.colorbar(tpc, ax=ax)
plt.tight_layout()
plt.show()  

### Local postprocess for superconvergence ###

print("\nPerforming local postprocess...\n")

from referenceElement.createReferenceElementTriStar import create_reference_element_tri_star
from hdgCommon.HDGPostProcess import HDG_postprocess

t0 = time.perf_counter()
referenceElement_star = create_reference_element_tri_star(referenceElement)
u_star = HDG_postprocess(X, T, u, -q, referenceElement_star)
ms_postprocess = 1000.0 * (time.perf_counter() - t0)
print("Postprocess time (ms):", ms_postprocess)
"""print("\nPostprocessed solution (u_star):\n", u_star) """
### Plots postprocess solution ###

from Utilities.plotPostProcessedSolution import plot_postprocessed_solution

fig, ax, tpc = plot_postprocessed_solution(X, T, u_star, referenceElement_star, n_deg_ref=20)

plt.colorbar(tpc, ax=ax)
plt.tight_layout()
plt.show()

### Errors ###
#########################################################

from Utilities.computeL2Norm import compute_L2_norm
from Utilities.computeL2NormPostProcessed import compute_L2_norm_postprocess

import numpy as np

### Define the analytical solution (as a function) ###
u_ex = analytical_poisson  # Assume this is a Python function: def analytical_poisson(xy): ...

### Error for the HDG solution ###
t0 = time.perf_counter()
error = compute_L2_norm(referenceElement, X, T, u, u_ex)

### Error for the postprocessed solution ###
error_post = compute_L2_norm_postprocess(referenceElement_star, X, T, u_star, u_ex)
ms_errors  = 1000.0 * (time.perf_counter() - t0)
print("Error computation time (ms):", ms_errors)
print(f"Error HDG = {error:.6e}")
print(f"Error HDG postprocessed = {error_post:.6e}")
print()

# total (igual que en MATLAB: suma de sub-etapas)
ms_total = float(ms_preprocess + ms_assembly + ms_dirichlet_reduce +
                 ms_solve_global + ms_elem_solve + ms_postprocess + ms_errors)

# ==== EXTRACTION + SAVE (Python CPU) =========================================
import os, time, platform, getpass
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat

# --- User / run identifiers (match MATLAB schema)
method     = 'python_cpu'
timestamp  = time.strftime('%Y%m%dT%H%M%S')
mesh_id    = f'mesh{matrix_number}'
run_id     = f'{mesh_id}_P{degree}_{timestamp}'

root_dir   = os.getcwd()
out_root   = os.path.join(root_dir, 'results', method)
dirs = {
    'root': out_root,
    'runs': os.path.join(out_root, 'runs'),
    'rows': os.path.join(out_root, 'rows'),
    'locals': os.path.join(out_root, 'locals')
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

exp_path = os.path.join(out_root, 'experiment_global.mat')

# --- Helper: equivalent mesh size h_equiv (triangles)
def compute_mesh_h_equiv_py(X, T):
    x1, y1 = X[T[:,0],0], X[T[:,0],1]
    x2, y2 = X[T[:,1],0], X[T[:,1],1]
    x3, y3 = X[T[:,2],0], X[T[:,2],1]
    area = 0.5*np.abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    a_eq = np.sqrt(4*area/np.sqrt(3.0))
    return area, float(np.mean(a_eq))

_, h_equiv = compute_mesh_h_equiv_py(X, T)

# --- Timings (wrap your existing stages)
# Si quieres tiempos finos por etapa, mide con time.perf_counter() donde corresponda.
# Aquí asumimos que ya tienes:
#   ms_preprocess, ms_assembly, ms_dirichlet_reduce, ms_solve_global,
#   ms_elem_solve, ms_postprocess, ms_errors, y calculamos total si hace falta.
def _ms(v): return float(v)
timings = dict(
    preprocess      = _ms(ms_preprocess),
    assembly        = _ms(ms_assembly),
    dirichlet_reduce= _ms(ms_dirichlet_reduce),
    solve_global    = _ms(ms_solve_global),
    elem_solve      = _ms(ms_elem_solve),
    postprocess     = _ms(ms_postprocess),
    errors          = _ms(ms_errors),
)
ms_total = float(sum(timings.values()))

# --- Algebraic metrics & DoFs (match MATLAB)
nOfElements       = int(T.shape[0])
nOfInteriorFaces  = int(infoFaces["intFaces"].shape[0])
nOfExteriorFaces  = int(infoFaces["extFaces"].shape[0])
nOfFaceNodes      = int(degree + 1)
Np                = int((degree+1)*(degree+2)//2)

dofs_DG_total     = int(nOfElements * Np)
dofs_HDG_total    = int((nOfInteriorFaces + nOfExteriorFaces) * nOfFaceNodes)
dofs_HDG_unknowns = int(nOfInteriorFaces * nOfFaceNodes)

dimK  = int(K_reduced.shape[0])          # tras reducción (traza interior)
nnzK  = int(K_reduced.nnz)
densK = float(nnzK) / max(1, dimK*dimK)

# --- Accuracy (match MATLAB field names)
Error     = float(error)
ErrorPost = float(error_post)

# --- Percent breakdown (match MATLAB)
pct = {k: 100.0*v/ms_total for k,v in timings.items()}

# --- Locals for Figure 2 (patterns 0/1)
# (a) A1: K_reduced (ya reducido por Dirichlet)
A1 = K_reduced.copy().astype(np.float64)
A1.data[:] = 1.0

# (b) Zinv: bloque-diagonal con KKe locales (patrón)
# (c) Zminv: bloque-diagonal con (-All) (patrón)
def _block_diag_ones(blocks):
    if len(blocks) == 0: 
        return sparse.csr_matrix((0,0))
    B = sparse.block_diag(blocks, format='csr')
    B.data[:] = 1.0
    return B

# Limitar nº de bloques solo para que el .mat no sea gigante (igual que MATLAB)
bsz = 3 * nOfFaceNodes
ne_plot = min(max(1, int(np.ceil(80.0/bsz))), len(QQ))  # usa tamaño de cara ~
Zinv_blocks = []
Zminv_blocks = []
if 'KKe_cells' in globals() and KKe_cells is not None:
    for e in range(ne_plot):
        Zinv_blocks.append(sparse.csr_matrix(KKe_cells[e]))
if 'All_cells' in globals() and All_cells is not None:
    for e in range(ne_plot):
        Zminv_blocks.append(sparse.csr_matrix(-All_cells[e]))

Zinv  = _block_diag_ones(Zinv_blocks)   # patrón de Z^{-1}
Zminv = _block_diag_ones(Zminv_blocks)  # patrón de Z_m^{-1}

locals_path = os.path.join(dirs['locals'], f'locals_{mesh_id}_k{degree}.mat')
savemat(locals_path, {
    'A1':   A1, 
    'Zinv': Zinv,
    'Zminv':Zminv
}, do_compression=True)

# --- Platform specs (paridad con MATLAB 3.3)
specs = dict(
    method          = method,
    python_version  = platform.python_version(),
    numpy_version   = np.__version__,
    node            = platform.node(),
    system          = platform.system(),
    release         = platform.release(),
    machine         = platform.machine(),
    processor       = platform.processor(),
    user            = getpass.getuser()
)

# --- Build 'run' structure (nested like MATLAB)
run = dict(
    meta   = dict(run_id=run_id, method=method, mesh_id=mesh_id, degree=int(degree), timestamp=timestamp),
    mesh   = dict(Ne=int(nOfElements), h_equiv=float(h_equiv)),
    faces  = dict(nInterior=int(nOfInteriorFaces), nExterior=int(nOfExteriorFaces), nFaceNodes=int(nOfFaceNodes)),
    algebra= dict(Np_per_elem=int(Np), dofs_DG_total=int(dofs_DG_total),
                  dofs_HDG_total=int(dofs_HDG_total), dofs_HDG_unknowns=int(dofs_HDG_unknowns),
                  K_dim=int(dimK), K_nnz=int(nnzK), K_density=float(densK)),
    timings_ms = dict(total=ms_total, **timings),
    timings_pct= dict(**pct),
    accuracy   = dict(L2_u=Error, L2_u_star=ErrorPost),
    platform   = specs
)

# --- Flat 'row' for the accumulator table (same fields as MATLAB)
row = dict(
  method=method, mesh_id=mesh_id, degree=int(degree),
  Ne=int(nOfElements), h_equiv=float(h_equiv), Np=int(Np), faceNodes=int(nOfFaceNodes),
  dofs_DG_total=int(dofs_DG_total), dofs_HDG_total=int(dofs_HDG_total), dofs_HDG_unknowns=int(dofs_HDG_unknowns),
  K_dim=int(dimK), K_nnz=int(nnzK), K_density=float(densK),
  L2_u=float(Error), L2_u_star=float(ErrorPost),
  ms_total=ms_total,
  ms_preprocess=timings['preprocess'], ms_assembly=timings['assembly'],
  ms_dirichlet_reduce=timings['dirichlet_reduce'], ms_solve_global=timings['solve_global'],
  ms_elem_solve=timings['elem_solve'], ms_postprocess=timings['postprocess'], ms_errors=timings['errors'],
  pct_preprocess=pct['preprocess'], pct_assembly=pct['assembly'], pct_dirichlet_reduce=pct['dirichlet_reduce'],
  pct_solve_global=pct['solve_global'], pct_elem_solve=pct['elem_solve'],
  pct_postprocess=pct['postprocess'], pct_errors=pct['errors']
)

# --- Save per-run artifacts
savemat(os.path.join(dirs['runs'], f'run_{run_id}.mat'), {'run': run}, do_compression=True)
savemat(os.path.join(dirs['rows'], f'row_{run_id}.mat'), {'row': row}, do_compression=True)

# --- Rebuild global experiment (MATLAB-friendly): struct arrays
def _collect_struct_array_from_mat(folder, varname):
    """Load all *.mat in folder, pick 'varname' and return list of dicts."""
    out = []
    if not os.path.isdir(folder):
        return out
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith('.mat'):
            try:
                D = loadmat(os.path.join(folder, fn), squeeze_me=True, struct_as_record=False)
                if varname in D:
                    # Convert Matlab structs (if any) to plain dicts; else already dict
                    v = D[varname]
                    out.append(v if isinstance(v, dict) else _matobj_to_dict(v))
            except Exception:
                pass
    return out

def _matobj_to_dict(matobj):
    """Recursively convert scipy.io loadmat MATLAB objects into Python dicts."""
    d = {}
    for name in dir(matobj):
        if name.startswith('__'): 
            continue
        val = getattr(matobj, name)
        if callable(val): 
            continue
        if isinstance(val, np.ndarray) and val.dtype==np.object_ and val.size==0:
            continue
        if isinstance(val, (np.void,)) or str(type(val)).endswith("mat_struct'>"):
            d[name] = _matobj_to_dict(val)
        else:
            d[name] = val
    return d

# Gather all rows & runs
rows_list = _collect_struct_array_from_mat(dirs['rows'], 'row')
runs_list = _collect_struct_array_from_mat(dirs['runs'], 'run')

mesh_ids = sorted(list({r['mesh_id'] for r in rows_list}))
degrees  = sorted(list({int(r['degree']) for r in rows_list}))

experiment = dict(
    meta      = dict(created=timestamp, updated=timestamp, method=method, version=1),
    runs      = np.array(runs_list, dtype=object),   # MATLAB lo verá como struct array
    table     = np.array(rows_list, dtype=object),   # idem; en MATLAB: struct2table(E.table)
    mesh_ids  = np.array(mesh_ids, dtype=object),
    degrees   = np.array(degrees, dtype=float)
)

savemat(exp_path, {'experiment': experiment}, do_compression=True)

print(f'[OK] Appended run "{run_id}" and rebuilt experiment: {exp_path}')
print(f'     Locals for Fig.2: {locals_path}')
