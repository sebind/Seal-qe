import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.build import surface
from ase.spacegroup import get_spacegroup
from collections import Counter
import py3Dmol
from io import StringIO
import seekpath
import spglib
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.electronic_structure.plotter import plot_brillouin_zone

st.set_page_config(layout="wide")
st.title("SeAl Pro — Intelligent Quantum ESPRESSO GUI")

# ==========================================================
# Utility Functions
# ==========================================================

def parse_structure(file):
    if file.name.endswith(".cif"):
        return read(file, format="cif")
    elif file.name.endswith(".xyz"):
        return read(file, format="xyz")
    return None


def build_formula(atoms):
    counts = Counter(atoms.get_chemical_symbols())
    return "".join(f"{k}{counts[k] if counts[k]>1 else ''}" for k in sorted(counts))


def show_structure(atoms):
    xyz = StringIO()
    write(xyz, atoms, format="xyz")
    view = py3Dmol.view(width=500, height=500)
    view.addModel(xyz.getvalue(), "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()
    return view


# ================= Crystal Identification =================

def get_crystal_system(number):
    if 1 <= number <= 2:
        return "Triclinic"
    elif 3 <= number <= 15:
        return "Monoclinic"
    elif 16 <= number <= 74:
        return "Orthorhombic"
    elif 75 <= number <= 142:
        return "Tetragonal"
    elif 143 <= number <= 167:
        return "Trigonal"
    elif 168 <= number <= 194:
        return "Hexagonal"
    elif 195 <= number <= 230:
        return "Cubic"
    return "Unknown"


def analyze_crystal(atoms):
    lattice = atoms.cell.array
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    return {
        "spacegroup": dataset["international"],
        "number": dataset["number"],
        "pointgroup": dataset["pointgroup"],
        "crystal_system": get_crystal_system(dataset["number"])
    }


# ================= Automatic K-Point Suggestion =================

def guess_metallic(atoms):
    metals = ["Fe", "Co", "Ni", "Cr", "Mn", "Cu"]
    return any(m in atoms.get_chemical_symbols() for m in metals)


def suggest_kmesh(atoms, metallic=False):
    recip = atoms.cell.reciprocal()
    lengths = recip.lengths()

    density = 45 if metallic else 30
    kmesh = [max(1, int(l * density / (2*np.pi))) for l in lengths]

    cell_lengths = atoms.cell.lengths()
    if max(cell_lengths) > 20:
        kmesh[2] = 1

    return kmesh


# ================= Pseudopotential Suggestion =================

def suggest_pseudo(atoms, functional, metallic=False, dftu=False):
    if functional == "HSE":
        family = "SSSP_precision"
    elif metallic:
        family = "SSSP_precision"
    else:
        family = "SSSP_efficiency"

    if dftu:
        family += "_relativistic"

    species = sorted(set(atoms.get_chemical_symbols()))
    return {s: f"{family}/{s}.UPF" for s in species}


# ================= Automatic k-Path =================

def generate_kpath(atoms):
    lattice = atoms.cell.array
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    path_data = seekpath.get_path((lattice, positions, numbers))
    return path_data["point_coords"], path_data["path"]


def qe_kpath_block(kpoints, path, n=20):
    lines = ["K_POINTS crystal_b", str(len(path))]
    for segment in path:
        label = segment[0]
        coords = kpoints[label]
        lines.append(
            f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {n} ! {label}"
        )
    return "\n".join(lines)


# ================= Band & DOS Plot =================

def plot_bands(file):
    data = np.loadtxt(file)
    k = data[:,0]
    energies = data[:,1:]
    fig, ax = plt.subplots()
    for i in range(energies.shape[1]):
        ax.plot(k, energies[:,i])
    ax.axhline(0, linestyle="--")
    return fig


def plot_dos(file):
    data = np.loadtxt(file)
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1])
    ax.axvline(0, linestyle="--")
    return fig


# ==========================================================
# Sidebar Controls
# ==========================================================

with st.sidebar:

    st.header("Calculation")

    calculation = st.selectbox("calculation",
                               ["scf", "relax", "vc-relax", "bands"])
    functional = st.selectbox("XC Functional",
                               ["PBE", "PBEsol", "HSE"])

    spin = st.checkbox("Spin Polarized")
    dftu = st.checkbox("DFT+U")
    U = st.number_input("Hubbard U", 0.0, 10.0, 4.0) if dftu else 0.0

    auto_k = st.checkbox("Auto Monkhorst-Pack", True)
    auto_pp = st.checkbox("Auto Pseudopotential", True)

    ecutwfc = st.number_input("ecutwfc", 10.0, 200.0, 40.0)
    ecutrho = st.number_input("ecutrho", 50.0, 1000.0, 320.0)


# ==========================================================
# Main
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload CIF / XYZ", type=["cif", "xyz"])

    if uploaded:
        atoms = parse_structure(uploaded)
        st.success("Structure Loaded")
        st.write("Formula:", build_formula(atoms))

        analysis = analyze_crystal(atoms)
        st.write("Crystal System:", analysis["crystal_system"])
        st.write("Space Group:",
                 f"{analysis['spacegroup']} ({analysis['number']})")
        st.write("Point Group:", analysis["pointgroup"])

        view = show_structure(atoms)
        st.components.v1.html(view._make_html(), height=500)


with col2:
    if uploaded:

        metallic = guess_metallic(atoms)

        if auto_k:
            kmesh = suggest_kmesh(atoms, metallic)
            st.info(f"Suggested k-mesh: {kmesh}")

        if auto_pp:
            pp_dict = suggest_pseudo(atoms, functional, metallic, dftu)
            st.write("Suggested Pseudopotentials:")
            for k,v in pp_dict.items():
                st.write(f"{k} → {v}")

        st.subheader("Automatic Band k-Path")
        kpoints, path = generate_kpath(atoms)
        qe_path = qe_kpath_block(kpoints, path)
        st.code(qe_path)

        st.subheader("Band Plot")
        band_file = st.file_uploader("Upload bands.dat", type=["dat"])
        if band_file:
            with open("bands.dat","wb") as f:
                f.write(band_file.read())
            st.pyplot(plot_bands("bands.dat"))

        st.subheader("DOS Plot")
        dos_file = st.file_uploader("Upload dos.dat", type=["dat"])
        if dos_file:
            with open("dos.dat","wb") as f:
                f.write(dos_file.read())
            st.pyplot(plot_dos("dos.dat"))
