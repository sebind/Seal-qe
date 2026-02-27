import streamlit as st
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.build import surface
from ase.spacegroup import get_spacegroup
from collections import Counter
import py3Dmol
import seekpath
from io import StringIO

st.set_page_config(layout="wide")
st.title("SeAl Pro — Quantum ESPRESSO Workflow GUI")

# ======================================================
# ----------------- Utility Functions ------------------
# ======================================================

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


def auto_kmesh(atoms, density=0.2):
    lengths = atoms.cell.lengths()
    return [max(1, int(1/(l*density))) for l in lengths]


def generate_qe_input(atoms, settings):
    nat = len(atoms)
    species = sorted(set(atoms.get_chemical_symbols()))

    qe = []
    qe.append("&CONTROL")
    qe.append(f"  calculation='{settings['calculation']}'")
    qe.append(f"  prefix='{settings['prefix']}'")
    qe.append("  pseudo_dir='./pseudo/'")
    qe.append("  outdir='./tmp/'")
    qe.append("/")

    qe.append("&SYSTEM")
    qe.append("  ibrav=0")
    qe.append(f"  nat={nat}")
    qe.append(f"  ntyp={len(species)}")
    qe.append(f"  ecutwfc={settings['ecutwfc']}")
    qe.append(f"  ecutrho={settings['ecutrho']}")
    qe.append(f"  input_dft='{settings['functional']}'")

    if settings["spin"]:
        qe.append("  nspin=2")
        qe.append("  starting_magnetization(1)=0.5")

    if settings["dftu"]:
        qe.append("  lda_plus_u=.true.")
        for i in range(len(species)):
            qe.append(f"  Hubbard_U({i+1})={settings['U']}")

    if settings["functional"] == "HSE":
        qe.append("  exx_fraction=0.25")
        qe.append("  screening_parameter=0.11")

    qe.append("/")

    qe.append("&ELECTRONS")
    qe.append("  conv_thr=1.0d-6")
    qe.append("/")

    qe.append("ATOMIC_SPECIES")
    for s in species:
        qe.append(f"{s} 1.0 {s}.UPF")

    qe.append("CELL_PARAMETERS angstrom")
    for v in atoms.cell.array:
        qe.append(f"{v[0]} {v[1]} {v[2]}")

    qe.append("ATOMIC_POSITIONS angstrom")
    for s, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
        qe.append(f"{s} {pos[0]} {pos[1]} {pos[2]}")

    if settings["calculation"] == "bands":
        qe.append("K_POINTS automatic")
        qe.append("4 4 4 1 1 1")
    else:
        k = settings["kmesh"]
        qe.append("K_POINTS automatic")
        qe.append(f"{k[0]} {k[1]} {k[2]} 1 1 1")

    return "\n".join(qe)


def plot_bands(file):
    data = np.loadtxt(file)
    k = data[:,0]
    energies = data[:,1:]

    fig, ax = plt.subplots()
    for i in range(energies.shape[1]):
        ax.plot(k, energies[:,i])
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    return fig


def plot_dos(file):
    data = np.loadtxt(file)
    energy = data[:,0]
    dos = data[:,1]

    fig, ax = plt.subplots()
    ax.plot(energy, dos)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    return fig


def generate_phonon_input(prefix):
    return f"""
&INPUTPH
 prefix='{prefix}'
 outdir='./tmp/'
 fildyn='dynmat'
/
0.0 0.0 0.0
"""


# ======================================================
# ---------------------- Sidebar -----------------------
# ======================================================

with st.sidebar:
    st.header("Calculation Settings")

    calculation = st.selectbox("Calculation", ["scf", "relax", "vc-relax", "bands"])
    functional = st.selectbox("XC Functional", ["PBE", "PBEsol", "HSE"])
    ecutwfc = st.number_input("ecutwfc", 10.0, 200.0, 40.0)
    ecutrho = st.number_input("ecutrho", 50.0, 1000.0, 320.0)
    spin = st.checkbox("Spin Polarized")
    dftu = st.checkbox("DFT+U")
    U = st.number_input("Hubbard U (eV)", 0.0, 10.0, 4.0) if dftu else 0.0
    auto_k = st.checkbox("Auto k-mesh", value=True)

# ======================================================
# ---------------------- Main --------------------------
# ======================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload CIF or XYZ")
    uploaded = st.file_uploader("Upload structure", type=["cif", "xyz"])

    if uploaded:
        atoms = parse_structure(uploaded)
        st.success("Structure Loaded")
        st.write("Formula:", build_formula(atoms))

        try:
            sg = get_spacegroup(atoms)
            st.write("Space Group:", sg)
        except:
            st.warning("Symmetry detection failed")

        view = show_structure(atoms)
        st.components.v1.html(view._make_html(), height=500)

        if st.checkbox("Generate Slab"):
            hkl = tuple(map(int, st.text_input("Miller indices", "1 0 0").split()))
            layers = st.number_input("Layers", 1, 10, 3)
            atoms = surface(atoms, hkl, layers)
            st.success("Slab generated")


with col2:
    if uploaded:
        prefix = st.text_input("Prefix", build_formula(atoms))

        if auto_k:
            kmesh = auto_kmesh(atoms)
        else:
            kmesh = (
                st.number_input("kx", 1, 20, 4),
                st.number_input("ky", 1, 20, 4),
                st.number_input("kz", 1, 20, 4),
            )

        settings = {
            "calculation": calculation,
            "prefix": prefix,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "functional": functional,
            "spin": spin,
            "dftu": dftu,
            "U": U,
            "kmesh": kmesh
        }

        qe_input = generate_qe_input(atoms, settings)

        st.subheader("QE Input")
        st.code(qe_input)
        st.download_button("Download QE Input", qe_input, file_name="qe.in")

        st.subheader("Phonon Input")
        ph_input = generate_phonon_input(prefix)
        st.code(ph_input)
        st.download_button("Download ph.in", ph_input, file_name="ph.in")

        st.subheader("Band Plot")
        band_file = st.file_uploader("Upload bands.dat.gnu", type=["dat"])
        if band_file:
            with open("bands.dat", "wb") as f:
                f.write(band_file.read())
            fig = plot_bands("bands.dat")
            st.pyplot(fig)

        st.subheader("DOS Plot")
        dos_file = st.file_uploader("Upload dos.dat", type=["dat"])
        if dos_file:
            with open("dos.dat", "wb") as f:
                f.write(dos_file.read())
            fig = plot_dos("dos.dat")
            st.pyplot(fig)
