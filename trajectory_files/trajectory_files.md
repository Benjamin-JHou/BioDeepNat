## 🧬 Molecular Dynamics Simulation Protocol

### 🖥️ Simulation Setup 
- **Software**: AMBER suite 🧪   **Simulation Duration**: 100 ns per protein ⏱️

### 🌐 System Configuration 
- **Boundary Conditions**: Periodic (PBC) 🔄  **Simulation Box**: Cubic 📦  **Solvent Model**: Explicit, TIP3P water model 💧
- **Ionic Environment**: Na+ and Cl- ions ⚡  Salt concentration: 0.15 M (physiological ionic strength) 🧂 Purpose: System neutrality ⚖️

### 🔬 Force Fields 
- **Proteins and Nucleic Acids**: AMBER99SB-ILDN 🧬   **Small-molecule Ligands**: General AMBER Force Field (GAFF) 🔗

### 🔬 Simulation Protocol 
1. **Energy Minimization**: Initial configurations optimized 📉 2. **Temperature Control**: - Algorithm: V-rescale 🌡️  Temperature: 310 K 🔥

### 📊 Trajectory Data 
- **Recording Frequency**: Every 10 ps ⏲️
- [ADCY8_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/ADCY8_trajectory.dcd)
- [CBX2_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/CBX2_trajectory.dcd)
- [DDC_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/DDC_trajectory.dcd)
- [DNAH11_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/DNAH11_trajectory.dcd)
- [DOCK1_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/DOCK1_trajectory.dcd)
- [KIFBP_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/KIFBP_trajectory.dcd)
- [LNX1_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/LNX1_trajectory.dcd)
- [PPP2R5C_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/PPP2R5C_trajectory.dcd)
- [RPTOR_trajectory.dcd](https://github.com/Benjamin-JHou/BioDeepNat/blob/main/trajectory_files/RPTOR_trajectory.dcd)


