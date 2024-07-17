## Molecular Dynamics Simulation Protocol

### Simulation Setup
- **Software**: AMBER suite
- **Hardware**: High-performance computing cluster
- **Simulation Duration**: 100 ns per protein
- **Number of Proteins**: 9

### System Configuration
- **Boundary Conditions**: Periodic (PBC)
- **Simulation Box**: Cubic
- **Solvent Model**: Explicit, TIP3P water model
- **Ionic Environment**: 
  - Na+ and Cl- ions
  - Salt concentration: 0.15 M (physiological ionic strength)
  - Purpose: System neutrality

### Force Fields
- **Proteins and Nucleic Acids**: AMBER99SB-ILDN
- **Small-molecule Ligands**: General AMBER Force Field (GAFF)

### Simulation Protocol
1. **Energy Minimization**: Initial configurations optimized
2. **Temperature Control**:
   - Algorithm: V-rescale
   - Temperature: 310 K (37°C, human physiological conditions)

### Trajectory Data
- **Recording Frequency**: Every 10 ps
- **Data Usage**: Root-mean-square deviation (RMSD) calculations
