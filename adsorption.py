"""
Author: Chao Xu
Date: 2023-12-04
Email: xu.chao@northeastern.edu
Description: This script calculates adsorption energy of a surface adsorption system
"""

import sys
from ase.build import add_adsorbate
from ase.optimize import BFGS, MDMin
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory
from time import time
import yaml
import getopt

class AdsorptionCalc:
    """
    This is a class for calculating adsorption energy in a surface adsorption system
    Adsorption energy is defined as E_ads = E_system - E_slab - E_gas
    where E_system, E_slab, and E_gas are the energy of the system, metal slab, and
    the gas adsorbate
    """
    def __init__(self, input_file:str):
        """
        input_file: str, The input is an yaml input file path with parameters for the DFT calculation
        """
        self.yml_path = input_file
        self.restart = False
        self.bond_atom_index = None
        self.restart_traj_path = None
        with open(self.yml_path) as f:
            self.inputs = yaml.load(f, Loader=yaml.FullLoader)
        self.logfile = self.inputs['log_path']
        self.fmax = self.inputs['fmax']
        self.adsorbate_file = self.inputs['adsorbate_path']
        self.restart = self.inputs['restart']
        self.slab_file = self.inputs['slab_path']
        self.height = self.inputs['height']
        self.bond_atom_id = self.inputs['bond_atom_index']
        self.site = self.inputs['position']
        self.offset = self.inputs['offset']
        self.restart_traj_path = self.inputs['restart_traj_path']
        self.calculator = self.inputs['calculator']
        self.pseudo = self.inputs['pseudopotentials']
        self.gpu_num = self.inputs['gpu_number']
        self.kpts = self.inputs['kpoints']
        self.output = self.inputs['output_traj_path']

    def adsorption(self):
        start = time()

        if self.restart == False:
            adsorbate = Trajectory(self.adsorbate_file)[-1]
            adsorbate.set_tags([2] * len(adsorbate))
            adslab = Trajectory(self.slab_file)[-1]

            # freeze bottom layers to simulate the surface
            fix_bottom_layers = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag==0)])
            adslab.set_constraint(fix_bottom_layers)
            
            # 'ontop', 'bridge', 'fcc', 'hcp'
            # https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#ase.build.add_adsorbate
            if self.bond_atom_id is None:
                add_adsorbate(adslab, adsorbate, height=self.height, position=self.site, offset=self.offset)
            else:
                add_adsorbate(adslab, adsorbate, height=self.height, position=self.site, offset=self.offset, mol_index=self.bond_atom_id)

        else:
            adslab = Trajectory(self.restart_traj_path)[-1]
        
        cores = self.gpu_num
        
        # command to launch Quantum Espresso on multiple GPUs
        command = f"mpirun -n {cores} /work/westgroup/chao/qe_gpu/7.0-gpu/bin/pw.x -input espresso.pwi -npool 1 -ndiag 1 > espresso.pwo"

        espresso = Espresso(
            command=command,
            pseudopotentials=self.pseudo,
            tstress=True,
            tprnfor=True,
            kpts=self.kpts,
            disk_io='none',
            input_data=self.calculator,
        )
        
        adslab.calc = espresso
        opt = MDMin(adslab, logfile=self.logfile, trajectory=self.output)
        opt.run(fmax=0.5)

        opt = BFGS(adslab, logfile=self.logfile, trajectory=self.output)
        opt.run(fmax=self.fmax)
        
        #calculate the adsorption energy
        total = Trajectory(self.output)[-1]
        metal= Trajectory(self.slab_file)[-1]
        adsorption_energy = total.get_potential_energy() - adsorbate.get_potential_energy() - metal.get_potential_energy()
        
        end = time()
        duration = end - start
        
        with open(self.logfile, 'a') as f:
            f.write(f'Adsorption energy: {adsorption_energy}')
            f.write(f'Completed in {duration} seconds\n')
        
def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('adsorption.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('command should be adsorption.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    AdsorptionCalc(inputfile).adsorption()
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
