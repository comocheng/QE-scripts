import sys
import os
from shutil import copyfile
from ase.build import bulk, fcc111, add_adsorbate
from ase.io import read, write
from ase import Atoms
from ase.optimize import BFGS, LBFGS, MDMin
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
import ase.io
from ase.io.espresso import read_espresso_out
from ase.io.trajectory import Trajectory
from ase.io.ulm import InvalidULMFileError
from time import time
import yaml
import getopt
from finetuna.atomistic_methods import Relaxation
from finetuna.online_learner.online_learner import OnlineLearner
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from ase.constraints import FixAtoms, FixCartesian

def adsorption(yml_file:str):
    """
    yml_file: str, The input is an yaml input file path with parameters for the calculation
    """
    with open(yml_file) as f:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    
    start = time()
    
    logfile = inputs['log_path']
    
    fmax = inputs['fmax']  # eV/A
    
    adsorbate_file = inputs['adsorbate_path']
    
    slab_file = inputs['slab_path']
    
    if inputs['restart'] == False:
        # read adsorbate trajectory
        traj = Trajectory(adsorbate_file)
        adsorbate = traj[-1]
        adsorbate.set_tags([2] * len(adsorbate))
        
        height = inputs['height']
        
        # # read metal slab trajectory
        adslab = Trajectory(slab_file)[-1]
        
        # freeze bottom layers
        fix_bottom_layers = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag==0)])
        adslab.set_constraint(fix_bottom_layers)
        
        for i in range(inputs['adsorbate_number']):
        # 'ontop', 'bridge', 'fcc', 'hcp'
        # https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#ase.build.add_adsorbate
            if inputs['bond_atom_index'] is None:
                add_adsorbate(adslab, adsorbate, height=height, position=inputs['position'], offset=inputs['offset'][i])
            else:
                add_adsorbate(adslab, adsorbate, height=height, position=inputs['position'], offset=inputs['offset'][i], mol_index=inputs['bond_atom_index'])
    else:
        adslab = Trajectory(inputs['restart_traj_path'])[-1]

    espresso_settings = inputs['calculator']
    
    pseudopotentials = inputs['pseudopotentials']
    
    cores = inputs['gpu_number']
    
    command = f"mpirun -n {cores} /work/westgroup/chao/qe_gpu/7.0-gpu/bin/pw.x -input espresso.pwi -npool 1 -ndiag 1 > espresso.pwo"

    espresso = Espresso(
        command=command,
        pseudopotentials=pseudopotentials,
        tstress=True,
        tprnfor=True,
        kpts=inputs['kpoints'],
        disk_io='none',
        input_data=espresso_settings,
    )
    
    traj_file = inputs['output_traj_path']

    ml_potential = FinetunerCalc(
        checkpoint_path="/work/westgroup/chao/finetuna_models/s2ef/all/gemnet_t_direct_h512_all.pt",  # change this path to your gemnet checkpoint,
        mlp_params={
            "tuner": {
                "unfreeze_blocks": [
                    "out_blocks.3.seq_forces",
                    "out_blocks.3.scale_rbf_F",
                    "out_blocks.3.dense_rbf_F",
                    "out_blocks.3.out_forces",
                    "out_blocks.2.seq_forces",
                    "out_blocks.2.scale_rbf_F",
                    "out_blocks.2.dense_rbf_F",
                    "out_blocks.2.out_forces",
                    "out_blocks.1.seq_forces",
                    "out_blocks.1.scale_rbf_F",
                    "out_blocks.1.dense_rbf_F",
                    "out_blocks.1.out_forces",
                ],
                "num_threads": 8,
            },
            "optim": {
                "batch_size": 1,
                "num_workers": 0,
                "max_epochs": 400,
                "lr_initial": 0.0003,
                "factor": 0.9,
                "eval_every": 1,
                "patience": 3,
                "checkpoint_every": 100000,
                "scheduler_loss": "train",
                "weight_decay": 0,
                "eps": 1e-8,
                "optimizer_params": {
                    "weight_decay": 0,
                    "eps": 1e-8,
                },
            },
            "task": {
                "primary_metric": "loss",
            },
        },
    )

    learner = OnlineLearner(
        learner_params={
            "query_every_n_steps": 10,
            "num_initial_points": 1,
            "fmax_verify_threshold": fmax,
            "uncertain_tol": 2,
            "dyn_uncertain_tol": 2,
            "stat_uncertain_tol": 2,
            "uncertainty_metric": "energy"
        },
        parent_dataset=[],
        ml_potential=ml_potential,
        parent_calc=espresso,
        mongo_db=None,
        optional_config=None,
    )
    
    relaxer = Relaxation(
        initial_geometry=adslab, optimizer=BFGS, fmax=fmax, steps=None, maxstep=0.2
    )
    relaxer.run(
        calc=learner,
        filename=traj_file,
        replay_traj="parent_only",
        max_parent_calls=None,
        check_final=True,
        online_ml_fmax=learner.fmax_verify_threshold,
    )
    
    #calculate the adsorption energy
    # total = Trajectory(traj_file)[-1]
    # metal= Trajectory(slab_file)[-1]
    # ads = Trajectory(adsorbate_file)[-1]
    # adsorption_energy = total.get_potential_energy() - ads.get_potential_energy() - metal.get_potential_energy()
    # adsorption_energy = total.get_potential_energy() - ads.get_potential_energy() - len(metal) / 3 * (-14955.550740)
    
    end = time()
    duration = end - start
    print(f"Done in {duration} seconds!")
        
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
    adsorption(inputfile)
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
