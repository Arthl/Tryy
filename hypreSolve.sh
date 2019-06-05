# Go to hypreHOME
HYPREHOME=$1
cd $HYPREHOME

NPROC=$2

# Call hypreSolver
time mpirun -np $NPROC hypreMatlabSolve
