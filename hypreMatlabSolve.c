/*
 Read a linear system assembled in Matlab
 Solve the linear system using hypre
 Interface: Linear-Algebraic (IJ)
 Available solvers: 0  - AMG-PCG
                    1  - ParaSails-PCG
                    20 - AMG-GMRES
		    21 - ParaSails-GMRES
		    30 - AMG-FlexGMRES
		    31 - ParaSails-FlexGMRES

 Copyright: M. Giacomini (2017) / S. Zlotnik / A. Lustman (2019)
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

#include "vis.c"

// Contains the header to read binary files (not implemented yet)
#include "binary.h"



int main (int argc, char *argv[])
{
    int time_index;
    time_index = hypre_InitializeTiming("Setup matrix and vectors");
    hypre_BeginTiming(time_index);
    HYPRE_Int i;
    HYPRE_Int iErr = 0;

    int num_iterations;
    double final_res_norm;


    HYPRE_Int myid, num_procs, dummy;
    HYPRE_Int solver_id;

    HYPRE_Int first_local_row, last_local_row;
    HYPRE_Int first_local_col, last_local_col, local_num_cols;

    HYPRE_Int first_G_row, last_G_row;
    HYPRE_Int first_G_col, last_G_col, local_G_cols;
    
    HYPRE_Real *values;

    HYPRE_IJMatrix ij_K;
    HYPRE_ParCSRMatrix parcsr_K;
    HYPRE_IJMatrix ij_G;
    HYPRE_ParCSRMatrix parcsr_G;
    HYPRE_IJVector ij_f;
    HYPRE_ParVector par_f;
    HYPRE_IJVector ij_h;
    HYPRE_ParVector par_h;
    HYPRE_IJVector ij_vel;
    HYPRE_ParVector par_vel;
    HYPRE_IJVector ij_pres;
    HYPRE_ParVector par_pres;

    HYPRE_IJVector ij_D_1;
    HYPRE_ParVector par_D_1;
    
    HYPRE_IJVector ij_D_2;
    HYPRE_ParVector par_D_2;
    
    HYPRE_IJVector ij_residual;
    HYPRE_ParVector par_residual;
    
    HYPRE_IJVector ij_TransFirst;
    HYPRE_ParVector par_TransFirst;
    
    HYPRE_IJVector ij_TransSecond;
    HYPRE_ParVector par_TransSecond;

    char saveMatsAs;
    
    FILE *fset;
    
    void *object;
    
    HYPRE_Solver solver, precond;
    

    /* Read dimension of the system and chosen solver from file */
    fset = fopen("solverPar.dat","r"); // read mode
    if( fset == NULL )
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
    hypre_fscanf(fset, "%d\n", &solver_id);
    hypre_fscanf(fset, "%d\n", &num_procs);
    hypre_fscanf(fset, "%e\n", &dummy);
    hypre_fscanf(fset, "%c\n", &saveMatsAs);
    fclose(fset);

    
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
    /* Read the matrix previously assembled
      <filename>  = IJ.A.out to read in what has been printed out (processor numbers are omitted). */
    if (saveMatsAs=='a')
    {
      iErr = HYPRE_IJMatrixRead( "matrixK", MPI_COMM_WORLD, HYPRE_PARCSR, &ij_K );
    }
    else
    {
      iErr = HYPRE_IJMatrixRead_binary( "matrixK", MPI_COMM_WORLD, HYPRE_PARCSR, &ij_K );
    }


    if (iErr) 
    {
        hypre_printf("ERROR: Problem reading in the system matrix!\n");
        exit(1);
    }
    /* Get dimension info */
    iErr = HYPRE_IJMatrixGetLocalRange( ij_K,
                                        &first_local_row, &last_local_row ,
                                        &first_local_col, &last_local_col );
    
    local_num_cols = last_local_col - first_local_col + 1;
    /* Get the parcsr matrix object to use */
    iErr += HYPRE_IJMatrixGetObject( ij_K, &object);
    parcsr_K = (HYPRE_ParCSRMatrix) object;

    /* Read the matrix previously assembled
      <filename>  = IJ.A.out to read in what has been printed out (processor numbers are omitted). */
    if (saveMatsAs=='a')
    {
      iErr = HYPRE_IJMatrixRead( "matrixG", MPI_COMM_WORLD, HYPRE_PARCSR, &ij_G );
    }
    else
    {
      iErr = HYPRE_IJMatrixRead_binary( "matrixG", MPI_COMM_WORLD, HYPRE_PARCSR, &ij_G );
    }


    if (iErr) 
    {
        hypre_printf("ERROR: Problem reading in the system matrix!\n");
        exit(1);
    }
    /* Get dimension info */
    iErr = HYPRE_IJMatrixGetLocalRange( ij_G,
                                        &first_G_row, &last_G_row ,
                                        &first_G_col, &last_G_col );
    
    local_G_cols = last_G_col - first_G_col + 1;
    /* Get the parcsr matrix object to use */
    iErr += HYPRE_IJMatrixGetObject( ij_G, &object);
    parcsr_G = (HYPRE_ParCSRMatrix) object;
    
    
    /*  Read the RHS previously assembled */
    iErr = HYPRE_ParVectorRead(MPI_COMM_WORLD, "vectorF.0", &par_f);
    if (iErr)
    {
      hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
      exit(1);
    }
    ij_f = NULL;


    /*  Read the RHS previously assembled */
    iErr = HYPRE_ParVectorRead(MPI_COMM_WORLD, "vectorH.0", &par_h);
    if (iErr)
    {
        hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
        exit(1);
    }
    ij_h = NULL;


    /* Create the initial solution and set it to zero */
    /* The Velocity Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_vel);
    HYPRE_IJVectorSetObjectType(ij_vel, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_vel);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_vel, local_num_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_vel, &object );
    par_vel = (HYPRE_ParVector) object;

    /* The Pressure Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_G_col, last_G_col, &ij_pres);
    HYPRE_IJVectorSetObjectType(ij_pres, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_pres);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_G_cols);
    for (i = 0; i < local_G_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_pres, local_G_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_pres, &object );
    par_pres = (HYPRE_ParVector) object;

    /* The d_1 Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_D_1);
    HYPRE_IJVectorSetObjectType(ij_D_1, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_D_1);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_D_1, local_num_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_D_1, &object );
    par_D_1 = (HYPRE_ParVector) object;

    /* The d_2 Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_G_col, last_G_col, &ij_D_2);
    HYPRE_IJVectorSetObjectType(ij_D_2, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_D_2);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_G_cols);
    for (i = 0; i < local_G_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_D_2, local_G_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_D_2, &object );
    par_D_2 = (HYPRE_ParVector) object;

    /* The residual Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_G_col, last_G_col, &ij_residual);
    HYPRE_IJVectorSetObjectType(ij_residual, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_residual);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_G_cols);
    for (i = 0; i < local_G_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_residual, local_G_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_residual, &object );
    par_residual = (HYPRE_ParVector) object;


    // Usage of temporary arrays and values, never declared in the Uzawa algorithm
    /* Temporary_V Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_TransFirst);
    HYPRE_IJVectorSetObjectType(ij_TransFirst, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_TransFirst);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
    for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_TransFirst, local_num_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_TransFirst, &object );
    par_TransFirst = (HYPRE_ParVector) object;

    /* Temporary_P Vector */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_G_col, last_G_col, &ij_TransSecond);
    HYPRE_IJVectorSetObjectType(ij_TransSecond, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_TransSecond);
    /* Initialize the guess vector */
    values = hypre_CTAlloc(HYPRE_Real, local_G_cols);
    for (i = 0; i < local_G_cols; i++)
      values[i] = 0.;
    HYPRE_IJVectorSetValues(ij_TransSecond, local_G_cols, NULL, values);
    hypre_TFree(values);
    /* Get the parcsr vector object to use */
    iErr = HYPRE_IJVectorGetObject( ij_TransSecond, &object );
    par_TransSecond = (HYPRE_ParVector) object;

    // Create the assigned variables for the resolution
    HYPRE_Real *init_res, *alpha, *beta, *gamma, *gamma_old, *dConv;
    HYPRE_Real *abba;

    init_res = hypre_CTAlloc(HYPRE_Real, 1);
    alpha = hypre_CTAlloc(HYPRE_Real, 1);
    beta = hypre_CTAlloc(HYPRE_Real, 1);
    gamma = hypre_CTAlloc(HYPRE_Real, 1);
    gamma_old = hypre_CTAlloc(HYPRE_Real, 1);
    dConv = hypre_CTAlloc(HYPRE_Real, 1);

    abba = hypre_CTAlloc(HYPRE_Real, 1);

    hypre_EndTiming(time_index);
    hypre_PrintTiming("Time to initiate matrix and vector", MPI_COMM_WORLD);
    hypre_FinalizeTiming(time_index);
    hypre_ClearTiming();

    /* Variables used for the loop */
    int it, max_it;
    it = 1;
    max_it = 5000;
    double tol_r, tol_d;
    tol_r = 1e-2;
    tol_d = 1e-2;

    /* Before the loop*/
    if (solver_id < 10)
      {

	time_index = hypre_InitializeTiming("PCG - Setup Precond");
	hypre_BeginTiming(time_index);

        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, max_it); /* max iterations */
        HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
        HYPRE_PCGSetPrintLevel(solver, 0); /* prints out the iteration info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */


	if (solver_id == 0) //AMG Precond
	{
		HYPRE_BoomerAMGCreate(&precond);
		HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
		HYPRE_BoomerAMGSetCoarsenType(precond, 6);
		HYPRE_BoomerAMGSetOldDefault(precond); 
		HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
		HYPRE_BoomerAMGSetNumSweeps(precond, 1);
		HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
		HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

		// Efficiency factors
		HYPRE_BoomerAMGSetInterpType(precond, 7);
		HYPRE_BoomerAMGSetTruncFactor(precond, 4);
		HYPRE_BoomerAMGSetAggNumLevels(precond, 2);

		HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
				(HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
	}
	else if (solver_id == 1) //ParaSails Precond
	{
		int      sai_max_levels = 1;
		double   sai_threshold = 0.1;
		double   sai_filter = 0.05;
		int      sai_sym = 1;

		HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

		/* Set some parameters (See Reference Manual for more parameters) */
		HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
		HYPRE_ParaSailsSetFilter(precond, sai_filter);
		HYPRE_ParaSailsSetSym(precond, sai_sym);
		HYPRE_ParaSailsSetLogging(precond, 3);

		/* Set the PCG preconditioner */
		HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
				(HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);
	}

	HYPRE_ParCSRPCGSetup(solver, parcsr_K, par_f, par_vel);

	hypre_EndTiming(time_index);
        hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();
        time_index = hypre_InitializeTiming("PCG - Solve");
        hypre_BeginTiming(time_index);

        HYPRE_ParCSRPCGSolve(solver, parcsr_K, par_f, par_vel);

        hypre_EndTiming(time_index);
        hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        /* Destroy solver */
        //HYPRE_ParCSRPCGDestroy(solver);

        //if (solver_id == 0)
        //{
	//   HYPRE_BoomerAMGDestroy(precond);
        //}
        //else
        //{
        //   HYPRE_ParaSailsDestroy(precond);
        //}
      }
    else if ((solver_id > 12) && (solver_id < 22))
    {
	time_index = hypre_InitializeTiming("GMRES - Precond Setup");
	hypre_BeginTiming(time_index);

	/* Create solver */
	HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

	/* Set some parameters (See Reference Manual for more parameters) */
	HYPRE_GMRESSetMaxIter(solver, max_it); /* max iterations */
	HYPRE_GMRESSetKDim(solver, 30);
	HYPRE_GMRESSetTol(solver, 1e-7); /* conv. tolerance */
	HYPRE_GMRESSetPrintLevel(solver, 0); /* print solve info */
	HYPRE_GMRESSetLogging(solver, 1); /* needed to get run info later */

	if (solver_id == 20) // AMG Precond
	{
		HYPRE_BoomerAMGCreate(&precond);
		HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
		HYPRE_BoomerAMGSetCoarsenType(precond, 6);
		HYPRE_BoomerAMGSetOldDefault(precond); 
		HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
		HYPRE_BoomerAMGSetNumSweeps(precond, 1);
		HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
		HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

		// Efficiency factors
		HYPRE_BoomerAMGSetInterpType(precond, 7);
		HYPRE_BoomerAMGSetTruncFactor(precond, 4);
		HYPRE_BoomerAMGSetAggNumLevels(precond, 2);

		/* Set the PCG preconditioner */
		HYPRE_GMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
	}
	else if (solver_id == 21) // ParaSails
	{
		int      sai_max_levels = 1;
		double   sai_threshold = 0.1;
		double   sai_filter = 0.01;
		int      sai_sym = 1;

		HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

		/* Set some parameters (See Reference Manual for more parameters) */
		HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
		HYPRE_ParaSailsSetFilter(precond, sai_filter);
		HYPRE_ParaSailsSetSym(precond, sai_sym);
		HYPRE_ParaSailsSetLogging(precond, 3);

		/* Set the PCG preconditioner */
		HYPRE_GMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
				  (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);
	}

	HYPRE_ParCSRGMRESSetup(solver, parcsr_K, par_f, par_vel);

	hypre_EndTiming(time_index);
	hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
	hypre_FinalizeTiming(time_index);
	hypre_ClearTiming();
	time_index = hypre_InitializeTiming("GMRES - Precond Solve");
	hypre_BeginTiming(time_index);

	/* Now setup and solve! */
	HYPRE_ParCSRGMRESSolve(solver, parcsr_K, par_f, par_vel);

	hypre_EndTiming(time_index);
	hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
	hypre_FinalizeTiming(time_index);
	hypre_ClearTiming();


	/* Run info - needed logging turned on */
	HYPRE_GMRESGetNumIterations(solver, &num_iterations);
	HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

	/* Destory solver and preconditioner */
	//HYPRE_ParCSRGMRESDestroy(solver);

	//if (solver_id == 20)
	//{
	//  HYPRE_BoomerAMGDestroy(precond);
	//}
	//else
	//{
	//   HYPRE_ParaSailsDestroy(precond);
	//}

      }
       /* Flexible GMRES with AMG Preconditioner */
    else if ((solver_id > 22) && (solver_id < 32))
      {
	time_index = hypre_InitializeTiming("FlexGMRES - Precond Setup");
	hypre_BeginTiming(time_index);

        int    restart = 30;

        /* Create solver */
        HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_FlexGMRESSetKDim(solver, restart);
        HYPRE_FlexGMRESSetMaxIter(solver, max_it); /* max iterations */
        HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
        HYPRE_FlexGMRESSetPrintLevel(solver, 0); /* print solve info */
        HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */

	if (solver_id == 30) // AMG Precond
	{
		HYPRE_BoomerAMGCreate(&precond);
		HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
		HYPRE_BoomerAMGSetCoarsenType(precond, 6);
		HYPRE_BoomerAMGSetOldDefault(precond);
		HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
		HYPRE_BoomerAMGSetNumSweeps(precond, 1);
		HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
		HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

		HYPRE_BoomerAMGSetInterpType(precond, 7);
		HYPRE_BoomerAMGSetTruncFactor(precond, 4);
		HYPRE_BoomerAMGSetAggNumLevels(precond, 2);

        	HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
	}
	else if (solver_id == 31) //ParaSails
	{
		int      sai_max_levels = 1;
		double   sai_threshold = 0.1;
		double   sai_filter = 0.05;
		int      sai_sym = 1;

		HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

		/* Set some parameters (See Reference Manual for more parameters) */
		HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
		HYPRE_ParaSailsSetFilter(precond, sai_filter);
		HYPRE_ParaSailsSetSym(precond, sai_sym);
		HYPRE_ParaSailsSetLogging(precond, 3);

		/* Set the PCG preconditioner */
		HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
				(HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);
	}

        HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_K, par_f, par_vel);

        hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();
        time_index = hypre_InitializeTiming("FlexGMRES - Solve");
        hypre_BeginTiming(time_index);

        HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_K, par_f, par_vel);

        hypre_EndTiming(time_index);
        hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        /* Run info - needed logging turned on */
        HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
        HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

        /* Destory solver and preconditioner */
        //HYPRE_ParCSRFlexGMRESDestroy(solver);
	//if (solver_id == 30)
	//{
	//  HYPRE_BoomerAMGDestroy(precond);
	//}
	//else
	//{
	//   HYPRE_ParaSailsDestroy(precond);
	//}
      }

    else
      {
        if (myid ==0) printf("Invalid solver id specified.\n");
      }

	if (num_iterations == max_it)
	{
	    printf("\n");
	    printf("Maximum number of iterations reached, convergence not acquired\n");
	    return(-1);
	}
	if (myid == 0)
	{
	 printf("\n");
	 printf("Iterations = %d\n", num_iterations);
	 printf("Final Relative Residual Norm = %e\n", final_res_norm);
	 printf("\n");
	}

    // bi_prod = <f,f> + <h,h>
    //HYPRE_ParVectorInnerProd(par_f, par_f, bi_prod);//+ inner(h,h) but equal to 0 !!!!!!!
    //HYPRE_ParVectorInnerProd(par_h, par_h, abba);
    //bi_prod[0] += abba[0];

    // res = GT*u
    HYPRE_ParCSRMatrixMatvecT( 1.0 , parcsr_G , par_vel , 0.0 , par_residual );

    HYPRE_ParVectorCopy( par_residual, par_D_2 );

    // < res, res >
    HYPRE_ParVectorInnerProd(par_residual, par_residual, gamma);
    init_res[0] = gamma[0];
   
    if (myid == 0)
    {
    printf("Initial Norm of the residual is %f\n", pow(gamma[0],0.5));
    }

  // Loop Starts Here
  for (it = 1; it < max_it; it++) 
    { 
      
    // transFirst = G*d_2 
    HYPRE_ParCSRMatrixMatvec( 1.0 , parcsr_G , par_D_2 , 0.0 , par_TransFirst );


    if (solver_id < 10)
      {
	// INSERT HERE

	HYPRE_ParCSRPCGSetup(solver, parcsr_K, par_TransFirst, par_D_1);
        HYPRE_ParCSRPCGSolve(solver, parcsr_K, par_TransFirst, par_D_1);

        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        /* Destroy solver */
        //HYPRE_ParCSRPCGDestroy(solver);

        //if (solver_id == 0)
        //{
	//   HYPRE_BoomerAMGDestroy(precond);
        //}
        //else
        //{
        //   HYPRE_ParaSailsDestroy(precond);
        //}
      } 
    else if ((solver_id > 12) && (solver_id < 22))
    {

	HYPRE_ParCSRGMRESSetup(solver, parcsr_K, par_TransFirst, par_D_1);
	HYPRE_ParCSRGMRESSolve(solver, parcsr_K, par_TransFirst, par_D_1);

	/* Run info - needed logging turned on */
	HYPRE_GMRESGetNumIterations(solver, &num_iterations);
	HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

	/* Destory solver and preconditioner */
	//HYPRE_ParCSRGMRESDestroy(solver);
	//if (solver_id == 20)
	//{
	//  HYPRE_BoomerAMGDestroy(precond);
	//}
	//else
	//{
	//   HYPRE_ParaSailsDestroy(precond);
	//}
      }
       /* Flexible GMRES with AMG Preconditioner */
    else if ((solver_id > 22) && (solver_id < 32))
      {

        HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_K, par_TransFirst, par_D_1);
        HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_K, par_TransFirst, par_D_1);

        /* Run info - needed logging turned on */
        HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
        HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

	/* Destory solver and preconditioner */
        //HYPRE_ParCSRFlexGMRESDestroy(solver);
        //if (solver_id == 30)
        //{
	//   HYPRE_BoomerAMGDestroy(precond);
        //}
        //else
        //{
        //   HYPRE_ParaSailsDestroy(precond);
        //}
      }

    else
      {
        if (myid ==0) printf("Invalid solver id specified.\n");
      }

    if (num_iterations == max_it)
      {
	printf("\n");
	printf("Maximum number of iterations reached, convergence not acquired\n");
	return(-1);
      }
    if (myid == 0)
      {
	printf("\n");
	printf("Iterative solver Iterations = %d, ", num_iterations);
	printf("Final Relative Residual Norm = %e\n", final_res_norm);
	printf("\n");
      }

    // transFirst = Gmatrix * d_2vector;
    HYPRE_ParCSRMatrixMatvec( 1.0 , parcsr_G , par_D_2 , 0.0 , par_TransFirst );

    // <GT*d_2, d_1>
    HYPRE_ParVectorInnerProd(par_TransFirst, par_D_1, abba);
    alpha[0] = gamma[0] / abba[0];
    gamma_old[0] = gamma[0];

    // pSol = pSol + alpha * d_2vector;
    hypre_ParVectorAxpy( alpha[0], par_D_2, par_pres);

    //uSol = uSol - alpha * d_1vector;
    hypre_ParVectorAxpy( -alpha[0], par_D_1, par_vel);

    // Used for convergence over d_1
    HYPRE_ParVectorCopy( par_D_1, par_TransFirst );
    HYPRE_ParVectorScale( alpha[0], par_TransFirst );
    HYPRE_ParVectorInnerProd(par_TransFirst, par_TransFirst, beta);
    HYPRE_ParVectorInnerProd(par_vel,par_vel, abba);
    dConv[0] = sqrt(beta[0]/abba[0]);

    // transSecond = GT * d_1vector;
    HYPRE_ParCSRMatrixMatvecT( 1.0 , parcsr_G , par_D_1 , 0.0 , par_TransSecond );

    // residual = residual - alpha * transSecond;
    hypre_ParVectorAxpy( -alpha[0], par_TransSecond, par_residual);

    HYPRE_ParVectorInnerProd(par_residual, par_residual, abba);
    abba[0] = pow(abba[0],0.5);

    if (myid == 0)
      {  
        printf("Uzawa it %d\n: Relative residual is %.5e, relative direction is %.5e\n",it, abba[0]/sqrt(init_res[0]), dConv[0]);
      }


    if (((abba[0]/sqrt(init_res[0]))<tol_r) && (dConv[0] < tol_d))
      {
        HYPRE_ParVectorPrint( par_vel , "solution_Vel.0" );
        HYPRE_ParVectorPrint( par_pres , "solution_Pres.0" );
        HYPRE_ParVectorInnerProd(par_vel, par_vel, alpha);
        HYPRE_ParVectorInnerProd(par_pres, par_pres, beta);

	// Precond - Solver Destroyer
	if (solver_id < 10)
		{
		HYPRE_ParCSRPCGDestroy(solver);
		}
	else if ((solver_id > 12) && (solver_id < 22))
		{
		HYPRE_ParCSRGMRESDestroy(solver);
		}
	else if ((solver_id > 22) && (solver_id < 32))
		{
		HYPRE_ParCSRFlexGMRESDestroy(solver);
		}
	if ((solver_id == 0) || (solver_id == 20) || (solver_id == 30))
		{
		   HYPRE_BoomerAMGDestroy(precond);
		}
	else if ((solver_id == 1) || (solver_id == 21) || (solver_id == 31))
		{
		   HYPRE_ParaSailsDestroy(precond);
		}
        if (myid ==0)
        {
          printf("Congrats on the convergence, %e\n", abba[0]);
          printf("Dot prod of velocity, %e\n", alpha[0]);
          printf("Dot prod of the pressure, %e\n", beta[0]);
        }

	    /* Clean up */
	HYPRE_IJMatrixDestroy(ij_K);
	HYPRE_IJMatrixDestroy(ij_G);
	HYPRE_IJVectorDestroy(ij_f);
	HYPRE_IJVectorDestroy(ij_h);
	HYPRE_IJVectorDestroy(ij_vel);
	HYPRE_IJVectorDestroy(ij_pres);


	HYPRE_IJVectorDestroy(ij_D_1);
	HYPRE_IJVectorDestroy(ij_D_2);
	HYPRE_IJVectorDestroy(ij_residual);
	HYPRE_IJVectorDestroy(ij_TransFirst);
	HYPRE_IJVectorDestroy(ij_TransSecond);

	/* Finalize MPI*/
	if (myid == 0)
	{
	printf("\n Linear system correctly solved.");
	}
	MPI_Finalize();

	return(0);
      }

    HYPRE_ParVectorInnerProd(par_residual, par_residual, gamma);
    beta[0] = gamma[0] / gamma_old[0];

    // d_2vector = residual + beta * d_2vector;
    HYPRE_ParVectorScale( beta[0], par_D_2);
    hypre_ParVectorAxpy( 1.0, par_residual, par_D_2);


 }

   return(0);
}

