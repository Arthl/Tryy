#include "binary.h"

#include <stdio.h>

void binary() {
}


HYPRE_Int 
HYPRE_IJMatrixRead_binary( const char     *filename,
                           MPI_Comm        comm,
                           HYPRE_Int       type,
                           HYPRE_IJMatrix *matrix_ptr )
{
   HYPRE_IJMatrix  matrix;
   HYPRE_Int       ilower, iupper, jlower, jupper;
   HYPRE_Int       nnz, iVal;
   HYPRE_Int       ncols, I, J;
   HYPRE_Real      value, realI, realJ; /* no complex number allowed */
   HYPRE_Int       myid, ret;
   char            new_filename[255];
   FILE           *file;

   hypre_MPI_Comm_rank(comm, &myid);
   
   hypre_sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "rb")) == NULL)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   fread(&ilower, sizeof(ilower), 1, file);
   fread(&iupper, sizeof(iupper), 1, file);
   fread(&jlower, sizeof(jlower), 1, file);
   fread(&jupper, sizeof(jupper), 1, file);
   fread(&nnz,    sizeof(nnz),    1, file); /* added by sz */

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);

   HYPRE_IJMatrixSetObjectType(matrix, type);
   HYPRE_IJMatrixInitialize(matrix);

   ncols = 1;
   while( !feof(file)) 
   {
      ret = fread(&realI,     sizeof(realI),     1, file);
      ret = fread(&realJ,     sizeof(realJ),     1, file);
      ret = fread(&value, sizeof(value), 1, file);

      I = (HYPRE_Int) realI;
      J = (HYPRE_Int) realJ;

      if (I < ilower || I > iupper)
         HYPRE_IJMatrixAddToValues(matrix, 1, &ncols, &I, &J, &value);
      else
         HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
   }

   HYPRE_IJMatrixAssemble(matrix);

   fclose(file);

   *matrix_ptr = matrix;

   return hypre_error_flag;
}
