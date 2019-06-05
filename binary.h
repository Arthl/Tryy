#ifndef BINARY_HEADER
#define BINARY_HEADER

#include "_hypre_parcsr_ls.h"


HYPRE_Int HYPRE_IJMatrixRead_binary(const char     *filename,
                             MPI_Comm        comm,
                             HYPRE_Int       type,
                             HYPRE_IJMatrix *matrix);


#ifdef __cplusplus
}
#endif

#endif

