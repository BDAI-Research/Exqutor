#ifndef VECTOR_H
#define VECTOR_H

#define VECTOR_MAX_DIM 16000

#define VECTOR_SIZE(_dim)		(offsetof(Vector, x) + sizeof(float)*(_dim))
#define DatumGetVector(x)		((Vector *) PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x)	DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x)	PG_RETURN_POINTER(x)

#include "executor/execdesc.h"

typedef struct Vector
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for future use, always zero */
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;

// List*		vectorIndexOids;

typedef struct VectorSearchResult
{
	Node *node;
	Oid oid;
	float4 original_tuples;
	float4 processed_tuples;
	QueryDesc *queryDesc;
}		VectorSearchResult;

extern bool reuse_computation;

Vector	   *InitVector(int dim);
void		PrintVector(char *msg, Vector * vector);
int			vector_cmp_internal(Vector * a, Vector * b);


#endif
