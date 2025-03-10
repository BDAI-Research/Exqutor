#include "postgres.h"

#include <math.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif


#include "bitutils.h"
#include "bitvec.h"
#include "catalog/pg_type.h"
#include "common/shortest_dec.h"
#include "fmgr.h"
#include "halfutils.h"
#include "halfvec.h"
#include "hnsw.h"
#include "ivfflat.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "port.h"				/* for strtof() */
#include "sparsevec.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/lsyscache.h"
#include "utils/numeric.h"
#include "vector.h"

#include "optimizer/planner.h"
#include "executor/executor.h"
#include "nodes/plannodes.h"
#include "commands/defrem.h"
#include "access/relation.h"
#include "optimizer/paths.h"
#include "catalog/index.h"
#include "nodes/makefuncs.h"
#include "tcop/tcopprot.h"
#include "parser/parser.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "utils/ruleutils.h"
#include "parser/parsetree.h"

#include "utils/syscache.h"
#include "catalog/pg_class.h"
#include "catalog/pg_am.h"
#include "catalog/namespace.h"
#include "executor/spi.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM < 130000
#define TYPALIGN_DOUBLE 'd'
#define TYPALIGN_INT 'i'
#endif

#define STATE_DIMS(x) (ARR_DIMS(x)[0] - 1)
#define CreateStateDatums(dim) palloc(sizeof(Datum) * (dim + 1))

#if defined(USE_TARGET_CLONES) && !defined(__FMA__)
#define VECTOR_TARGET_CLONES __attribute__((target_clones("default", "fma")))
#else
#define VECTOR_TARGET_CLONES
#endif

PG_MODULE_MAGIC;
Datum vector_out(PG_FUNCTION_ARGS);

extern void (*set_baserel_rows_estimate_hook)(PlannerInfo *root, RelOptInfo *rel);

/* Planner and Execution */
static PlannedStmt *pgvector_plan_planner(Query *parse, const char *query_string, int cursorOptions, ParamListInfo boundParams);
static void pgvector_ExecutorRun(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once);
static void pgvector_ExecutorEnd(QueryDesc *queryDesc);
static void pgvector_set_baserel_rows_estimate_hook(PlannerInfo *root, RelOptInfo *rel);

/* Vector Index and Search */
static bool check_for_vector_search(QueryDesc *queryDesc, Node *node);
static bool check_vector_range_query(PlannerInfo *root, RelOptInfo *rel);
static bool has_vector_column(PlannerInfo *root, RelOptInfo *rel);
static PlannedStmt *CreatePlannedStmtForVectorSearchNodes(QueryDesc *queryDesc, List *hnswNodes);

/* Cardinality Estimation and Sampling */
static double estimate_cardinality_with_sampling(double total_rows);
static void get_true_cardinality_for_vector_query(QueryDesc *queryDesc, PlanState *planstate);
static double get_relation_row_count(QueryDesc *queryDesc, Index scanrelid);
static char *get_tablename_from_scan(QueryDesc *queryDesc, Index scanrelid);

/* Query Analysis and Table Counting */
static void count_total_tables(Query *query, int *table_count);
static void count_tables_in_quals(Node *quals, int *table_count);

/* Query Error Learning and Update */
static void load_qerrors_array(const char *table_name, Datum *Qerrors_array, int *Qerrors_count, float8 *sample_size, float8 *v_grad, float8 *learning_rate);
static void update_qerrors_array(const char *table_name, Datum *Qerrors_array, int Qerrors_count, float8 sample_size, float8 v_grad, float8 learning_rate);
static int compare_float8(const void *a, const void *b);
static float8 get_median(Datum *Qerrors_array, int length);

static planner_hook_type prev_planner = NULL;
static ExecutorRun_hook_type prev_ExecutorRun = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd = NULL;

static Query *original_query = NULL;
static char *original_query_string = NULL;
static int original_cursorOptions = 0;
static ParamListInfo original_boundParams = NULL;
static bool is_first_execution = true;
static bool need_vector_cardinality_estimation = false;
static bool ordering_needed = false;
static bool is_sampling_active = false;

static PlannedStmt *originalPlannedStmt;
static PlannedStmt *original_es_plannedstmt;
static PlannedStmt *newPlannedStmt;


static List *vector_cardinality_results;
static Oid hnswOid = 0;
static Oid vectorOid = 0;

static char *vector_table_name = NULL;
static char *vector_column_name = NULL;
static char *vector_str = NULL;
static float range_distance_value = 0.0;
static double model_estimated_rows = -1.0;
static double estimated_sample_rows = -1.0;
static bool is_vector_range_query = false;

bool reuse_computation = false;

static bool allow_sample_size_update = true;
static float8 sample_size = 385;
static int sample_update_cycle = 50;
static double learning_rate = 0.1;
static double lr_lambda = 0.99;
static double momentum = 0.9;
static double true_cardinality = 0.0;
static double vector_table_size = 0.0;
static double alpha = 50;
static double beta = 1.5;
static double v_grad = 0;
static Datum *Qerrors_array = NULL;
static int Qerrors_count = 0;


/*
 * Initialize index options and variables
 */
PGDLLEXPORT void _PG_init(void);
void
_PG_init(void)
{
	BitvecInit();
	HalfvecInit();
	HnswInit();
	IvfflatInit();
	prev_planner = planner_hook;
	planner_hook = pgvector_plan_planner;
	prev_ExecutorRun = ExecutorRun_hook;
	ExecutorRun_hook = pgvector_ExecutorRun;
	prev_ExecutorEnd = ExecutorEnd_hook;
	ExecutorEnd_hook = pgvector_ExecutorEnd;
	set_baserel_rows_estimate_hook = pgvector_set_baserel_rows_estimate_hook;
	DefineCustomRealVariable("vector.sample_size",
                             "Sets the sample size percentage for TABLESAMPLE.",
                             "Valid range is 0..table_size.",
                             &sample_size,
                             385,   
                             1, 
                             DBL_MAX,  
                             PGC_USERSET,
                             0,
                             NULL, NULL, NULL);
	DefineCustomIntVariable(
		"vector.sample_update_cycle",
		"Sets the update cycle for vector sampling.",
		"Valid range is 1..1000.",
		&sample_update_cycle,
		50,    
		1,     
		1000,  
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);
	DefineCustomBoolVariable(
        "vector.update_sample_size",  
        "Enable or disable user update of sample size.", 
        NULL,  
        &allow_sample_size_update,  
        true,  
        PGC_USERSET,  
        0,  
        NULL,  
        NULL,  
        NULL   
    );
}


static PlannedStmt *
pgvector_plan_planner(Query *parse, const char *query_string, int cursorOptions, ParamListInfo boundParams)
{
	PlannedStmt *result;
	if (hnswOid == 0)
	{
		hnswOid = get_index_am_oid("hnsw", true);
	}

	if (vectorOid == 0)
	{
		vectorOid = TypenameGetTypid("vector");
	}

	if (Qerrors_array == NULL) 
	{
		Qerrors_array = palloc(sizeof(Datum) * sample_update_cycle);
		for (int i = 0; i < sample_update_cycle; i++)
		{
			Qerrors_array[i] = Float8GetDatum(0.0);
		}
	}

	int table_count = 0;
    count_total_tables(parse, &table_count);
	if (table_count > 2)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(TopMemoryContext);
		ordering_needed = true;
		original_query = (Query *)copyObject(parse);
		original_query_string = (char *)query_string;
		original_cursorOptions = cursorOptions;
		original_boundParams = boundParams;
		MemoryContextSwitchTo(oldCtx);
	}
	
	if (prev_planner)
	{
		result = prev_planner(parse, query_string, cursorOptions, boundParams);
	}
	else
	{
		result = standard_planner(parse, query_string, cursorOptions, boundParams);
	}
	return result;
}


static void
pgvector_ExecutorRun(QueryDesc *queryDesc, ScanDirection direction,
                     uint64 count, bool execute_once)
{	
	if (is_first_execution && need_vector_cardinality_estimation)
	{
		if (IsParallelWorker())
			return;

		vector_cardinality_results = NIL;
		is_first_execution = false;
		Plan *planTree = queryDesc->plannedstmt->planTree;
		// find vector search nodes and estimate cardinality
		bool other_node_exists = check_for_vector_search(queryDesc, (Node *) planTree);
		if (vector_cardinality_results)
		{
			// replan with estimated cardinality
			if (prev_planner)
			{
				newPlannedStmt = prev_planner(original_query, original_query_string, original_cursorOptions, original_boundParams);
			}
			else
			{
				newPlannedStmt = standard_planner(original_query, original_query_string, original_cursorOptions, original_boundParams);
			}
			reuse_computation = false;
			originalPlannedStmt = queryDesc->plannedstmt;
			original_es_plannedstmt = queryDesc->estate->es_plannedstmt;
			queryDesc->plannedstmt = newPlannedStmt;
            queryDesc->estate->es_plannedstmt = newPlannedStmt;
            ExecutorFinish(queryDesc);
            if (prev_ExecutorEnd)
				prev_ExecutorEnd(queryDesc);
			else
				standard_ExecutorEnd(queryDesc);
            ExecutorStart(queryDesc, 0);
			need_vector_cardinality_estimation = false;
			// execute with optimal plan
			standard_ExecutorRun(queryDesc, direction, count, execute_once);
		}
		else
		{
			if (prev_ExecutorRun)
				prev_ExecutorRun(queryDesc, direction, count, execute_once);
			else
				standard_ExecutorRun(queryDesc, direction, count, execute_once);
			return;
		}
	}
	else
	{
		if (prev_ExecutorRun)
			prev_ExecutorRun(queryDesc, direction, count, execute_once);
		else
			standard_ExecutorRun(queryDesc, direction, count, execute_once);
		return;
	}
}


static void 
pgvector_ExecutorEnd(QueryDesc *queryDesc)
{
	if (is_first_execution)
	{
		ordering_needed = false;
		need_vector_cardinality_estimation = false;
		model_estimated_rows = -1.0;
		estimated_sample_rows = -1.0;
	}
	else if (is_first_execution == false && need_vector_cardinality_estimation == false)
	{
		// update sample size
		if (estimated_sample_rows != -1.0 && allow_sample_size_update == true)
		{
			get_true_cardinality_for_vector_query(queryDesc, queryDesc->planstate);
			double Q_error = Max(estimated_sample_rows / true_cardinality, true_cardinality / estimated_sample_rows);
			Qerrors_array[Qerrors_count] = Float8GetDatum(Q_error);
			if (Qerrors_count == sample_update_cycle - 1)
			{
				double median_qerror = get_median(Qerrors_array, sample_update_cycle);
				double grad = alpha * (median_qerror - beta) - (100-alpha) * (sample_size / vector_table_size);
				v_grad = momentum * v_grad + learning_rate * grad;
				learning_rate = learning_rate * lr_lambda;
				sample_size = sample_size + v_grad;

			}
			Qerrors_count++;
			if (Qerrors_count == sample_update_cycle)
			{
				Qerrors_count = 0;
			}
			estimated_sample_rows = -1.0;
			update_qerrors_array(vector_table_name, Qerrors_array, Qerrors_count, sample_size, v_grad, learning_rate);
		}
		ordering_needed = false;
		list_free_deep(vector_cardinality_results);
		vector_cardinality_results = NIL;
		is_first_execution = true;
		model_estimated_rows = -1.0;
		
		is_vector_range_query = false;
		reuse_computation = true;
	}

	if (prev_ExecutorEnd)
		prev_ExecutorEnd(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}


static void
pgvector_set_baserel_rows_estimate_hook(PlannerInfo *root, RelOptInfo *rel)
{
	if (has_vector_column(root, rel))
	{
		if (is_first_execution && ordering_needed && !is_sampling_active)
		{
			if (is_vector_range_query == false)
			{
				is_vector_range_query = check_vector_range_query(root, rel);
				if (is_vector_range_query)
				{
					need_vector_cardinality_estimation = true;
				}
			}
		}
		else if (!is_first_execution)
		{
			if (vector_cardinality_results)
			{
				Oid relid = root->simple_rte_array[rel->relid]->relid;
				ListCell *lc;
				// get results of vector search cardinality estimation
				foreach(lc, vector_cardinality_results)
				{
					VectorSearchResult *result = (VectorSearchResult *) lfirst(lc);
					Oid tableoid = result->oid;
					if (tableoid == relid)
					{
						rel->rows = clamp_row_est(result->processed_tuples);
						return;
					}
				}
			}
			
		}
	}

	set_baserel_rows_estimate_standard(root, rel);
}


static bool
check_for_vector_search(QueryDesc *queryDesc, Node *node)
{
	bool other_node_exists = false;
	int vector_tuples_processed = 0;
    if (node == NULL)
	{
		return false;
	}
        
	Plan *plan = (Plan *)node;
	
    if (IsA(node, IndexScan) || IsA(node, IndexOnlyScan))
    {
        IndexScan *indexScan = (IndexScan *) node;
        Oid indexid = indexScan->indexid;

        Relation indexRel = relation_open(indexid, NoLock);
        Oid indexAmOid = indexRel->rd_rel->relam;
		relation_close(indexRel, AccessShareLock);
        // check if the index is HNSW index
        if (indexAmOid == hnswOid)
        {
			Oid tableoid = IndexGetRelation(indexid, true);
			reuse_computation = true;
			List *hnswNodes = NIL;
			hnswNodes = lappend(hnswNodes, node);
			// execute HNSW vector search
			PlannedStmt *hnswPlannedStmt = CreatePlannedStmtForVectorSearchNodes(queryDesc, hnswNodes);
			QueryDesc *hnswQueryDesc = CreateQueryDesc(hnswPlannedStmt,
													queryDesc->sourceText,
													queryDesc->snapshot,
													queryDesc->crosscheck_snapshot,
													None_Receiver,
													queryDesc->params,
													queryDesc->queryEnv,
													0);
			ExecutorStart(hnswQueryDesc, 0);
			ExecutorRun(hnswQueryDesc, ForwardScanDirection, 0L, true);
			vector_tuples_processed = hnswQueryDesc->estate->es_processed;
			VectorSearchResult *hnswResult = palloc(sizeof(VectorSearchResult));
			hnswResult->node = node;
			hnswResult->oid = tableoid;
			hnswResult->processed_tuples = vector_tuples_processed;
			vector_cardinality_results = lappend(vector_cardinality_results, hnswResult);
			
			ExecutorFinish(hnswQueryDesc);
			if (prev_ExecutorEnd)
				prev_ExecutorEnd(hnswQueryDesc);
			else
				standard_ExecutorEnd(hnswQueryDesc);
			FreeQueryDesc(hnswQueryDesc);

			return other_node_exists;
        }
		else
		{
			other_node_exists = true;
		}
    }
	else if (IsA(node, SubqueryScan))
	{
		SubqueryScan *subqueryScanNode = (SubqueryScan *) node;
		
		if (check_for_vector_search(queryDesc, (Node *) subqueryScanNode->subplan))
			other_node_exists = true;
	}
	else if (IsA(node, SeqScan))
	{
		SeqScan *seqScanNode = (SeqScan *) node;

		RangeTblEntry *rte = rt_fetch(seqScanNode->scan.scanrelid, queryDesc->estate->es_range_table);
		
		Oid tableoid = rte->relid;

		Relation seqScanRel = relation_open(tableoid, AccessShareLock);
		char *tableName = get_rel_name(tableoid);
		// execute sampling for vector table
		if (strcmp(tableName, vector_table_name) == 0)
		{
			if (is_vector_range_query)
			{
				
				double total_rows = seqScanRel->rd_rel->reltuples;
				if (allow_sample_size_update == true)
				{
					load_qerrors_array(vector_table_name, Qerrors_array, &Qerrors_count, &sample_size, &v_grad, &learning_rate);
				}
				estimated_sample_rows = estimate_cardinality_with_sampling(total_rows);
				VectorSearchResult *sampling_result = palloc(sizeof(VectorSearchResult));
				sampling_result->node = node;
				sampling_result->oid = tableoid;
				sampling_result->processed_tuples = estimated_sample_rows;
				vector_cardinality_results = lappend(vector_cardinality_results, sampling_result);
			}
		}
		else {
			other_node_exists = true;
		}
		relation_close(seqScanRel, AccessShareLock);

		return other_node_exists;
	}
	else
	{
		other_node_exists = true;
	}

    if (check_for_vector_search(queryDesc, (Node *) plan->lefttree))
		other_node_exists = true;

    if (check_for_vector_search(queryDesc, (Node *) plan->righttree))
		other_node_exists = true;

    return other_node_exists;
}


static bool
check_vector_range_query(PlannerInfo *root, RelOptInfo *rel)
{
	bool vector_range_query = false;
	RangeTblEntry *rte;
	Oid relid = root->simple_rte_array[rel->relid]->relid;
	

	ListCell *lc;
	foreach(lc, rel->baserestrictinfo)
    {
        RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);
        Expr *clause = rinfo->clause;

        if (IsA(clause, OpExpr))
        {
            OpExpr *opexpr = (OpExpr *) clause;

            Oid opno = opexpr->opno;
            const char *opname = get_opname(opno);

            if (strcmp(opname, "<") == 0 || strcmp(opname, ">") == 0 ||
                strcmp(opname, "<=") == 0 || strcmp(opname, ">=") == 0)
            {
                Expr *left_expr = (Expr *) linitial(opexpr->args);
                Expr *right_expr = (Expr *) lsecond(opexpr->args);

				
                if (IsA(left_expr, OpExpr))
                {
                    OpExpr *left_opexpr = (OpExpr *) left_expr;
                    const char *left_opname = get_opname(left_opexpr->opno);

                    if (strcmp(left_opname, "<->") == 0)
                    {
						vector_range_query = true;
						if (IsA(left_opexpr, OpExpr))
						{
							OpExpr *left = (OpExpr *) left_opexpr;
							Expr *left_expr1 = (Expr *) linitial(left->args);
							Expr *right_expr2 = (Expr *) lsecond(left->args);
							if (IsA(left_expr1, Var))
							{
								Var *var = (Var *) left_expr1;
								rte = planner_rt_fetch(var->varno, root);
								vector_column_name = get_rte_attribute_name(rte, var->varattno);
							}
							if (IsA(right_expr2, Const))
							{
								Const *const_val = (Const *) right_expr2;
								
								if(const_val->consttype == vectorOid)
								{
									Vector *vector = (Vector *) DatumGetPointer(const_val->constvalue);
									vector_str = DatumGetCString(DirectFunctionCall1(vector_out, PointerGetDatum(vector)));
									vector_table_name = get_rel_name(relid);
								}
							}
						}
                    }
					if (IsA(right_expr, Const))
					{
						Const *const_val = (Const *) right_expr;

						if (const_val->consttype == FLOAT8OID)
						{
							range_distance_value = DatumGetFloat8(const_val->constvalue);
						}
					}
                }
            }
        }
    }
	return vector_range_query;
}


static bool
has_vector_column(PlannerInfo *root, RelOptInfo *rel)
{
    RangeTblEntry *rte = root->simple_rte_array[rel->relid];
	if (rte == NULL || rte->rtekind != RTE_RELATION)
    {
        return false;
    }
    Oid relid = rte->relid;
    Relation relation;
    TupleDesc tupleDesc;
    bool has_vector = false;

    relation = relation_open(relid, AccessShareLock);
    tupleDesc = RelationGetDescr(relation);

    for (int i = 0; i < tupleDesc->natts; i++)
    {
        Form_pg_attribute attr = TupleDescAttr(tupleDesc, i);

        if (!attr->attisdropped)
        {
            if (attr->atttypid == vectorOid)
            {
                has_vector = true;
                break;
            }
        }
    }

    /* Close the relation */
    relation_close(relation, AccessShareLock);
    return has_vector;
}


static PlannedStmt *
CreatePlannedStmtForVectorSearchNodes(QueryDesc *queryDesc, List *hnswNodes)
{
    PlannedStmt *newPlannedStmt = makeNode(PlannedStmt);

    newPlannedStmt->commandType = CMD_SELECT;  
    newPlannedStmt->queryId = queryDesc->plannedstmt->queryId;
    newPlannedStmt->hasReturning = false;
    newPlannedStmt->hasModifyingCTE = false;
    newPlannedStmt->canSetTag = true;
    newPlannedStmt->transientPlan = false;
    newPlannedStmt->dependsOnRole = false;
    newPlannedStmt->parallelModeNeeded = false;

    Plan *topPlan = (Plan *) linitial(hnswNodes);
    newPlannedStmt->planTree = topPlan;

    newPlannedStmt->rtable = queryDesc->plannedstmt->rtable;
    newPlannedStmt->subplans = NIL;
    newPlannedStmt->resultRelations = NIL;
    newPlannedStmt->rowMarks = NIL;
    newPlannedStmt->relationOids = queryDesc->plannedstmt->relationOids;
    newPlannedStmt->invalItems = queryDesc->plannedstmt->invalItems;
	newPlannedStmt->permInfos = queryDesc->plannedstmt->permInfos;
    return newPlannedStmt;
}


static double
estimate_cardinality_with_sampling(double total_rows)
{
    StringInfoData query;
    int ret;
    double count_result = 0.0;
    bool isnull;
    is_sampling_active = true;

    initStringInfo(&query);

    double sample_ratio = sample_size / total_rows * 100;
    appendStringInfo(&query,
                     "SELECT COUNT(*)::float FROM (SELECT %s FROM %s TABLESAMPLE SYSTEM(%f)) p "
                     "WHERE p.%s <-> '%s' < %f;",
                     vector_column_name, vector_table_name, sample_ratio,
                     vector_column_name, vector_str, range_distance_value);

    SPI_connect();
    ret = SPI_execute(query.data, true, 0);

    if (ret != SPI_OK_SELECT)
    {
        SPI_finish();
    }

    if (SPI_processed > 0)
    {
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        HeapTuple tuple = SPI_tuptable->vals[0];
        Datum count_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);

        if (!isnull)
        {
            count_result = DatumGetFloat8(count_datum);
        }
    }

    SPI_finish();

    pfree(query.data);
    is_sampling_active = false;

    if (count_result == 0)
    {
        count_result = 1;
    }

    count_result = count_result / sample_ratio * 100;

    return count_result;
}


static void 
get_true_cardinality_for_vector_query(QueryDesc *queryDesc, PlanState *planstate) 
{
    if (planstate == NULL) {
        return;
    }

    Plan *plan = planstate->plan;
    Instrumentation *instrument = planstate->instrument;

    if (IsA(plan, SeqScan) || IsA(plan, IndexScan) || IsA(plan, BitmapHeapScan)) {
        char *relname = get_tablename_from_scan(queryDesc, ((Scan *)plan)->scanrelid);

		if (strcmp(relname, vector_table_name) == 0){
			true_cardinality = instrument->ntuples;
			if (vector_table_size == 0.0)
			{
				vector_table_size = get_relation_row_count(queryDesc, ((Scan *)plan)->scanrelid);
			}
			return;
		}
    }

    if (outerPlanState(planstate)) {
        get_true_cardinality_for_vector_query(queryDesc, outerPlanState(planstate));
    }
    if (innerPlanState(planstate)) {
        get_true_cardinality_for_vector_query(queryDesc, innerPlanState(planstate));
    }

}


static double 
get_relation_row_count(QueryDesc *queryDesc, Index scanrelid) 
{
    if (queryDesc == NULL || queryDesc->plannedstmt == NULL || queryDesc->plannedstmt->rtable == NULL) {
        elog(WARNING, "Invalid QueryDesc or scanrelid");
        return 0.0;
    }

    List *rtable = queryDesc->plannedstmt->rtable;
    if (scanrelid == 0 || scanrelid > list_length(rtable)) {
        elog(WARNING, "Invalid scanrelid: %u", scanrelid);
        return 0.0;
    }

    RangeTblEntry *rte = list_nth(rtable, scanrelid - 1);

    if (rte == NULL || rte->relid == InvalidOid) {
        elog(WARNING, "RangeTblEntry not found or invalid relid for scanrelid: %u", scanrelid);
        return 0.0;
    }

    Oid relid = rte->relid;
    HeapTuple tuple;
    Form_pg_class relform;
    double row_count = 0.0;

    tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(relid));
    if (HeapTupleIsValid(tuple)) {
        relform = (Form_pg_class) GETSTRUCT(tuple);
        row_count = relform->reltuples;  
        ReleaseSysCache(tuple);
    } else {
        elog(WARNING, "Relation with OID %u not found in pg_class", relid);
    }

    return row_count;
}

static char *
get_tablename_from_scan(QueryDesc *queryDesc, Index scanrelid) 
{
    if (scanrelid == 0 || queryDesc->plannedstmt->rtable == NULL) {
        return NULL;
    }

    List *rtable = queryDesc->plannedstmt->rtable;
    RangeTblEntry *rte = list_nth(rtable, scanrelid - 1);

    if (rte->relid != InvalidOid) {
        return get_rel_name(rte->relid);
    }

    return NULL;
}


static void 
count_total_tables(Query *query, int *table_count)
{
    ListCell *lc;

    foreach(lc, query->rtable)
    {
        RangeTblEntry *rte = (RangeTblEntry *) lfirst(lc);

        if (rte->rtekind == RTE_RELATION)
        {
            (*table_count)++;
        }
		
        else if (rte->rtekind == RTE_SUBQUERY)
        {
            count_total_tables(rte->subquery, table_count);
        }
    }

    if (query->jointree && query->jointree->quals)
    {
        count_tables_in_quals(query->jointree->quals, table_count);
    }
}

static void 
count_tables_in_quals(Node *quals, int *table_count)
{
    if (quals == NULL)
        return;

    if (IsA(quals, SubLink))
    {
        SubLink *sublink = (SubLink *) quals;
        Query *subquery = (Query *) sublink->subselect;
        count_total_tables(subquery, table_count);
    }
    else if (IsA(quals, BoolExpr)) 
    {
        BoolExpr *bool_expr = (BoolExpr *) quals;
        ListCell *lc;

        foreach(lc, bool_expr->args)
        {
            Node *arg = (Node *) lfirst(lc);
            count_tables_in_quals(arg, table_count);
        }
    }
}


static void 
load_qerrors_array(const char *table_name, Datum *Qerrors_array, int *Qerrors_count, float8 *sample_size, float8 *v_grad, float8 *learning_rate) 
{

    SPI_connect();

    char *select_query = "SELECT sample_size, recent_qerrors, qerror_count, v_grad, learning_rate FROM pgvector_qerror WHERE table_name = $1";
    Oid argtypes[1] = {TEXTOID};
    Datum values[1] = {CStringGetTextDatum(table_name)};

    int ret = SPI_execute_with_args(select_query, 1, argtypes, values, NULL, false, 0);

    if (ret == SPI_OK_SELECT && SPI_processed > 0) {
        bool isnull;

        Datum sample_size_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
        *sample_size = isnull ? 385 : DatumGetFloat8(sample_size_datum);

        Datum recent_qerrors = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull);

        Datum qerror_count_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull);
        *Qerrors_count = DatumGetInt32(qerror_count_datum);

        Datum v_grad_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull);
        *v_grad = isnull ? 0.0 : DatumGetFloat8(v_grad_datum);

        Datum learning_rate_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5, &isnull);
        *learning_rate = isnull ? 0.1 : DatumGetFloat8(learning_rate_datum);

        ArrayType *array = DatumGetArrayTypeP(recent_qerrors);
        Datum *elements;
        bool *nulls;
        int num_elements;

        deconstruct_array(array, FLOAT8OID, sizeof(float8), true, 'd', &elements, &nulls, &num_elements);

        MemoryContext oldCtx = MemoryContextSwitchTo(TopMemoryContext);

        for (int i = 0; i < num_elements; i++) {
            Qerrors_array[i] = elements[i];
        }

        MemoryContextSwitchTo(oldCtx);

        pfree(elements);
    } else {
        elog(WARNING, "No rows found for table_name: %s", table_name);
    }

    SPI_finish();
}

static void 
update_qerrors_array(const char *table_name, Datum *Qerrors_array, int Qerrors_count, float8 sample_size, float8 v_grad, float8 learning_rate) 
{
    ArrayType *array = construct_array(
        Qerrors_array,     
        sample_update_cycle,     
        FLOAT8OID,         
        sizeof(float8),    
        true,              
        'd'                
    );

    SPI_connect();

    char *update_query = "UPDATE pgvector_qerror "
                         "SET recent_qerrors = $1, qerror_count = $2, sample_size = $3, v_grad = $4, learning_rate = $5 "
                         "WHERE table_name = $6";

    Oid argtypes[6] = {FLOAT8ARRAYOID, INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, TEXTOID};
    Datum values[6] = {
        PointerGetDatum(array),             
        Int32GetDatum(Qerrors_count),       
        Float8GetDatum(sample_size),        
        Float8GetDatum(v_grad),             
        Float8GetDatum(learning_rate),      
        CStringGetTextDatum(table_name)     
    };

    int ret = SPI_execute_with_args(update_query, 6, argtypes, values, NULL, false, 0);

    if (ret == SPI_OK_UPDATE && SPI_processed == 0) {
        char *insert_query = "INSERT INTO pgvector_qerror (recent_qerrors, qerror_count, sample_size, v_grad, learning_rate, table_name) "
                             "VALUES ($1, $2, $3, $4, $5, $6)";

        SPI_execute_with_args(insert_query, 6, argtypes, values, NULL, false, 0);
    }

    SPI_finish();
}

static int 
compare_float8(const void *a, const void *b) 
{
    float8 fa = *(const float8 *)a;
    float8 fb = *(const float8 *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static float8 
get_median(Datum *Qerrors_array, int length) 
{
    float8 *values = palloc(length * sizeof(float8));
	float8 median;
    for (int i = 0; i < length; i++) {
        values[i] = DatumGetFloat8(Qerrors_array[i]);
    }

    qsort(values, length, sizeof(float8), compare_float8);

    
    if (length % 2 == 1) {
        median = values[length / 2];
    } else {
        median = (values[length / 2 - 1] + values[length / 2]) / 2.0;
    }

    pfree(values);
    return median;
}




/*
 * Ensure same dimensions
 */
static inline void
CheckDims(Vector * a, Vector * b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", a->dim, b->dim)));
}

/*
 * Ensure expected dimensions
 */
static inline void
CheckExpectedDim(int32 typmod, int dim)
{
	if (typmod != -1 && typmod != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", typmod, dim)));
}

/*
 * Ensure valid dimensions
 */
static inline void
CheckDim(int dim)
{
	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	if (dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector cannot have more than %d dimensions", VECTOR_MAX_DIM)));
}

/*
 * Ensure finite element
 */
static inline void
CheckElement(float value)
{
	if (isnan(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("NaN not allowed in vector")));

	if (isinf(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("infinite value not allowed in vector")));
}

/*
 * Allocate and initialize a new vector
 */
Vector *
InitVector(int dim)
{
	Vector	   *result;
	int			size;

	size = VECTOR_SIZE(dim);
	result = (Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	return result;
}

/*
 * Check for whitespace, since array_isspace() is static
 */
static inline bool
vector_isspace(char ch)
{
	if (ch == ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f')
		return true;
	return false;
}

/*
 * Check state array
 */
static float8 *
CheckStateArray(ArrayType *statearray, const char *caller)
{
	if (ARR_NDIM(statearray) != 1 ||
		ARR_DIMS(statearray)[0] < 1 ||
		ARR_HASNULL(statearray) ||
		ARR_ELEMTYPE(statearray) != FLOAT8OID)
		elog(ERROR, "%s: expected state array", caller);
	return (float8 *) ARR_DATA_PTR(statearray);
}

#if PG_VERSION_NUM < 120003
static pg_noinline void
float_overflow_error(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
			 errmsg("value out of range: overflow")));
}

static pg_noinline void
float_underflow_error(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
			 errmsg("value out of range: underflow")));
}
#endif

/*
 * Convert textual representation to internal representation
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_in);
Datum
vector_in(PG_FUNCTION_ARGS)
{
	char	   *lit = PG_GETARG_CSTRING(0);
	int32		typmod = PG_GETARG_INT32(2);
	float		x[VECTOR_MAX_DIM];
	int			dim = 0;
	char	   *pt = lit;
	Vector	   *result;

	while (vector_isspace(*pt))
		pt++;

	if (*pt != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type vector: \"%s\"", lit),
				 errdetail("Vector contents must start with \"[\".")));

	pt++;

	while (vector_isspace(*pt))
		pt++;

	if (*pt == ']')
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	for (;;)
	{
		float		val;
		char	   *stringEnd;

		if (dim == VECTOR_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("vector cannot have more than %d dimensions", VECTOR_MAX_DIM)));

		while (vector_isspace(*pt))
			pt++;

		/* Check for empty string like float4in */
		if (*pt == '\0')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));

		errno = 0;

		/* Use strtof like float4in to avoid a double-rounding problem */
		/* Postgres sets LC_NUMERIC to C on startup */
		val = strtof(pt, &stringEnd);

		if (stringEnd == pt)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));

		/* Check for range error like float4in */
		if (errno == ERANGE && isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("\"%s\" is out of range for type vector", pnstrdup(pt, stringEnd - pt))));

		CheckElement(val);
		x[dim++] = val;

		pt = stringEnd;

		while (vector_isspace(*pt))
			pt++;

		if (*pt == ',')
			pt++;
		else if (*pt == ']')
		{
			pt++;
			break;
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));
	}

	/* Only whitespace is allowed after the closing brace */
	while (vector_isspace(*pt))
		pt++;

	if (*pt != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type vector: \"%s\"", lit),
				 errdetail("Junk after closing right brace.")));

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
		result->x[i] = x[i];

	PG_RETURN_POINTER(result);
}

#define AppendChar(ptr, c) (*(ptr)++ = (c))
#define AppendFloat(ptr, f) ((ptr) += float_to_shortest_decimal_bufn((f), (ptr)))

/*
 * Convert internal representation to textual representation
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_out);
Datum
vector_out(PG_FUNCTION_ARGS)
{
	Vector	   *vector = PG_GETARG_VECTOR_P(0);
	int			dim = vector->dim;
	char	   *buf;
	char	   *ptr;

	/*
	 * Need:
	 *
	 * dim * (FLOAT_SHORTEST_DECIMAL_LEN - 1) bytes for
	 * float_to_shortest_decimal_bufn
	 *
	 * dim - 1 bytes for separator
	 *
	 * 3 bytes for [, ], and \0
	 */
	buf = (char *) palloc(FLOAT_SHORTEST_DECIMAL_LEN * dim + 2);
	ptr = buf;

	AppendChar(ptr, '[');

	for (int i = 0; i < dim; i++)
	{
		if (i > 0)
			AppendChar(ptr, ',');

		AppendFloat(ptr, vector->x[i]);
	}

	AppendChar(ptr, ']');
	*ptr = '\0';

	PG_FREE_IF_COPY(vector, 0);
	PG_RETURN_CSTRING(buf);
}

/*
 * Print vector - useful for debugging
 */
void
PrintVector(char *msg, Vector * vector)
{
	char	   *out = DatumGetPointer(DirectFunctionCall1(vector_out, PointerGetDatum(vector)));

	elog(INFO, "%s = %s", msg, out);
	pfree(out);
}

/*
 * Convert type modifier
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_typmod_in);
Datum
vector_typmod_in(PG_FUNCTION_ARGS)
{
	ArrayType  *ta = PG_GETARG_ARRAYTYPE_P(0);
	int32	   *tl;
	int			n;

	tl = ArrayGetIntegerTypmods(ta, &n);

	if (n != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid type modifier")));

	if (*tl < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type vector must be at least 1")));

	if (*tl > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type vector cannot exceed %d", VECTOR_MAX_DIM)));

	PG_RETURN_INT32(*tl);
}

/*
 * Convert external binary representation to internal representation
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_recv);
Datum
vector_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	int32		typmod = PG_GETARG_INT32(2);
	Vector	   *result;
	int16		dim;
	int16		unused;

	dim = pq_getmsgint(buf, sizeof(int16));
	unused = pq_getmsgint(buf, sizeof(int16));

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	if (unused != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected unused to be 0, not %d", unused)));

	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
	{
		result->x[i] = pq_getmsgfloat4(buf);
		CheckElement(result->x[i]);
	}

	PG_RETURN_POINTER(result);
}

/*
 * Convert internal representation to the external binary representation
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_send);
Datum
vector_send(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint(&buf, vec->dim, sizeof(int16));
	pq_sendint(&buf, vec->unused, sizeof(int16));
	for (int i = 0; i < vec->dim; i++)
		pq_sendfloat4(&buf, vec->x[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * Convert vector to vector
 * This is needed to check the type modifier
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector);
Datum
vector(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	int32		typmod = PG_GETARG_INT32(1);

	CheckExpectedDim(typmod, vec->dim);

	PG_RETURN_POINTER(vec);
}

/*
 * Convert array to vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(array_to_vector);
Datum
array_to_vector(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum	   *elemsp;
	int			nelemsp;

	if (ARR_NDIM(array) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be 1-D")));

	if (ARR_HASNULL(array) && array_contains_nulls(array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain nulls")));

	get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
	deconstruct_array(array, ARR_ELEMTYPE(array), typlen, typbyval, typalign, &elemsp, NULL, &nelemsp);

	CheckDim(nelemsp);
	CheckExpectedDim(typmod, nelemsp);

	result = InitVector(nelemsp);

	if (ARR_ELEMTYPE(array) == INT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetInt32(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT8OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat8(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat4(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == NUMERICOID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat4(DirectFunctionCall1(numeric_float4, elemsp[i]));
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported array type")));
	}

	/*
	 * Free allocation from deconstruct_array. Do not free individual elements
	 * when pass-by-reference since they point to original array.
	 */
	pfree(elemsp);

	/* Check elements */
	for (int i = 0; i < result->dim; i++)
		CheckElement(result->x[i]);

	PG_RETURN_POINTER(result);
}

/*
 * Convert vector to float4[]
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_to_float4);
Datum
vector_to_float4(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	Datum	   *datums;
	ArrayType  *result;

	datums = (Datum *) palloc(sizeof(Datum) * vec->dim);

	for (int i = 0; i < vec->dim; i++)
		datums[i] = Float4GetDatum(vec->x[i]);

	/* Use TYPALIGN_INT for float4 */
	result = construct_array(datums, vec->dim, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);

	pfree(datums);

	PG_RETURN_POINTER(result);
}

/*
 * Convert half vector to vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(halfvec_to_vector);
Datum
halfvec_to_vector(PG_FUNCTION_ARGS)
{
	HalfVector *vec = PG_GETARG_HALFVEC_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;

	CheckDim(vec->dim);
	CheckExpectedDim(typmod, vec->dim);

	result = InitVector(vec->dim);

	for (int i = 0; i < vec->dim; i++)
		result->x[i] = HalfToFloat4(vec->x[i]);

	PG_RETURN_POINTER(result);
}

VECTOR_TARGET_CLONES static float
VectorL2SquaredDistance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		diff = ax[i] - bx[i];

		distance += diff * diff;
	}

	return distance;
}

/*
 * Get the L2 distance between vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(l2_distance);
Datum
l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8(sqrt((double) VectorL2SquaredDistance(a->dim, a->x, b->x)));
}

/*
 * Get the L2 squared distance between vectors
 * This saves a sqrt calculation
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_l2_squared_distance);
Datum
vector_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorL2SquaredDistance(a->dim, a->x, b->x));
}

VECTOR_TARGET_CLONES static float
VectorInnerProduct(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += ax[i] * bx[i];

	return distance;
}

/*
 * Get the inner product of two vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(inner_product);
Datum
inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorInnerProduct(a->dim, a->x, b->x));
}

/*
 * Get the negative inner product of two vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_negative_inner_product);
Datum
vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) -VectorInnerProduct(a->dim, a->x, b->x));
}

VECTOR_TARGET_CLONES static double
VectorCosineSimilarity(int dim, float *ax, float *bx)
{
	float		similarity = 0.0;
	float		norma = 0.0;
	float		normb = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		similarity += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

/*
 * Get the cosine distance between two vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(cosine_distance);
Datum
cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	double		similarity;

	CheckDims(a, b);

	similarity = VectorCosineSimilarity(a->dim, a->x, b->x);

#ifdef _MSC_VER
	/* /fp:fast may not propagate NaN */
	if (isnan(similarity))
		PG_RETURN_FLOAT8(NAN);
#endif

	/* Keep in range */
	if (similarity > 1)
		similarity = 1.0;
	else if (similarity < -1)
		similarity = -1.0;

	PG_RETURN_FLOAT8(1.0 - similarity);
}

/*
 * Get the distance for spherical k-means
 * Currently uses angular distance since needs to satisfy triangle inequality
 * Assumes inputs are unit vectors (skips norm)
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_spherical_distance);
Datum
vector_spherical_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	double		distance;

	CheckDims(a, b);

	distance = (double) VectorInnerProduct(a->dim, a->x, b->x);

	/* Prevent NaN with acos with loss of precision */
	if (distance > 1)
		distance = 1;
	else if (distance < -1)
		distance = -1;

	PG_RETURN_FLOAT8(acos(distance) / M_PI);
}

/* Does not require FMA, but keep logic simple */
VECTOR_TARGET_CLONES static float
VectorL1Distance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += fabsf(ax[i] - bx[i]);

	return distance;
}

/*
 * Get the L1 distance between two vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(l1_distance);
Datum
l1_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorL1Distance(a->dim, a->x, b->x));
}

/*
 * Get the dimensions of a vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_dims);
Datum
vector_dims(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);

	PG_RETURN_INT32(a->dim);
}

/*
 * Get the L2 norm of a vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_norm);
Datum
vector_norm(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	double		norm = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < a->dim; i++)
		norm += (double) ax[i] * (double) ax[i];

	PG_RETURN_FLOAT8(sqrt(norm));
}

/*
 * Normalize a vector with the L2 norm
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(l2_normalize);
Datum
l2_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	double		norm = 0;
	Vector	   *result;
	float	   *rx;

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0; i < a->dim; i++)
		norm += (double) ax[i] * (double) ax[i];

	norm = sqrt(norm);

	/* Return zero vector for zero norm */
	if (norm > 0)
	{
		for (int i = 0; i < a->dim; i++)
			rx[i] = ax[i] / norm;

		/* Check for overflow */
		for (int i = 0; i < a->dim; i++)
		{
			if (isinf(rx[i]))
				float_overflow_error();
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * Add vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_add);
Datum
vector_add(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] + bx[i];

	/* Check for overflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Subtract vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_sub);
Datum
vector_sub(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] - bx[i];

	/* Check for overflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Multiply vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_mul);
Datum
vector_mul(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] * bx[i];

	/* Check for overflow and underflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();

		if (rx[i] == 0 && !(ax[i] == 0 || bx[i] == 0))
			float_underflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Concatenate vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_concat);
Datum
vector_concat(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	Vector	   *result;
	int			dim = a->dim + b->dim;

	CheckDim(dim);
	result = InitVector(dim);

	for (int i = 0; i < a->dim; i++)
		result->x[i] = a->x[i];

	for (int i = 0; i < b->dim; i++)
		result->x[i + a->dim] = b->x[i];

	PG_RETURN_POINTER(result);
}

/*
 * Quantize a vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(binary_quantize);
Datum
binary_quantize(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	VarBit	   *result = InitBitVector(a->dim);
	unsigned char *rx = VARBITS(result);

	for (int i = 0; i < a->dim; i++)
		rx[i / 8] |= (ax[i] > 0) << (7 - (i % 8));

	PG_RETURN_VARBIT_P(result);
}

/*
 * Get a subvector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(subvector);
Datum
subvector(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	int32		start = PG_GETARG_INT32(1);
	int32		count = PG_GETARG_INT32(2);
	int32		end;
	float	   *ax = a->x;
	Vector	   *result;
	int			dim;

	if (count < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	/*
	 * Check if (start + count > a->dim), avoiding integer overflow. a->dim
	 * and count are both positive, so a->dim - count won't overflow.
	 */
	if (start > a->dim - count)
		end = a->dim + 1;
	else
		end = start + count;

	/* Indexing starts at 1, like substring */
	if (start < 1)
		start = 1;
	else if (start > a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	dim = end - start;
	CheckDim(dim);
	result = InitVector(dim);

	for (int i = 0; i < dim; i++)
		result->x[i] = ax[start - 1 + i];

	PG_RETURN_POINTER(result);
}

/*
 * Internal helper to compare vectors
 */
int
vector_cmp_internal(Vector * a, Vector * b)
{
	int			dim = Min(a->dim, b->dim);

	/* Check values before dimensions to be consistent with Postgres arrays */
	for (int i = 0; i < dim; i++)
	{
		if (a->x[i] < b->x[i])
			return -1;

		if (a->x[i] > b->x[i])
			return 1;
	}

	if (a->dim < b->dim)
		return -1;

	if (a->dim > b->dim)
		return 1;

	return 0;
}

/*
 * Less than
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_lt);
Datum
vector_lt(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) < 0);
}

/*
 * Less than or equal
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_le);
Datum
vector_le(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) <= 0);
}

/*
 * Equal
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) == 0);
}

/*
 * Not equal
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) != 0);
}

/*
 * Greater than or equal
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_ge);
Datum
vector_ge(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) >= 0);
}

/*
 * Greater than
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_gt);
Datum
vector_gt(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) > 0);
}

/*
 * Compare vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_cmp);
Datum
vector_cmp(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_INT32(vector_cmp_internal(a, b));
}

/*
 * Accumulate vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_accum);
Datum
vector_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *statearray = PG_GETARG_ARRAYTYPE_P(0);
	Vector	   *newval = PG_GETARG_VECTOR_P(1);
	float8	   *statevalues;
	int16		dim;
	bool		newarr;
	float8		n;
	Datum	   *statedatums;
	float	   *x = newval->x;
	ArrayType  *result;

	/* Check array before using */
	statevalues = CheckStateArray(statearray, "vector_accum");
	dim = STATE_DIMS(statearray);
	newarr = dim == 0;

	if (newarr)
		dim = newval->dim;
	else
		CheckExpectedDim(dim, newval->dim);

	n = statevalues[0] + 1.0;

	statedatums = CreateStateDatums(dim);
	statedatums[0] = Float8GetDatum(n);

	if (newarr)
	{
		for (int i = 0; i < dim; i++)
			statedatums[i + 1] = Float8GetDatum((double) x[i]);
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			double		v = statevalues[i + 1] + x[i];

			/* Check for overflow */
			if (isinf(v))
				float_overflow_error();

			statedatums[i + 1] = Float8GetDatum(v);
		}
	}

	/* Use float8 array like float4_accum */
	result = construct_array(statedatums, dim + 1,
							 FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	pfree(statedatums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Combine vectors or half vectors (also used for halfvec_combine)
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_combine);
Datum
vector_combine(PG_FUNCTION_ARGS)
{
	/* Must also update parameters of halfvec_combine if modifying */
	ArrayType  *statearray1 = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *statearray2 = PG_GETARG_ARRAYTYPE_P(1);
	float8	   *statevalues1;
	float8	   *statevalues2;
	float8		n;
	float8		n1;
	float8		n2;
	int16		dim;
	Datum	   *statedatums;
	ArrayType  *result;

	/* Check arrays before using */
	statevalues1 = CheckStateArray(statearray1, "vector_combine");
	statevalues2 = CheckStateArray(statearray2, "vector_combine");

	n1 = statevalues1[0];
	n2 = statevalues2[0];

	if (n1 == 0.0)
	{
		n = n2;
		dim = STATE_DIMS(statearray2);
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
			statedatums[i] = Float8GetDatum(statevalues2[i]);
	}
	else if (n2 == 0.0)
	{
		n = n1;
		dim = STATE_DIMS(statearray1);
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
			statedatums[i] = Float8GetDatum(statevalues1[i]);
	}
	else
	{
		n = n1 + n2;
		dim = STATE_DIMS(statearray1);
		CheckExpectedDim(dim, STATE_DIMS(statearray2));
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
		{
			double		v = statevalues1[i] + statevalues2[i];

			/* Check for overflow */
			if (isinf(v))
				float_overflow_error();

			statedatums[i] = Float8GetDatum(v);
		}
	}

	statedatums[0] = Float8GetDatum(n);

	result = construct_array(statedatums, dim + 1,
							 FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	pfree(statedatums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Average vectors
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(vector_avg);
Datum
vector_avg(PG_FUNCTION_ARGS)
{
	ArrayType  *statearray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *statevalues;
	float8		n;
	uint16		dim;
	Vector	   *result;

	/* Check array before using */
	statevalues = CheckStateArray(statearray, "vector_avg");
	n = statevalues[0];

	/* SQL defines AVG of no values to be NULL */
	if (n == 0.0)
		PG_RETURN_NULL();

	/* Create vector */
	dim = STATE_DIMS(statearray);
	CheckDim(dim);
	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
	{
		result->x[i] = statevalues[i + 1] / n;
		CheckElement(result->x[i]);
	}

	PG_RETURN_POINTER(result);
}

/*
 * Convert sparse vector to dense vector
 */
PGDLLEXPORT PG_FUNCTION_INFO_V1(sparsevec_to_vector);
Datum
sparsevec_to_vector(PG_FUNCTION_ARGS)
{
	SparseVector *svec = PG_GETARG_SPARSEVEC_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;
	int			dim = svec->dim;
	float	   *values = SPARSEVEC_VALUES(svec);

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	result = InitVector(dim);
	for (int i = 0; i < svec->nnz; i++)
		result->x[svec->indices[i]] = values[i];

	PG_RETURN_POINTER(result);
}
