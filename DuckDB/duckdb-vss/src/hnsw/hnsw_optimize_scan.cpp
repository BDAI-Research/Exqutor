#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/optimizer/build_probe_side_optimizer.hpp"
#include "duckdb/storage/data_table.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexScanOptimizer : public OptimizerExtension {
public:
	HNSWIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool findFilterOperatorWithArray_distance(ClientContext &context,
													unique_ptr<LogicalOperator> &plan,
													unique_ptr<LogicalOperator>* &filter_out,
													unique_ptr<LogicalOperator>* &parent_out,
													float *threshold,
													string *vector_string,
													int &join_count) {
		auto current_child = &plan;
		unique_ptr<LogicalOperator>* parent = nullptr;

		while ((*current_child)->type != LogicalOperatorType::LOGICAL_GET) {
			if ((*current_child)->children.empty()) {
				return false;
			}

			if ((*current_child)->type == LogicalOperatorType::LOGICAL_FILTER) {
				auto &filter = (*current_child)->Cast<LogicalFilter>();
				for (auto &expr : filter.expressions) {
					if (expr->type == ExpressionType::COMPARE_LESSTHAN || expr->type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
						auto &compare_expr = expr->Cast<BoundComparisonExpression>();

						if (compare_expr.left->type == ExpressionType::BOUND_FUNCTION) {
							auto &func_expr = compare_expr.left->Cast<BoundFunctionExpression>();

							if (func_expr.function.name == "array_distance") {
								auto &right_expr = func_expr.children[1];
								*vector_string = "array_value(" + right_expr->Cast<BoundConstantExpression>().value.ToString().substr(1, right_expr->Cast<BoundConstantExpression>().value.ToString().size() - 2) + ")";

								*threshold = compare_expr.right->Cast<BoundConstantExpression>().value.GetValue<float>();

								filter_out = current_child;
								parent_out = parent;
								return true;
							}
						}
					}
				}
				parent = current_child;
				current_child = &(*current_child)->children[0];
			} else if ((*current_child)->children.size() == 2) {
				join_count++;
				auto left = &(*current_child)->children[0];
				auto right = &(*current_child)->children[1];
				bool left_found = findFilterOperatorWithArray_distance(context, *left, filter_out, parent_out, threshold, vector_string, join_count);
				bool right_found = findFilterOperatorWithArray_distance(context, *right, filter_out, parent_out, threshold, vector_string, join_count);
				return left_found || right_found;
			} else {
				parent = current_child;
				current_child = &(*current_child)->children[0];
			}
		}
		return false;
	}


	static bool TryOptimizeTopN(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}

		auto &top_n = op.Cast<LogicalTopN>();

		if (top_n.orders.size() != 1) {
			// We can only optimize if there is a single order by expression right now
			return false;
		}

		const auto &order = top_n.orders[0];

		if (order.type != OrderType::ASCENDING) {
			// We can only optimize if the order by expression is ascending
			return false;
		}

		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			// The expression has to reference the child operator (a projection with the distance function)
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		// find the expression that is referenced
		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			// The child has to be a projection
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();

		// This the expression that is referenced by the order by expression
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		// The projection must sit on top of a get
		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
		auto &get = get_ptr->Cast<LogicalGet>();
		// Check if the get is a table scan
		if (get.function.name != "seq_scan") {
			return false;
		}

		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			// Cant push down!
			return false;
		}

		// We have a top-n operator on top of a table scan
		// We can replace the function with a custom index scan (if the table has a custom index)

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			// We can only replace the scan if the table is a duck table
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		// Find the index
		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!hnsw_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!hnsw_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size = hnsw_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, hnsw_index, top_n.limit, std::move(query_vector));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		// If there are no table filters pushed down into the get, we can just replace the get with the index scan
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = HNSWIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);
		if (get.table_filters.filters.empty()) {

			// Remove the TopN operator
			plan = std::move(top_n.children[0]);
			return true;
		}

		// Otherwise, things get more complicated. We need to pullup the filters from the table scan as our index scan
		// does not support regular filter pushdown.
		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &column_ids = get.GetColumnIds();
		for (const auto &entry : get.table_filters.filters) {
			idx_t column_id = entry.first;
			auto &type = get.returned_types[column_id];
			bool found = false;
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (column_ids[i] == column_id) {
					column_id = i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InternalException("Could not find column id for filter");
			}
			auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
			new_filter->expressions.push_back(entry.second->ToExpression(*column));
		}
		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);

		// Remove the TopN operator
		plan = std::move(top_n.children[0]);
		return true;
	}	

	static bool TryOptimizeOrderBy(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		auto &order_by = plan->Cast<LogicalOrder>();
		const auto &order = order_by.orders[0];
		idx_t rows = 0;
		float threshold = -1.0f;
		idx_t table_rows = 0;
		string vector_string;

		if (order.type != OrderType::ASCENDING || order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		if (order_by.children.size() != 1 || order_by.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			return false;
		}

		auto &projection = order_by.children.front()->Cast<LogicalProjection>();
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		unique_ptr<LogicalOperator>* filter = nullptr;
		unique_ptr<LogicalOperator>* parent = nullptr;
		int join_count = 0;

		bool found_range_filter = findFilterOperatorWithArray_distance(context, plan, filter, parent, &threshold, &vector_string, join_count);

		if (!found_range_filter) {
			return false;
		}

		auto &get = filter->get()->children[0]->Cast<LogicalGet>(); 

		if (get.function.name != "seq_scan") {
			return false;
		}
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();


		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;
		idx_t ef_search = 64;
		Value hnsw_ef_search_opt;
		if (context.TryGetCurrentSetting("hnsw_ef_search", hnsw_ef_search_opt)) {
			if (!hnsw_ef_search_opt.IsNull() && hnsw_ef_search_opt.type() == LogicalType::BIGINT) {
				auto val = hnsw_ef_search_opt.GetValue<int64_t>();
				if (val > 0) {
					ef_search = static_cast<idx_t>(val);
				}
			}
		}
		
		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
			
			bindings.clear();
			if (projection_expr->type == ExpressionType::BOUND_FUNCTION) {
				auto &func_expr = projection_expr->Cast<BoundFunctionExpression>();
			} 

			if (!hnsw_index.TryMatchDistanceFunction(projection_expr, bindings)) {
				return false;
			}
			unique_ptr<Expression> index_expr;
			if (!hnsw_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
					return false;
				}
			}

			const auto vector_size = hnsw_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}
			if (threshold == -1.0f) {
				rows = hnsw_index.ScanOnly(query_vector.get(), ef_search, context);
			} else {
				rows = hnsw_index.ScanAndFilter(query_vector.get(), ef_search, context, threshold);
			}

			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, hnsw_index, ef_search, std::move(query_vector));
			return true;
		});


		if (!bind_data) {
			return false;
		}

		get.function = HNSWIndexScanFunction::GetFunction();		
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality = true;
		get.estimated_cardinality = rows * 5;
		get.bind_data = std::move(bind_data);
		
		plan = std::move(order_by.children[0]);
		JoinOrderOptimizer optimizer(context);
		plan = optimizer.Optimize(std::move(plan));
		BuildProbeSideOptimizer build_probe_side_optimizer(context, *plan);
		build_probe_side_optimizer.VisitOperator(*plan);
		return true;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a TopN operator
		auto &op = *plan;

		if (op.type == LogicalOperatorType::LOGICAL_TOP_N) {
			return TryOptimizeTopN(context, plan);
		} else if (op.type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			return TryOptimizeOrderBy(context, plan);
		} else {
			return false;
		}
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {

		auto ok = TryOptimize(context, plan);
		// Recursively optimize the children
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];

				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "hnsw_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];

						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							// The binding is not referenced
							// We can remove this expression. But positionality matters so just replace with int.
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_hnsw_scan = OptimizeChildren(input.context, plan);
		if (did_use_hnsw_scan) {
			MergeProjections(plan);
		}
	}
};

//-----------------------------------------------------------------------------
// Register
//-----------------------------------------------------------------------------
void HNSWModule::RegisterScanOptimizer(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(HNSWIndexScanOptimizer());
}

} // namespace duckdb