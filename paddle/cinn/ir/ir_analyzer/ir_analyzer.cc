// Copyright (c) 2023 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/expr_visitors.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/schedule/schedule_desc.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/platform/enforce.h"
namespace cinn {
namespace ir {
namespace analyzer {

using cinn::ir::stmt::BlockRef;
using cinn::ir::stmt::Schedule;
using cinn::ir::stmt::StmtRef;

bool HasSchedStmt(const std::vector<BlockRef>& root_blocks,
                  const std::string& sched_name) {
  for (auto& it_block : root_blocks) {
    const auto& find_res = FindSchedStmt(it_block, sched_name);
    if (!find_res.empty()) {
      PADDLE_ENFORCE_EQ(
          find_res.size(),
          1U,
          ::common::errors::InvalidArgument(
              "There should not be more than 1 Schedule stmt with "
              "identical name!"));
      return true;
    }
  }
  return false;
}

std::vector<stmt::For> GetLoops(const std::vector<BlockRef>& root_blocks,
                                const std::string& sched_name) {
  Schedule target_sched = GetSchedStmt(root_blocks, sched_name);
  return GetLoops(root_blocks, target_sched);
}

std::vector<stmt::For> GetLoops(const std::vector<BlockRef>& root_blocks,
                                const Schedule& target_sched) {
  std::vector<stmt::For> result;
  std::string sched_name = target_sched->name();

  for (auto& it_block : root_blocks) {
    const auto& find_loops = FindParentLoops(it_block, target_sched);
    if (!find_loops.empty()) {
      if (!result.empty()) {
        std::stringstream ss;
        ss << "Find block with name: \n"
           << sched_name << " appeared in more than one AST!";
        PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      }
      result = find_loops;
    }
  }

  return result;
}

std::vector<Schedule> GetAllSchedStmts(
    const std::vector<BlockRef>& root_blocks) {
  std::vector<Schedule> result;
  for (auto& it_block : root_blocks) {
    const auto& find_res = FindSchedStmt(it_block);
    result.insert(result.end(), find_res.begin(), find_res.end());
  }
  PADDLE_ENFORCE_EQ(result.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "Didn't find any Schedule stmt in root_blocks."));
  return result;
}

std::vector<Schedule> GetChildSchedStmts(const StmtRef& stmt) {
  PADDLE_ENFORCE_EQ(
      stmt.isa<For>() || stmt.isa<Schedule>(),
      true,
      ::common::errors::InvalidArgument(
          "The stmt must be convertible to either Schedule or For."));
  BlockRef block = stmt->block_fields()[0];
  return FindSchedStmt(block);
}

Schedule GetSchedStmt(const std::vector<BlockRef>& root_blocks,
                      const std::string& sched_name) {
  for (auto& it_block : root_blocks) {
    const auto& find_res = FindSchedStmt(it_block, sched_name);
    if (!find_res.empty()) {
      PADDLE_ENFORCE_EQ(
          find_res.size(),
          1U,
          ::common::errors::InvalidArgument(
              "There should not be more than 1 Schedule stmt with "
              "identical name!"));
      return find_res[0];
    }
  }
  std::stringstream ss;
  ss << "Didn't find a block with name " << sched_name
     << " in this ScheduleModule!";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

Schedule GetRootSchedStmt(const BlockRef& root_block) {
  PADDLE_ENFORCE_EQ(root_block->stmts().size(),
                    1U,
                    ::common::errors::InvalidArgument(
                        "The root block must have exactly one stmt."));
  PADDLE_ENFORCE_EQ((root_block->stmts()[0]).isa<Schedule>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The first stmt in the block must be Schedule."));
  Schedule res = root_block->stmts()[0].as<Schedule>();
  return res;
}

Schedule GetRootSchedStmt(const std::vector<BlockRef>& root_blocks,
                          const StmtRef& stmt) {
  for (auto& it_block : root_blocks) {
    BlockRef cur_block = stmt->GetParentBlockRef();
    while (cur_block.defined() && cur_block->GetParentStmtRef().defined()) {
      cur_block = cur_block->GetParentStmtRef()->GetParentBlockRef();
    }
    if (cur_block == it_block) {
      return GetRootSchedStmt(it_block);
    }
  }
  std::stringstream ss;
  ss << "Didn't find stmt \n" << stmt << "when GetRootSchedule.";
  PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
}

DeviceAPI GetDeviceAPI(const std::vector<BlockRef>& root_blocks) {
  DeviceAPI res;
  bool find_for_stmt = false;
  stmt::InterruptibleVisit(
      root_blocks.front(),
      [&](const StmtRef& stmt) -> stmt::VisitResult {
        if (stmt.isa<stmt::For>()) {
          find_for_stmt = true;
          res = stmt.as<stmt::For>()->device_api();
          return stmt::VisitResult::interrupt();
        }
        return stmt::VisitResult::advance();
      },
      [&](const StmtRef& stmt) -> stmt::VisitResult {
        return stmt::VisitResult::advance();
      });
  PADDLE_ENFORCE_EQ(
      find_for_stmt,
      true,
      ::common::errors::InvalidArgument(
          "GetDeviceAPI failed. Didn't find any For stmt in the root_blocks."));
  return res;
}

stmt::For AddUnitLoop(const std::vector<BlockRef>& root_blocks,
                      const Schedule& target_sched) {
  std::string sched_name = target_sched->name();
  BlockRef parent_block = target_sched->GetParentBlockRef();
  std::vector<StmtRef> new_stmts;
  stmt::For res;

  for (const auto& stmt : parent_block->stmts()) {
    if (stmt.isa<Schedule>() && stmt.as<Schedule>()->name() == sched_name) {
      res = stmt::For(ir::Var(cinn::common::UniqName("ix")),
                      ir::Expr(0),
                      ir::Expr(1),
                      ir::ForType::Serial,
                      ir::DeviceAPI::UNK,
                      BlockRef({stmt}));
      new_stmts.push_back(res);
    } else {
      new_stmts.push_back(stmt);
    }
  }
  parent_block->set_stmts(new_stmts);

  return res;
}

stmt::Store GetStoreOfSchedStmt(const Schedule& target_sched) {
  stmt::Store res;
  stmt::InterruptibleVisit(
      target_sched,
      [&](const StmtRef& stmt) {
        if (stmt.isa<stmt::Store>()) {
          res = stmt.as<stmt::Store>();
          return stmt::VisitResult::interrupt();
        }
        return stmt::VisitResult::advance();
      },
      [&](const StmtRef& stmt) { return stmt::VisitResult::advance(); });
  return res;
}

Tensor GetStoreTensorOfSchedStmt(const Schedule& target_sched) {
  stmt::Store find_store = GetStoreOfSchedStmt(target_sched);
  PADDLE_ENFORCE_NOT_NULL(
      find_store->tensor().as_tensor(),
      ::common::errors::InvalidArgument(
          "The tensor must be convertible to Tensor type."));
  return find_store->tensor().as_tensor_ref();
}

std::unordered_map<std::string, std::unordered_map<ir::Var, stmt::For>>
CollectVarToForMap(const std::vector<BlockRef>& root_blocks,
                   const std::vector<Schedule>& schedules) {
  std::unordered_map<std::string, std::unordered_map<ir::Var, stmt::For>>
      for_map;
  for (const Schedule& schedule : schedules) {
    std::string sched_name = schedule->name();
    std::vector<stmt::For> for_stmts = GetLoops(root_blocks, schedule);
    for (const auto& for_stmt : for_stmts) {
      for_map[sched_name][for_stmt->loop_var()] = for_stmt;
      VLOG(6) << "for_map.insert: <" << sched_name << ", "
              << for_stmt->loop_var()->name << ">";
    }
  }
  return for_map;
}

std::unordered_map<ir::Var, ir::Expr> GetIterVarToValueOfSchedStmt(
    const Schedule& target_sched) {
  PADDLE_ENFORCE_EQ(
      target_sched->iter_values().size(),
      target_sched->iter_vars().size(),
      ::common::errors::InvalidArgument(
          "The size of iter_values should be equal to the size of "
          "iter_vars in the Schedule stmt!"));
  std::unordered_map<ir::Var, ir::Expr> iter_var2iter_values;
  for (size_t i = 0; i < target_sched->iter_values().size(); ++i) {
    iter_var2iter_values.emplace(target_sched->iter_vars()[i],
                                 target_sched->iter_values()[i]);
  }
  return iter_var2iter_values;
}

template <>
ir::Expr ReplaceVarWithExpr(const ir::Expr& source,
                            const std::vector<ir::Var>& candidates,
                            const std::vector<ir::Expr>& targets) {
  PADDLE_ENFORCE_EQ(
      candidates.size(),
      targets.size(),
      ::common::errors::InvalidArgument(
          "In ReplaceExpr, the size of Vars to be replaces must "
          "be equal to the size of targets Exprs! Please check."));
  ir::Expr copied = ir::ir_utils::IRCopy(source);
  if (candidates.empty()) return copied;
  std::map<Var, Expr, CompVar> replacing_map;
  for (int i = 0; i < candidates.size(); ++i) {
    // If the Var to be candidates is equal to the candidate, we skip it.
    if (targets[i].is_var() && targets[i].as_var_ref() == candidates[i])
      continue;
    replacing_map[candidates[i]] = targets[i];
  }
  MappingVarToExprMutator mapper(replacing_map);
  mapper(&copied);
  return copied;
}

template <>
StmtRef ReplaceVarWithExpr(const StmtRef& source,
                           const std::vector<ir::Var>& candidates,
                           const std::vector<ir::Expr>& targets) {
  StmtRef copied = source;  // stmt does not need deep copy.
  const auto& ReplaceInStmt = [&](StmtRef stmt) {
    switch (stmt->stmt_type()) {
#define __(stmt__)                                      \
  case StmtNodeTy::stmt__:                              \
    MutateExpr(stmt.as<stmt::stmt__>(), [&](Expr* e) {  \
      *e = ReplaceVarWithExpr(*e, candidates, targets); \
    });                                                 \
    break;
      NODETY_FORALL_STMT(__)
#undef __
      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Deadcode, not supported StmtNodeTy"));
    }
  };
  stmt::Mutate(
      copied, [&](StmtRef stmt) { ReplaceInStmt(stmt); }, [&](StmtRef stmt) {});
  return copied;
}

template <typename T>
T ExpandIterVar(const T& source, const Schedule& iter_info_sched) {
  return ReplaceVarWithExpr<T>(
      source, iter_info_sched->iter_vars(), iter_info_sched->iter_values());
}
template Expr ExpandIterVar(const Expr& expr, const Schedule& iter_info_sched);
template StmtRef ExpandIterVar(const StmtRef& stmt,
                               const Schedule& iter_info_sched);

template <typename T>
T CanonicalizeLoopVar(const T& source, const std::vector<stmt::For>& loops) {
  std::vector<ir::Var> loop_vars;
  std::vector<ir::Expr> new_loop_vars;
  for (int i = 0; i < loops.size(); i++) {
    const auto& loop_var = loops[i]->loop_var();
    loop_vars.push_back(loop_var);

    ir::Var new_loop_var = ir::ir_utils::IRCopy(loop_var);
    new_loop_var->name = kLoopVar + std::to_string(i);
    new_loop_vars.push_back(new_loop_var);
  }
  return ReplaceVarWithExpr<T>(source, loop_vars, new_loop_vars);
}

template Expr CanonicalizeLoopVar(const Expr& source,
                                  const std::vector<stmt::For>& loops);
template StmtRef CanonicalizeLoopVar(const StmtRef& source,
                                     const std::vector<stmt::For>& loops);

std::unordered_set<ir::Var> GetReduceIterVars(const Schedule& target_sched) {
  const std::vector<ir::Var>& iter_vars = target_sched->iter_vars();
  std::unordered_set<ir::Var> reduce_vars;
  for (int i = 0; i < iter_vars.size(); ++i) {
    if (iter_vars[i]->is_reduce_axis) {
      reduce_vars.insert(iter_vars[i]);
    }
  }
  return reduce_vars;
}

bool IsReductionSchedStmt(const stmt::Schedule& target_sched) {
  for (const ir::Var& var : target_sched->iter_vars()) {
    if (var->is_reduce_axis) {
      return true;
    }
  }
  return false;
}

bool IsBroadcastSchedStmt(const stmt::Schedule& target_sched) {
  stmt::Store store_stmt = GetStoreOfSchedStmt(target_sched);
  const ir::Load* load = store_stmt->value().As<ir::Load>();
  if (load == nullptr) {
    return false;
  }
  // each load index can be found in store index and maintain relative order
  const auto IsIndexZero = [](const ir::Expr& e) -> bool {
    return e.is_constant() && e.get_constant() == 0;
  };
  int num_load_index_zero = 0;
  for (size_t i = 0; i < load->indices.size(); ++i) {
    if (IsIndexZero(load->indices[i]) && i < store_stmt->indices().size() &&
        !IsIndexZero(store_stmt->indices()[i])) {
      ++num_load_index_zero;
      continue;
    }
    bool found = false;
    for (size_t j = i; j < store_stmt->indices().size(); ++j) {
      const ir::_Var_* load_var = load->indices[i].as_var();
      const ir::_Var_* store_var = store_stmt->indices()[j].as_var();
      if (load_var == nullptr || store_var == nullptr) {
        return false;
      }
      if (load_var->name == store_var->name) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return load->indices.size() - num_load_index_zero <
         store_stmt->indices().size();
}

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> result;
  for (const ir::Expr& e : indices) {
    if (e.is_constant()) {
      std::string var_name =
          cinn::UniqName("constant" + static_cast<int>(e.get_constant()));
      result.emplace_back(e, e, var_name, /* is_reduce = */ false);
    } else if (e.As<ir::_Var_>() != nullptr) {
      ir::Expr copy_e = ir::ir_utils::IRCopy(e);
      ir::_Var_* var_ref = copy_e.As<ir::_Var_>();
      result.emplace_back(ir::Var(var_ref));
    } else {
      std::string var_name = cinn::UniqName("expr");
      common::cas_intervals_t var_intervals;
      bool is_reduce = false;
      ir::ir_utils::CollectIRNodes(e, [&](const ir::Expr* x) {
        if (x->As<ir::_Var_>() != nullptr) {
          ir::Var var = x->as_var_ref();
          var_intervals.insert(
              {var->name,
               common::CasInterval{var->lower_bound, var->upper_bound}});
          if (var->is_reduce_axis) is_reduce = true;
        }
        return false;
      });
      common::SymbolicExprAnalyzer analyzer(var_intervals);
      result.emplace_back(
          analyzer.LowerBound(e), analyzer.UpperBound(e), var_name, is_reduce);
    }
  }
  return result;
}

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
