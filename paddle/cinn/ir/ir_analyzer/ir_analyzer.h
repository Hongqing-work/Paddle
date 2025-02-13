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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {
namespace analyzer {

bool HasSchedStmt(const std::vector<stmt::BlockRef>& root_blocks,
                  const std::string& sched_name);

std::vector<stmt::For> GetLoops(const std::vector<stmt::BlockRef>& root_blocks,
                                const std::string& sched_name);

std::vector<stmt::For> GetLoops(const std::vector<stmt::BlockRef>& root_blocks,
                                const stmt::Schedule& target_sched);

std::vector<stmt::Schedule> GetAllSchedStmts(
    const std::vector<stmt::BlockRef>& root_blocks);

std::vector<stmt::Schedule> GetChildSchedStmts(const stmt::StmtRef& stmt);

stmt::Schedule GetSchedStmt(const std::vector<stmt::BlockRef>& root_blocks,
                            const std::string& sched_name);

/**
 * Get the root schedule stmt (i.e. Schedule(root)) from `root_block`.
 * The `root_block` must be the root block of ScheduleModule.
 */
stmt::Schedule GetRootSchedStmt(const stmt::BlockRef& root_block);

stmt::Schedule GetRootSchedStmt(const std::vector<stmt::BlockRef>& root_blocks,
                                const stmt::StmtRef& stmt);

DeviceAPI GetDeviceAPI(const std::vector<stmt::BlockRef>& root_blocks);

stmt::For AddUnitLoop(const std::vector<stmt::BlockRef>& root_blocks,
                      const stmt::Schedule& target_sched);

stmt::Store GetStoreOfSchedStmt(const stmt::Schedule& target_sched);

Tensor GetStoreTensorOfSchedStmt(const stmt::Schedule& target_sched);

std::unordered_map<std::string, std::unordered_map<ir::Var, stmt::For>>
CollectVarToForMap(const std::vector<stmt::BlockRef>& root_blocks,
                   const std::vector<stmt::Schedule>& schedules);

std::unordered_map<ir::Var, ir::Expr> GetIterVarToValueOfSchedStmt(
    const stmt::Schedule& target_sched);

template <typename T>
T ReplaceVarWithExpr(const T& source,
                     const std::vector<ir::Var>& candidates,
                     const std::vector<ir::Expr>& targets);

/**
 * Expand the iter_vars in `stmt` to the iter_values of `iter_info_sched`.
 */
template <typename T>
T ExpandIterVar(const T& source, const stmt::Schedule& iter_info_sched);

constexpr char* kLoopVar = "loop_var_";

/**
 * Replace the loop_vars in `source` to the canonicalized form such that the
 * loop_var of loop[i] has name `loop_var_i`.
 */
template <typename T>
T CanonicalizeLoopVar(const T& source, const std::vector<stmt::For>& loops);

std::unordered_set<ir::Var> GetReduceIterVars(
    const stmt::Schedule& target_sched);

bool IsReductionSchedStmt(const stmt::Schedule& target_sched);

bool IsBroadcastSchedStmt(const stmt::Schedule& target_sched);

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices);

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
