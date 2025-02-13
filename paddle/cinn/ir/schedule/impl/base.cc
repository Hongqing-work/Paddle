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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/common/enforce.h"
/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with CINN_IR_SCHEDULE_BEGIN, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)              \
  }                                                      \
  catch (const utils::ErrorHandler& err_handler) {       \
    PADDLE_THROW(::common::errors::InvalidArgument(      \
        err_handler.FormatErrorMessage(err_msg_level))); \
  }

using cinn::ir::stmt::BlockRef;
using cinn::ir::stmt::For;
using cinn::ir::stmt::Schedule;
using cinn::ir::stmt::StmtRef;

namespace cinn {
namespace ir {

void DyScheduleImpl::MergeBlocks() {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "MergeBlocks";
  std::ostringstream os;
  auto blocks = this->GetModule().GetBlocks();
  if (blocks.size() <= 1U) return;
  PADDLE_ENFORCE_EQ(
      blocks[0]->stmts().size(),
      1U,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<MergeBlocks>.\n"
          "[Error info] Block[0] of sched_module should have only one stmt!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));
  PADDLE_ENFORCE_EQ(
      blocks[0]->stmts().at(0).isa<Schedule>(),
      true,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<MergeBlocks>.\n"
          "[Error info] Block[0] of sched_module should be Block with only one "
          "stmt which is a Schedule stmt!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  std::vector<BlockRef> merged_block;
  merged_block.push_back(blocks[0]->stmts()[0].as<Schedule>()->body());
  VLOG(3) << "Before merging, blocks[0] is : " << blocks[0];
  for (int i = 1; i < blocks.size(); ++i) {
    std::vector<Schedule> root_schedules;
    stmt::Visit(
        blocks[i],
        [&](const StmtRef& stmt) {
          if (stmt.isa<Schedule>()) {
            Schedule cur_schedule = stmt.as<Schedule>();
            if (cur_schedule->iter_values().empty()) {
              root_schedules.push_back(cur_schedule);
            }
          }
        },
        [&](const StmtRef& stmt) {});
    PADDLE_ENFORCE_EQ(
        root_schedules.size(),
        1U,
        ::common::errors::InvalidArgument("Number of root block should be 1"));
    for (auto& it_schedule : root_schedules) {
      auto& block_body = it_schedule->body();
      merged_block.push_back(block_body);
    }
  }
  std::vector<StmtRef> merged_block_stmts;
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has \n" << block;
    for (const auto& stmt : block->stmts()) merged_block_stmts.push_back(stmt);
  }
  Schedule merged_schedule = blocks[0]->stmts()[0].as<Schedule>();
  merged_schedule->set_body(BlockRef{merged_block_stmts});
  blocks[0]->set_stmts({merged_schedule});
  VLOG(3) << "After merging, blocks[0] is : " << blocks[0];
  blocks.erase(blocks.begin() + 1, blocks.end());
  this->SetBlocks(blocks);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

bool DyScheduleImpl::HasSchedStmt(const std::string& sched_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "HasSchedStmt";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::HasSchedStmt(blocks, sched_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<stmt::For> DyScheduleImpl::GetLoops(
    const Schedule& target_sched) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetLoops";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetLoops(blocks, target_sched);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<stmt::For> DyScheduleImpl::GetLoops(
    const std::string& block_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetLoops";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetLoops(blocks, block_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Schedule> DyScheduleImpl::GetAllSchedStmts() const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetAllSchedStmts";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetAllSchedStmts(blocks);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Schedule> DyScheduleImpl::GetChildSchedStmts(
    const StmtRef& stmt) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetChildSchedStmts";
  std::ostringstream os;
  return analyzer::GetChildSchedStmts(stmt);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Schedule DyScheduleImpl::GetSchedStmt(const std::string& sched_name) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetSchedStmt";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetSchedStmt(blocks, sched_name);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::GetRootSchedStmt(const StmtRef& stmt) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetRootSchedStmt";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetRootSchedStmt(blocks, expr);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

DeviceAPI DyScheduleImpl::GetDeviceAPI() const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "GetDeviceAPI";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::GetDeviceAPI(blocks);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Annotate(const Expr& block,
                              const std::string& key,
                              const attr_t& value) {
  CINN_IR_SCHEDULE_BEGIN();

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<Annotate>.\n"
          "[Error info] Expr parameter 'block' must be a "
          "ScheduleBlockRealize!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<Annotate>.\n"
          "[Error info] Expr parameter 'block' must be a ScheduleBlockRealize "
          "with a defined ScheduleBlock!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  auto copied_block = ir::ir_utils::IRCopy(block);
  auto* schedule_block = copied_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  schedule_block->attrs.emplace(key, value);
  this->Replace(block, copied_block);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::Unannotate(Expr& block,
                                const std::string& ann_key) {  // NOLINT
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "Unannotate";

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<Unannotate>.\n"
          "[Error info] Expr parameter 'block' must be a "
          "ScheduleBlockRealize!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<Unannotate>.\n"
          "[Error info] Expr parameter 'block' must be a ScheduleBlockRealize "
          "with a defined ScheduleBlock!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  auto* schedule_block = block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  if (schedule_block->attrs.count(ann_key)) {
    schedule_block->attrs.erase(ann_key);
  } else {
    LOG(WARNING) << "[IRScheduleWarning] In the schedule primitive <"
                 << primitive
                 << ">, can't find annotation with key: " << ann_key;
    return;
  }
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                              const Expr& block_target) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CopyTransformAndLoopInfo";

  PADDLE_ENFORCE_NOT_NULL(
      block.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<CopyTransformAndLoopInfo>.\n"
          "[Error info] Expr parameter 'block' must be a "
          "ScheduleBlockRealize!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  PADDLE_ENFORCE_NOT_NULL(
      block_target.As<ir::ScheduleBlockRealize>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<CopyTransformAndLoopInfo>.\n"
          "[Error info] Expr parameter 'block_target' must be a "
          "ScheduleBlockRealize!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  auto blocks = this->GetModule().GetBlocks();

  PADDLE_ENFORCE_EQ(
      blocks.size(),
      1U,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<CopyTransformAndLoopInfo>.\n"
          "[Error info] Size of blocks of current module must be 1!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  auto expr = blocks[0];
  auto vars = block.As<ir::ScheduleBlockRealize>()
                  ->schedule_block.As<ir::ScheduleBlock>()
                  ->iter_vars;
  auto vars_target = block_target.As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ir::ScheduleBlock>()
                         ->iter_vars;
  auto old_iter_values = block.As<ir::ScheduleBlockRealize>()->iter_values;
  auto iter_values_target =
      block_target.As<ir::ScheduleBlockRealize>()->iter_values;
  std::vector<Expr> new_iter_values;
  for (int i = 0; i < vars.size() && i < vars_target.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        vars[i]->upper_bound.defined() && vars_target[i]->upper_bound.defined(),
        true,
        ::common::errors::InvalidArgument(
            "[IRScheduleError] An error occurred in the schedule primitive "
            "<CopyTransformAndLoopInfo>.\n"
            "[Error info] Upper bound of iter_vars in both Expr parameter "
            "'block' and Expr parameter 'block_target' must be defined!\n"
            "[Module info] The Module of current schedule is: %s.",
            sched_module_.GetBlocks()));

    if (vars[i]->upper_bound.is_constant() &&
        vars_target[i]->upper_bound.is_constant() &&
        vars[i]->upper_bound.get_constant() ==
            vars_target[i]->upper_bound.get_constant() &&
        !vars[i]->is_reduce_axis && !vars_target[i]->is_reduce_axis) {
      new_iter_values.push_back(iter_values_target[i]);
      VLOG(3) << "new_iter_values.push_back " << iter_values_target[i];
    } else {
      break;
    }
  }

  PADDLE_ENFORCE_EQ(
      !new_iter_values.empty(),
      true,
      ::common::errors::InvalidArgument([&]() -> std::string {
        std::ostringstream oss;
        oss << "[IRScheduleError] An error occurred in the schedule primitive <"
            << primitive << ">.\n"
            << "[Error info] Cannot CopyTransformAndLoopInfo since shape[0] of "
               "source and target is not equal! "
            << vars[0]->upper_bound << " vs " << vars_target[0]->upper_bound
            << "\n"
            << "[Module info] The Module of current schedule is: "
            << sched_module_.GetBlocks() << ".";
        return oss.str();
      }()));

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop =
        ir::ir_utils::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
          if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
          return x->as_var();
        });
  }

  PADDLE_ENFORCE_EQ(!used_target_loop_vars.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <CopyTransformAndLoopInfo>.\n"
                        "[Error info] Cannot CopyTransformAndLoopInfo since "
                        "there is no loop var in the new_iter_values!\n"
                        "[Module info] The Module of current schedule is: %s.",
                        sched_module_.GetBlocks()));

  std::vector<Expr> used_target_loops;
  auto expr_copy = ir::ir_utils::IRCopy(expr);
  for (auto& var : used_target_loop_vars) {
    auto find_loop_var = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr_copy,
        [&](const Expr* x) {
          return x->As<ir::For>() && x->As<ir::For>()->loop_var->name == var &&
                 Contains(*x, block_target);
        },
        true);
    PADDLE_ENFORCE_EQ(
        find_loop_var.size(),
        1U,
        ::common::errors::InvalidArgument(
            "[IRScheduleError] An error occurred in the schedule "
            "primitive <CopyTransformAndLoopInfo>.\n"
            "[Error info] Number of loop with iter_var which is "
            "used in ScheduleBlockRealize for indexing in "
            "Exprs[0] of sched_modules must be 1!\n"
            "[Module info] The Module of current schedule is: %s.",
            sched_module_.GetBlocks()));
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(
      used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
        return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
      });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        old_iter_values[i].as_var() != nullptr,
        true,
        ::common::errors::InvalidArgument([&]() -> std::string {
          std::ostringstream oss;
          oss << "[IRScheduleError] An error occurred in the "
                 "schedule primitive <"
              << primitive << ">.\n"
              << "[Error info] iter_vars[" << i
              << "] in Expr parameter 'block' must be vars!\n"
              << "[Module info] The Module of current schedule is: "
              << sched_module_.GetBlocks() << ".";
          return oss.str();
        }()));
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= static_cast<int>(old_iter_values.size())) {
    new_loop = ir::ir_utils::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    PADDLE_ENFORCE_EQ(
        old_iter_values[changed_loop_num].as_var() != nullptr,
        true,
        ::common::errors::InvalidArgument([&]() -> std::string {
          std::ostringstream oss;
          oss << "[IRScheduleError] An error occurred in the "
                 "schedule primitive <"
              << primitive << ">.\n"
              << "[Error info] iter_vars[" << changed_loop_num
              << "] in Expr parameter 'block' must be vars!\n"
              << "[Module info] The Module of current schedule is: "
              << sched_module_.GetBlocks() << ".";
          return oss.str();
        }()));

    auto old_var = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<ir::For>() &&
                 x->As<ir::For>()->loop_var->name == old_var->name &&
                 Contains(*x, block);
        },
        true);
    PADDLE_ENFORCE_EQ(
        find_partial_loop.size(),
        1U,
        ::common::errors::InvalidArgument([&]() -> std::string {
          std::ostringstream oss;
          oss << "[IRScheduleError] An error occurred in the schedule "
                 "primitive <"
              << primitive << ">.\n"
              << "[Error info] Number of loop with iter_var which is "
              << old_var->name << " should be 1 in Exprs[0] of sched_module!\n"
              << "[Module info] The Module of current schedule is: "
              << sched_module_.GetBlocks() << ".";
          return oss.str();
        }()));
    new_loop = ir::ir_utils::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop,
        [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); },
        true);
    PADDLE_ENFORCE_EQ(
        find_schedule_block.size(),
        1U,
        ::common::errors::InvalidArgument(
            "[IRScheduleError] An error occurred in the schedule "
            "primitive <CopyTransformAndLoopInfo>.\n"
            "[Error info] Number of ScheduleBlockRealize in "
            "partial_loop should be 1!\n"
            "[Module info] The Module of current schedule is: %s.",
            sched_module_.GetBlocks()));

    Expr sch_block = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;

  PADDLE_ENFORCE_EQ(
      !used_target_loops.empty(),
      true,
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<CopyTransformAndLoopInfo>.\n"
          "[Error info] Cannot CopyTransformAndLoopInfo since there is no loop "
          "which uses vars in the new_iter_values in Expr[0] of sched_module!\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  Expr res;
  if (used_target_loops.size() == 1) {
    auto for_loop = used_target_loops[0].As<ir::For>();
    res = For::Make(for_loop->loop_var,
                    for_loop->min,
                    for_loop->extent,
                    for_loop->for_type(),
                    for_loop->device_api,
                    new_loop,
                    for_loop->vectorize_info(),
                    for_loop->bind_info());
  } else {
    Expr outer_loop = used_target_loops.front();
    Expr inner_loop = used_target_loops.back();
    inner_loop.As<ir::For>()->body = Block::Make({new_loop});
    res = outer_loop;
  }
  VLOG(3) << "res is : " << res;
  std::vector<Expr> all_loops = this->GetLoops(block);

  PADDLE_ENFORCE_EQ(!all_loops.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <CopyTransformAndLoopInfo>.\n"
                        "[Error info] Cannot CopyTransformAndLoopInfo since "
                        "there is no loop in Expr parameter 'block'!\n"
                        "[Module info] The Module of current schedule is: %s.",
                        sched_module_.GetBlocks()));

  this->Replace(all_loops[0], res);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

void DyScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "CopyTransformAndLoopInfo";
  std::ostringstream os;
  auto block = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SampleCategorical";
  std::ostringstream os;

  PADDLE_ENFORCE_EQ(candidates.size(),
                    probs.size(),
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <SampleCategorical>.\n"
                        "[Error info] vector<int> params(candidates) and "
                        "vector<int> params(probs) must "
                        "have same size in SampleCategorical!\n"
                        "[Module info] The Module of current schedule is: %s.",
                        sched_module_.GetBlocks()));

  int seed_idx = utils::SampleDiscreteFromDistribution(probs, rand_seed);
  auto result = candidates[seed_idx];
  Expr result_expr(result);
  return result_expr;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

std::vector<Expr> DyScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "SamplePerfectTile";

  PADDLE_ENFORCE_NOT_NULL(
      loop.As<ir::For>(),
      ::common::errors::InvalidArgument(
          "[IRScheduleError] An error occurred in the schedule primitive "
          "<SampleCategorical>.\n"
          "[Error info] Expr parameter 'loop' should be a For loop.\n"
          "[Module info] The Module of current schedule is: %s.",
          sched_module_.GetBlocks()));

  PADDLE_ENFORCE_GE(n,
                    2,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <SamplePerfectTile>.\n"
                        "[Error info] The number of tile factors (n) should be "
                        "at least 2, but got %d.\n"
                        "[Module info] The Module of current schedule is: %s.",
                        n,
                        sched_module_.GetBlocks()));

  PADDLE_ENFORCE_GE(max_innermost_factor,
                    1,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <SamplePerfectTile>.\n"
                        "[Error info] The max innermost factor should be at "
                        "least 1, but got %d.\n"
                        "[Module info] The Module of current schedule is: %s.",
                        max_innermost_factor,
                        sched_module_.GetBlocks()));

  PADDLE_ENFORCE_EQ(cinn::common::is_zero(loop.As<ir::For>()->min),
                    true,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <SamplePerfectTile>.\n"
                        "[Error info] The For loop should start from 0.\n"
                        "[Module info] The Module of current schedule is: %s.",
                        sched_module_.GetBlocks()));

  int loop_extent = GetLoopExtent(loop);
  std::vector<int> innermost_factors;
  for (int i = max_innermost_factor; i >= 1; --i) {
    if (loop_extent % i == 0) {
      innermost_factors.push_back(i);
    }
  }

  // 检查是否找到合适的 innermost_factor
  PADDLE_ENFORCE_EQ(!innermost_factors.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "[IRScheduleError] An error occurred in the schedule "
                        "primitive <SamplePerfectTile>.\n"
                        "[Error info] No innermost factor found for loop "
                        "extent %d with max_innermost_factor %d.\n"
                        "[Module info] The Module of current schedule is: %s.",
                        loop_extent,
                        max_innermost_factor,
                        sched_module_.GetBlocks()));

  int innermost_factor = innermost_factors[utils::SampleUniformInt(
      0, innermost_factors.size(), rand_seed)];
  auto result = SampleTile(rand_seed, n - 1, loop_extent / innermost_factor);
  std::vector<Expr> result_expr;
  for (auto& factor : result) {
    result_expr.push_back(Expr(factor));
  }
  result_expr.push_back(Expr(innermost_factor));
  return result_expr;
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

Expr DyScheduleImpl::AddUnitLoop(const Expr& block) const {
  CINN_IR_SCHEDULE_BEGIN();
  std::string primitive = "AddUnitLoop";
  std::ostringstream os;
  auto blocks = sched_module_.GetBlocks();
  return analyzer::AddUnitLoop(blocks, block);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
}

}  // namespace ir
}  // namespace cinn
