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

#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/pass/pass_manager.h"

using cinn::ir::stmt::BlockRef;
using cinn::ir::stmt::StmtRef;
namespace cinn {
namespace ir {
namespace {
bool ReplaceRecursive(BlockRef block, const StmtRef& src, const StmtRef& tgt) {
  std::vector<StmtRef> new_stmts;
  bool replaced = false;
  for (const auto& stmt : block->stmts()) {
    if (stmt == src) {
      new_stmts.push_back(tgt);
      replaced = true;
    } else {
      new_stmts.push_back(stmt);
    }
    if (!replaced) {
      for (BlockRef child_block : stmt->block_fields()) {
        if (ReplaceRecursive(child_block, src, tgt)) {
          replaced = true;
          break;
        }
      }
    }
    if (replaced) break;
  }
  block->set_stmts(new_stmts);
  return replaced;
}
}  // namespace

/**
 * Replace a stmt to another stmt.
 * @param src_stmt The stmt to be changed.
 * @param tgt_stmt The stmt we want.
 */
void ScheduleBase::Replace(const StmtRef& src_stmt, const StmtRef& tgt_stmt) {
  std::vector<BlockRef> root_blocks = sched_module_.GetBlocks();
  for (BlockRef& block : root_blocks) {
    ReplaceRecursive(block, src_stmt, tgt_stmt);
  }
}

}  // namespace ir
}  // namespace cinn
