import functools
import logging
from collections import Counter
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import code_execution
import mutmut
import numpy as np
from code_execution.utils import swallow_io

logger = logging.getLogger(__name__)
mutmut.mutations_by_type.pop("expr_stmt", None)

MAX_RERUNS = 20


def mutate(context: mutmut.Context) -> Tuple[str, int]:
    """
    :return: tuple of mutated source code and number of mutations performed
    """
    try:
        result = mutmut.parse(context.source, error_recovery=False)
    except Exception:
        print(
            "Failed to parse {}. Internal error from parso follows.".format(
                context.filename
            )
        )
        print("----------------------------------")
        raise
    mutate_list_of_nodes(result, context=context)
    mutated_source = result.get_code().replace(" not not ", " ")
    if context.remove_newline_at_end:
        assert mutated_source[-1] == "\n"
        mutated_source = mutated_source[:-1]

    # If we said we mutated the code, check that it has actually changed
    if context.performed_mutation_ids:
        if context.source == mutated_source:
            raise RuntimeError(
                "Mutation context states that a mutation occurred but the "
                "mutated source remains the same as original"
            )
    context.mutated_source = mutated_source
    return mutated_source, len(context.performed_mutation_ids)


def mutate_node(node, context: mutmut.Context):
    context.stack.append(node)
    try:
        if node.type in ("tfpdef", "import_from", "import_name"):
            return

        if (
            node.type == "atom_expr"
            and node.children
            and node.children[0].type == "name"
            and node.children[0].value == "__import__"
        ):
            return

        if node.start_pos[0] - 1 != context.current_line_index:
            context.current_line_index = node.start_pos[0] - 1
            context.index = (
                0  # indexes are unique per line, so start over here!
            )

        if node.type == "expr_stmt":
            if (
                node.children[0].type == "name"
                and node.children[0].value.startswith("__")
                and node.children[0].value.endswith("__")
            ):
                if node.children[0].value[2:-2] in mutmut.dunder_whitelist:
                    return

        # Avoid mutating pure annotations
        if node.type == "annassign" and len(node.children) == 2:
            return

        if hasattr(node, "children"):
            mutate_list_of_nodes(node, context=context)

            # this is just an optimization to stop early
            if (
                context.performed_mutation_ids
                and context.mutation_id != mutmut.ALL
            ):
                return

        mutation = mutmut.mutations_by_type.get(node.type)

        if mutation is None:
            return

        for key, value in sorted(mutation.items()):
            old = getattr(node, key)
            if context.exclude_line():
                continue

            new = value(
                context=context,
                node=node,
                value=getattr(node, "value", None),
                children=getattr(node, "children", None),
            )

            if isinstance(new, list) and not isinstance(old, list):
                # multiple mutations
                new_list = new
            else:
                # one mutation
                new_list = [new]

            # go through the alternate mutations in reverse as they may have
            # adverse effects on subsequent mutations, this ensures the last
            # mutation applied is the original/default/legacy mutmut mutation
            for new in reversed(new_list):
                assert not callable(new)
                if new is not None and new != old:
                    if hasattr(mutmut.mutmut_config, "pre_mutation_ast"):
                        mutmut.mutmut_config.pre_mutation_ast(context=context)
                    if context.should_mutate(node):
                        old_node_info = (
                            node.start_pos,
                            node.end_pos,
                            node.get_code(),
                        )
                        mut_id = context.mutation_id_of_current_index

                        setattr(node, key, new)

                        context.performed_mutation_ids.append(
                            ((*old_node_info, node.get_code()), mut_id)
                        )
                    context.index += 1
                # this is just an optimization to stop early
                if (
                    context.performed_mutation_ids
                    and context.mutation_id != mutmut.ALL
                ):
                    return
    finally:
        context.stack.pop()


def mutate_list_of_nodes(node, context: mutmut.Context):
    return_annotation_started = False

    for child_node in node.children:
        if child_node.type == "operator" and child_node.value == "->":
            return_annotation_started = True

        if (
            return_annotation_started
            and child_node.type == "operator"
            and child_node.value == ":"
        ):
            return_annotation_started = False

        if return_annotation_started:
            continue

        mutate_node(child_node, context=context)

        # this is just an optimization to stop early
        if context.performed_mutation_ids and context.mutation_id != mutmut.ALL:
            return


def list_mutations(context: mutmut.Context):
    assert context.mutation_id == mutmut.ALL
    mutate(context)
    return context.performed_mutation_ids


def sample_one_mutant(
    mutations,
    prog,
    rng,
    max_mutations_per_program=5,
    allow_same_line_mutations=False,
):
    num_sample = 1
    if len(mutations) > 1:
        num_sample = rng.choice(
            range(1, min(len(mutations) + 1, max_mutations_per_program + 1)), 1
        )[0]

    if not allow_same_line_mutations:
        to_use = []
        used_lines = set()
        idx_pool = list(range(len(mutations)))
        while idx_pool and len(to_use) < num_sample:

            new_mut_idx = rng.choice(idx_pool, 1)[0]
            new_mut = mutations[new_mut_idx]
            to_use.append(new_mut_idx)
            used_lines.add(new_mut[1].line_number)
            idx_pool = list(
                filter(
                    lambda x: mutations[x][1].line_number not in used_lines,
                    idx_pool,
                )
            )

    else:
        to_use = rng.choice(
            range(len(mutations)),
            size=num_sample,
            replace=False,
        )
    changes = []
    out = prog
    for i in sorted(to_use):
        _, mut_id = mutations[i]

        ctx = mutmut.Context(source=out, mutation_id=mut_id, index=mut_id.index)
        try:
            with swallow_io():
                new_mutant, _ = mutate(ctx)
        except Exception as e:
            continue
        if not ctx.performed_mutation_ids:
            continue

        node, _ = ctx.performed_mutation_ids[0]
        out = new_mutant
        changes.append(
            {
                "start": list(node[0]),
                "end": list(node[1]),
                "new": ctx.mutated_source,
                "line_number": mut_id.line_number,
                "line": mut_id.line,
                "index": mut_id.index,
            }
        )
    return out, changes


def get_mutations(
    program,
    rng,
    num_sets=5,
    max_mutations_per_program=5,
    allow_same_line_mutations=False,
    allow_invalid_syntax=False,
) -> Tuple[str, Counter]:
    """Returns the list of"""
    ctx = mutmut.Context(source=program["solution"])

    mutations = list_mutations(ctx)

    if not mutations:
        return {
            "program_id": program["program_id"],
            "problem_id": program["problem_id"],
            "mutants": [],
            "mutant_changes": [],
        }
    try:
        mutants = []
        mutant_changes = []
        retries = 0
        while len(mutants) < num_sets and retries < MAX_RERUNS:
            mutated, changes = sample_one_mutant(
                mutations,
                program["solution"],
                rng,
                max_mutations_per_program,
                allow_same_line_mutations,
            )
            with swallow_io():
                if (
                    not allow_invalid_syntax
                    and code_execution.safe_ast_parse(mutated) is None
                ):
                    retries += 1
                    continue

            if mutated in mutants or mutated == program["solution"]:
                retries += 1
                continue
            mutants.append(mutated)
            mutant_changes.append(changes)
    except:
        return {
            "program_id": program["program_id"],
            "problem_id": program["problem_id"],
            "mutants": [],
            "mutant_changes": [],
        }

    return {
        "program_id": program["program_id"],
        "problem_id": program["problem_id"],
        "mutants": mutants,
        "mutant_changes": mutant_changes,
    }


def _mutate_program(
    program: Dict,
    rng,
    max_mutations_per_program: int,
    allow_invalid_syntax: bool,
    allow_same_line_mutations: bool,
    to_return: int = 1,
    num_attempts: int = 1,
) -> List[Dict]:
    try:
        error = None
        out = []
        ctx = mutmut.Context(source=program["code"])

        mutations = list_mutations(ctx)
        while num_attempts > 0 and len(out) < to_return:
            mutated, changes = sample_one_mutant(
                mutations=mutations,
                prog=program["code"],
                rng=rng,
                max_mutations_per_program=max_mutations_per_program,
                allow_same_line_mutations=allow_same_line_mutations,
            )
            with swallow_io():
                if (
                    not allow_invalid_syntax
                    and code_execution.safe_ast_parse(mutated) is None
                ):
                    num_attempts -= 1
                    continue

            if (
                mutated in program["prior_mutants"]
                or mutated == program["code"]
                or mutated in {o["mutant"] for o in out}
            ):
                num_attempts -= 1
                continue

            out.append(
                {
                    "mutant": mutated,
                    "changes": changes,
                }
            )
    except Exception as e:
        error = str(e)
    return program["id"], out, error


def create_mutations(
    seed: int,
    all_problems: List[Dict],
    program_executor: Callable[[List[Dict], List[Dict], int], List[Dict]],
    target_mutants: int,
    mutations_per: int,
    num_workers: int,
    max_tries: int,
    per_round: int,
    round_attempts: int = 10,
    max_rounds: int = 50,
    allow_same_line_mutations: bool = False,
    allow_invalid_syntax: bool = False,
):
    logger.info("Making list of programs to try to mutate")
    logger.info(f"{mutations_per=}")
    logger.info(f"{target_mutants=}")

    prog_list = {}
    for idx, problem in enumerate(all_problems):
        for p_idx, pred in enumerate(problem["predictions"]):
            if pred["passed"]:
                prog_list[(idx, p_idx)] = {
                    "id": (idx, p_idx),
                    "code": pred["code"],
                    "prior_mutants": set(),
                }

    logger.info(f"Initial list of programs to mutate: {len(prog_list):,}")
    rng = np.random.default_rng(seed)
    valid_mutants = defaultdict(list)
    try_count = Counter()
    num_rounds = 1
    while num_rounds <= max_rounds and len(prog_list) > 0:
        logger.info(
            f"Starting mutation round {num_rounds} with {len(prog_list)} programs"
        )
        mutation_fn = functools.partial(
            _mutate_program,
            rng=rng,
            max_mutations_per_program=mutations_per,
            allow_invalid_syntax=allow_invalid_syntax,
            allow_same_line_mutations=allow_same_line_mutations,
            to_return=per_round,
            num_attempts=round_attempts,
        )
        raw_mutants = code_execution.run_in_parallel(
            mutation_fn,
            prog_list.values(),
            num_workers=num_workers,
        )
        mutants = []
        to_remove = set()
        while raw_mutants:
            pid, sampled_mutants, error = raw_mutants.pop()
            if error:
                logger.error(f"Error mutating program {pid}: {error}")
                to_remove.add(pid)
                continue
            if len(sampled_mutants) > 0:
                for m in sampled_mutants:
                    mutants.append({"pid": pid, **m})
            else:
                try_count[pid] += 1
                if try_count[pid] >= max_tries:
                    to_remove.add(pid)
        logger.debug(f"Programs at max tries: {to_remove}")

        mutants = program_executor(
            mutants=mutants,
        )

        logger.info(f"Filtering {len(mutants):,} mutants to find failures")
        valid_mutants_found = 0
        for res in mutants:
            prog_list[res["pid"]]["prior_mutants"].add(res["mutant"])
            if res["passed"]:
                try_count[res["pid"]] += 1
                if try_count[res["pid"]] >= max_tries:
                    to_remove.add(res["pid"])
            else:
                valid_mutants[res["pid"]].append(
                    (res["mutant"], res["changes"])
                )

                valid_mutants_found += 1
                if len(valid_mutants[res["pid"]]) >= target_mutants:
                    to_remove.add(res["pid"])
        logger.info(
            f"Found {valid_mutants_found:,} valid mutants, "
            f"{len(to_remove):,} programs to remove"
        )
        for i in to_remove:
            prog_list.pop(i)
        num_rounds += 1

    out = defaultdict(lambda: {"original_pred": [], "mutants": []})
    for (prob_id, pred_id), mutants in valid_mutants.items():
        for m, c in mutants:
            out[prob_id]["mutants"].append(m)
            out[prob_id]["original_pred"].append(pred_id)

    return out
