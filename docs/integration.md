# Integration

AOP keeps provider work isolated from the main worktree. Checkpoint records a completed task's
changes, and integration rebases those task commits onto the branch currently checked out in the
main worktree.

## Checkpoint a task

```sh
aop checkpoint task-a -m "Implement parser"
```

`checkpoint` commits all tracked, staged, and untracked changes in the task worktree. It refuses
unresolved conflicts, whitespace errors, empty changes, missing Git author identity, and a task
with an active AOP run. The checkpoint record includes the task base, parent commits, resulting
commit, and successful AOP run IDs associated with the task.

## Integrate task commits

```sh
aop integrate task-a
```

`integrate` rebases exactly the task commits onto current main. AOP owns the mechanical Git
operations: starting and continuing the rebase, recording final validation edits, and
fast-forwarding main. It verifies the recorded base, linear history, clean starting state, unchanged
main branch, and fast-forward ancestry.

When a commit conflicts, AOP resumes the task's latest authoring session in its original execution
profile. The author resolves file content and runs relevant tests inside the isolated worktree. AOP
then stages the resolution and continues the rebase. This repeats for every conflicting commit.
After the rebase, the author receives one final isolated validation turn before AOP fast-forwards
main.

Author continuations retain the original profile, normally `edit`, and are not responsible for Git
metadata or the main worktree. Use `--timeout` to override the original run timeout.

A successful integration updates the task's recorded base and writes an audit record linking the
original commits, rebased commits, conflict-resolution runs, and final validation run. Keep the task
worktree for additional work, or remove it after success:

```sh
aop integrate task-a --remove-worktree
```

AOP never stashes changes, force-updates a branch, or decides conflict content. A failed integration
does not delete the task. Task and integration locks prevent concurrent operations from changing
the same state.

See the [CLI guide](cli.md) for creating tasks, running providers, and inspecting retained state.
