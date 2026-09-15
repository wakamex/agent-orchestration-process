"""Large synthetic prompts must bypass the OS argument-size boundary."""

from agent_orchestration_process.runner import AgentRunner, DeepSeekHarnessAdapter
from agent_orchestration_process.worktrees import WorktreeManager


def test_dsh_large_prompt_run_and_resume(repository, fake_dsh):
    runner = AgentRunner(
        WorktreeManager.discover(repository), DeepSeekHarnessAdapter(str(fake_dsh))
    )
    prompt = "--literal\n" + "café '$value'\n" * 50000
    first = runner.run(task="large-dsh", profile="sealed", prompt=prompt)
    assert first.succeeded, first.error
    assert prompt in first.final_message
    followup = "--followup\n" + "résumé `literal`\n" * 45000
    resumed = runner.resume(run_id=first.run_id, prompt=followup)
    assert resumed.succeeded, resumed.error
    assert resumed.session_id == first.session_id
    assert followup in resumed.final_message
