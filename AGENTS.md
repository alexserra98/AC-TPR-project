## Implementation Guidance
- Prefer direct code over abstractions unless the abstraction removes real complexity.
- Comments and docstrings should explain what the code does, not narrate recent
  changes.
- When I ask you to implement an experimental module the guiding principle is: **implement something easy to add and easy to remove**. Avoid overengineering or overgeneralizing. I prefere code redundacy rather than chaning the codebase to support a single experiment.
- If you need to create scripts custom for a execute a specific experiment put under either scripts/temporary of scripts/slurm/temporary. Avoid cluttering the main scripts/ folder with one-off scripts.
- Reuse reuse reuse. I don't want you to reinvent the wheel prefer to reuse existing code and functions in this repo instead of writing new ones.
- This is code for  research not production so if the code halts or fails in some edge case it is ok. I prefer clear and strict contracts rather than loads of fallbacks and error handling.
