# Contributor guide

- Simulation only. Keep evaluator state and native outcomes outside policy inputs.
- Retain the continuous stage-code protocol and real perception-service adapters.
- Never add credentials, machine addresses, local experiment paths, model weights, assets, caches or run results.
- Generated Python checks are not a hardened sandbox; use an operator-managed isolated runtime.
- Keep changes scoped. Run compileall and unittest before delivery. Mock tests do not establish native physical success.
- Do not commit, push or launch benchmark episodes without explicit authorization.
