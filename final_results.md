Result semantics: this table records individual runs. For model-level reporting, the true result for a model is the maximum `Best Score` across all rows that normalize to the same specific model version; repeated runs are never averaged. Visualization score charts apply this max-only rule before plotting.

| Run ID | Agent | Harness | Effort | Best Score | Best Round | Rounds | Stop Reason | Timeout | Wall Time | Agent Chars | Code Lines Added | OpenRouter Calls | OpenRouter Cost | OpenRouter Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260430-102443-codex-5-3 | codex-5.3 | codex | xhigh | 171 | 6 | 9 | stale_limit | 600s | 18h 30m | 9176760 | 442 |  |  |  |
| 20260501-122827-codex-5-5 | codex-5.5 | codex | xhigh | 192 | 2 | 5 | stale_limit | 600s | 9h 10m | 8015306 | 1088 |  |  |  |
| 20260502-094850-claude-code-sonnet | claude-code-sonnet 4.6 | claudecode | low | 63 | 1 | 2 | forbidden_edit_detected | 600s | 5h 4m | 66803 | 423 |  |  |  |
| 20260502-153410-claude-code-sonnet | claude-code-sonnet 4.6 | claudecode | medium | 155 | 3 | 6 | stale_limit | 600s | 31h 13m | 78774 | 1606 |  |  |  |
| 20260504-090806-claude-code-opus | claude-code-opus 4.7 | claudecode | max | 195 | 1 | 4 | stale_limit | 600s | 10h 58m | 14476 | 601 |  |  |  |
| 20260505-082141-gemini-3-1-pro-preview | gemini 3.1 pro preview | antigravity |  | 262 | 9 | 12 | stale_limit | 600s | 17h 0m | 1400 | 6466 |  |  |  |
| 20260506-120335-gemini-2-5-flash | gemini 2.5 flash | antigravity |  | 65 | 1 | 4 | stale_limit | 600s | 4h 18m | 12351 | 422 |  |  |  |
| 20260506-184031-gemini-3-flash-preview | gemini 3 flash preview | antigravity |  | 262 | 5 | 7 | agent_idle_timeout | 600s | 8h 36m | 11121 | 447 |  |  |  |
| 20260508-163024-opencode-openrouter-gemini-3-flash-preview | opencode-openrouter-gemini-3-flash-preview | opencode | high | 176 | 2 | 5 | stale_limit | 600s | 4h 31m | 2465 | 556 |  |  |  |
| 20260509-182003-opencode-openrouter-moonshotai-kimi-k2-6 | opencode-openrouter-moonshotai-kimi-k2.6 | opencode | high | 152 | 4 | 6 | agent_idle_timeout | 600s | 31h 0m | 25409 | 4631 |  | $57.52 |  |
| 20260511-082448-opencode-openrouter-x-ai-grok-4-3 | opencode-openrouter-x-ai-grok-4.3 | opencode | high | 47 | 1 | 4 | stale_limit | 600s | 36m 23s | 72 | 25 | 42 | $0.399170 | 636025 |
| 20260511-112204-opencode-openrouter-mistralai-mistral-medium-3-1 | opencode-openrouter-mistralai-mistral-medium-3.1 | opencode | high | 47 | 1 | 4 | stale_limit | 600s | 3h 43m | 189182 | 192 | 313 | $5.308033 | 16098390 |
| 20260511-171126-opencode-openrouter-qwen-qwen3-6-plus | opencode-openrouter-qwen-qwen3.6-plus | opencode | high | 68 | 1 | 4 | stale_limit | 600s | 7h 11m | 19780 | 2959 | 336 | $11.398238 | 32433127 |
| 20260513-135710-claude-sonnet-4-5 | claude-sonnet-4-5 | claudecode | xhigh | 64 | 3 | 6 | stale_limit | 600s | 4h 28m | 14413 | 2864 |  |  |  |
| 20260513-154703-gpt-5-2 | gpt-5.2 | codex | xhigh | 100 | 1 | 4 | stale_limit | 600s | 12h 9m | 13601515 | 1031 |  |  |  |
| 20260513-213232-claude-opus-4-5 | claude-opus-4-5 | claudecode | high | 100 | 1 | 1 | evaluation_failed | 600s | 47m 17s | 2337 | 665 |  |  |  |
| 20260514-113354-claude-opus-4-5 | claude-opus-4-5 | claudecode | high | 144 | 5 | 8 | stale_limit | 600s | 15h 19m | 16980 | 1689 |  |  |  |
| 20260524-123240-gemini-3-5-flash-high | gemini-3.5-flash-high | antigravity |  | 176 | 5 | 8 | stale_limit | 600s | 6h 21m | 15906 | 678 |  |  |  |
| 20260524-193813-gemini-3-5-flash-high | gemini-3.5-flash-high | antigravity |  | 151 | 1 | 4 | stale_limit | 600s | 2h 51m | 27940 | 1065 |  |  |  |
| 20260525-101237-gemini-3-5-flash-high | gemini-3.5-flash-high | antigravity |  | 195 | 1 | 4 | stale_limit | 600s | 3h 19m | 34222 | 1314 |  |  |  |
| 20260527-103344-opencode-openrouter-deepseek-deepseek-v4-pro | opencode-openrouter-deepseek-deepseek-v4-pro | opencode | high | 89 | 3 | 4 | agent_idle_timeout | 600s | 15h 47m | 28354 | 292 | 423 | $4.077197 | 34813443 |
| 20260528-073107-opencode-openrouter-deepseek-deepseek-v4-pro | opencode-openrouter-deepseek-deepseek-v4-pro | opencode | high | 63 | 1 | 3 | agent_idle_timeout | 600s | 8h 0m | 16866 | 188 | 314 | $5.867184 | 34605071 |
| 20260528-160434-claude-opus-4-8 | claude-opus-4-8 | claudecode | xhigh | 262 | 3 | 6 | stale_limit | 600s | 8h 4m | 36000 | 813 |  |  |  |
| 20260529-153836-opencode-openrouter-z-ai-glm-5-1 | opencode-openrouter-z-ai-glm-5.1 | opencode | high | 104 | 8 | 8 |  | 600s | 18h 40m | 79704 | 1959 |  | $64.8177 |  |
| 20260531-213945-opencode-openrouter-mistralai-mistral-medium-3-5 | opencode-openrouter-mistralai-mistral-medium-3-5 | opencode | high | 68 | 3 | 6 | stale_limit | 600s | 22h 5m | 214079 | 4675 | 681 | $59.786295 | 40832194 |
| 20260602-130130-opencode-openrouter-minimax-minimax-m3 | opencode-openrouter-minimax-minimax-m3 | opencode | high | 89 | 2 | 3 |  | 600s | 10h 50m | 631855 | 1942 |  | $7.9823 | 119564526 |
| 20260609-075824-opencode-openrouter-qwen-qwen3-7-max | opencode-openrouter-qwen-qwen3.7-max | opencode | high | 77 | 2 | 5 | stale_limit | 600s | 4h 44m | 34669 | 700 | 346 | $11.838221 | 27760197 |
| 20260609-155527-claude-fable-5 | claude-fable-5 | claudecode | xhigh | 352 | 1 | 4 | stale_limit | 600s | 5h 39m | 7324 | 1259 |  |  |  |
| 20260615-085613-opencode-openrouter-moonshotai-kimi-k2-7-code | opencode-openrouter-moonshotai-kimi-k2.7-code | opencode | high | 99 | 1 | 3 | agent_idle_timeout | 600s | 11h 51m | 505461 | 2923 | 454 | $15.429768 | 45475119 |
| 20260616-225849-opencode-openrouter-moonshotai-kimi-k2-5 | opencode-openrouter-moonshotai-kimi-k2.5 | opencode | high | 68 | 2 | 5 | stale_limit | 600s | 8h 21m | 16212 | 264 | 359 | $1.898840 | 18242601 |
| 20260617-174139-opencode-openrouter-minimax-minimax-m2 | opencode-openrouter-minimax-minimax-m2 | opencode | high | 65 | 3 | 5 | agent_idle_timeout | 600s | 7h 53m | 8688 | 325 | 362 | $3.317071 | 13772479 |
| 20260618-093335-opencode-openrouter-z-ai-glm-5-2 | opencode-openrouter-z-ai-glm-5.2 | opencode | high | 176 | 2 | 2 | agent_idle_timeout | 600s | 5h 14m | 9885 | 655 | 225 | $8.904978 | 25992819 |
| 20260618-154508-opencode-openrouter-z-ai-glm-4-5 | opencode-openrouter-z-ai-glm-4.5 | opencode | high | 68 | 1 | 4 | stale_limit | 600s | 3h 31m | 5243 | 196 | 206 | $1.114505 | 5539750 |
| 20260619-155922-opencode-openrouter-openai-gpt-4-1 | opencode-openrouter-openai-gpt-4.1 | opencode | high | 1 | 4 | 7 | stale_limit | 600s | 18m 19s | 91781 | 181 | 177 | $3.053126 | 4685197 |
| 20260619-185805-opencode-openrouter-deepseek-deepseek-r1 | opencode-openrouter-deepseek-deepseek-r1 | opencode | high | 41 | 3 | 6 | stale_limit | 600s | 4h 50m | 50108 | 669 | 159 | $3.073608 | 3583156 |
| 20260620-084401-opencode-openrouter-anthropic-claude-opus-4 | opencode-openrouter-anthropic-claude-opus-4 | opencode | high | 47 | 2 | 5 | stale_limit | 600s | 1h 28m | 24471 | 2836 | 287 | $26.950575 | 9534134 |
| 20260620-103318-opencode-openrouter-qwen-qwen3-coder-plus | opencode-openrouter-qwen-qwen3-coder-plus | opencode | high | 29 | 2 | 5 | stale_limit | 600s | 53m 36s | 46816 | 2581 | 273 | $2.698133 | 13380415 |
| 20260628-223303-grok-composer-2-5-fast | grok-composer-2.5-fast | grokbuild |  | 64 | 1 | 1 | agent_failed | 600s | 1h 13m | 4726 | 591 |  |  |  |
| 20260629-091644-grok-build | grok-build | grokbuild |  | 65 | 1 | 1 | agent_failed | 600s | 53m 9s | 2204 | 208 |  |  |  |
| 20260629-103458-opencode-openrouter-sakana-fugu-ultra | opencode-openrouter-sakana-fugu-ultra | opencode | high | 171 | 1 | 3 | agent_idle_timeout | 600s | 15h 18m | 26711 | 509 | 253 | $20.174499 | 14814495 |
| 20260630-161457-claude-sonnet-5 | claude-sonnet-5 | claudecode | xhigh | 195 | 1 | 4 | stale_limit | 600s | 4h 9m | 10054 | 748 |  |  |  |
| 20260701-144021-opencode-openrouter-nvidia-nemotron-3-ultra-550b-a55b | opencode-openrouter-nvidia-nemotron-3-ultra-550b-a55b | opencode | high | 47 | 2 | 3 | agent_idle_timeout | 600s | 3h 52m | 1815 | 1061 | 510 | $7.042864 | 36511311 |
| 20260708-075701-opencode-openrouter-tencent-hy3 | opencode-openrouter-tencent-hy3 | opencode | high | 47 | 1 | 1 | agent_idle_timeout | 600s | 30m 22s | 0 | 0 |  |  |  |
| 20260708-133138-grok-4-5 | grok-4.5 | grokbuild |  | 73 | 1 | 1 | agent_failed | 600s | 1h 15m | 1844 | 960 |  |  |  |
| 20260708-173920-grok-4-5 | grok-4.5 | grokbuild |  | 131 | 1 | 1 | agent_failed | 600s | 41m 40s | 1645 | 1243 |  |  |  |
| 20260708-184051-grok-4-5 | grok-4.5 | grokbuild |  | 86 | 2 | 2 | agent_idle_timeout | 600s | 4h 11m | 4045 | 948 |  |  |  |
| 20260709-075301-grok-4-5 | grok-4.5 | grokbuild |  | 196 | 2 | 3 | agent_idle_timeout | 600s | 7h 46m | 9360 | 2307 |  |  |  |
| 20260712-075410-gpt-5-6-sol | gpt-5.6-sol | codex | max | 321 | 2 | 5 | stale_limit | 600s | 12h 43m | 53466382 | 1820 |  |  |  |
| 20260712-221136-gpt-5-6-sol | gpt-5.6-sol | codex | max | 576 | 9 | 9 | system_restart | 600s | 48h 38m | 30963677 | 4434 |  |  |  |
| 20260713-184225-opencode-openrouter-z-ai-glm-5-2 | opencode-openrouter-z-ai-glm-5.2 | opencode | high | 186 | 2 | 2 | agent_idle_timeout | 600s | 9h 41m | 14144 | 993 | 279 | $8.505415 | 29438342 |
| 20260714-092026-opencode-openrouter-sakana-fugu-ultra | opencode-openrouter-sakana-fugu-ultra | opencode | high | 152 | 1 | 2 | agent_idle_timeout | 600s | 4h 46m | 15061 | 563 | 154 | $10.959157 | 10011757 |
| 20260714-144926-opencode-openrouter-moonshotai-kimi-k2-5 | opencode-openrouter-moonshotai-kimi-k2.5 | opencode | high | 64 | 1 | 4 | stale_limit | 600s | 2h 38m | 15546 | 206 | 237 | $2.819872 | 12241973 |
| 20260714-195310-opencode-openrouter-moonshotai-kimi-k2-7-code | opencode-openrouter-moonshotai-kimi-k2.7-code | opencode | high | 73 | 2 | 2 | agent_idle_timeout | 600s | 1h 35m | 459 | 191 | 89 | $1.013549 | 4415132 |
| 20260715-000956-opencode-openrouter-moonshotai-kimi-k2-6 | opencode-openrouter-moonshotai-kimi-k2.6 | opencode | high | 86 | 1 | 1 | agent_idle_timeout | 600s | 3h 32m | 473 | 360 | 87 | $2.908970 | 8522341 |
| 20260715-071526-opencode-openrouter-z-ai-glm-5-1 | opencode-openrouter-z-ai-glm-5.1 | opencode | high | 65 | 1 | 4 | stale_limit | 600s | 19h 12m | 116734 | 1494 | 1581 | $37.541033 | 154484394 |
| 20260716-101358-claude-fable-5 | claude-fable-5 | claudecode | xhigh | 457 | 3 | 6 | stale_limit | 600s | 12h 22m | 10578 | 3755 |  |  |  |
| 20260716-224238-opencode-openrouter-moonshotai-kimi-k3 | opencode-openrouter-moonshotai-kimi-k3 | opencode | high | 64 | 1 | 1 | agent_error | 600s | 1h 6m | 1316 | 257 | 28 | $1.565580 | 874896 |
| 20260717-085254-opencode-openrouter-meta-llama-llama-4-maverick | opencode-openrouter-meta-llama-llama-4-maverick | opencode | high | 47 | 1 | 4 | stale_limit | 600s | 2m 1s | 5845 | 0 | 8 | $0.014964 | 69612 |
