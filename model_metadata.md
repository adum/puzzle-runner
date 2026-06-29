# Model Metadata

Last researched: June 29, 2026.

This table normalizes runner and provider prefixes from `final_results.md` to the underlying public model. Origins are mapped by the model developer's home region: America, China, or Europe.

| Model family | Specific model version | Release date | Origin | Open weights |
| --- | --- | --- | --- | --- |
| ChatGPT / Codex | GPT-4.1 | April 14, 2025 | America | false |
| ChatGPT / Codex | GPT-5.2 | December 11, 2025 | America | false |
| ChatGPT / Codex | GPT-5.3-Codex | February 5, 2026 | America | false |
| ChatGPT / Codex | GPT-5.5 (Codex) | April 23, 2026 | America | false |
| Claude | Claude Opus 4 | May 22, 2025 | America | false |
| Claude | Claude Sonnet 4.5 | September 29, 2025 | America | false |
| Claude | Claude Opus 4.5 | November 24, 2025 | America | false |
| Claude | Claude Sonnet 4.6 | February 17, 2026 | America | false |
| Claude | Claude Opus 4.7 | April 16, 2026 | America | false |
| Claude | Claude Opus 4.8 | May 28, 2026 | America | false |
| Claude | Claude Fable 5 | June 9, 2026 | America | false |
| Gemini | Gemini 2.5 Flash | June 17, 2025 | America | false |
| Gemini | Gemini 3 Flash Preview | December 17, 2025 | America | false |
| Gemini | Gemini 3.1 Pro Preview | February 19, 2026 | America | false |
| Gemini | Gemini 3.5 Flash (High) | May 19, 2026 | America | false |
| Grok | Grok 4.3 | April 17, 2026 | America | false |
| Grok | Grok Composer 2.5 Fast | June 1, 2026 | America | false |
| GLM | GLM-4.5 | July 28, 2025 | China | true |
| GLM | GLM-5.1 | April 7, 2026 | China | true |
| GLM | GLM-5.2 | June 16, 2026 | China | true |
| Kimi | Kimi K2.5 | January 27, 2026 | China | true |
| Kimi | Kimi K2.6 | April 20, 2026 | China | true |
| Kimi | Kimi K2.7 Code | June 12, 2026 | China | true |
| MiniMax | MiniMax M2 | October 23, 2025 | China | true |
| MiniMax | MiniMax M3 | May 31, 2026 | China | true |
| Mistral | Mistral Medium 3.1 | August 12, 2025 | Europe | false |
| Mistral | Mistral Medium 3.5 | April 29, 2026 | Europe | true |
| Qwen | Qwen3.6-Plus | April 2, 2026 | China | false |
| Qwen | Qwen3 Coder Plus | September 23, 2025 | China | false |
| Qwen | Qwen3.7-Max | May 21, 2026 | China | false |
| DeepSeek | DeepSeek R1 | January 20, 2025 | China | true |
| DeepSeek | DeepSeek V4 Pro | April 24, 2026 | China | true |

## Notes

- GPT-4.1 uses OpenAI's official API launch date and remains false because OpenAI has not published public model weights.
- GPT-5.5 was publicly rolling out in ChatGPT and Codex on April 23, 2026; API availability followed on April 24, 2026.
- Claude Opus 4 uses Anthropic's official Claude 4 launch date and remains false because it is a hosted proprietary model.
- Gemini 2.5 Flash uses the GA `gemini-2.5-flash` release date, not earlier preview model IDs.
- Grok 4.3 uses the first reported public paid beta date. Public reports also cite April 24, 2026 as the move to production and April 30, 2026 as full API availability; recheck this if xAI publishes a canonical release note or model card.
- Grok Composer 2.5 Fast uses xAI's official Composer 2.5 announcement date. It remains false because xAI describes it as available inside Grok Build for SuperGrok and X Premium+ users, not as public model weights.
- MiniMax M2 and M3 use OpenRouter's listed release dates. MiniMax M3's official launch post is dated June 1, 2026, but OpenRouter lists earlier public availability.
- Claude Fable 5 uses Anthropic's official launch post date and remains false because it is a hosted proprietary model.
- Open weights means public model weights are available now. MiniMax M2 and M3 are true because MiniMax publishes public Hugging Face/GitHub model weights for both.
- Kimi K2.5 uses OpenRouter's listed release date and is true for open weights because Moonshot's GitHub and Hugging Face pages say the model weights are released under the Modified MIT License.
- Kimi K2.7 Code uses OpenRouter's listed release date and is true for open weights because the Hugging Face model card says the model weights are released under the Modified MIT License.
- Qwen3.6-Plus remains false because it is the hosted API model tested here; Alibaba separately open-sourced Qwen3.6-35B-A3B after the Plus launch.
- Qwen3 Coder Plus uses OpenRouter's listed release date; it remains false because the tested Plus API model is described as Alibaba's proprietary version of the open-source Qwen3 Coder 480B A35B.
- Qwen3.7-Max uses OpenRouter's listed release date; it remains false because the tested Max API model is proprietary/hosted.
- DeepSeek R1 uses DeepSeek's public launch date and is true for open weights because DeepSeek publishes the model weights under the MIT License.
- GLM-4.5 uses Z.AI technical blog release date and is true for open weights because Z.AI and Hugging Face publish the model weights under an MIT license.
- GLM-5.2 uses OpenRouter's listed release date for the tested API model and is true for open weights because Z.AI and Hugging Face publish the model weights under an MIT license.

## Sources

- OpenAI: [GPT-4.1](https://openai.com/index/gpt-4-1/), [GPT-5.2](https://openai.com/index/introducing-gpt-5-2/), [GPT-5.3-Codex](https://openai.com/index/introducing-gpt-5-3-codex/), [GPT-5.5](https://openai.com/index/introducing-gpt-5-5/)
- Anthropic: [Claude 4](https://www.anthropic.com/news/claude-4), [Claude Sonnet 4.5](https://www.anthropic.com/news/claude-sonnet-4-5), [Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5), [Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6), [Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7), [Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8), [Claude Fable 5](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- Google / DeepMind: [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash), [Gemini 3 Flash](https://blog.google/products/gemini/gemini-3-flash/), [Gemini 3.1 Pro](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/), [Gemini 3.5 Flash](https://deepmind.google/models/model-cards/gemini-3-5-flash/)
- xAI / Grok: [Grok 4.3 API rollout report](https://www.aibars.net/en/library/ai-news/details/839794306796621824), [Grok 4.3 production rollout report](https://www.theautonomous.net/issue/002-april-29-2026/xai-grok-4-3-multiagent-realtime), [Composer 2.5 announcement](https://x.ai/news/composer-2-5)
- Z.AI / GLM: [GLM-4.5 technical blog](https://z.ai/blog/glm-4.5), [GLM-4.5 Hugging Face model card](https://huggingface.co/zai-org/GLM-4.5), [OpenRouter GLM-4.5 model page](https://openrouter.ai/z-ai/glm-4.5), [GLM-5.1 release notes](https://docs.z.ai/release-notes/new-released), [GLM-5.1 Hugging Face model card](https://huggingface.co/zai-org/GLM-5.1), [OpenRouter GLM-5.1 model page](https://openrouter.ai/z-ai/glm-5.1-20260406/benchmarks), [Z.AI GLM-5.2 docs](https://docs.z.ai/guides/llm/glm-5.2), [GLM-5.2 Hugging Face model card](https://huggingface.co/zai-org/GLM-5.2), [OpenRouter GLM-5.2 model page](https://openrouter.ai/z-ai/glm-5.2)
- Moonshot AI / Kimi: [Kimi K2.5 GitHub repo](https://github.com/MoonshotAI/Kimi-K2.5), [Kimi K2.5 Hugging Face model card](https://huggingface.co/moonshotai/Kimi-K2.5), [OpenRouter Kimi K2.5 model page](https://openrouter.ai/moonshotai/kimi-k2.5), [Kimi K2.6 model page](https://www.kimi.com/ai-models/kimi-k2-6), [Kimi K2.6 Hugging Face model card](https://huggingface.co/moonshotai/Kimi-K2.6), [Kimi K2.7 Code API docs](https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart), [Kimi K2.7 Code Hugging Face model card](https://huggingface.co/moonshotai/Kimi-K2.7-Code), [OpenRouter Kimi K2.7 Code model page](https://openrouter.ai/moonshotai/kimi-k2.7-code), [Kimi research index](https://www.kimi.com/blog/kimi-k2.5)
- MiniMax: [MiniMax M2 GitHub repo](https://github.com/MiniMax-AI/MiniMax-M2), [MiniMax M2 Hugging Face model card](https://huggingface.co/MiniMaxAI/MiniMax-M2), [OpenRouter MiniMax M2 model page](https://openrouter.ai/minimax/minimax-m2), [MiniMax M3 launch post](https://www.minimax.io/blog/minimax-m3), [MiniMax M3 GitHub repo](https://github.com/MiniMax-AI/MiniMax-M3), [MiniMax M3 Hugging Face model card](https://huggingface.co/MiniMaxAI/MiniMax-M3), [OpenRouter MiniMax M3 model page](https://openrouter.ai/minimax/minimax-m3/performance)
- Mistral AI: [Mistral Medium 3.1 model card](https://docs.mistral.ai/models/model-cards/mistral-medium-3-1-25-08), [Mistral Medium 3.5 model card](https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04), [Mistral Medium 3.5 Hugging Face model card](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B), [Mistral Medium 3.5 launch post](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5)
- Alibaba Cloud / Qwen: [Qwen3.6-Plus launch post](https://www.alibabacloud.com/blog/alibaba-unveils-qwen3-6-plus-to-accelerate-agentic-ai-deployment-for-enterprises-and-alibaba%E2%80%99s-ai-applications_603005), [Qwen3.6-35B-A3B open-weights post](https://www.alibabacloud.com/blog/alibaba-open-sources-qwen3-6-35b-a3b-wan2-7-tops-design-arena_603042), [OpenRouter Qwen3 Coder Plus model page](https://openrouter.ai/qwen/qwen3-coder-plus), [Qwen3.7 blog](https://qwen.ai/blog?id=qwen3.7), [OpenRouter Qwen3.7-Max model page](https://openrouter.ai/qwen/qwen3.7-max)
- DeepSeek: [DeepSeek R1 GitHub repo](https://github.com/deepseek-ai/DeepSeek-R1), [DeepSeek R1 Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek V4 Pro Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [V4 preview release notes](https://api-docs.deepseek.com/news/news260424), [transparency center](https://www.deepseek.com/en/transparency/)
