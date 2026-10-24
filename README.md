Distributed-inference feasibility bound
botnet_inference_throughput.py is a pure-arithmetic calculator. It accompanies the research article Autonomous Is Not Autonomy: What it would actually take for an AI to turn off the lights, and it exists to make one of that article's claims checkable: that a frontier-scale mixture-of-experts (MoE) model, if its weights were spread across many compromised consumer machines, would decode so slowly that the "distributed superintelligence" scenario does not survive contact with the numbers.

The script performs no inference, opens no network connections, and contains no code that interacts with any system. It takes published hardware and model figures, applies the communication cost of sharded decoding, and prints a report. Every assumption is a named constant at the top of the file, chosen in the direction that favours the hypothetical attacker, so the output is an upper bound on what such a system could achieve, not an estimate of what it would.

Running it
Python 3.8 or later, standard library only.

python -I botnet_inference_throughput.py
The -I flag runs Python in isolated mode so the interpreter ignores the working directory and user site-packages. The script prints to standard output and writes nothing.

What the report contains
Model and capacity. A 700B-class MoE shaped like DeepSeek-V3 (hidden size 7,168, 60 layers, 256 routed experts per layer, 8 active per token). At four-bit weights the model is about 326 GiB, so roughly 81 hosts at 4 GB usable memory each are needed just to hold it. Capacity is granted to the attacker immediately: individual experts are only about 21 MiB.
Per-token network volume. Under the cheapest workable arrangement, expert-parallel with point-to-point dispatch and combine, each layer moves about 224 KiB and each token about 13.1 MiB, across 60 sequential synchronisation barriers.
Single-stream decode throughput in tokens per hour, swept over residential upload speed (10 to 1,000 Mbps) and round-trip time (20 to 100 ms), for three cases: full uplink, uplink throttled to 15 percent to avoid notice, and the throttled case with FP8 activations.
Bandwidth-bound or latency-bound. Above roughly 37 Mbps at 50 ms the system is pinned by latency across the 60 barriers, so a hundredfold bandwidth increase buys about double the throughput.
Batching steelman. Perfect batching across many concurrent sequences, which turns the aggregate ceiling into a bandwidth limit, compared against DeepSeek's published serving figures for the same model class.
Cross-check against measured systems. Petals (BLOOM-176B, pipeline-parallel, open internet) and Anthropic's fifteen-device run (Gemma4-26B-A4B, pipeline-parallel). Both use a different topology from the one modelled here, so they are directional checks, not validation.
Speculative decoding, granted at a fourfold multiplier, with the three reasons that multiplier is generous.
Datacenter comparison against NVLink interconnect figures.
Headline figures and operational translation. At 35 Mbps and 50 ms, about 586 tokens per hour unthrottled and about 150 throttled. The translation uses three round working assumptions, labelled as such: about 2,000 tokens per agentic reasoning step, 50,000 per agent working day, and 10 million per campaign the size of a documented AI-assisted espionage operation.
Worse variants (sharded attention, ring and tree collectives), the cost of fault tolerance, and the one steelman that works: a 30B dense model running locally on a single consumer GPU with no interconnect at all, which is the realistic threat and a different one.
Changing the assumptions
Every input is a named constant near the top of the file: model shape, weight and activation precision, usable memory per host, upload-speed and round-trip cases, the stealth duty cycle, datacenter interconnect figures, the Petals table, and the speculative-decoding multiplier. Change a constant and rerun. The point of publishing the script is that anyone who disagrees with an assumption can see exactly what it changes.

What is not modelled
All of these would make the attacker's position worse, not better, and none is included:

recovery after a dropped host (re-routing, re-sending, re-electing holders)
a latency distribution rather than a point round-trip time; tail latency dominates a barrier
KV-cache placement and growth for long agentic contexts
NAT traversal, relay hops and connection-setup costs
Sources
DeepSeek-V3 public model configuration, and DeepSeek's published 24-hour serving statistics for DeepSeek-V3/R1 (February 2025, deepseek-ai/open-infra-index)
Borzunov et al., Petals: Collaborative Inference and Fine-tuning of Large Models, Table 3
Speculative Decoding in Decentralized LLM Inference, arXiv:2511.11733
BloomBee, arXiv:2604.21072
NVIDIA GB200 NVL72 NVLink specifications
Anthropic Frontier Red Team, AI models may be capable of self-replicating without access to a datacenter, 12 October 2026
Scope
This is a feasibility bound written for defenders and for readers trying to evaluate claims about autonomous AI. It demonstrates why one widely repeated scenario fails on throughput, and it is published so that the demonstration can be checked and argued with.
