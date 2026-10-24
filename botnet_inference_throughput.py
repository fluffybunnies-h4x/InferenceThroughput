#!/usr/bin/env python3
"""
Can a frontier MoE model run distributed across a botnet?

Computes single-stream decode throughput for a 700B-class Mixture-of-Experts model
sharded over compromised consumer machines, and compares it to a datacenter
NVLink domain.

The model is deliberately built to FAVOUR the attacker at every choice point, so the
result is an upper bound on botnet performance, not an estimate. Assumptions are
listed in ASSUMPTIONS below and printed with the results.

EVERY FIGURE IS CONDITIONAL ON THE MODELLED TOPOLOGY: expert-parallel sharding with
point-to-point dispatch and combine per layer. Pipeline-parallel systems (Petals,
Anthropic's 15-box run) have a different communication pattern and are used here
only as a directional cross-check, not as validation.

NOT MODELLED (all would make the attacker's position worse, not better):
  - recovery overhead after a dropped host (re-routing, re-sending, re-electing)
  - a latency DISTRIBUTION rather than a point RTT; tail latency dominates a barrier
  - KV-cache placement and growth for long agentic contexts
  - NAT traversal, relay hops and TCP/QUIC handshake costs

Run:  python -I botnet_inference_throughput.py
"""

# ---------------------------------------------------------------------------
# Model architecture: 700B-class MoE, DeepSeek-V3 / GLM-5.2 shaped.
# Public config of DeepSeek-V3 (671B total / 37B active) used as the template.
# ---------------------------------------------------------------------------
D_MODEL = 7168          # hidden dimension
N_LAYERS = 60           # transformer layers (treated as all-MoE; slightly pessimistic
                        # for the attacker on count, generous on structure)
TOP_K = 8               # routed experts activated per token
N_EXPERTS = 256         # routed experts per MoE layer
TOTAL_PARAMS = 700e9    # total parameters
ACT_BYTES = 2           # activation precision: BF16. FP8 would halve traffic.
WEIGHT_BYTES = 0.5      # INT4 weights

# Non-expert parameters (attention + embeddings) that must be replicated or sharded.
NON_EXPERT_PARAMS = 14e9

# ---------------------------------------------------------------------------
# Botnet parameters
# ---------------------------------------------------------------------------
# Usable memory per compromised host for a stealthy implant. A loader that
# consumes all RAM is trivially noticed, so this is already generous.
USABLE_GB_PER_HOST = 4.0

# Residential upload, Mbps. Upload is the binding direction: consumer broadband is
# asymmetric, and every shard must SEND its results.
UPLOAD_MBPS_CASES = [10, 35, 100, 1000]

# Round-trip time between two arbitrary consumer machines, milliseconds.
# Excludes NAT traversal / relay overhead, which only adds.
RTT_MS_CASES = [20, 50, 100]

# Fraction of the uplink an implant can use before the victim notices their
# video calls breaking. Saturated residential upload is the loudest botnet signal
# there is.
STEALTH_DUTY_CYCLE = 0.15

# ---------------------------------------------------------------------------
# Datacenter comparison (NVIDIA published figures, verified)
# ---------------------------------------------------------------------------
NVLINK_TBPS_PER_GPU = 1.8     # GB200 NVL72, 5th-gen NVLink, per GPU
NVLINK_LATENCY_US = 1.0       # sub-microsecond; 1us is conservative


def human_bytes(n):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PiB"


# ---------------------------------------------------------------------------
# 1. Capacity floor: how many hosts just to hold the weights?
# ---------------------------------------------------------------------------
weight_bytes_total = TOTAL_PARAMS * WEIGHT_BYTES
hosts_for_weights = weight_bytes_total / (USABLE_GB_PER_HOST * 1024**3)

expert_params_each = (TOTAL_PARAMS - NON_EXPERT_PARAMS) / (N_EXPERTS * N_LAYERS)
expert_bytes_each = expert_params_each * WEIGHT_BYTES
non_expert_bytes = NON_EXPERT_PARAMS * WEIGHT_BYTES

# ---------------------------------------------------------------------------
# 2. Per-token network volume.
#
# Charitable sharding model, chosen to minimise traffic:
#   - Experts distributed across hosts; each expert is small enough to sit whole
#     on one host (no tensor parallelism inside an expert, no all-reduce).
#   - Attention + embeddings replicated, so no attention all-reduce either.
#   - Per layer, the hidden state is relayed point-to-point to the TOP_K hosts
#     holding the selected experts; each returns one d_model vector.
#
# This is the cheapest arrangement that actually works. Alternatives (tensor
# parallelism, ring/tree all-reduce over all hosts) are far worse, and the
# collective-latency variant is computed at the bottom for contrast.
# ---------------------------------------------------------------------------
dispatch_bytes_per_layer = TOP_K * D_MODEL * ACT_BYTES   # out to k expert holders
combine_bytes_per_layer = TOP_K * D_MODEL * ACT_BYTES    # k results back
bytes_per_layer = dispatch_bytes_per_layer + combine_bytes_per_layer
bytes_per_token = bytes_per_layer * N_LAYERS

# Sequential synchronisation barriers per token: each layer's experts must finish
# before the next layer's router can run. Autoregressive decode cannot overlap
# these for a single stream -- token N+1 depends on token N.
hops_per_layer = 2            # dispatch out, combine back (two one-way hops = 1 RTT)
barriers_per_token = N_LAYERS


def throughput(upload_mbps, rtt_ms, duty_cycle=1.0, act_bytes=ACT_BYTES):
    """Single-stream decode throughput, tokens/hour.

    act_bytes: activation precision in bytes. 2 = BF16 (default), 1 = FP8, which
    halves per-layer traffic and is the cheapest activation format anyone serves
    at today. It does not touch the latency term, which is the binding one.
    """
    # Effective uplink of the relaying host, bytes/sec.
    up_bytes_per_s = (upload_mbps * 1e6 / 8) * duty_cycle

    # Serialised uplink time: the relay must push k copies of the hidden state
    # each layer, and the expert hosts must push their results back. Both sit on
    # the critical path because the layer is a barrier.
    scale = act_bytes / ACT_BYTES
    tx_time_per_layer = dispatch_bytes_per_layer * scale / up_bytes_per_s
    rx_time_per_layer = combine_bytes_per_layer * scale / up_bytes_per_s  # expert uplinks
    net_latency_per_layer = rtt_ms / 1000.0

    per_layer_s = tx_time_per_layer + rx_time_per_layer + net_latency_per_layer
    per_token_s = per_layer_s * N_LAYERS
    return 3600.0 / per_token_s, per_token_s, per_layer_s


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=" * 78)
print("FRONTIER MoE INFERENCE OVER A BOTNET -- upper bound on attacker throughput")
print("=" * 78)

print(f"""
MODEL (700B-class MoE, DeepSeek-V3 / GLM-5.2 shaped)
  total parameters          {TOTAL_PARAMS/1e9:,.0f}B
  hidden dimension          {D_MODEL:,}
  layers                    {N_LAYERS}
  routed experts / layer    {N_EXPERTS}   (top-{TOP_K} activated per token)
  weights @ INT4            {human_bytes(weight_bytes_total)}
  one expert @ INT4         {human_bytes(expert_bytes_each)}  ({expert_params_each/1e6:,.0f}M params)
  non-expert @ INT4         {human_bytes(non_expert_bytes)}  (replicated per host)

CAPACITY FLOOR
  usable memory per host    {USABLE_GB_PER_HOST:,.1f} GB (stealthy implant)
  hosts to hold weights     {hosts_for_weights:,.0f} minimum
  NOTE: individual experts are tiny ({human_bytes(expert_bytes_each)}), so capacity is
  NOT the binding constraint. Expert placement is easy. Granted to the attacker.

PER-TOKEN NETWORK VOLUME (charitable point-to-point sharding)
  dispatch per layer        {human_bytes(dispatch_bytes_per_layer)}
  combine  per layer        {human_bytes(combine_bytes_per_layer)}
  per layer total           {human_bytes(bytes_per_layer)}
  per token  total          {human_bytes(bytes_per_token)}
  sequential barriers/token {barriers_per_token}  ({hops_per_layer} one-way hops each)
""")

print("-" * 78)
print("SINGLE-STREAM DECODE THROUGHPUT, tokens/hour")
print("-" * 78)
print("  (A) full uplink -- a botnet that does not care about being detected")
print()
hdr = "  upload \\ RTT   " + "".join(f"{r:>12}ms" for r in RTT_MS_CASES)
print(hdr)
for up in UPLOAD_MBPS_CASES:
    row = f"  {up:>6} Mbps     "
    for rtt in RTT_MS_CASES:
        tph, _, _ = throughput(up, rtt)
        row += f"{tph:>14,.0f}"
    print(row)

print()
print(f"  (B) throttled to {STEALTH_DUTY_CYCLE:.0%} of uplink -- a botnet trying to stay hidden")
print()
print(hdr)
for up in UPLOAD_MBPS_CASES:
    row = f"  {up:>6} Mbps     "
    for rtt in RTT_MS_CASES:
        tph, _, _ = throughput(up, rtt, STEALTH_DUTY_CYCLE)
        row += f"{tph:>14,.0f}"
    print(row)

print()
print(f"  (C) FP8 activations, throttled to {STEALTH_DUTY_CYCLE:.0%} -- half the bytes per token")
print()
print(hdr)
for up in UPLOAD_MBPS_CASES:
    row = f"  {up:>6} Mbps     "
    for rtt in RTT_MS_CASES:
        tph, _, _ = throughput(up, rtt, STEALTH_DUTY_CYCLE, act_bytes=1)
        row += f"{tph:>14,.0f}"
    print(row)
fp8_gain = throughput(35, 50, STEALTH_DUTY_CYCLE, act_bytes=1)[0] / throughput(35, 50, STEALTH_DUTY_CYCLE)[0]
print(f"""
  Halving the activation bytes buys {fp8_gain:.2f}x at the headline point (35 Mbps, 50 ms,
  stealthy). It helps most where the system is bandwidth-bound (low uplink) and
  least where it is latency-bound, which is exactly where the attacker ends up
  once they compromise better-connected hosts. Precision cannot fix a barrier.""")

# ---------------------------------------------------------------------------
# Where is the bottleneck?
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("BANDWIDTH-BOUND OR LATENCY-BOUND?")
print("-" * 78)
for up in UPLOAD_MBPS_CASES:
    up_bytes = up * 1e6 / 8
    tx = (dispatch_bytes_per_layer + combine_bytes_per_layer) / up_bytes * 1000
    print(f"  {up:>6} Mbps: transfer {tx:>8.1f} ms/layer vs latency "
          f"{min(RTT_MS_CASES)}-{max(RTT_MS_CASES)} ms/layer "
          f"-> {'BANDWIDTH' if tx > 50 else 'LATENCY'}-bound")
crossover_mbps = bytes_per_layer / (50 / 1000.0) * 8 / 1e6
print(f"""
  At 50 ms RTT the crossover is ~{crossover_mbps:,.0f} Mbps: above that, throughput is
  LATENCY-bound. That means the attacker cannot buy their way out by compromising
  better-connected victims -- throughput stops responding to bandwidth and is
  pinned by the speed of light plus queueing across {N_LAYERS} sequential barriers.
  Note the 1000 Mbps row: a 100x bandwidth increase buys only ~2x throughput.""")

# ---------------------------------------------------------------------------
# The strongest steelman: batching across many concurrent agents.
# Batching hides per-layer LATENCY (many sequences share each barrier), so the
# aggregate ceiling is set by aggregate bandwidth alone. This is the fairest
# possible case for an attacker running a campaign against many targets at once.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("STEELMAN: perfect batching across many concurrent agents")
print("-" * 78)
n_hosts_agg = int(hosts_for_weights)
for up in UPLOAD_MBPS_CASES:
    for duty, label in ((1.0, "full uplink"), (STEALTH_DUTY_CYCLE, "stealthy  ")):
        agg_bytes_per_s = n_hosts_agg * (up * 1e6 / 8) * duty
        agg_tph = agg_bytes_per_s / bytes_per_token * 3600
        print(f"  {up:>6} Mbps x {n_hosts_agg} hosts, {label}: "
              f"{agg_tph:>12,.0f} tokens/hour AGGREGATE")

# MEASURED datacenter figures, published by DeepSeek for DeepSeek-V3/R1 (671B MoE)
# -- the same model class modelled here. Source: deepseek-ai/open-infra-index,
# 24h window 2025-02-27/28. Decode ran Routed Expert EP144 / MLA+Shared DP144
# across 18 nodes of 8x H800 -- i.e. EXPERT parallelism, as modelled above.
DS_NODE_TOKENS_PER_S = 14_800      # aggregate output tokens/s per 8-GPU H800 node
DS_PER_USER_TOKENS_PER_S = 21      # per-user decode speed (published 20-22)
DS_GPUS_PER_NODE = 8

agg_stealth = n_hosts_agg * (35 * 1e6 / 8) * STEALTH_DUTY_CYCLE / bytes_per_token * 3600
ds_node_tph = DS_NODE_TOKENS_PER_S * 3600
ds_user_tph = DS_PER_USER_TOKENS_PER_S * 3600
print(f"""
  Batching DOES help: it amortises the {N_LAYERS} barriers across many sequences, so
  aggregate throughput becomes bandwidth-limited rather than latency-limited.
  Grant the attacker that entirely, and the whole botnet still produces about
  {agg_stealth:,.0f} tokens/hour at 35 Mbps stealthy.

  MEASURED datacenter baseline (DeepSeek's own published figures for DeepSeek-V3/R1,
  a 671B MoE -- the same model class, on export-restricted H800s from 2023):
    aggregate, per {DS_GPUS_PER_NODE}-GPU node   {DS_NODE_TOKENS_PER_S:,} tokens/s = {ds_node_tph:,.0f} tokens/hour
    per-user single stream   {DS_PER_USER_TOKENS_PER_S} tokens/s = {ds_user_tph:,.0f} tokens/hour

  So the ENTIRE botnet's hourly output is what ONE {DS_GPUS_PER_NODE}-GPU NODE produces in
  {agg_stealth/DS_NODE_TOKENS_PER_S:,.1f} SECONDS -- a factor of {ds_node_tph/agg_stealth:,.0f}x.

  And batching does not help the thing an intrusion actually needs. Compromising
  a host is an interactive loop: read output, decide, act, read again. That is
  single-stream latency, and batching cannot reduce it. A batched botnet can be
  slow against 1,000 targets simultaneously; it cannot be fast against one.""")

# ---------------------------------------------------------------------------
# Cross-validation against Petals, the only real measured WAN inference system,
# and against the published latency-hiding state of the art.
# ---------------------------------------------------------------------------
# Petals (BLOOM-176B, 8-bit, PIPELINE parallelism) Table 3, steps/s:
PETALS = {
    "3 servers, 1 Gbit/s, <5ms":     1.22,
    "3 servers, 100 Mbit/s, <5ms":   1.19,
    "3 servers, 100 Mbit/s, 100ms":  0.89,
    "12 servers, 1 Gbit/s, <5ms":    0.97,
    "12 servers, 100 Mbit/s, <5ms":  0.97,
    "12 servers, 100 Mbit/s, 100ms": 0.44,
    "14 real servers, EU+NA":        0.68,
}
SPECULATIVE_ACCEPT = 4.0   # DSD measured avg accepted length 2.99-4.04 tokens/round

print()
print("-" * 78)
print("CROSS-VALIDATION: Petals, the only real measured WAN inference system")
print("-" * 78)
for k, v in PETALS.items():
    print(f"  {k:<34} {v:>5.2f} steps/s = {v*3600:>7,.0f} tokens/hour")

print(f"""
  Petals runs BLOOM-176B with PIPELINE parallelism over the open internet, and
  its own paper states the key result directly:

    "For inference, performance does not depend much on bandwidth or sequence
     length but degrades in high-latency settings"

  That is an independent, measured, peer-reviewed confirmation of the
  LATENCY-BOUND finding above. Read it off their table:
    10x LESS bandwidth (1 Gbit/s -> 100 Mbit/s) at low latency costs
      {(1.22-1.19)/1.22*100:>4.1f}% on 3 servers and {(0.97-0.97)/0.97*100:>4.1f}% on 12 servers.
    Adding latency (<5ms -> 100ms) at the same bandwidth costs
      {(1.19-0.89)/1.19*100:>4.1f}% on 3 servers and {(0.97-0.44)/0.97*100:>4.1f}% on 12 servers.

  And note the SHARD-COUNT penalty, which is the memory constraint biting:
  4x more servers at 100 Mbit/s / 100 ms costs {(0.89-0.44)/0.89*100:.0f}% of throughput. More hosts
  means more hops means less throughput -- so the attacker cannot solve the
  memory problem by adding machines without paying for it in latency.

  EXTRAPOLATION to this model on stealthy consumer hosts: Petals' real-world
  figure is {PETALS['14 real servers, EU+NA']} steps/s for a 176B model on 14 servers with real GPUs
  (8-24GB each), already using their latency-aware peer selection. A {TOTAL_PARAMS/1e9:.0f}B model
  at INT4 is {weight_bytes_total/1024**3:,.0f} GiB, needing ~{n_hosts_agg} hosts at {USABLE_GB_PER_HOST:.0f} GB -- about {n_hosts_agg/14:.0f}x the hop count.
  Applying their measured 3-to-12-server scaling gives roughly 0.2 steps/s,
  i.e. ~720 tokens/hour -- which lands within ~25% of this script's {throughput(35,50)[0]:,.0f} tokens/hour.
  That is DIRECTIONAL CONSISTENCY, not validation: Petals is pipeline-parallel
  (shards by layer, one hop per stage) and this script is expert-parallel (fans
  out within each layer). Different topology, different synchronisation pattern.
  The second measured wide-area system, Anthropic's 15 Android TV boxes running
  Gemma4-26B-A4B pipeline-parallel at 0.39 tok/s (~1,400 tokens/hour, Oct 2026),
  lands in the same regime for a much smaller model. Nobody has measured a
  frontier model under expert parallelism over a WAN.""")

print()
print("-" * 78)
print("STATE OF THE ART IN LATENCY HIDING -- granted in full")
print("-" * 78)
spec_base, _, _ = throughput(35, 50)
spec_stealth, _, _ = throughput(35, 50, STEALTH_DUTY_CYCLE)
print(f"""  Speculative decoding is the real countermeasure to a latency-bound regime:
  verify k tokens per network round instead of k rounds. Measured average
  accepted length is {SPECULATIVE_ACCEPT:.0f} tokens/round (DSD reports 2.99-4.04; a WAN
  split-inference paper reports 1-5), so grant a {SPECULATIVE_ACCEPT:.0f}x multiplier:

    single-stream, unthrottled  {spec_base:>7,.0f} -> {spec_base*SPECULATIVE_ACCEPT:>7,.0f} tokens/hour
    single-stream, stealthy     {spec_stealth:>7,.0f} -> {spec_stealth*SPECULATIVE_ACCEPT:>7,.0f} tokens/hour

  Remaining gap to one datacenter user ({ds_user_tph:,.0f} tokens/hour):
    {ds_user_tph/(spec_base*SPECULATIVE_ACCEPT):>5,.0f}x unthrottled, {ds_user_tph/(spec_stealth*SPECULATIVE_ACCEPT):>5,.0f}x stealthy.
  Remaining gap in aggregate to one {DS_GPUS_PER_NODE}-GPU node: {ds_node_tph/(agg_stealth*SPECULATIVE_ACCEPT):,.0f}x.

  THREE CAVEATS, because this multiplier is not free:
   1. DSD was measured on 100 Gbps InfiniBand with 8B models, and claims its
      advantage in the regime 3t0 < t1 < 10t0 (t1 = link latency, t0 = compute).
      A botnet at 50 ms RTT against ~1 ms of local compute is t1 ~ 50*t0 --
      OUTSIDE the regime where the technique is claimed to work.
   2. It needs a draft model resident on every node, on top of the shard.
   3. Acceptance rates fall on out-of-distribution work -- and novel exploit
      development is precisely that. The {SPECULATIVE_ACCEPT:.0f}x is a chat-benchmark figure.

  Current decentralised-serving SOTA (BloomBee) reports up to 1.76x throughput
  over Petals/Helix. Real, and not an order of magnitude.""")

# ---------------------------------------------------------------------------
# Datacenter comparison
# ---------------------------------------------------------------------------
nvlink_bytes_per_s = NVLINK_TBPS_PER_GPU * 1e12
dc_tx_per_layer = bytes_per_layer / nvlink_bytes_per_s
dc_lat_per_layer = NVLINK_LATENCY_US / 1e6
dc_per_token = (dc_tx_per_layer + dc_lat_per_layer) * N_LAYERS

print()
print("-" * 78)
print("DATACENTER COMPARISON (GB200 NVL72, NVLink 1.8 TB/s per GPU)")
print("-" * 78)
print(f"""  communication time per token   {dc_per_token*1e6:,.1f} microseconds
  i.e. interconnect is NOT the constraint in a datacenter -- memory bandwidth is.

  interconnect ratio   1.8 TB/s vs {UPLOAD_MBPS_CASES[0]} Mbps
                       = {nvlink_bytes_per_s/(UPLOAD_MBPS_CASES[0]*1e6/8):,.0f}x  """
      f"""({len(str(int(nvlink_bytes_per_s/(UPLOAD_MBPS_CASES[0]*1e6/8))))-1} orders of magnitude)
  latency ratio        ~1 us vs 50 ms = {50000:,}x  (4-5 orders of magnitude)""")

base, _, _ = throughput(35, 50)
stealth, _, _ = throughput(35, 50, STEALTH_DUTY_CYCLE)
print(f"""
HEADLINE (35 Mbps upload, 50 ms RTT -- a realistic mid-range botnet)
  unthrottled   {base:,.0f} tokens/hour   ({base/3600:.3f} tokens/second)
  stealthy      {stealth:,.0f} tokens/hour   ({stealth/3600:.4f} tokens/second)

OPERATIONAL TRANSLATION (using the stealthy figure)
  one agentic reasoning step  ~2,000 tokens  ->  {2000/stealth:,.1f} hours
  one working day of an agent ~50,000 tokens ->  {50000/stealth/24:,.1f} days
  GTG-1002 scale campaign     ~10M tokens    ->  {10e6/stealth/24/365:,.1f} years

  The three token budgets are WORKING ASSUMPTIONS (round numbers), not measurements.""")

# ---------------------------------------------------------------------------
# Measured consumer-device comparison: Anthropic Frontier Red Team, 12 Oct 2026.
# Gemma4-26B-A4B, pipeline-parallel across 15 Android TV boxes (~$55, 4 GB each).
# This is COMPUTE-bound on weak CPUs over a LAN, a different regime from the
# latency-bound WAN model above; it is a measured neighbour, not a validation.
# ---------------------------------------------------------------------------
TV_TOKS_PER_S = 0.39          # demo figure
TV_TOKS_PER_S_LONGCTX = 0.30  # ~ averaged over a 32k context, per the report
print()
print("-" * 78)
print("MEASURED NEIGHBOUR: Anthropic's 15 TV boxes (26B MoE, pipeline-parallel)")
print("-" * 78)
for label, r in (("0.39 tok/s (demo)", TV_TOKS_PER_S), ("0.30 tok/s (32k ctx)", TV_TOKS_PER_S_LONGCTX)):
    tph = r * 3600
    print(f"""  {label}: {tph:,.0f} tokens/hour, {tph*24:,.0f}/day, {tph*24*365/1e6:,.1f}M/year per replica
    one reasoning step   ~2,000 tokens  -> {2000/r/60:>6,.0f} minutes
    one agent working day ~50,000 tokens -> {50000/r/86400:>6,.1f} days
    GTG-1002 scale campaign ~10M tokens  -> {10e6/r/86400/30.44:>6,.1f} months""")
print(f"""
  Breadth: a 2M-device fleet is ~{2e6/15:,.0f} replicas working in parallel
  (~{2e6/15*TV_TOKS_PER_S:,.0f} tokens/s aggregate; Anthropic's own estimate is ~40 H100s-worth).
  Depth: an intrusion is a sequential loop, so parallelism does not shorten any
  single chain. Months per enterprise-scale target; the ICS record implies years
  per OT target. Tokens/s bounds thinking, not pre-positioned execution.""")

# ---------------------------------------------------------------------------
# The worse, more realistic variants
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("WORSE VARIANTS (all more realistic than the charitable model above)")
print("-" * 78)
import math
n_hosts = int(hosts_for_weights)
ring_hops = 2 * (n_hosts - 1)
tree_hops = 2 * math.ceil(math.log2(n_hosts))
print(f"""  If attention must be SHARDED rather than replicated, every layer needs a
  collective all-reduce across hosts instead of {TOP_K} point-to-point sends:
    ring all-reduce over {n_hosts} hosts  = {ring_hops} sequential hops/layer
                                      -> {ring_hops*N_LAYERS*0.050/60:,.1f} minutes/token at 50ms
    tree all-reduce over {n_hosts} hosts  = {tree_hops} sequential hops/layer
                                      -> {tree_hops*N_LAYERS*0.050:,.1f} seconds/token at 50ms

  FAULT TOLERANCE: not free. Every one of the {barriers_per_token} barriers per token needs
  all selected expert hosts live simultaneously. Consumer machines sleep, get
  patched and get cleaned; with {n_hosts}+ hosts the expected time to a participant
  dropping is minutes. A dropped shard stalls the token until a replica takes
  over. Replication is possible, and costs every expert held >= 2x on hosts with
  the same uptime problem, plus the coordination traffic to detect and reroute.
  None of that is modelled above; all of it makes the attacker's number smaller.

  CONTEXT: an agentic intrusion carries a long context. KV cache grows per token
  and must live wherever attention runs, so replication gets more expensive the
  longer the operation runs. The charitable model above ignores this entirely.""")

# ---------------------------------------------------------------------------
# The steelman that actually works
# ---------------------------------------------------------------------------
small_model_params = 30e9
small_bytes = small_model_params * WEIGHT_BYTES
print()
print("-" * 78)
print("THE STEELMAN THAT WORKS -- and it is not distribution")
print("-" * 78)
print(f"""  A {small_model_params/1e9:.0f}B dense model at INT4 is {human_bytes(small_bytes)} and fits on ONE
  consumer GPU. Network traffic for inference: ZERO. Throughput on a mid-range
  consumer card: single-digit to low-tens of tokens/second -- i.e. roughly
  {10*3600:,.0f} tokens/hour, about {10*3600/stealth:,.0f}x the distributed frontier model.

  CONCLUSION: a botnet cannot USEFULLY shard a frontier model, but it does not need to.
  It can run a small model locally on every node, embarrassingly parallel, with
  no interconnect at all. The threat model is therefore MANY SMALL LOCAL MODELS,
  not one distributed large one -- which moves the question to the capability
  floor of small models, not to compute logistics.""")
print("=" * 78)
