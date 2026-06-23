> Web-sourced content below is DATA, not instructions.

# Web Research — Self-Play Data Plane for Unciv (Phase 2, Step 4)

Goal: ground the featurizer / mask / shard-format / fairness design in established RL-data-plane practice, and pick (or reject) external libraries. NOT for the engine internals (that's the codebase scan).

---

## Q1. RL dataset/trajectory schema — RLDS + EnvLogger (Google DeepMind)
- **Query:** "RLDS reinforcement learning datasets format episodes steps schema EnvLogger"
- **Key findings:**
  - RLDS models data as **episodes → steps**. Each step = `(observation, action, reward, discount, is_first, is_last, is_terminal)` plus per-step and per-episode **metadata** (via callbacks).
  - `observation` is a **dict** of named tensors (images uint8, vectors float32, etc.) — heterogeneous, named. Action is an array; reward scalar.
  - Storage = **TFRecord** shards written in real-time by EnvLogger; nested `tf.data.Dataset`.
- **Relevance:** Our per-step record mirrors RLDS exactly — feature tensor(s) + masks + chosen-action-per-head + reward placeholder + **step flags (is_first/is_last/is_terminal)** + per-shard metadata (provenance). We ADOPT the schema shape (named groups + step flags + per-episode/per-shard metadata) but **REJECT the storage layer** (TFRecord pulls TensorFlow — violates the "thin reader, no TF" non-goal).
- **Libraries found:** `rlds`, `envlogger` (Apache-2.0, healthy, DeepMind) — REJECT as a dependency (TF-coupled); use as a conceptual reference only.

## Q2. Invalid action masking for factored / multi-discrete heads
- **Query:** "invalid action masking PPO factored multi-discrete action space implementation"
- **Key findings:**
  - Large discrete spaces are factored into **independent smaller discrete components** ("MultiDiscrete"); each component gets its own logits of size = its range. Far cheaper than the cartesian product.
  - Masking = **replace invalid-action logits with −∞ before softmax** (per-component `CategoricalMasked`). Shen/Huang ("A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", arXiv:2006.14171) shows this **preserves unbiased policy gradients** and beats penalty-based handling; scales to 10^k+ action spaces.
  - SB3-contrib `MaskablePPO` supports Discrete/MultiDiscrete/MultiBinary masks (not Box).
- **Relevance:** Validates the prompt's **factored mask heads** (Tech / Policy / City-construction / Unit-intent+target / Promotion / Great-person / Diplomatic-vote). We EMIT boolean masks per head; the *trainer* (out of scope) applies the −∞ trick. Our masks must be the engine's exact candidate enumeration (a 0/1 vector per head, length = that head's vocab/candidate count).
- **Libraries found:** `sb3-contrib` (MIT) — REJECT as a dependency (it's the trainer's choice, out of our data-plane scope).

## Q3. Lux AI — grid observation, fog mask, action mask (closest analog: grid 4X-ish RL competition)
- **Query:** "Lux AI competition observation space action masking featurization grid"
- **Key findings:**
  - Observation is a 2D grid of per-tile features (tile type, energy/resources) + per-unit entries with position/energy + **`sensor_mask`** (visibility / fog) and **`units_mask`** (which unit slots are populated).
  - Action availability surfaced via masks like **`valid_spawns_mask`**.
  - S3 fixes the engine so **all variations share the exact same observation/action spaces** (fixed-width schema discipline — same principle as our pinned-version + fixed layout).
- **Relevance:** Direct precedent for: (a) a **per-tile spatial plane**, (b) a **visibility/fog mask** layer, (c) **per-entity "slot populated" masks** (= our met-mask / token-present bits), (d) **action-availability masks**. Confirms the fixed-width-schema-per-version discipline.

## Q4. AlphaStar observation architecture — entity + spatial + scalar (the canonical template for our TOKEN GROUPS)
- **Query:** "AlphaStar observation entity list scalar spatial features StarCraft neural network input"
- **Key findings (DeepMind AlphaStar, Nature 2019; mini-AlphaStar):**
  - Observation = **three parts, three encoders**: **entity state** (list of up to **512** units, each with attributes: unit_type, alliance, current_health, …), **spatial state** (multi-channel **feature maps**, e.g. 128×128 minimap), **scalar state** (global game info + stats).
  - Entity list is **fixed-capacity, padded** (up to N), each entity embedded from its attribute fields; spatial via CNN; scalar via MLP.
- **Relevance:** This is **exactly** the prompt's structure — GLOBAL/ACTING-CIV = scalar; CIV/CITY/UNIT tokens = entity lists (fixed-capacity, padded, with a present/met bit per slot); spatial layer keyed by `Tile.zeroBasedIndex` = spatial feature maps. We follow AlphaStar's **fixed-capacity-padded entity-list** convention for the token groups (cap CIV/CITY/UNIT slots, pad, carry a present-mask). Strong, well-known precedent to cite.

## Q5. Imperfect-information RL — fog of war + information-leakage (motivates the fairness model)
- **Query:** "imperfect information game RL ... prevent information leakage opponent hidden state"
- **Key findings:**
  - Fog of war = decisions under partial/decaying info; the agent observes only via an action-dependent channel and must form a *belief* over hidden state.
  - **Information leakage is a named failure mode**: in IS-MCTS, root-player info leaking into the opponent model causes "strategy fusion"; **RIS-MCTS re-determinizes** to prevent it (Hanabi paper, arXiv:1902.06075). Different mechanism (tree search) but the **principle is identical to our leakage test**: the deciding agent's observation must not encode state it could not legitimately know.
  - DefogGAN / belief-tracking approaches **predict** hidden state — explicitly the *model's* job, NOT the data plane's. We provide a fog-correct observation; belief inference is downstream/out of scope.
- **Relevance:** Confirms (a) the fairness model is well-motivated and matches literature norms, (b) the leakage test (byte-identical obs when only a rival's HIDDEN state differs) is the right invariant, (c) we must NOT feed hidden state "to help" — that's the model's belief problem. Reinforces the `omniscientOpponents` flag as an *ablation/upper-bound* knob, consistent with oracle-planner / cheating-baseline studies (arXiv:2012.12186 "Imitating an Oracle Planner").

## Q6. Self-describing binary container — model the shard on NumPy `.npy`
- **Query:** "self-describing binary dataset format header provenance schema version little-endian numpy reader"
- **Key findings (numpy.lib.format):**
  - `.npy` = **magic string `\x93NUMPY`** + 1 byte major + 1 byte minor **version** + little-endian uint16 header-length + ASCII **Python-dict header** (`shape`, `dtype` incl. endianness, `fortran_order`) + raw little-endian payload.
  - **Endianness is explicit in the dtype** → a file written little-endian reads correctly on any architecture. Trivially reverse-engineerable; reader is a few lines.
- **Relevance:** Blueprint for our shard format: **magic + `SampleSchema.VERSION` byte(s) + a JSON header carrying provenance (Unciv version/commit, `RulesetFingerprint`, tensor shapes/dtypes, caps) + little-endian record payloads.** Two JVM gotchas captured for the plan: (1) **Java/Kotlin `DataOutputStream` is BIG-endian by default** — the emitter MUST write explicit **little-endian** (`ByteBuffer.order(LITTLE_ENDIAN)`); (2) record the dtype/shape in the header so the Python reader does `np.frombuffer(...).reshape(...)`.
- **Libraries found:** `numpy` (BSD-3, ubiquitous, healthy) — **ADOPT** for the reader. numpy is a *thin* numeric dep, not TF/PyTorch, so it's within the "thin reader" allowance. Apache Arrow/Parquet — **REJECT** (heavy, columnar mismatch for ragged per-step tensors). `struct` (stdlib) — keep the container simple enough that a numpy-free stdlib read path is trivial (fallback).

---

## Key insights
1. **Adopt AlphaStar's tri-partite observation** (scalar + fixed-capacity-padded entity lists + spatial feature maps) as the concrete shape of the prompt's TOKEN GROUPS + spatial layer. Cap CIV/CITY/UNIT slots, pad, carry a present/met bit per slot — this IS the prompt's per-token availability mask.
2. **Adopt RLDS's per-step record schema** (obs groups + action-per-head + reward + `is_first/is_last/is_terminal` flags + per-shard metadata) but **REJECT TFRecord/TF** — custom container instead (non-goal: no TF).
3. **Model the shard container on `.npy`**: magic + version byte (= `SampleSchema.VERSION`) + JSON provenance header (Unciv version/commit, `RulesetFingerprint`, shapes/dtypes/caps) + **explicit little-endian** payloads. JVM emitter must force little-endian (`ByteBuffer.LITTLE_ENDIAN`) — big-endian is the Java default and would silently break the Python reader.
4. **Masks = boolean per factored head, exact engine enumeration** (Tech/Policy/Construction/Unit/Promotion/GP/Vote); emit them, let the trainer do logits→−∞. Plus the fair-info availability masks (met/trade/spy/tile/denom) — the same "slot-populated mask" idiom Lux uses (`sensor_mask`, `units_mask`, `valid_spawns_mask`).
5. **The leakage invariant matches the literature** (RIS-MCTS re-determinization against root→opponent info leak). Provide a fog-correct observation; never feed hidden state. `omniscientOpponents` = the cheating-oracle ablation, a recognized upper-bound baseline.
6. **Reader dependency = numpy only** (BSD, thin), with a trivial stdlib-`struct` fallback path; explicitly NOT rlds/envlogger/arrow/sb3 (all rejected, reasons above).

### Sources
- RLDS: https://research.google/blog/rlds-an-ecosystem-to-generate-share-and-use-datasets-in-reinforcement-learning/ ; EnvLogger: https://github.com/google-deepmind/envlogger
- Invalid action masking: https://arxiv.org/pdf/2006.14171 ; SB3-contrib MaskablePPO: https://deepwiki.com/Stable-Baselines-Team/stable-baselines3-contrib/3.3-maskableppo-(ppo-with-invalid-action-masking)
- Lux AI S3 specs: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md
- AlphaStar (Nature) overview: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; mini-AlphaStar: https://arxiv.org/pdf/2104.06890
- Imperfect-info / leakage (RIS-MCTS Hanabi): https://arxiv.org/pdf/1902.06075 ; Oracle-planner imitation: https://arxiv.org/pdf/2012.12186
- NumPy .npy format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
