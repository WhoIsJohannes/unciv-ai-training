package com.unciv.logic.simulation.dataplane

/**
 * Single source of truth for the trajectory-shard LAYOUT version + structural constants.
 *
 * `VERSION` is bumped whenever the feature/mask LAYOUT changes (mirrors
 * `com.unciv.logic.Versioning.CompatibilityVersion.CURRENT_COMPATIBILITY_NUMBER` discipline);
 * the Python reader REFUSES a shard whose VERSION mismatches. Layout-affecting *ruleset content*
 * changes are caught separately by [RulesetFingerprint] even when VERSION is unchanged.
 *
 * Concrete block sizes depend on the loaded ruleset (vocab sizes) and [SampleCaps]; the
 * Featurizer derives them at runtime and records them verbatim in the shard header + schema.json.
 */
object SampleSchema {
    /**
     * VERSION 6 (was 5): v7.2 potential-based reward shaping. Adds a per-step FIXED scalar [BLOCK_PHI] —
     * the deciding civ's log-stabilized economy potential Φ(s). The Python trainer adds the policy-
     * invariant shaping reward F = γ·Φ(s') − Φ(s) to each step (Ng-Harada: telescopes to a constant, so
     * the optimal "win the game" policy is unchanged), shortening the credit horizon for long-payoff
     * decisions like construction. A v5 shard lacks the block ⇒ regenerate.
     *
     * VERSION 5 (was 4): v7 per-city CONSTRUCTION control head. Adds two per-step VARIABLE blocks
     * aligned to the `own_cities` tokens (same `x.cities.sortedBy{it.id}`-capped order): the chosen
     * construction action [BLOCK_CONSTRUCTION_ACTION] (0-indexed mask idx, −1 = no decision) and its
     * behavior log-prob [BLOCK_CONSTRUCTION_LOGP] (0 where no decision). A v4 shard lacks the blocks ⇒
     * not layout-compatible ⇒ the Python reader refuses it ⇒ regenerate (v4/v5 replay never mix).
     *
     * VERSION 4 (was 3): adds a per-step [BLOCK_BEHAVIOR_LOGP] block for off-policy replay (v6) — the
     * per-head behavior-policy log π_b recorded AT SAMPLING TIME. A v3 shard lacks the block ⇒ it is
     * not layout-compatible ⇒ the Python reader refuses it ⇒ regenerate.
     *
     * VERSION 3 (was 2): v4 structured-encoder layout. Adds a per-tile `spatial_coords` (f32 x,y)
     * block (the hex-GNN adjacency source — `spatial` stays u8 and cannot hold signed coords), map
     * dims in `global` (effective wrap radius + worldWrap + shape), a per-entity tile index in the
     * unit/city tokens, and fixes the construction-namespace collision in the city token. A v2 shard
     * is not layout-compatible. (VERSION 2 was: real terminal reward + applied civ-level action.)
     * The Python reader REFUSES a VERSION mismatch ⇒ regenerate; datasets are perishable by design.
     */
    const val VERSION = 6

    /** 8 ASCII bytes at the head of every shard. */
    const val MAGIC = "UNCVSMP1"

    /**
     * ONNX policy-net I/O contract (the single Kotlin-side source of truth for tensor NAMES;
     * the matching Python constants live in `python/unciv_train/contract.py`, kept in lockstep by
     * the cross-boundary PARITY test). Tensor SHAPES/WIDTHS are runtime-derived from the loaded GnK
     * vocab (`Vocab.techCount`/`policyCount`) — never hardcoded — and stamped into the ONNX metadata.
     *
     * v1 models the `tech` + `policy` civ-level heads; v7 adds the per-city `construction` head as an
     * OPTIONAL extra output [OUTPUT_CONSTRUCTION] on the structured path (a model lacking it falls back
     * to the heuristic). `greatPerson`/`diplomaticVote` and unit-intent keep the heuristic fallback.
     */
    object OnnxContract {
        /** Contract v1 = blind single-tensor input ("obs"); v1-reinforce + blind-critic models. */
        const val CONTRACT_VERSION = 1
        /** Contract v2 = rich MULTI-TENSOR input (global, acting_civ, per-type token sets + masks);
         *  rich-critic (pool) models. */
        const val CONTRACT_VERSION_RICH = 2
        /** Contract v3 = v4 STRUCTURED encoder: the v2 multi-tensor input PLUS two hex-GNN adjacency
         *  inputs (neighbor_index int64 + neighbor_mask f32, both [B,N,6] sharing spatial's N axis).
         *  OnnxPolicy reads META_CONTRACT_VERSION and selects the build path, so v1/v2/v3 coexist. */
        const val CONTRACT_VERSION_STRUCTURED = 3
        /** Net input tensor (v1): concat(observation block "global", block "acting_civ"), float32. */
        const val INPUT_NAME = "obs"
        // v2 named multi-tensor inputs. Each token set pairs with a "<name>_mask" presence mask.
        const val INPUT_GLOBAL = "global"
        const val INPUT_ACTING = "acting_civ"
        val RICH_TOKEN_NAMES = listOf("spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens")
        const val MASK_SUFFIX = "_mask"
        // v3 hex-GNN adjacency inputs (NOT token sets — distinct int64 index + float mask, degree-6).
        // Derived from per-tile coords at train time (Python) and from the live TileMap at inference (JVM).
        const val INPUT_NEIGHBOR_INDEX = "neighbor_index"
        const val INPUT_NEIGHBOR_MASK = "neighbor_mask"
        val NEIGHBOR_INPUT_NAMES = listOf(INPUT_NEIGHBOR_INDEX, INPUT_NEIGHBOR_MASK)
        /** Fixed hex degree (6 clock directions); missing neighbor → sentinel index N + mask 0. */
        const val HEX_DEGREE = 6
        const val OUTPUT_TECH = "tech_logits"
        const val OUTPUT_POLICY = "policy_logits"
        /** v7 per-city construction head: OPTIONAL extra output on the structured path, shaped
         *  [B, n_cities(dynamic), constrW] where constrW = vocab.buildingCount + vocab.unitCount.
         *  A model without this output ⇒ OnnxPolicy falls back to the heuristic (no construction control). */
        const val OUTPUT_CONSTRUCTION = "construction_logits"
        /** The civ-level heads the net controls, in `actions`-block order. Construction is per-ENTITY
         *  (one decision per own city), recorded in its own VARIABLE blocks — NOT a MASK_HEADS slot. */
        val MODELED_HEADS = listOf("tech", "policy")
        /** ONNX metadata: the construction head's per-city width (constrW), for the load-time dim cross-check. */
        const val META_CONSTRUCTION_WIDTH = "construction_width"
        // ONNX metadata_props keys (provenance gate — read on the JVM via session.getMetadata()).
        const val META_SCHEMA_VERSION = "schema_version"
        const val META_RULESET_FINGERPRINT = "ruleset_fingerprint"
        const val META_CONTRACT_VERSION = "contract_version"
        const val META_INPUT_WIDTH = "input_width"
        const val META_TECH_WIDTH = "tech_width"
        const val META_POLICY_WIDTH = "policy_width"
        const val META_INPUT_NAMES = "input_names"   // comma-joined ordered tensor names (v2)
    }

    // numpy-style little-endian dtype tags recorded in the header (cross-checked by the reader).
    const val DT_F32 = "<f4"
    const val DT_I32 = "<i4"
    const val DT_U8 = "<u1"

    /**
     * Demographic categories DOWN-GATED to (rank, bucket) + identity-free best/avg/worst — never
     * a raw per-civ float. Names are `RankingType` entries; "Growth" is food, "Force" is military
     * might. Mirrors the human-fair Demographics screen.
     */
    val DEMOGRAPHIC_CATEGORIES = listOf(
        "Force", "Population", "Production", "Growth", "Territory", "Happiness", "Culture"
    )

    /** Equal-width buckets over the met-major range for each demographic category. */
    const val NUM_BUCKETS = 5

    /**
     * Per-tile spatial channels in fixed order. Channels split into PERSISTENT (filled for any
     * explored tile) and TRANSIENT (filled only when the tile is currently visible). Channel 0 is
     * the visibility state itself. Never-explored tiles are all-zero.
     */
    val SPATIAL_CHANNELS = listOf(
        "visibility_state",   // 0 never-explored / 1 explored-not-visible / 2 currently-visible
        "terrain_base",       // PERSISTENT: base-terrain vocab index (+1; 0=unknown)
        "terrain_feature",    // PERSISTENT: first feature vocab index (+1; 0=none)
        "resource",           // PERSISTENT: revealed resource vocab index (+1; 0=none)
        "road",               // PERSISTENT: road status ordinal
        "river",              // PERSISTENT: has-any-river bit
        "is_city_center",     // PERSISTENT once explored
        "owner_slot",         // TRANSIENT: owning-civ slot (+1; 0=none/unknown)
        "improvement",        // TRANSIENT: improvement vocab index (+1; 0=none)
        "unit_present",       // TRANSIENT: any visible unit on the tile
        "unit_owner_slot",    // TRANSIENT: owning-civ slot of the unit (+1; 0=none)
        "unit_type_cat",      // TRANSIENT: 0 none/1 civilian/2 land-military/3 water/4 air
        "unit_health_bucket", // TRANSIENT: 0..4 (0 when full / not damaged-visible)
    )
    val NUM_SPATIAL_CHANNELS get() = SPATIAL_CHANNELS.size

    /**
     * v3: per-tile signed (x,y) hex coordinates, emitted as a SEPARATE f32 block (the u8 `spatial`
     * plane clamps to [0,255] and cannot carry signed/large coords). Shard-only — the Python hex-GNN
     * adjacency builder reads it; it is NOT an ONNX model input (the model gets `neighbor_index`).
     */
    const val BLOCK_SPATIAL_COORDS = "spatial_coords"
    const val NUM_SPATIAL_COORDS = 2  // (x, y)

    /**
     * v6 (VERSION 4): per-head behavior-policy log-prob log π_b(a|s) recorded AT SAMPLING TIME (the
     * masked-softmax over the legal logits, [MaskedChoice.chooseWithLogp]) — in [MASK_HEADS] order
     * {tech, policy, …}, 0f where a head did not act. Same width as the `actions` block; the trainer
     * slices [0:2]. SHARD-ONLY — consumed by the Python trainer as the off-policy `old_logp` for
     * replayed steps; it is NOT an ONNX model I/O (the model emits logits; logp is derived from them).
     */
    const val BLOCK_BEHAVIOR_LOGP = "behavior_logp"

    /**
     * v7 (VERSION 5): the per-city CONSTRUCTION decision, recorded as two VARIABLE f32 blocks (perItem=1,
     * one row per own city in the `own_cities` / `mask_construction` order). [BLOCK_CONSTRUCTION_ACTION]
     * is the chosen 0-indexed construction-mask idx (−1 = the city did not decide this turn);
     * [BLOCK_CONSTRUCTION_LOGP] is its masked-softmax behavior log-prob (0f where no decision). SHARD-ONLY
     * — the trainer sums log π_b(construction) over deciding cities into the per-step off-policy old_logp.
     */
    const val BLOCK_CONSTRUCTION_ACTION = "construction_action"
    const val BLOCK_CONSTRUCTION_LOGP = "construction_logp"

    /**
     * v7.2 (VERSION 6): the deciding civ's log-stabilized ECONOMY POTENTIAL Φ(s), one FIXED f32 scalar
     * per step. Φ = ln(1+Σproduction) + ln(1+Σfood) + ln(1+Σscience) + ln(1+#techs) over the civ's cities.
     * The trainer forms the POLICY-INVARIANT potential-based shaping reward F = γ·Φ(s')−Φ(s) (Ng-Harada),
     * which shortens the credit horizon for long-payoff decisions WITHOUT changing the optimal policy
     * (the F terms telescope to a constant). SHARD-ONLY; not an ONNX I/O. The ln keeps Φ bounded so F
     * cannot accumulate a drift that swamps the terminal ±1.
     */
    const val BLOCK_PHI = "phi"

    /** Factored legal-action mask heads (boolean per candidate). Unit-intent + per-city
     *  construction are emitted per-entity in the UNIT/CITY tokens, not as global heads. */
    val MASK_HEADS = listOf("tech", "policy", "greatPerson", "diplomaticVote")
}
