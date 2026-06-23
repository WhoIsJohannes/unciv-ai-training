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
     * VERSION 2 (was 1): trajectory shards now carry a real terminal reward + an end-of-game
     * terminal step per civ (was a hardcoded `reward=0f`, `isTerminal=0` placeholder). The recorded
     * civ-level action is now ALSO applied to the game (the policy DRIVES tech+policy for controlled
     * civs — see [DataPlaneHooks]), so a v1 shard (label-only, no reward) is not training-compatible.
     * The Python reader REFUSES a VERSION mismatch ⇒ regenerate; datasets are perishable by design.
     */
    const val VERSION = 2

    /** 8 ASCII bytes at the head of every shard. */
    const val MAGIC = "UNCVSMP1"

    /**
     * ONNX policy-net I/O contract (the single Kotlin-side source of truth for tensor NAMES;
     * the matching Python constants live in `python/unciv_train/contract.py`, kept in lockstep by
     * the cross-boundary PARITY test). Tensor SHAPES/WIDTHS are runtime-derived from the loaded GnK
     * vocab (`Vocab.techCount`/`policyCount`) — never hardcoded — and stamped into the ONNX metadata.
     *
     * v1 models ONLY the `tech` + `policy` civ-level heads; `greatPerson`/`diplomaticVote` and all
     * per-entity heads keep the heuristic/RandomPolicy fallback.
     */
    object OnnxContract {
        const val CONTRACT_VERSION = 1
        /** Net input tensor: concat(observation block "global", block "acting_civ"), float32. */
        const val INPUT_NAME = "obs"
        const val OUTPUT_TECH = "tech_logits"
        const val OUTPUT_POLICY = "policy_logits"
        /** The civ-level heads the net controls in v1, in `actions`-block order. */
        val MODELED_HEADS = listOf("tech", "policy")
        // ONNX metadata_props keys (provenance gate — read on the JVM via session.getMetadata()).
        const val META_SCHEMA_VERSION = "schema_version"
        const val META_RULESET_FINGERPRINT = "ruleset_fingerprint"
        const val META_CONTRACT_VERSION = "contract_version"
        const val META_INPUT_WIDTH = "input_width"
        const val META_TECH_WIDTH = "tech_width"
        const val META_POLICY_WIDTH = "policy_width"
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

    /** Factored legal-action mask heads (boolean per candidate). Unit-intent + per-city
     *  construction are emitted per-entity in the UNIT/CITY tokens, not as global heads. */
    val MASK_HEADS = listOf("tech", "policy", "greatPerson", "diplomaticVote")
}
