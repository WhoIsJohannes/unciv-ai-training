package com.unciv.logic.simulation.dataplane

import com.unciv.logic.GameInfo
import com.unciv.logic.battle.CityCombatant
import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.map.MapShape
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.ui.screens.victoryscreen.RankingType

/**
 * Builds the fog-correct observation for a deciding civ X (AlphaStar-style: scalar + entity lists +
 * spatial planes). Fairness lives in [FairOpponentModel] (civ tokens) and the tile/spy gates here;
 * the ONLY switch that changes observations is `config.omniscientOpponents`.
 *
 * SIZE: entity lists store ONLY present entities (VARIABLE blocks, fixed width per token) — padding
 * to the caps is the data loader's job, not storage. Masks + the spatial plane use u8. This keeps a
 * Tiny game in the hundreds-of-KB, dominated by the actual map (not megabytes of zero padding).
 * Present entities are emitted in CANONICAL order (civs/cities/units by id) so position is no channel.
 */
class Featurizer(private val gameInfo: GameInfo, val vocab: Vocab, val config: SampleConfig) {

    private val ruleset = gameInfo.ruleset
    private val caps = config.caps
    private val fair = FairOpponentModel(vocab, ruleset, config)
    private val demographics = SampleSchema.DEMOGRAPHIC_CATEGORIES.map { RankingType.valueOf(it) }
    private val channels = SampleSchema.NUM_SPATIAL_CHANNELS

    val civTokenWidth get() = fair.tokenWidth
    private val cityTokenWidth = 17  // v3: +centerTile.zeroBasedIndex (entity↔GNN-node co-location)
    private val unitTokenWidth = 9   // v3: +currentTile.zeroBasedIndex

    fun observe(x: Civilization): Observation {
        var overflow = false
        val omni = config.omniscientOpponents
        val demoCtx = DemographicsContext(x, demographics)

        // ---- present opponent civs (met, or all if omniscient), canonical by civID ----
        val presentCivs = gameInfo.civilizations
            .filter { it !== x && (it.isMajorCiv() || it.isCityState) && (omni || x.knows(it)) }
            .sortedBy { it.civID }
            .let { if (it.size > caps.maxCivTokens) { overflow = true; it.take(caps.maxCivTokens) } else it }
        val presentCivIndex = presentCivs.withIndex().associate { (i, c) -> c.civID to i }
        val ownerSlot = { civId: String -> if (civId == x.civID) 255 else (presentCivIndex[civId]?.plus(1) ?: 0) }

        val civTokens = FloatArray(presentCivs.size * civTokenWidth)
        presentCivs.forEachIndexed { i, c ->
            System.arraycopy(fair.encode(x, c, demoCtx), 0, civTokens, i * civTokenWidth, civTokenWidth)
        }

        // ---- diplo edges among present civs (relationshipLevel ordinal+1), present×present ----
        val n = presentCivs.size
        val diplo = FloatArray(n * n)
        for (i in 0 until n) for (j in 0 until n) {
            if (i == j) continue
            val lvl = presentCivs[i].getDiplomacyManager(presentCivs[j])?.relationshipLevel()?.ordinal
            if (lvl != null) diplo[i * n + j] = (lvl + 1).toFloat()
        }

        // ---- own cities (EXACT) ----
        val ownCityList = x.cities.sortedBy { it.id }
            .let { if (it.size > caps.maxOwnCities) { overflow = true; it.take(caps.maxOwnCities) } else it }
        val ownCities = FloatArray(ownCityList.size * cityTokenWidth)
        ownCityList.forEachIndexed { i, c -> writeCityToken(ownCities, i * cityTokenWidth, c, x, true, ownerSlot) }

        // ---- opponent cities (tile-gated) + spy-gated stealable-tech rows ----
        val visibleOppCities = gameInfo.civilizations.asSequence()
            .filter { it !== x }.flatMap { it.cities.asSequence() }
            .filter { omni || x.viewableTiles.contains(it.getCenterTile()) }
            .sortedBy { it.id }.toList()
            .let { if (it.size > caps.maxVisOppCities) { overflow = true; it.take(caps.maxVisOppCities) } else it }
        val oppCities = FloatArray(visibleOppCities.size * cityTokenWidth)
        val oppCityOwners = ArrayList<String>(); val oppCityValues = ArrayList<FloatArray>()
        val spyRows = ArrayList<FloatArray>()  // one techCount-wide u8 row per spy-occupied city
        visibleOppCities.forEachIndexed { i, city ->
            writeCityToken(oppCities, i * cityTokenWidth, city, x, false, ownerSlot)
            oppCityOwners.add(city.civ.civID)
            oppCityValues.add(oppCities.copyOfRange(i * cityTokenWidth, i * cityTokenWidth + cityTokenWidth))
            val hasSpy = omni || x.espionageManager.getSpiesInCity(city).any { it.isSetUp() }
            if (hasSpy) {
                val row = FloatArray(vocab.techCount)
                for (t in x.espionageManager.getTechsToSteal(city.civ)) vocab.tech(t).takeIf { it >= 0 }?.let { row[it] = 1f }
                spyRows.add(row)
            }
        }
        val spyTech = flatten(spyRows, vocab.techCount)

        // ---- own + opponent units (tile-gated) ----
        val ownUnitList = x.units.getCivUnits().sortedBy { it.currentTile.zeroBasedIndex }.toList()
            .let { if (it.size > caps.maxOwnUnits) { overflow = true; it.take(caps.maxOwnUnits) } else it }
        val ownUnits = FloatArray(ownUnitList.size * unitTokenWidth)
        ownUnitList.forEachIndexed { i, u -> writeUnitToken(ownUnits, i * unitTokenWidth, u, x, true, ownerSlot) }

        val visibleOppUnits = (if (omni) gameInfo.tileMap.tileList.asSequence() else x.viewableTiles.asSequence())
            .flatMap { it.getUnits() }.filter { it.civ !== x }
            .sortedBy { it.currentTile.zeroBasedIndex }.toList()
            .let { if (it.size > caps.maxVisOppUnits) { overflow = true; it.take(caps.maxVisOppUnits) } else it }
        val oppUnits = FloatArray(visibleOppUnits.size * unitTokenWidth)
        val oppUnitOwners = ArrayList<String>(); val oppUnitValues = ArrayList<FloatArray>()
        visibleOppUnits.forEachIndexed { i, u ->
            writeUnitToken(oppUnits, i * unitTokenWidth, u, x, false, ownerSlot)
            oppUnitOwners.add(u.civ.civID)
            oppUnitValues.add(oppUnits.copyOfRange(i * unitTokenWidth, i * unitTokenWidth + unitTokenWidth))
        }

        // ---- masks ----
        val constrW = vocab.buildingCount + vocab.unitCount
        val construction = FloatArray(ownCityList.size * constrW)
        ownCityList.forEachIndexed { i, c ->
            val m = LegalActionMasks.constructionMask(c, vocab); for (k in m.indices) if (m[k]) construction[i * constrW + k] = 1f
        }
        val promoW = vocab.promotionCount
        val promotion = FloatArray(ownUnitList.size * promoW)
        ownUnitList.forEachIndexed { i, u ->
            val m = LegalActionMasks.promotionMask(u, vocab); for (k in m.indices) if (m[k]) promotion[i * promoW + k] = 1f
        }
        val voteMask = FloatArray(presentCivs.size) // 1 if a known major (votable) — present-civ aligned
        presentCivs.forEachIndexed { i, c -> if (c.isMajorCiv()) voteMask[i] = 1f }

        val blocks = listOf(
            fixedF32("global", buildGlobal(x, demoCtx)),
            fixedF32("acting_civ", buildActingCiv(x)),
            varF32("civ_tokens", civTokenWidth, civTokens),
            varU8("diplo_edges", maxOf(n, 1), diplo),                 // present×present matrix
            varF32("own_cities", cityTokenWidth, ownCities),
            varF32("opp_cities", cityTokenWidth, oppCities),
            varU8("spy_stealable_tech", vocab.techCount, spyTech),
            varF32("own_units", unitTokenWidth, ownUnits),
            varF32("opp_units", unitTokenWidth, oppUnits),
            fixedU8("spatial", buildSpatial(x, ownerSlot)),
            fixedF32(SampleSchema.BLOCK_SPATIAL_COORDS, buildSpatialCoords()),  // v3: signed (x,y), GNN adjacency source
            fixedU8("mask_tech", boolF(LegalActionMasks.techMask(x, vocab))),
            fixedU8("mask_policy", boolF(LegalActionMasks.policyMask(x, vocab, ruleset))),
            fixedU8("mask_greatPerson", boolF(LegalActionMasks.greatPersonMask(x, vocab))),
            varU8("mask_diplomaticVote", 1, voteMask),
            varU8("mask_construction", constrW, construction),
            varU8("mask_promotion", promoW, promotion),
        )
        return Observation(blocks, presentCivIndex, civTokenWidth, oppCityOwners, oppCityValues,
            oppUnitOwners, oppUnitValues, overflow)
    }

    // ---- block factory helpers ----
    private fun fixedF32(name: String, a: FloatArray) = Observation.Block(name, SampleSchema.DT_F32, BlockKind.FIXED, 0, a)
    private fun fixedU8(name: String, a: FloatArray) = Observation.Block(name, SampleSchema.DT_U8, BlockKind.FIXED, 0, a)
    private fun varF32(name: String, perItem: Int, a: FloatArray) = Observation.Block(name, SampleSchema.DT_F32, BlockKind.VARIABLE, perItem, a)
    private fun varU8(name: String, perItem: Int, a: FloatArray) = Observation.Block(name, SampleSchema.DT_U8, BlockKind.VARIABLE, perItem, a)
    private fun boolF(b: BooleanArray) = FloatArray(b.size) { if (b[it]) 1f else 0f }
    private fun flatten(rows: List<FloatArray>, w: Int): FloatArray {
        val out = FloatArray(rows.size * w); rows.forEachIndexed { i, r -> System.arraycopy(r, 0, out, i * w, w) }; return out
    }

    // ---- block builders ----

    private fun buildGlobal(x: Civilization, demoCtx: DemographicsContext): FloatArray {
        val agg = FloatArray(demographics.size * 3)
        demographics.forEachIndexed { i, cat ->
            val s = demoCtx.perCategory.getValue(cat)
            agg[i * 3] = s.best; agg[i * 3 + 1] = s.avg; agg[i * 3 + 2] = s.worst
        }
        // v3 map dims: the Python hex-GNN adjacency builder needs the EFFECTIVE wrap radius
        // (rectangular maps wrap by width/2, hex by radius — see TileMap.getIfTileExistsOrNull),
        // plus the worldWrap bit + shape ordinal. Read by NAMED schema field (mapDims), not a raw offset.
        val mp = gameInfo.tileMap.mapParameters
        val effWrapRadius = if (mp.shape == MapShape.rectangular) mp.mapSize.width / 2 else mp.mapSize.radius
        val shapeOrdinal = when (mp.shape) {
            MapShape.rectangular -> 0f
            MapShape.flatEarth -> 2f
            else -> 1f   // hexagonal (default)
        }
        val head = floatArrayOf(
            gameInfo.turns.toFloat(), x.tech.era.eraNumber.toFloat(),
            gameInfo.tileMap.tileList.size.toFloat(),
            x.getKnownCivs().count { it.isMajorCiv() }.toFloat(),
            gameInfo.civilizations.count { it.isMajorCiv() && it.isAlive() }.toFloat(),
            effWrapRadius.toFloat(),
            if (mp.worldWrap) 1f else 0f,
            shapeOrdinal,
        )
        return head + agg
    }

    private fun buildActingCiv(x: Civilization): FloatArray {
        val s = x.stats.statsForNextTurn
        val head = floatArrayOf(
            x.gold.toFloat(), s.gold, s.science, s.culture, s.faith, s.food, s.production,
            x.getHappiness().toFloat(), x.tech.era.eraNumber.toFloat(),
            x.cities.size.toFloat(), x.units.getCivUnitsSize().toFloat(),
            x.calculateTotalScore().toInt().toFloat(), x.tech.researchedTechnologies.size.toFloat(),
        )
        val ownTech = FloatArray(vocab.techCount)
        for (t in x.tech.techsResearched) vocab.tech(t).takeIf { it >= 0 }?.let { ownTech[it] = 1f }
        val ownPolicy = FloatArray(vocab.policyCount)
        for (p in x.policies.adoptedPolicies) vocab.policy(p).takeIf { it >= 0 }?.let { ownPolicy[it] = 1f }
        val ownBranch = FloatArray(vocab.policyBranchCount)
        for (p in x.policies.adoptedPolicies) vocab.policyBranch(p).takeIf { it >= 0 }?.let { ownBranch[it] = 1f }
        return head + ownTech + ownPolicy + ownBranch
    }

    private fun writeCityToken(arr: FloatArray, off: Int, city: City, x: Civilization, isOwn: Boolean,
                               ownerSlot: (String) -> Int) {
        val w = TokenSlice(arr, off, cityTokenWidth)
        w.put(1f)
        w.put(if (isOwn) 1f else 0f)
        w.put(ownerSlot(city.civ.civID))
        w.put(city.population.population)
        w.put(CityCombatant(city).getDefendingStrength(null))
        w.put(((city.health.coerceIn(0, 200)) / 200f * 4f).toInt())
        w.put(city.getCenterTile().airUnits.size)
        val rel = city.religion.getMajorityReligionName()
        w.put((rel?.let { vocab.religion(it) + 1 }) ?: 0)
        w.put(if (city.isInResistance()) 1f else 0f)
        w.put(if (city.isPuppet) 1f else 0f)
        w.put(if (city.isBeingRazed) 1f else 0f)
        val hasSpy = config.omniscientOpponents ||
            (!isOwn && x.espionageManager.getSpiesInCity(city).any { it.isSetUp() })
        w.put(if (hasSpy) 1f else 0f)
        w.put(city.getCenterTile().zeroBasedIndex)   // v3: tile index → co-locate entity with its GNN node
        if (isOwn || hasSpy) {
            // v3 fix: collision-free building∪unit code (see Vocab.constructionCode).
            w.put(vocab.constructionCode(city.cityConstructions.currentConstructionName()))
            w.put(city.cityConstructions.getBuiltBuildings().count())
        }
    }

    private fun writeUnitToken(arr: FloatArray, off: Int, u: MapUnit, x: Civilization, isOwn: Boolean,
                               ownerSlot: (String) -> Int) {
        val w = TokenSlice(arr, off, unitTokenWidth)
        val capital = x.getCapital()?.getCenterTile()
        val t = u.currentTile
        w.put(1f)
        w.put(if (isOwn) 1f else 0f)
        w.put(ownerSlot(u.civ.civID))
        w.put(unitTypeCat(u))
        w.put((u.health.coerceIn(0, 100) / 100f * 4f).toInt())
        w.put(if (capital != null) t.position.x.toInt() - capital.position.x.toInt() else 0)
        w.put(if (capital != null) t.position.y.toInt() - capital.position.y.toInt() else 0)
        w.put(u.promotions.getAvailablePromotions().count())
        w.put(t.zeroBasedIndex)   // v3: tile index → co-locate entity with its GNN node
    }

    private fun unitTypeCat(u: MapUnit): Int = when {
        u.baseUnit.isCivilian() -> 1
        u.baseUnit.isAirUnit() -> 4
        u.baseUnit.isWaterUnit -> 3
        else -> 2
    }

    /** Per-tile spatial planes (u8), keyed by zeroBasedIndex. owner_slot: 0 none, 1..N present civ, 255 self. */
    private fun buildSpatial(x: Civilization, ownerSlot: (String) -> Int): FloatArray {
        val tiles = gameInfo.tileMap.tileList
        val out = FloatArray(tiles.size * channels)
        val omni = config.omniscientOpponents
        for (tile in tiles) {
            val base = tile.zeroBasedIndex * channels
            if (base < 0 || base + channels > out.size) continue
            val visible = omni || x.viewableTiles.contains(tile)
            val explored = visible || x.hasExplored(tile)
            out[base] = if (visible) 2f else if (explored) 1f else 0f
            if (!explored) continue
            out[base + 1] = ((vocab.terrain(tile.baseTerrain) + 1).coerceAtMost(255)).toFloat()
            out[base + 2] = ((tile.terrainFeatures.firstOrNull()?.let { vocab.terrain(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
            out[base + 3] = ((tile.resource?.let { vocab.resource(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
            out[base + 4] = tile.roadStatus.ordinal.toFloat()
            out[base + 5] = if (tile.hasBottomRightRiver || tile.hasBottomRiver || tile.hasBottomLeftRiver) 1f else 0f
            out[base + 6] = if (tile.isCityCenter()) 1f else 0f
            if (!visible) continue
            out[base + 7] = (tile.getOwner()?.civID?.let { ownerSlot(it) } ?: 0).toFloat()
            out[base + 8] = ((tile.improvement?.let { vocab.improvement(it) + 1 } ?: 0).coerceAtMost(255)).toFloat()
            val unit = tile.getUnits().firstOrNull()
            if (unit != null) {
                out[base + 9] = 1f
                out[base + 10] = (ownerSlot(unit.civ.civID)).toFloat()
                out[base + 11] = unitTypeCat(unit).toFloat()
                out[base + 12] = (unit.health.coerceIn(0, 100) / 100f * 4f).toFloat()
            }
        }
        return out
    }

    /**
     * v3: per-tile signed (x,y) hex coords as a SEPARATE f32 block (u8 `spatial` clamps [0,255] and
     * can't carry signed/large coords). Always written (position is static, fog-independent). The
     * Python hex-GNN adjacency builder reads this; it is NOT an ONNX model input.
     */
    private fun buildSpatialCoords(): FloatArray {
        val tiles = gameInfo.tileMap.tileList
        val w = SampleSchema.NUM_SPATIAL_COORDS
        val out = FloatArray(tiles.size * w)
        for (tile in tiles) {
            val base = tile.zeroBasedIndex * w
            if (base < 0 || base + w > out.size) continue
            out[base] = tile.position.x.toFloat()
            out[base + 1] = tile.position.y.toFloat()
        }
        return out
    }

    private class TokenSlice(val arr: FloatArray, val off: Int, val width: Int) {
        private var c = 0
        fun put(v: Float) { if (c < width) arr[off + c] = v; c++ }
        fun put(v: Int) = put(v.toFloat())
    }
}
