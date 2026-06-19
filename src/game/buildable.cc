#include "game/buildable.h"

#include <algorithm>
#include <string>
#include <unordered_map>

namespace odai::game {

int turnsToBuild(int productionCost, int accumulated, int perTurn) {
    const int remaining = productionCost - accumulated;
    if (remaining <= 0) return 0;
    const int rate = std::max(1, perTurn);
    return (remaining + rate - 1) / rate;  // Ceiling division.
}

const std::vector<BuildableItem>& defaultBuildables() {
    static const std::vector<BuildableItem> kItems = {
        // Units -- icons from assets/icons/unit_icons.json.
        {"warrior",  "Warrior",  "warrior",    BuildableKind::Unit,      40, "warrior"},
        {"spearman", "Spearman", "spearman",   BuildableKind::Unit,      60, "spearman"},
        {"archer",   "Archer",   "archer",     BuildableKind::Unit,      60, "archer"},
        {"scout",    "Scout",    "scout",      BuildableKind::Unit,      25, "scout"},
        {"settler",  "Settler",  "settler",    BuildableKind::Unit,      80, "settler"},
        {"builder",  "Builder",  "builder",    BuildableKind::Unit,      50, "builder"},
        // Buildings -- reuse yield icons (food/production/gold/culture/science).
        {"granary",  "Granary",  "food",       BuildableKind::Building,  60,  "granary"},
        {"smithy",   "Smithy",   "production", BuildableKind::Building,  80,  "smithy"},
        {"market",   "Market",   "gold",       BuildableKind::Building, 100,  "market"},
        {"temple",   "Temple",   "culture",    BuildableKind::Building,  90,  "temple"},
        {"library",  "Library",  "science",    BuildableKind::Building,  90,  "library"},
    };
    return kItems;
}

// ---------------------------------------------------------------------------
// CivPedia articles -- rich-text markup, ASCII-safe.
// ---------------------------------------------------------------------------
// Color palette (matches the CivPedia window):
//   Section headers : <color=#9a7a3a>
//   Stat labels     : <color=#c8b888>
//   Item name refs  : <color=#c06820>
// ---------------------------------------------------------------------------

const std::string& getPediaArticle(const std::string& id) {
    static const std::unordered_map<std::string, std::string> kArticles = {

        // ------------------------------------------------------------------ //
        // UNITS
        // ------------------------------------------------------------------ //

        {"spearman",
            "The <tip=Spearman -- 30 HP, melee. +100% vs mounted units.>"
            "[icon=spearman 18]<b><color=#c06820>Spearman</color></b></tip>"
            " is an <i>Ancient Era</i> melee unit, the backbone of every early army."
            " Its long spear forms an impenetrable wall against charging horsemen.\n\n"
            "<b><color=#9a7a3a>Combat Statistics</color></b>\n"
            "<color=#c8b888>Strength</color>     <b>7</b>\n"
            "<color=#c8b888>Hit Points</color>   <b>30</b>\n"
            "<color=#c8b888>Movement</color>     <b>2</b>\n"
            "<color=#c8b888>Production</color>   <b>60</b>\n\n"
            "<b><color=#9a7a3a>Abilities</color></b>\n"
            "- <b>Phalanx</b> -- +<b>100%</b> combat strength against"
            " <tip=Cavalry -- fast mounted units, weak to spears.>"
            "[icon=cavalry 18]<b><color=#c06820>Cavalry</color></b></tip>"
            " and Chariots.\n"
            "- <b>Zone of Control</b> -- Enemy units spend all remaining movement"
            " when leaving an adjacent tile.\n\n"
            "<b><color=#9a7a3a>Weaknesses</color></b>\n"
            "Spearmen struggle against dedicated anti-infantry."
            " <b><color=#c06820>Swordsmen</color></b> backed by Iron cut through spear"
            " formations, and ranged fire from"
            " <tip=Archer -- ranged unit, 2 range, weak in melee.>"
            "[icon=archer 18]<b><color=#c06820>Archers</color></b></tip>"
            " can whittle down a phalanx before it reaches melee range."
            " Siege equipment on high ground is also devastating.\n\n"
            "<b><color=#9a7a3a>Upgrade Path</color></b>\n"
            "Spearman -> <i>Pikeman</i> -> <i>Musketman</i>\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The spear is humanity's oldest purpose-built weapon."
            " Sumerian phalanxes at Lagash (c. 2450 BCE) drilled tight formations of"
            " spear-bearers locked shield to shield."
            " Greek city-states refined this into the <i>hoplite</i> phalanx --"
            " eight ranks deep, each man overlapping his neighbour's aspis -- a wall of"
            " bronze that shattered Persian cavalry at Marathon in 490 BCE.\n\n"
            "Macedonian king Philip II lengthened the shaft into the <i>sarissa</i>,"
            " a pike up to six metres long. His son Alexander carried the phalanx to"
            " the edges of the known world, defeating every opponent from Granicus to"
            " Gaugamela. The Romans replaced the phalanx with the flexible <i>maniple</i>"
            " system, but the spearman never truly vanished -- Swiss <i>Reislaeufer</i>"
            " pikemen were still breaking French chivalry in the 15th century CE."},

        {"warrior",
            "[icon=warrior 18] <b><color=#c06820>Warriors</color></b> are the most basic"
            " military unit -- disciplined foot soldiers armed with clubs, axes, or short"
            " swords. Cheap and fast to produce, they provide cities with essential early"
            " defence and can explore nearby terrain in relative safety.\n\n"
            "<b><color=#9a7a3a>Combat Statistics</color></b>\n"
            "<color=#c8b888>Strength</color>    <b>5</b>\n"
            "<color=#c8b888>Hit Points</color>  <b>30</b>\n"
            "<color=#c8b888>Movement</color>    <b>2</b>\n"
            "<color=#c8b888>Production</color>  <b>40</b>\n\n"
            "<b><color=#9a7a3a>Upgrade Path</color></b>\n"
            "Warrior -> <i>Swordsman</i> -> <i>Musketman</i>\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The warrior predates civilisation itself. Evidence of organised violence --"
            " skeletons with healed and unhealed weapon injuries buried together -- appears"
            " in hunter-gatherer cemeteries as early as 10,000 BCE. What changed with the"
            " rise of cities was not the fact of warfare but its scale and organisation.\n\n"
            "Sumerian city-states fielded the first true armies around 2500 BCE: professional"
            " soldiers drawing palace rations, equipped with standardised copper-tipped spears"
            " and heavy leather cloaks. The Stele of the Vultures, commemorating Lagash's"
            " victory over Umma, shows a phalanx of helmeted warriors advancing in lockstep"
            " behind a wall of overlapping shields -- a formation that would define infantry"
            " combat for four thousand years.\n\n"
            "The warrior's essential nature changed little across this span. Whether Aztec"
            " <i>cuachic</i> veterans or medieval English billmen, the close-combat"
            " infantryman remained the unit that took and held ground, and no cavalry or"
            " archery could substitute for men willing to stand and fight at arm's reach."},

        {"archer",
            "[icon=archer 18] The <b><color=#c06820>Archer</color></b> is the first ranged"
            " unit of the Ancient Era -- a disciplined bowman who strikes enemies from two"
            " tiles away without entering melee range. Archers excel at softening enemy"
            " formations before an assault and at defending cities from elevated positions.\n\n"
            "<b><color=#9a7a3a>Combat Statistics</color></b>\n"
            "<color=#c8b888>Strength (melee)</color>   <b>4</b>\n"
            "<color=#c8b888>Strength (ranged)</color>  <b>5</b>\n"
            "<color=#c8b888>Hit Points</color>         <b>25</b>\n"
            "<color=#c8b888>Range</color>              <b>2</b>\n"
            "<color=#c8b888>Movement</color>           <b>2</b>\n"
            "<color=#c8b888>Production</color>         <b>60</b>\n\n"
            "<b><color=#9a7a3a>Abilities</color></b>\n"
            "- <b>Ranged Attack</b> -- Can attack any tile within 2 hexes without"
            " moving into it.\n"
            "- <b>Vulnerable in Melee</b> -- -50% combat strength when attacked in"
            " close combat.\n\n"
            "<b><color=#9a7a3a>Upgrade Path</color></b>\n"
            "Archer -> <i>Crossbowman</i> -> <i>Field Cannon</i>\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The composite bow -- made from laminated wood, bone, and sinew -- was the"
            " high-technology weapon of the ancient world. Developed on the Eurasian steppe"
            " around 2000 BCE, it stored three times the energy of a simple wooden stave"
            " and was compact enough to fire from horseback. Hittite and Egyptian armies"
            " that mastered it gained a decisive battlefield edge.\n\n"
            "At Kadesh (1274 BCE) Egyptian archers rode paired in chariots: one man drove,"
            " one fired. Ramesses II's account describes chariot squadrons making repeated"
            " passes at Hittite infantry flanks, pouring arrows into formations that could"
            " not effectively strike back.\n\n"
            "The English longbow is perhaps the most studied ranged weapon in history."
            " At Crecy (1346) and Agincourt (1415), English archers loosed twelve arrows"
            " per minute against French knights at ranges exceeding 200 metres."
            " Longbowmen's skeletons show deformed spines and asymmetrically enlarged left"
            " arms from decades of training begun in childhood."},

        {"scout",
            "[icon=scout 18] The <b><color=#c06820>Scout</color></b> is a lightly armed,"
            " fast-moving reconnaissance unit trained to navigate difficult terrain and reveal"
            " hidden lands. Scouts explore farther and faster than any other unit, uncovering"
            " natural wonders, city-state locations, and rival civilisations before they"
            " find you.\n\n"
            "<b><color=#9a7a3a>Combat Statistics</color></b>\n"
            "<color=#c8b888>Strength</color>    <b>2</b>\n"
            "<color=#c8b888>Hit Points</color>  <b>20</b>\n"
            "<color=#c8b888>Movement</color>    <b>3</b>\n"
            "<color=#c8b888>Production</color>  <b>25</b>\n\n"
            "<b><color=#9a7a3a>Abilities</color></b>\n"
            "- <b>Woodsman</b> -- Ignores movement penalties in forests and jungles.\n"
            "- <b>Keen Eyes</b> -- Sight radius extended by 1 tile.\n"
            "- <b>Evasion</b> -- Cannot be targeted when a stronger friendly unit is"
            " adjacent.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "Every military culture developed specialist scouts chosen for endurance,"
            " woodcraft, and the discipline to observe without being seen. Roman"
            " <i>exploratores</i> rode ahead of the legions, mapping roads, locating water,"
            " and counting enemy campfires. Native American scouts serving the US Army in"
            " the 1870s could track a single horse across bare rock and estimate how many"
            " days old the trail was.\n\n"
            "The consequences of poor reconnaissance were catastrophic. Crassus led three"
            " Roman legions into the Syrian desert at Carrhae in 53 BCE without adequate"
            " scouting, blundering into a Parthian force ideally suited to the open terrain."
            " His army was destroyed -- 20,000 dead, 10,000 captured -- in one of Rome's"
            " worst ever defeats."},

        {"settler",
            "[icon=settler 18] The <b><color=#c06820>Settler</color></b> carries a group"
            " of colonists equipped to establish a new city. Founding a settlement costs the"
            " origin city 1 Population but opens an entirely new site for growth, buildings,"
            " and resource exploitation. Settlers cannot attack or defend.\n\n"
            "<b><color=#9a7a3a>Statistics</color></b>\n"
            "<color=#c8b888>Strength</color>    <b>0</b>  (civilian, cannot fight)\n"
            "<color=#c8b888>Movement</color>    <b>2</b>\n"
            "<color=#c8b888>Production</color>  <b>80</b>\n\n"
            "<b><color=#9a7a3a>Notes</color></b>\n"
            "- Settlers cannot enter territory owned by another civilisation.\n"
            "- Escort Settlers through dangerous terrain -- they are defenceless.\n"
            "- Scout a location before settling; city placement determines tile yields.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The decision to establish a new settlement was among the most consequential a"
            " community could make. Successful colonisation required a founding group large"
            " enough to be self-sustaining, knowledge of the destination's resources, and"
            " either peaceful relations with existing inhabitants or the military strength"
            " to displace them.\n\n"
            "Greek <i>apoikia</i> (colonial settlements) were carefully planned expeditions:"
            " an oracle at Delphi consulted, a leader (<i>oikist</i>) chosen, and a sacred"
            " flame carried from the mother city to light the new hearth. Between 750 and"
            " 550 BCE the Greeks established over 400 such settlements across the"
            " Mediterranean and Black Sea -- spreading language and culture from Spain to"
            " the Crimea.\n\n"
            "The Roman <i>colonia</i> served a dual purpose: absorbing surplus population"
            " and projecting military power. Veterans rewarded with land grants in conquered"
            " territory became farmer-soldiers holding terrain the legions had won."
            " Colchester, Gloucester, Lincoln, and York were all Roman <i>coloniae</i>;"
            " their street plans shaped those cities for two thousand years."},

        {"builder",
            "[icon=builder 18] The <b><color=#c06820>Builder</color></b> is a civilian unit"
            " with a limited number of build charges used to construct terrain improvements"
            " -- Farms, Mines, Lumber Mills -- on tiles worked by nearby cities. Each"
            " improvement permanently boosts yields. Builders are consumed when all charges"
            " are spent.\n\n"
            "<b><color=#9a7a3a>Statistics</color></b>\n"
            "<color=#c8b888>Strength</color>      <b>0</b>  (civilian, cannot fight)\n"
            "<color=#c8b888>Movement</color>      <b>2</b>\n"
            "<color=#c8b888>Build Charges</color> <b>3</b>\n"
            "<color=#c8b888>Production</color>    <b>50</b>\n\n"
            "<b><color=#9a7a3a>Common Improvements</color></b>\n"
            "- <b>Farm</b> -- +1 Food on grassland and plains tiles.\n"
            "- <b>Mine</b> -- +1 Production on hills and resource tiles.\n"
            "- <b>Lumber Mill</b> -- +1 Production on forest tiles.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The transformation of raw landscape into productive farmland is the defining"
            " act of settled civilisation. Irrigation agriculture in Mesopotamia, begun"
            " around 6000 BCE, required coordinated community labour: surveying, digging"
            " canals, building levees, and managing water allocation across dozens of"
            " fields simultaneously.\n\n"
            "Egyptian officials tracked annual Nile flood levels from at least 3000 BCE,"
            " using them to predict crop yields months in advance. Fields were re-surveyed"
            " every year after the flood receded because boundary markers washed away"
            " -- a practical need that drove early advances in geometry. The word"
            " 'geometry' itself means <i>earth measurement</i>.\n\n"
            "Roman agricultural engineers divided conquered land into a grid of 200-acre"
            " squares (<i>centuriae</i>), reorganising entire landscapes for efficient"
            " farming. The grid of the Po Valley in northern Italy still follows Roman"
            " centuriation lines visible in satellite photographs two thousand years later."},

        // ------------------------------------------------------------------ //
        // BUILDINGS
        // ------------------------------------------------------------------ //

        {"granary",
            "[icon=food 18] The <b><color=#c06820>Granary</color></b> is one of humanity's"
            " earliest infrastructure achievements -- a communal storehouse that turns"
            " seasonal surplus into year-round security. Cities with a Granary grow faster"
            " and can support larger populations without hunger.\n\n"
            "<b><color=#9a7a3a>City Yields</color></b>\n"
            "<color=#c8b888>Food</color>        <b>+2</b>\n"
            "<color=#c8b888>Housing</color>     <b>+1</b>\n"
            "<color=#c8b888>Production</color>  <b>60</b>\n\n"
            "<b><color=#9a7a3a>Requirements</color></b>\n"
            "Unlocked at <i>Pottery</i>. Built in the City Center.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The first purpose-built granaries appear in the Levantine Pre-Pottery Neolithic,"
            " around 9500 BCE -- centuries before writing or the wheel. At Dhra' in modern"
            " Jordan, oval mud-brick pits were lined with woven grass to wick moisture away"
            " from stored grain.\n\n"
            "By 3500 BCE the Sumerians had elevated storage to an administrative science."
            " Temple precincts at Ur and Uruk maintained vast warehouse complexes staffed"
            " by scribes who tallied every shekel-weight of barley entering or leaving on"
            " clay tablets -- among the oldest written records ever found.\n\n"
            "Roman <i>horrea</i> (public granaries) were architectural marvels: floors raised"
            " on brick pillars to admit circulating air, narrow north-facing windows to"
            " minimise heat, and thick south walls against summer sun. Emperor Augustus"
            " reorganised the entire grain supply into a professional state administration"
            " -- the <i>cura annonae</i> -- recognising that feeding the capital was as"
            " much a military problem as a logistical one."},

        {"smithy",
            "[icon=production 18] The <b><color=#c06820>Smithy</color></b> is the industrial"
            " heart of an ancient city -- a forge where raw ore is smelted into tools,"
            " weapons, and building hardware. Cities with a Smithy complete construction"
            " projects significantly faster.\n\n"
            "<b><color=#9a7a3a>City Yields</color></b>\n"
            "<color=#c8b888>Production</color>        <b>+2</b>\n"
            "<color=#c8b888>Production</color> (cost)  <b>80</b>\n\n"
            "<b><color=#9a7a3a>Requirements</color></b>\n"
            "Unlocked at <i>Bronze Working</i>.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "Copper smelting first appears around 5000 BCE in the Balkans and Anatolia,"
            " where craftsmen discovered that malachite -- a vivid green ore -- could be"
            " reduced to workable metal in a sufficiently hot charcoal fire. Within a"
            " millennium, smiths learned that adding tin produced bronze: harder, sharper,"
            " and far more durable than copper alone.\n\n"
            "Bronze smithies were among the most closely guarded facilities of the ancient"
            " world. Mycenaean Linear B tablets record specialist smiths (<i>ka-ke-u</i>)"
            " receiving rations directly from the palace -- they were state assets, not"
            " free tradesmen. The island of Cyprus, whose very name gives us the word"
            " 'copper,' became so synonymous with metal production that controlling it"
            " was a strategic priority for every Mediterranean power.\n\n"
            "Iron smelting spread from Anatolia around 1200 BCE. Smiths who learned to"
            " carburise and quench wrought iron produced steel superior to any bronze."
            " Because iron ore is vastly more common, the iron smithy democratised"
            " metalwork and flooded armies with affordable weaponry, ending the Bronze"
            " Age monopoly of palace workshops."},

        {"market",
            "[icon=gold 18] The <b><color=#c06820>Market</color></b> is a hub of commercial"
            " activity where merchants, artisans, and travellers exchange goods, currency,"
            " and information. Cities with a Market generate substantially more gold each"
            " turn and may establish an additional Trade Route.\n\n"
            "<b><color=#9a7a3a>City Yields</color></b>\n"
            "<color=#c8b888>Gold</color>         <b>+3</b>\n"
            "<color=#c8b888>Trade Routes</color> <b>+1</b>\n"
            "<color=#c8b888>Production</color>   <b>100</b>\n\n"
            "<b><color=#9a7a3a>Requirements</color></b>\n"
            "Unlocked at <i>Currency</i>. Requires the Commercial Hub district.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The earliest markets were recurring festivals -- neutral ground where groups"
            " with otherwise hostile relations could exchange livestock, flint, and obsidian"
            " under the protection of custom. The word 'bazaar' comes from the Persian"
            " <i>bazar</i>, derived from an Old Iranian root meaning 'the place of prices.'\n\n"
            "Mesopotamian markets operated under sophisticated legal frameworks. The Code of"
            " Hammurabi (c. 1754 BCE) devoted dozens of clauses to commercial contracts:"
            " interest rates on grain loans, liability for lost merchandise, and penalties"
            " for merchants who falsified weights. Temple complexes served as both banks and"
            " commodity exchanges, issuing standardised silver ingots stamped with the temple"
            " seal as early currency.\n\n"
            "The Roman <i>forum</i> began as a combined market and civic space. By the"
            " Imperial period specialised markets had proliferated -- the <i>Forum"
            " Boarium</i> for cattle, the <i>Forum Piscarium</i> for fish. Trajan's Market"
            " (completed c. 113 CE) provided six levels of covered retail space housing over"
            " 150 individual shops: the world's first purpose-built shopping complex."},

        {"temple",
            "[icon=culture 18] The <b><color=#c06820>Temple</color></b> is a sacred space"
            " dedicated to the divine -- the spiritual centre of the city where citizens"
            " gather for rites, festivals, and communal identity. Cities with a Temple"
            " generate Faith each turn, drawing priests and scholars who enrich civic and"
            " cultural life.\n\n"
            "<b><color=#9a7a3a>City Yields</color></b>\n"
            "<color=#c8b888>Faith</color>       <b>+2</b>\n"
            "<color=#c8b888>Culture</color>     <b>+1</b>\n"
            "<color=#c8b888>Production</color>  <b>90</b>\n\n"
            "<b><color=#9a7a3a>Requirements</color></b>\n"
            "Unlocked at <i>Astrology</i>. Requires the Holy Site district.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The oldest known temple is Gobekli Tepe in southern Turkey, built around"
            " 9600 BCE -- before pottery, before agriculture, perhaps before permanent"
            " settlement. Its carved limestone pillars weighing up to 10 tonnes required"
            " organised communal labour on a scale that challenges earlier assumptions"
            " about pre-agricultural society.\n\n"
            "Mesopotamian <i>ziggurats</i> were not temples in the modern sense but"
            " artificial mountains -- raised platforms that brought the priest-king closer"
            " to heaven. The ziggurat of Ur, built by Ur-Nammu around 2100 BCE, soared"
            " 30 metres above the flat alluvial plain and was visible for miles, an"
            " assertion of divine and civic power simultaneously.\n\n"
            "The Parthenon (432 BCE) took the temple in a different direction: not a house"
            " of interior worship but an exterior monument. The cult statue of Athena stood"
            " within, but rituals happened on the steps outside. Its subtle proportional"
            " tricks -- slightly curved columns, inward-leaning walls, a bowed floor --"
            " exist entirely to make the eye perceive perfect geometry where none is"
            " actually present."},

        {"library",
            "[icon=science 18] The <b><color=#c06820>Library</color></b> is a repository"
            " of knowledge -- scrolls, tablets, and codices accumulated from philosophers,"
            " explorers, and merchants. Cities with a Library generate more Science each"
            " turn, accelerating research into new technologies.\n\n"
            "<b><color=#9a7a3a>City Yields</color></b>\n"
            "<color=#c8b888>Science</color>     <b>+2</b>\n"
            "<color=#c8b888>Production</color>  <b>90</b>\n\n"
            "<b><color=#9a7a3a>Requirements</color></b>\n"
            "Unlocked at <i>Writing</i>. Requires the Campus district.\n\n"
            "<b><color=#9a7a3a>Historical Notes</color></b>\n"
            "The oldest library yet discovered belongs not to Greece or Rome but to the"
            " Assyrian king Ashurbanipal (668-627 BCE), unearthed at Nineveh in the 1840s."
            " His palace library contained over 30,000 clay tablets covering medicine,"
            " astronomy, mythology, and administration -- including a copy of the <i>Epic"
            " of Gilgamesh</i> that preserved the flood narrative for three millennia until"
            " its modern rediscovery.\n\n"
            "The Great Library of Alexandria was less a single building than an institution:"
            " a research complex where Ptolemaic kings funded hundreds of scholars to live,"
            " argue, and write. Eratosthenes calculated the circumference of the Earth to"
            " within 2% and devised the first system for cataloguing books by genre and"
            " author. At its peak the collection may have held 700,000 scrolls.\n\n"
            "The Islamic House of Wisdom (<i>Bayt al-Hikma</i>), established in Baghdad"
            " around 830 CE, preserved and extended classical knowledge at a time when"
            " European access to Greek texts had largely vanished. Al-Khwarizmi's algebraic"
            " work there gave us both the word 'algorithm' (a Latinisation of his name)"
            " and the foundations of the mathematical notation used in every computer"
            " on Earth today."},
    };

    static const std::string kEmpty;
    auto it = kArticles.find(id);
    return it != kArticles.end() ? it->second : kEmpty;
}

}  // namespace odai::game
