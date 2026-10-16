"""
data.py - small synthetic benchmark for the plagiarism detector.

Ten short passages (five topics x two passages) written for this project. Each has a hand-written
paraphrase. From them we generate suspect documents of several kinds, with a known label for
every sentence:

    exact       the source copied word for word
    synonym     the source with many words swapped for synonyms (automatic, seeded)
    reorder     the source sentences shuffled
    paraphrase  the hand-written paraphrase (same meaning, different wording and structure)
    mixed       5 sentences: one copied, one paraphrased, three from an unrelated passage
    hard_neg    the sibling passage on the SAME topic (related, but not plagiarised)
    easy_neg    a passage on a different topic

hard_neg is the important control: a good detector must not flag text merely for being on-topic.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

SOURCES = [
    dict(
        id="bio_photosynthesis", topic="biology",
        sentences=[
            "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in sugar.",
            "It takes place mainly in the chloroplasts of leaf cells, where chlorophyll absorbs sunlight.",
            "During the light-dependent reactions, water molecules are split, releasing oxygen as a by-product and producing energy carriers.",
            "These carriers then power the Calvin cycle, which builds glucose from carbon dioxide taken in through tiny leaf pores called stomata.",
            "Almost all life on Earth depends on this process, since it supplies both the food chain and the oxygen in the atmosphere.",
        ],
        paraphrase=[
            "Green plants transform sunlight into sugar-based chemical fuel through a mechanism known as photosynthesis.",
            "The work happens chiefly inside leaf-cell chloroplasts, whose chlorophyll pigment captures incoming light.",
            "In the first stage, light drives the splitting of water, which yields oxygen as waste and generates molecules that carry energy.",
            "Those molecules fuel the Calvin cycle, a series of reactions that assembles glucose out of carbon dioxide entering via microscopic openings in the leaf known as stomata.",
            "Nearly every living thing relies on it, because it provides the foundation of food webs as well as the oxygen we breathe.",
        ],
    ),
    dict(
        id="bio_respiration", topic="biology",
        sentences=[
            "Cellular respiration is the set of reactions through which cells release energy from nutrients such as glucose.",
            "Most of it occurs in the mitochondria, which are often described as the power plants of the cell.",
            "Glucose is first broken down into smaller molecules, and the energy released is used to produce a molecule called ATP.",
            "When oxygen is available, this aerobic pathway yields far more ATP than fermentation, which works without oxygen.",
            "Animals, plants and most microbes all rely on respiration to keep their cells working.",
        ],
        paraphrase=[
            "Through a chain of reactions known as cellular respiration, cells extract usable energy from nutrients like glucose.",
            "The majority takes place inside mitochondria, frequently called the cell's powerhouses.",
            "To begin with, glucose is split into smaller compounds, and the energy freed in the process is invested in making ATP.",
            "In the presence of oxygen, this aerobic route generates much more ATP than fermentation, which proceeds in the absence of oxygen.",
            "Animals, plants and the bulk of microorganisms depend on it to keep their cells functioning.",
        ],
    ),
    dict(
        id="cs_blockchain", topic="computing",
        sentences=[
            "A blockchain is a shared digital ledger that records transactions across many computers so that no single party controls it.",
            "Transactions are grouped into blocks, and each block contains a cryptographic hash of the block that came before it.",
            "Because changing an old block would alter every hash that follows, tampering with the history is easy to detect.",
            "Participants in the network agree on which blocks to add using a consensus mechanism such as proof of work or proof of stake.",
            "Although best known for supporting cryptocurrencies, the technology is also being tested for supply chain tracking and record keeping.",
        ],
        paraphrase=[
            "Spread over numerous machines, a blockchain acts as a common electronic record book of transactions that nobody alone is able to dominate.",
            "Entries are bundled into units called blocks, with every unit storing the cryptographic fingerprint of its predecessor.",
            "Since rewriting a past block would invalidate all the fingerprints after it, any attempt to falsify the record can be spotted readily.",
            "The network's members settle which blocks get appended through a consensus protocol, for instance proof of work or proof of stake.",
            "Though it is most famous for underpinning digital currencies, experiments are also under way to use it for tracking goods through supply chains and for maintaining records.",
        ],
    ),
    dict(
        id="cs_publickey", topic="computing",
        sentences=[
            "Public-key cryptography uses a pair of mathematically linked keys, one that can be shared openly and one that must be kept secret.",
            "Anyone can use the public key to encrypt a message, but only the holder of the matching private key can decrypt it.",
            "This removes the need for two people to exchange a secret key in advance, which was the main weakness of older symmetric systems.",
            "The same idea works in reverse for digital signatures, where the private key signs a document and the public key verifies it.",
            "Common algorithms such as RSA rely on the fact that multiplying large prime numbers is easy while factoring the result is extremely hard.",
        ],
        paraphrase=[
            "Asymmetric encryption relies on two keys that are linked by mathematics: a public one anyone may see and a private one that its owner guards.",
            "A message locked with the public key can be unlocked solely by whoever possesses the corresponding private key.",
            "As a result, two parties no longer have to swap a shared secret beforehand, a drawback that plagued earlier symmetric schemes.",
            "Digital signatures apply the principle the other way around: the private key produces the signature on a document, and the public key checks that it is genuine.",
            "Well-known schemes like RSA depend on the asymmetry that multiplying big primes is simple, whereas recovering the factors from their product is extraordinarily difficult.",
        ],
    ),
    dict(
        id="hist_industrial", topic="history",
        sentences=[
            "The Industrial Revolution began in Britain in the late eighteenth century and transformed how goods were made.",
            "Steam engines and mechanized textile mills moved production out of homes and small workshops into large factories.",
            "Cities grew rapidly as workers left farms to look for jobs, often living in crowded and unhealthy conditions.",
            "New railways and canals lowered transport costs and linked distant markets for the first time.",
            "Over the following century, similar changes spread across Europe and North America and raised average living standards.",
        ],
        paraphrase=[
            "Britain was the birthplace of the Industrial Revolution, which started in the closing decades of the 1700s and changed the way products were manufactured.",
            "Production shifted from cottages and little workshops to enormous factories thanks to steam power and machine-driven textile mills.",
            "As people abandoned agricultural work in search of employment, towns expanded quickly, and many residents endured overcrowded, unhealthy housing.",
            "The arrival of railroads and canals made shipping cheaper and tied faraway markets together for the very first time.",
            "In the century that followed, comparable developments reached the rest of Europe and North America, lifting the typical standard of living.",
        ],
    ),
    dict(
        id="hist_printing", topic="history",
        sentences=[
            "Around 1440, Johannes Gutenberg introduced a printing press in Europe that used movable metal type.",
            "Earlier books had to be copied by hand, which made them slow to produce and far too expensive for most people.",
            "With the new press, a single workshop could produce hundreds of identical copies in a fraction of the time.",
            "Books became cheaper, literacy rates slowly rose, and new ideas travelled across the continent faster than ever before.",
            "Historians often link the press to the spread of the Reformation and the growth of modern science.",
        ],
        paraphrase=[
            "Johannes Gutenberg brought a press with movable metal type to Europe roughly in 1440.",
            "Prior to that, every book had to be transcribed by hand, so production was laborious and the price was beyond the reach of ordinary readers.",
            "Using the press, one workshop was able to turn out many identical copies in much less time.",
            "Reading material grew more affordable, literacy gradually climbed, and fresh ideas moved around Europe more quickly than at any previous point.",
            "Scholars frequently tie the invention to the rise of the Reformation and the development of modern science.",
        ],
    ),
    dict(
        id="earth_greenhouse", topic="earth science",
        sentences=[
            "The greenhouse effect is a natural process that warms the Earth's surface and makes the planet habitable.",
            "Gases such as carbon dioxide, methane and water vapour absorb heat radiated from the ground and send part of it back downward.",
            "Without this effect, the average temperature at the surface would be well below the freezing point of water.",
            "Burning fossil fuels and clearing forests have increased the concentration of these gases in the atmosphere since the industrial era.",
            "As a result, more heat is being trapped, and global average temperatures have risen over the past century.",
        ],
        paraphrase=[
            "A natural mechanism called the greenhouse effect heats the planet's surface and allows life to exist here.",
            "Carbon dioxide, methane and water vapour soak up the warmth emitted by the ground and redirect a portion of it toward the surface.",
            "If it did not occur, surface temperatures on average would sit far under the point at which water freezes.",
            "Since industrialisation, the combustion of coal, oil and gas along with deforestation has raised the atmospheric levels of those gases.",
            "Consequently, additional heat stays trapped, and worldwide mean temperatures have climbed during the last hundred years.",
        ],
    ),
    dict(
        id="earth_ocean", topic="earth science",
        sentences=[
            "Ocean currents are large, continuous flows of seawater driven by wind, differences in temperature and salinity, and the rotation of the Earth.",
            "Surface currents such as the Gulf Stream carry warm water from the tropics toward higher latitudes.",
            "This movement of heat helps keep the climate of western Europe milder than other regions at the same latitude.",
            "Deep currents form when cold, salty water becomes dense and sinks, slowly circulating around the globe over hundreds of years.",
            "Changes in these currents can affect weather patterns, fisheries, and the distribution of marine life.",
        ],
        paraphrase=[
            "Winds, contrasts in warmth and saltiness, and the planet's spin combine to propel vast, unbroken streams of seawater known as ocean currents.",
            "Near the surface, flows like the Gulf Stream haul tropical warmth towards the poles.",
            "Because of this transfer of heat, the climate in western Europe is gentler than that of other places lying at a similar latitude.",
            "In the depths, chilly and briny water grows heavy and descends, then drifts around the world for centuries.",
            "Alterations to these flows can influence weather, fishing grounds and where sea creatures live.",
        ],
    ),
    dict(
        id="econ_supply", topic="economics",
        sentences=[
            "The law of demand states that, all else being equal, consumers buy less of a product as its price rises.",
            "The law of supply says the opposite for producers, who are willing to offer more of a good when they can sell it at a higher price.",
            "The market price settles at the equilibrium point where the quantity buyers want matches the quantity sellers offer.",
            "If demand suddenly increases, shortages appear and prices tend to climb until supply catches up.",
            "If suppliers produce more than buyers want, surpluses build up and sellers are forced to cut prices.",
        ],
        paraphrase=[
            "According to the law of demand, when everything else stays the same, people purchase a smaller amount of something once it becomes more expensive.",
            "Producers behave in the reverse manner under the law of supply: they supply a larger quantity of a good if the selling price is higher.",
            "Prices stabilise at the equilibrium, the level where the amount consumers wish to buy equals the amount firms are ready to sell.",
            "When buyers' appetite jumps unexpectedly, goods become scarce and prices usually go up until output rises to meet it.",
            "When firms make more than customers are looking for, unsold stock accumulates and vendors have to lower what they charge.",
        ],
    ),
    dict(
        id="econ_inflation", topic="economics",
        sentences=[
            "Inflation is the general rise in prices across an economy over time, which reduces the purchasing power of money.",
            "It is usually measured by tracking the cost of a fixed basket of everyday goods and services, known as a consumer price index.",
            "Moderate inflation is often considered healthy, but rapid price increases can erode savings and make planning difficult.",
            "Central banks typically respond to high inflation by raising interest rates, which makes borrowing more expensive and slows spending.",
            "In extreme cases, called hyperinflation, prices can double within a few days and people lose faith in the currency.",
        ],
        paraphrase=[
            "When costs throughout an economy climb steadily as time passes, money buys less than before, a phenomenon called inflation.",
            "Analysts normally gauge it by monitoring what a set collection of ordinary goods and services costs, a measure referred to as the consumer price index.",
            "A mild degree of inflation is generally seen as beneficial, whereas fast price hikes can eat away at savings and complicate planning.",
            "When inflation runs high, central banks tend to lift interest rates, which makes loans pricier and dampens spending.",
            "In the most severe episodes, labelled hyperinflation, prices may double over a few days, and people stop trusting their currency.",
        ],
    ),
]

# single-word, same-part-of-speech, same-inflection swaps (each token is replaced at most once)
SYNONYMS = {
    "process": "procedure", "convert": "transform", "energy": "power", "mainly": "chiefly",
    "absorbs": "captures", "sunlight": "daylight", "molecules": "particles", "releasing": "emitting",
    "producing": "generating", "carriers": "transporters", "power": "drive", "builds": "constructs",
    "taken": "drawn", "tiny": "minute", "pores": "openings", "called": "named", "almost": "nearly",
    "depends": "relies", "supplies": "provides", "stored": "kept",
    "release": "emit", "nutrients": "foods", "occurs": "happens", "described": "portrayed",
    "broken": "split", "smaller": "lesser", "produce": "make", "yields": "delivers", "far": "much",
    "rely": "depend", "working": "functioning",
    "shared": "common", "digital": "electronic", "ledger": "register", "across": "throughout",
    "computers": "machines", "party": "entity", "controls": "governs", "grouped": "bundled",
    "contains": "holds", "changing": "altering", "alter": "modify", "every": "each",
    "tampering": "meddling", "history": "past", "easy": "simple", "detect": "spot",
    "participants": "members", "agree": "decide", "add": "append", "mechanism": "method",
    "although": "though", "supporting": "underpinning", "tested": "trialled", "tracking": "monitoring",
    "uses": "employs", "pair": "couple", "linked": "connected", "openly": "publicly",
    "secret": "confidential", "matching": "corresponding", "holder": "owner", "removes": "eliminates",
    "need": "requirement", "exchange": "swap", "main": "principal", "weakness": "flaw",
    "older": "earlier", "idea": "concept", "works": "operates", "verifies": "validates",
    "common": "popular", "large": "big", "extremely": "exceptionally", "hard": "difficult",
    "began": "started", "transformed": "changed", "goods": "products", "made": "manufactured",
    "mechanized": "automated", "mills": "plants", "production": "manufacturing", "moved": "shifted",
    "homes": "houses", "small": "little", "cities": "towns", "grew": "expanded", "rapidly": "quickly",
    "workers": "labourers", "jobs": "employment", "crowded": "overcrowded", "railways": "railroads",
    "lowered": "reduced", "transport": "shipping", "costs": "expenses", "distant": "faraway",
    "similar": "comparable", "spread": "extended", "raised": "lifted", "average": "typical",
    "introduced": "brought", "earlier": "previous", "copied": "transcribed", "expensive": "costly",
    "fraction": "portion", "slowly": "gradually", "rose": "climbed", "ideas": "notions",
    "travelled": "moved", "faster": "quicker", "historians": "scholars", "often": "frequently",
    "link": "connect", "warms": "heats", "absorb": "retain", "radiated": "emitted", "send": "return",
    "clearing": "removing", "increased": "raised", "concentration": "level", "trapped": "captured",
    "risen": "climbed", "global": "worldwide", "past": "last", "continuous": "uninterrupted",
    "flows": "streams", "driven": "propelled", "differences": "contrasts", "rotation": "spin",
    "carry": "transport", "toward": "towards", "movement": "transfer", "helps": "assists",
    "milder": "gentler", "regions": "areas", "form": "develop", "dense": "heavy", "sinks": "descends",
    "changes": "alterations", "affect": "influence",
    "states": "says", "consumers": "buyers", "buy": "purchase", "product": "item", "rises": "increases",
    "opposite": "reverse", "producers": "manufacturers", "willing": "prepared", "offer": "provide",
    "settles": "stabilises", "quantity": "amount", "matches": "equals", "suddenly": "abruptly",
    "increases": "grows", "shortages": "scarcities", "appear": "emerge", "suppliers": "vendors",
    "surpluses": "excesses", "forced": "compelled", "cut": "reduce",
    "general": "overall", "rise": "increase", "reduces": "lowers", "measured": "gauged",
    "fixed": "set", "basket": "collection", "everyday": "ordinary", "moderate": "mild",
    "considered": "regarded", "healthy": "beneficial", "rapid": "fast", "erode": "undermine",
    "difficult": "tough", "typically": "normally", "respond": "react", "raising": "lifting",
    "slows": "curbs", "extreme": "severe", "cases": "instances", "faith": "trust",
}

SYN_RATE = 0.8


def swap_synonyms(sentence: str, rng: random.Random) -> str:
    def repl(m):
        w = m.group(0)
        syn = SYNONYMS.get(w.lower())
        if syn is None or rng.random() > SYN_RATE:
            return w
        return syn.capitalize() if w[0].isupper() else syn

    # words attached to a hyphen ("by-product", "light-dependent") are left alone
    return re.sub(r"(?<![\w-])[A-Za-z]+(?![\w-])", repl, sentence)


@dataclass
class Case:
    id: str
    source_id: str
    topic: str
    category: str
    source_text: str
    suspect_text: str
    sentences: list  # intended suspect sentences
    labels: list  # 1 = derived from the source, 0 = not


POSITIVE_CATEGORIES = ("exact", "synonym", "reorder", "paraphrase", "mixed")
NEGATIVE_CATEGORIES = ("hard_neg", "easy_neg")
CATEGORIES = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES


def build_cases(seed: int = 0) -> list[Case]:
    cases = []
    for k, src in enumerate(SOURCES):
        rng = random.Random(seed * 1000 + k)
        sents, para = src["sentences"], src["paraphrase"]
        sibling = next(s for s in SOURCES if s["topic"] == src["topic"] and s["id"] != src["id"])
        other = rng.choice([s for s in SOURCES if s["topic"] != src["topic"]])

        shuffled = sents[:]
        while shuffled == sents:
            rng.shuffle(shuffled)

        variants = {
            "exact": (sents, [1] * 5),
            "synonym": ([swap_synonyms(s, rng) for s in sents], [1] * 5),
            "reorder": (shuffled, [1] * 5),
            "paraphrase": (para, [1] * 5),
            "mixed": ([other["sentences"][0], sents[1], other["sentences"][2], para[3], other["sentences"][4]],
                      [0, 1, 0, 1, 0]),
            "hard_neg": (sibling["sentences"], [0] * 5),
            "easy_neg": (other["sentences"], [0] * 5),
        }
        for cat, (sentences, labels) in variants.items():
            cases.append(
                Case(
                    id=f"{src['id']}:{cat}", source_id=src["id"], topic=src["topic"], category=cat,
                    source_text=" ".join(sents), suspect_text=" ".join(sentences),
                    sentences=sentences, labels=labels,
                )
            )
    return cases


def all_sentences() -> list[str]:
    """Every sentence in the data set (used to fit the offline stand-in embedder; no labels involved)."""
    out = []
    for s in SOURCES:
        out += s["sentences"] + s["paraphrase"]
    return out


def demo_samples() -> list[dict]:
    """Ready-made source/suspect pairs for the app's demo buttons."""
    cases = {c.id: c for c in build_cases()}
    pid = "bio_photosynthesis"
    names = {
        "exact": "Copy-paste (exact)",
        "synonym": "Synonym swap",
        "paraphrase": "Paraphrase",
        "mixed": "Partly copied",
        "hard_neg": "Same topic, original text",
    }
    return [
        dict(name=names[cat], source=cases[f"{pid}:{cat}"].source_text, suspect=cases[f"{pid}:{cat}"].suspect_text)
        for cat in names
    ]
