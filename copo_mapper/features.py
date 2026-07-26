from __future__ import annotations

import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
# Two groups are merged into one set:
#
# 1) A small standard-English function-word list (articles, prepositions,
#    conjunctions, pronouns, auxiliaries).  These carry no domain or Bloom
#    signal and only inflate raw token overlap between unrelated CO/PO pairs.
#
# 2) OBE boiler-plate that appears in almost every outcome statement and is
#    therefore uninformative for *discriminating* between outcomes:
#      - "ability"/"able"/"student(s)"/"learner(s)"/"course"/"end":
#        remnants of the "at the end of the course the student will be able
#        to ..." frame; pure filler.
#      - "various"/"given"/"different"/"basic": quantity/quality hedges
#        ("for a given grammar", "various techniques") that modify content
#        words but carry none of the content themselves.
#      - "concept(s)": nearly every CO is phrased "concepts of X"; the signal
#        lives entirely in X, so "concept" only adds spurious overlap.
#    Deliberately NOT stopworded: "understanding", "knowledge", "apply",
#    "use", "design", "analyze", ... — these are Bloom-level action words and
#    must survive filtering so detect_bloom (which runs on token_set output
#    in scoring.py) still sees them.
STOPWORDS: frozenset[str] = frozenset(
    {
        # --- standard English function words ---
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "by", "at", "from", "as", "into", "through", "during",
        "between", "about", "over", "under", "above", "is", "are", "was",
        "were", "be", "been", "being", "will", "shall", "would", "should",
        "can", "could", "may", "might", "must", "do", "does", "did", "have",
        "has", "had", "not", "no", "this", "that", "these", "those", "it",
        "its", "their", "there", "they", "them", "his", "her", "our", "your",
        "we", "you", "he", "she", "which", "who", "whom", "whose", "what",
        "when", "where", "how", "why", "such", "than", "then", "also",
        "other", "both", "each", "any", "all", "some", "so", "if", "etc",
        # --- OBE boiler-plate (see justification above) ---
        "ability", "abilities", "able", "student", "students", "learner",
        "learners", "course", "courses", "end", "various", "given",
        "different", "basic", "concept", "concepts",
    }
)


# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------
# A deliberately light, conservative, rule-based stemmer (pure stdlib).  The
# goal is NOT linguistic correctness but *consistency*: the same rules are
# applied to CO text, PO text, and every vocabulary entry below, so related
# surface forms collapse to one key ("developing"/"developed"/"develops"/
# "develop" -> "develop"; "translation"/"translating"/"translate" ->
# "translat").  Every rule carries a minimum-length guard so short content
# words are never mangled ("design" != "des", "use" stays "use").

# Double consonants that may be un-doubled after stripping -ing/-ed
# ("planning" -> "plann" -> "plan", "programmed" -> "programm" -> "program").
# "s" is excluded (so "assessing" -> "assess" keeps its ss) and vowels are
# irrelevant here.
_DOUBLE_CONSONANTS = frozenset("bdglmnprt")

# Plural clusters where stripping the whole "es" is correct
# ("matches" -> "match", "boxes" -> "box", "classes" -> "class").
_ES_CLUSTERS = ("sses", "xes", "zes", "ches", "shes")


def stem(token: str) -> str:
    """Reduce ``token`` to a light stem via conservative suffix rules.

    Rules (applied as a short pipeline, each documented inline):
      A. plural stripping   (ies->y, cluster-es->'', s->'')
      B. inflection/derivation (ization->ize, ation->ate, ied->y, ing->'',
         ed->'')
      C. consonant un-doubling, only after an -ing/-ed strip
      D. final-e normalisation (manage/managing both -> "manag")
    """
    t = token.lower()

    # --- A: plurals -------------------------------------------------------
    if t.endswith("ies") and len(t) >= 5:
        # "studies" -> "study", "theories" -> "theory".  Guard >=5 so "ties"
        # falls through to the plain -s rule instead of becoming "ty".
        t = t[:-3] + "y"
    elif t.endswith(_ES_CLUSTERS) and len(t) - 2 >= 3:
        # "matches" -> "match", "processes" -> "process".  Guard keeps the
        # stem at >=3 chars ("uses" falls through to the -s rule -> "use").
        t = t[:-2]
    elif t.endswith("s") and len(t) >= 4 and not t.endswith(("ss", "us", "is")):
        # "parsers" -> "parser", "algorithms" -> "algorithm".  The ss/us/is
        # guard protects "class", "analysis", "various", "syllabus".
        t = t[:-1]

    # --- B: inflection / derivation --------------------------------------
    stripped_participle = False
    if t.endswith("ization") and len(t) >= 8:
        # "organization" -> "organize" (then D -> "organiz", matching
        # "organizing" -> "organiz").
        t = t[:-7] + "ize"
    elif t.endswith("ation") and len(t) >= 8:
        # "translation" -> "translate", "generation" -> "generate" (then D
        # aligns them with "translating"/"generating").
        t = t[:-5] + "ate"
    elif t.endswith("ied") and len(t) >= 5:
        # "applied" -> "apply", "studied" -> "study".
        t = t[:-3] + "y"
    elif t.endswith("ing") and len(t) - 3 >= 3:
        # "developing" -> "develop", "designing" -> "design".  Guard keeps a
        # >=3-char stem so "using"/"doing"/"being" are left intact rather
        # than truncated to nonsense ("us", "do", "be").
        t = t[:-3]
        stripped_participle = True
    elif t.endswith("ed") and len(t) - 2 >= 3:
        # "designed" -> "design", "evaluated" -> "evaluat" (aligned with
        # "evaluate" via rule D).
        t = t[:-2]
        stripped_participle = True

    # --- C: un-double after -ing/-ed only ---------------------------------
    # "planning" -> "plann" -> "plan"; restricted to post-strip so genuine
    # double letters in base words ("skill", "class") are never touched.
    if (
        stripped_participle
        and len(t) >= 4
        and t[-1] == t[-2]
        and t[-1] in _DOUBLE_CONSONANTS
    ):
        t = t[:-1]

    # --- D: final-e normalisation -----------------------------------------
    # "manage" and "managing" must land on the same stem; stripping the
    # silent final e ("manage" -> "manag", "create" -> "creat") achieves
    # that.  Guard >=5 protects short words ("use", "code"); "ee" is kept so
    # "degree"/"see" stay intact.
    if len(t) >= 5 and t.endswith("e") and not t.endswith("ee"):
        t = t[:-1]

    return t


# ---------------------------------------------------------------------------
# Bloom's taxonomy action verbs
# ---------------------------------------------------------------------------
# Standard (revised) Bloom's taxonomy verb lists, ~15-25 surface forms per
# level.  Entries are stored as raw surface forms and matched through
# ``stem()`` (see _ACTION_VERB_STEMS), so "developing"/"developed" match
# "develop" automatically.  Several verbs legitimately appear at more than
# one level in the standard lists (e.g. "illustrate", "estimate",
# "produce"); a single verb matched at several levels resolves to the
# HIGHEST of those levels.
#
# ACTION_VERBS is the full reference list.  Matching, however, is NOT a
# plain "highest level wins over the whole bag of words" any more: many of
# these entries (value, model, measure, plan, test, design, ...) are
# usually NOUNS in outcome statements ("the value of assets", "Models of
# Investment") and must not bump the level when they appear mid-sentence.
# See BLOOM_NOUN_HOMOGRAPHS and detect_bloom for the leading-verb policy.
ACTION_VERBS: dict[str, set[str]] = {
    "remember": {
        "cite", "define", "identify", "label", "list", "match", "memorize",
        "name", "outline", "quote", "recall", "recite", "recognize",
        "record", "repeat", "reproduce", "retrieve", "state", "tabulate",
        "tell",
    },
    "understand": {
        "classify", "clarify", "comprehend", "convert", "describe",
        "discuss", "estimate", "exemplify", "explain", "express", "extend",
        "generalize", "illustrate", "infer", "interpret", "locate",
        "paraphrase", "predict", "report", "restate", "review", "summarize",
        "translate", "understand",
    },
    "apply": {
        "adapt", "apply", "calculate", "change", "choose", "compute",
        "demonstrate", "employ", "execute", "illustrate", "implement",
        "manipulate", "modify", "operate", "practice", "prepare", "produce",
        "schedule", "show", "simulate", "sketch", "solve", "use", "used",
        "using", "utilize",
    },
    "analyze": {
        "analyse", "analyze", "attribute", "categorize", "compare",
        "contrast", "correlate", "debate", "deconstruct", "deduce",
        "differentiate", "discriminate", "dissect", "distinguish",
        "examine", "experiment", "infer", "inspect", "investigate",
        "organize", "question", "test",
    },
    "evaluate": {
        "appraise", "argue", "assess", "conclude", "convince", "criticize",
        "critique", "decide", "defend", "estimate", "evaluate", "grade",
        "judge", "justify", "measure", "prioritize", "rank", "rate",
        "recommend", "score", "select", "support", "validate", "value",
        "verify",
    },
    "create": {
        "assemble", "build", "compose", "construct", "create", "design",
        "develop", "devise", "formulate", "generate", "hypothesize",
        "integrate", "invent", "make", "model", "originate", "plan",
        "produce", "propose", "synthesize", "write",
    },
}

# ---------------------------------------------------------------------------
# Bloom noun-homographs and derived (adjective/noun) forms
# ---------------------------------------------------------------------------
# Words from the standard verb lists above that, in CO/PO statements, are
# far more often NOUNS (or adjectives) than imperatives: "the value of
# assets", "Models of Investment", "decision support", "hypothesis test",
# "business plan", "growth rate".  Treating them as full action verbs
# anywhere in the sentence produced systematic level inflation under the
# old highest-wins rule ("Understand the value of assets" -> evaluate).
#
# Policy: a noun-homograph only counts as an action verb when it is the
# LEADING content word of the statement ("Design top-down parsers",
# "Measure riskiness of a stock") — outcome statements lead with their
# action verb, so leading position is strong evidence of verb-hood.  In
# the unordered fallback (no text available) they are a last-resort tier,
# consulted only when no unambiguous verb matched at all.
#
# "use" is listed because "uses of X" is a noun phrase; the unambiguous
# inflections "using"/"used" stem differently and stay regular verbs.
# "organize" is listed because stem() maps the ubiquitous noun
# "organization(s)" onto the same stem as the verb "organize".
BLOOM_NOUN_HOMOGRAPHS: frozenset[str] = frozenset(
    {
        "value", "values", "model", "measure", "plan", "rate", "score",
        "select", "support", "question", "test", "produce", "build",
        "design", "organize", "use",
    }
)

# Common adjective/noun derivations that unambiguously signal a Bloom level
# even though they are not verbs ("Foster ANALYTICAL thinking" -> analyze,
# "APPLICATION of DBMS" -> apply).  Consulted when no regular action verb
# is present.  ("creation" and "evaluation" need no entry: stem() already
# maps them onto the stems of "create"/"evaluate".)
BLOOM_DERIVED_FORMS: dict[str, str] = {
    "analytical": "analyze",
    "application": "apply",
    "creativity": "create",
    "comprehension": "understand",
}

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------
# Single-word terms are matched via stemmed-token intersection (so plural /
# gerund variants match automatically).  Multi-word terms (containing a
# space) cannot be matched by set intersection; they are matched as
# substrings of the normalised full text when detect_domains is given the
# optional ``text`` argument.  Each domain therefore also carries the
# distinctive single tokens of its key phrases (e.g. "lexical", "parser")
# so tokens-only callers such as scoring.py still get good coverage.
DOMAIN_TERMS: dict[str, set[str]] = {
    # --- original six domains, enriched ---
    "algorithms": {
        "algorithm", "algorithmic", "complexity", "optimization",
        "recursion", "sorting", "searching", "heuristic", "efficiency",
        "asymptotic", "greedy", "dynamic programming",
    },
    "databases": {
        "database", "sql", "query", "relational", "schema", "normalization",
        "transaction", "dbms", "nosql", "indexing", "warehouse",
    },
    "software": {
        "software", "system", "architecture", "deployment", "engineering",
        "testing", "debugging", "agile", "requirement", "module", "api",
        "programming", "code", "application", "maintenance", "uml",
    },
    "communication": {
        "communicate", "communication", "report", "presentation", "present",
        "write", "writing", "documentation", "listening", "speaking",
        "interpersonal", "articulate", "verbal",
    },
    "ethics": {
        "ethics", "ethical", "professional", "professionalism", "society",
        "societal", "impact", "responsibility", "integrity", "plagiarism",
        "sustainability", "legal", "moral", "privacy",
    },
    "experiments": {
        "experiment", "measure", "measurement", "hypothesis", "laboratory",
        "observation", "procedure", "instrumentation", "sampling",
        "analyze data", "experimental",
    },
    # --- new domains covering B.Tech CS data ---
    "compilers": {
        "compiler", "compilation", "lexical", "lexer", "analyzer", "parser",
        "parse", "parsing", "grammar", "syntax", "semantic", "translation",
        "tokenizer", "automata", "ambiguity", "intermediate",
        "code generation", "syntax directed", "regular expression",
    },
    "systems": {
        "operating", "os", "memory", "process", "thread", "scheduling",
        "kernel", "virtualization", "concurrency", "deadlock", "filesystem",
        "interrupt", "paging",
    },
    "networks": {
        "network", "networking", "protocol", "tcp", "ip", "routing",
        "switching", "dns", "socket", "wireless", "lan", "topology",
        "bandwidth",
    },
    "security": {
        "security", "cryptography", "encryption", "decryption",
        "authentication", "authorization", "firewall", "vulnerability",
        "attack", "cipher", "malware", "intrusion",
    },
    "ml_data": {
        "data", "model", "training", "learning", "machine", "analytics",
        "statistics", "statistical", "regression", "classification",
        "clustering", "neural", "prediction", "mining", "visualization",
        "dataset",
    },
    # --- new domains covering MBA data ---
    # NOTE: "directing" (of the classic POSDC management functions) is
    # deliberately omitted — its stem collides with CS usage such as
    # "syntax directed translation" and "directed graph".
    #
    # NOTE: the catch-all tokens "management", "managerial", "manager",
    # "decision", "strategy", "strategic", "planning", "organizing",
    # "organizational" and "organization" were deliberately REMOVED from
    # this domain.  In an MBA corpus they appear in nearly every CO and PO,
    # so they carried zero discriminative signal and made domain_overlap
    # fire on almost every pair (measured: 40%+ of a real management grid
    # was inflated to label 2 at semantic similarity 0).  A domain feature
    # exists to bridge *distinctive* same-field vocabulary; ubiquitous
    # field words are near-stopwords for that purpose.  Genuinely shared
    # surface vocabulary ("management" on both sides) is still rewarded via
    # token_overlap and the semantic backend.  What remains is the
    # discipline-distinctive core: POSDC function terms and governance
    # vocabulary, plus multi-word phrases matched against the full text.
    "management": {
        "staffing", "controlling", "governance", "stakeholder",
        "organizational behavior", "organisational behaviour",
        "strategic management", "management functions",
    },
    "finance": {
        "finance", "financial", "accounting", "budget", "budgeting", "cost",
        "costing", "investment", "capital", "audit", "taxation", "revenue",
        "valuation", "ratio", "ledger",
    },
    "marketing": {
        "marketing", "market", "consumer", "brand", "branding",
        "advertising", "sales", "pricing", "promotion", "segmentation",
        "customer", "retail",
    },
    "hr_leadership": {
        "leadership", "motivation", "recruitment", "teamwork", "team",
        "negotiation", "conflict", "appraisal", "compensation", "attrition",
        "human resource", "employee",
    },
    "entrepreneurship": {
        "entrepreneurship", "entrepreneur", "entrepreneurial", "startup",
        "venture", "innovation", "opportunity", "feasibility", "incubation",
        "business plan",
    },
}

# Domains whose single-word vocabulary contains everyday tech/business
# words ("data", "model", "learning"; "system", "application", "code")
# that fire on texts far outside the domain proper ("life-long learning"
# -> ml_data, "its application" -> software).  A match on such a domain is
# weaker evidence of true topical relatedness, so scoring halves the
# domain-overlap credit when the ONLY shared domain of a CO/PO pair is a
# single generic one.  Sharing a specific domain, or two or more domains,
# earns full credit.
GENERIC_DOMAINS: frozenset[str] = frozenset({"ml_data", "software"})


BLOOM_ORDER = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
BLOOM_INDEX = {level: i for i, level in enumerate(BLOOM_ORDER)}


# Pre-stemmed lookup tables.  Vocabulary entries pass through the same
# stem() as incoming tokens, guaranteeing consistent matching.
_ACTION_VERB_STEMS: dict[str, frozenset[str]] = {
    level: frozenset(stem(verb) for verb in verbs)
    for level, verbs in ACTION_VERBS.items()
}
_HOMOGRAPH_STEMS: frozenset[str] = frozenset(
    stem(word) for word in BLOOM_NOUN_HOMOGRAPHS
)
# Unambiguous verb stems per level: the standard lists minus the
# noun-homographs (which get the restricted treatment described above).
_STRONG_VERB_STEMS: dict[str, frozenset[str]] = {
    level: stems - _HOMOGRAPH_STEMS for level, stems in _ACTION_VERB_STEMS.items()
}
# Homograph stem -> highest level it appears at in ACTION_VERBS.
_HOMOGRAPH_LEVELS: dict[str, str] = {
    s: level
    for level in ACTION_VERBS  # low to high; later (higher) levels overwrite
    for s in (_ACTION_VERB_STEMS[level] & _HOMOGRAPH_STEMS)
}
_DERIVED_FORM_LEVELS: dict[str, str] = {
    stem(form): level for form, level in BLOOM_DERIVED_FORMS.items()
}
_DOMAIN_TOKEN_STEMS: dict[str, frozenset[str]] = {
    domain: frozenset(stem(term) for term in terms if " " not in term)
    for domain, terms in DOMAIN_TERMS.items()
}
_DOMAIN_PHRASES: dict[str, tuple[str, ...]] = {
    domain: tuple(term for term in terms if " " in term)
    for domain, terms in DOMAIN_TERMS.items()
}
_STOPWORD_STEMS: frozenset[str] = frozenset(stem(word) for word in STOPWORDS)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def token_set(text: str) -> set[str]:
    """Tokenise ``text`` into a set of stemmed, stopword-filtered tokens.

    Lower-cases and strips punctuation, drops stopwords (checked on the
    stemmed form, so "Students"/"abilities" are removed too), and stems the
    survivors.  Because scoring.py builds its Jaccard token overlap from
    this set, stemming here makes "Developing ... managerial" overlap with
    "develop ... management-related" without any change to scoring.py.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return {
        stemmed
        for token in tokens
        if (stemmed := stem(token)) not in _STOPWORD_STEMS
    }


def _stem_closure(tokens: Iterable[str]) -> set[str]:
    """Each token lower-cased as-is, plus its stem.

    Vocabulary lookups match against this union so both raw tokens
    ("database") and already-stemmed token_set output ("databas" — whose
    trailing s would be mis-stripped if stemmed a second time) resolve to
    the pre-stemmed vocabulary entries.
    """
    values = set()
    for token in tokens:
        lowered = token.lower()
        values.add(lowered)
        values.add(stem(lowered))
    return values


def _highest(levels: Iterable[str]) -> str:
    return max(levels, key=BLOOM_INDEX.__getitem__)


def _strong_verb_level(stem_value: str) -> str | None:
    """Highest level whose unambiguous verb stems contain ``stem_value``."""
    for level in reversed(BLOOM_ORDER):
        if stem_value in _STRONG_VERB_STEMS[level]:
            return level
    return None


def _detect_bloom_ordered(text: str) -> str:
    """Leading-verb-wins Bloom detection over the raw statement text.

    Outcome statements lead with their action verb(s): "Understand ...",
    "Develop ...", "Understanding and Applying ...".  The policy:

    1. Collect the LEADING VERB GROUP: consecutive unambiguous action
       verbs at the start of the statement (stopwords such as "ability",
       "to", "and" are transparent, so "Ability to develop ..." leads with
       "develop").  A noun-homograph (see BLOOM_NOUN_HOMOGRAPHS) may only
       OPEN the group ("Measure riskiness ..." -> evaluate); once any
       other content word has been seen it is treated as a noun
       ("Understand various Models ..." -> understand, not create).
       Compound statements resolve to the highest level in the group
       ("Understanding and Applying ..." -> apply), which is the level the
       outcome ultimately targets.
    2. If the statement does not lead with a verb, fall back to the
       highest unambiguous verb anywhere in the text, then to derived
       adjective/noun forms ("Foster Analytical ... thinking" -> analyze).
       Mid-text noun-homographs never count here.
    3. Default: "understand".
    """
    ordered = [
        s
        for token in _TOKEN_RE.findall(text.lower())
        if (s := stem(token)) not in _STOPWORD_STEMS
    ]

    leading: list[str] = []
    for position, s in enumerate(ordered):
        level = _strong_verb_level(s)
        if level is not None:
            leading.append(level)
            continue
        if position == 0 and s in _HOMOGRAPH_LEVELS:
            leading.append(_HOMOGRAPH_LEVELS[s])
            continue
        break
    if leading:
        return _highest(leading)

    strong = [lvl for s in ordered if (lvl := _strong_verb_level(s)) is not None]
    if strong:
        return _highest(strong)

    derived = [_DERIVED_FORM_LEVELS[s] for s in ordered if s in _DERIVED_FORM_LEVELS]
    if derived:
        return _highest(derived)
    return "understand"


def _detect_bloom_unordered(tokens: Iterable[str]) -> str:
    """Bag-of-words fallback when no raw text (hence no word order) exists.

    Tier 1: highest level with an unambiguous verb match.  Tier 2: derived
    adjective/noun forms.  Tier 3 (last resort): noun-homographs — with no
    order information a lone "produce"/"design" is still best read as a
    verb, but any unambiguous verb in the bag outranks it, so
    "understand ... value ..." resolves to understand, not evaluate.
    """
    token_stems = _stem_closure(tokens)
    for level in reversed(BLOOM_ORDER):
        if _STRONG_VERB_STEMS[level] & token_stems:
            return level
    derived = [
        level for s, level in _DERIVED_FORM_LEVELS.items() if s in token_stems
    ]
    if derived:
        return _highest(derived)
    homographs = [
        level for s, level in _HOMOGRAPH_LEVELS.items() if s in token_stems
    ]
    if homographs:
        return _highest(homographs)
    return "understand"


def detect_bloom(tokens: Iterable[str], text: str | None = None) -> str:
    """Return the Bloom level of an outcome statement.

    With the optional raw ``text`` (preferred; used by scoring.py), the
    LEADING verb group of the statement decides the level — pedagogically
    correct for outcome statements, which front-load their action verbs —
    and mid-sentence noun-homographs ("the value of assets", "Models of
    Investment") no longer inflate the level.  See _detect_bloom_ordered.

    Without ``text``, an order-free tiered match over ``tokens`` is used
    (unambiguous verbs, then derived forms, then noun-homographs); tokens
    are considered both as-is and stemmed, so raw tokens ("Developing")
    and pre-stemmed token_set output ("develop") both work.
    """
    if text is not None:
        return _detect_bloom_ordered(text)
    return _detect_bloom_unordered(tokens)


def bloom_distance(level_a: str, level_b: str) -> int:
    return abs(BLOOM_INDEX[level_a] - BLOOM_INDEX[level_b])


def detect_domains(tokens: Iterable[str], text: str | None = None) -> set[str]:
    """Return the set of domains matched by ``tokens``.

    Single-word vocabulary terms are matched on stems.  If the optional raw
    ``text`` is provided, multi-word terms ("code generation",
    "organizational behavior") are additionally matched as substrings of
    the normalised text.  The one-argument form keeps the original
    signature used by scoring.py.
    """
    token_stems = _stem_closure(tokens)
    normalized_text = _normalize(text) if text is not None else None

    matched = set()
    for domain, term_stems in _DOMAIN_TOKEN_STEMS.items():
        if term_stems & token_stems:
            matched.add(domain)
            continue
        if normalized_text is not None and any(
            phrase in normalized_text for phrase in _DOMAIN_PHRASES[domain]
        ):
            matched.add(domain)
    return matched


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
