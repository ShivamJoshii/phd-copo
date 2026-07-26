from __future__ import annotations

import unittest

from copo_mapper.features import (
    BLOOM_INDEX,
    BLOOM_ORDER,
    STOPWORDS,
    bloom_distance,
    detect_bloom,
    detect_domains,
    jaccard,
    stem,
    token_set,
)

# Real CO texts from the user's data (B.Tech compiler design + MBA).
CO_LEXICAL = "Develop lexical analyzer for a given grammar"
CO_PARSERS = "Design top-down and bottom-up parsers"
CO_SDT = "Develop syntax directed translation schemes"
CO_CODEGEN = "Develop algorithms to generate code"
CO_MBA_MGMT = "Developing understanding of managerial practices and their perspectives"
CO_MBA_OB = "Understanding and Applying the concepts of organizational behavior"


class StemTest(unittest.TestCase):
    def test_gerund_stripping(self) -> None:
        self.assertEqual(stem("developing"), stem("develop"))
        self.assertEqual(stem("designing"), stem("design"))
        self.assertEqual(stem("understanding"), stem("understand"))
        self.assertEqual(stem("applying"), stem("apply"))

    def test_past_tense_stripping(self) -> None:
        self.assertEqual(stem("designed"), stem("design"))
        self.assertEqual(stem("applied"), stem("apply"))
        self.assertEqual(stem("evaluated"), stem("evaluate"))

    def test_plurals(self) -> None:
        self.assertEqual(stem("parsers"), stem("parser"))
        self.assertEqual(stem("algorithms"), stem("algorithm"))
        self.assertEqual(stem("studies"), stem("study"))
        self.assertEqual(stem("matches"), stem("match"))
        self.assertEqual(stem("uses"), stem("use"))

    def test_ation_and_ization_align_with_verb_forms(self) -> None:
        self.assertEqual(stem("translation"), stem("translate"))
        self.assertEqual(stem("translation"), stem("translating"))
        self.assertEqual(stem("generation"), stem("generate"))
        self.assertEqual(stem("organization"), stem("organizing"))

    def test_final_e_alignment(self) -> None:
        self.assertEqual(stem("manage"), stem("managing"))
        self.assertEqual(stem("create"), stem("creating"))

    def test_double_consonant_undoubling(self) -> None:
        self.assertEqual(stem("planning"), stem("plan"))
        self.assertEqual(stem("programming"), stem("program"))

    def test_min_length_guards(self) -> None:
        # Short words must survive untouched: "design" != "des".
        self.assertEqual(stem("design"), "design")
        self.assertEqual(stem("use"), "use")
        self.assertEqual(stem("os"), "os")
        # Protected endings: ss / us / is are not plural -s.
        self.assertEqual(stem("class"), "class")
        self.assertEqual(stem("analysis"), "analysis")
        self.assertEqual(stem("various"), "various")
        # Base words with genuine double letters are never un-doubled.
        self.assertEqual(stem("skill"), "skill")

    def test_idempotent_on_vocabulary_stems(self) -> None:
        # detect_* re-stem incoming tokens, so stemming must be stable.
        for word in ["developing", "translation", "parsers", "organizing", "applied"]:
            once = stem(word)
            self.assertEqual(stem(once), once)


class TokenSetTest(unittest.TestCase):
    def test_stopwords_removed(self) -> None:
        tokens = token_set("At the end of the course the students will be able to design")
        self.assertEqual(tokens, {"design"})

    def test_stopwords_removed_on_stemmed_form(self) -> None:
        # "abilities" -> "ability" (stopword); "concepts" -> "concept".
        tokens = token_set("abilities and concepts of parsers")
        self.assertEqual(tokens, {"parser"})

    def test_tokens_are_stemmed_and_lowercased(self) -> None:
        tokens = token_set("Developing Parsers, translating grammars!")
        self.assertIn("develop", tokens)
        self.assertIn("parser", tokens)
        self.assertIn("translat", tokens)
        self.assertIn("grammar", tokens)

    def test_bloom_verbs_survive_filtering(self) -> None:
        # detect_bloom runs on token_set output in scoring.py, so action
        # verbs must not be stopworded away.
        for co in [CO_LEXICAL, CO_PARSERS, CO_SDT, CO_CODEGEN, CO_MBA_MGMT, CO_MBA_OB]:
            self.assertIn(detect_bloom(token_set(co)), BLOOM_ORDER)

    def test_understanding_is_not_a_stopword(self) -> None:
        self.assertNotIn("understanding", STOPWORDS)
        self.assertIn("understand", token_set("Understanding managerial practices"))


class BloomDetectionTest(unittest.TestCase):
    def test_compiler_cos_are_create_level(self) -> None:
        self.assertEqual(detect_bloom(token_set(CO_LEXICAL)), "create")
        self.assertEqual(detect_bloom(token_set(CO_PARSERS)), "create")
        self.assertEqual(detect_bloom(token_set(CO_SDT)), "create")
        self.assertEqual(detect_bloom(token_set(CO_CODEGEN)), "create")

    def test_gerund_verbs_match(self) -> None:
        # "Developing"/"Understanding"/"Applying" previously matched nothing
        # (exact-match only); the stemmer fixes that.
        self.assertEqual(detect_bloom(["understanding"]), "understand")
        self.assertEqual(detect_bloom(["applying"]), "apply")
        # POLICY: "develop" is an unambiguous action verb (not a noun-
        # homograph), so a lone "developing" reads as the verb -> create.
        self.assertEqual(detect_bloom(["developing"]), "create")

    def test_mba_ob_co_resolves_to_apply(self) -> None:
        # POLICY: compound leading verb groups ("Understanding and
        # Applying ...") resolve to the highest level in the group — the
        # level the outcome ultimately targets. Holds in both the ordered
        # (text) and unordered (token bag) modes.
        self.assertEqual(detect_bloom(token_set(CO_MBA_OB)), "apply")
        self.assertEqual(detect_bloom(token_set(CO_MBA_OB), text=CO_MBA_OB), "apply")

    def test_mba_mgmt_co_resolves_to_create(self) -> None:
        # POLICY (deliberate): "Developing understanding of managerial
        # practices ..." leads with the verb "Developing"; "understanding"
        # here is its object, but both sit in the leading verb group and
        # the group resolves to its highest level -> create. The unordered
        # fallback (highest unambiguous verb) agrees.
        level = detect_bloom(token_set(CO_MBA_MGMT))
        self.assertGreaterEqual(BLOOM_INDEX[level], BLOOM_INDEX["understand"])
        self.assertEqual(level, "create")
        self.assertEqual(detect_bloom(token_set(CO_MBA_MGMT), text=CO_MBA_MGMT), "create")

    def test_multi_level_verbs_resolve_to_highest(self) -> None:
        # "produce" appears in both apply and create lists. It is a noun-
        # homograph, but with a bare one-word bag there is no unambiguous
        # verb to outrank it, so the last-resort homograph tier reads it as
        # a verb at its highest level.
        self.assertEqual(detect_bloom(["produce"]), "create")
        # "estimate" appears in both understand and evaluate lists.
        self.assertEqual(detect_bloom(["estimate"]), "evaluate")

    def test_noun_homographs_do_not_bump_level(self) -> None:
        # C2 regression tests: mid-sentence noun usages of Bloom verbs
        # ("Models", "value") must not override the leading verb.
        self.assertEqual(
            detect_bloom(
                [], text="Understand various Models of Investment and its application"
            ),
            "understand",
        )
        self.assertEqual(
            detect_bloom(
                [], text="Understand the value of assets and manage investment portfolio"
            ),
            "understand",
        )
        # Unordered mode agrees: an unambiguous verb outranks homographs.
        self.assertEqual(
            detect_bloom(token_set("Understand the value of assets")), "understand"
        )

    def test_leading_noun_homograph_counts_as_verb(self) -> None:
        # A homograph that OPENS the statement is the action verb.
        self.assertEqual(
            detect_bloom([], text="Measure riskiness of a stock or a portfolio position"),
            "evaluate",
        )
        self.assertEqual(
            detect_bloom([], text="Design top-down and bottom-up parsers"), "create"
        )
        # Leading position is checked after stopword filtering, so OBE
        # filler ("Ability to ...") is transparent.
        self.assertEqual(
            detect_bloom([], text="Ability to develop Value based Leadership ability"),
            "create",
        )

    def test_derived_adjective_and_noun_forms(self) -> None:
        # MINOR fix: adjectival/nominal forms signal the level when no
        # regular action verb is present.
        self.assertEqual(
            detect_bloom(
                [],
                text="Foster Analytical and critical thinking abilities "
                "for data-based decision making",
            ),
            "analyze",
        )
        self.assertEqual(
            detect_bloom([], text="Application of DBMS for business process"), "apply"
        )
        self.assertEqual(detect_bloom(["creativity"]), "create")
        self.assertEqual(detect_bloom(["comprehension"]), "understand")

    def test_default_is_understand(self) -> None:
        self.assertEqual(detect_bloom(["quantum", "chromodynamics"]), "understand")
        self.assertEqual(detect_bloom([], text="Knowledge about the DBMS Technology"), "understand")

    def test_bloom_distance(self) -> None:
        self.assertEqual(bloom_distance("remember", "create"), 5)
        self.assertEqual(bloom_distance("apply", "apply"), 0)


class DomainDetectionTest(unittest.TestCase):
    def test_compiler_design_cos(self) -> None:
        # These matched NO domain with the old 6x4 vocabulary.
        self.assertIn("compilers", detect_domains(token_set(CO_LEXICAL)))
        self.assertIn("compilers", detect_domains(token_set(CO_PARSERS)))
        self.assertIn("compilers", detect_domains(token_set(CO_SDT)))
        self.assertIn("algorithms", detect_domains(token_set(CO_CODEGEN)))

    def test_mba_cos(self) -> None:
        # CHANGED (C1): the catch-all tokens ("management", "managerial",
        # "organizational", ...) were pruned from the management domain
        # because they fire on nearly every MBA CO/PO pair and inflated
        # labels. A generic "managerial practices" CO therefore no longer
        # maps to the domain at all ...
        self.assertEqual(detect_domains(token_set(CO_MBA_MGMT)), set())
        self.assertEqual(detect_domains(token_set(CO_MBA_MGMT), text=CO_MBA_MGMT), set())
        # ... while the distinctive phrase "organizational behavior" still
        # does (phrase matching needs the raw text).
        self.assertIn(
            "management", detect_domains(token_set(CO_MBA_OB), text=CO_MBA_OB)
        )

    def test_management_domain_keeps_distinctive_terms(self) -> None:
        self.assertIn("management", detect_domains(token_set("staffing and controlling")))
        text = "Corporate governance and stakeholder analysis"
        self.assertIn("management", detect_domains(token_set(text), text=text))

    def test_plural_and_capitalised_terms_match(self) -> None:
        self.assertIn("algorithms", detect_domains(token_set("Algorithms and complexity")))
        self.assertIn("databases", detect_domains(token_set("relational Databases")))

    def test_multiple_domains(self) -> None:
        domains = detect_domains(token_set("Design a database for financial accounting"))
        self.assertIn("databases", domains)
        self.assertIn("finance", domains)

    def test_bigram_matching_via_optional_text(self) -> None:
        text = "Apply syntax directed translation and code generation techniques"
        domains = detect_domains(token_set(text), text=text)
        self.assertIn("compilers", domains)
        ob_text = "Explain theories of organizational behavior"
        self.assertIn("management", detect_domains(token_set(ob_text), text=ob_text))

    def test_tokens_only_signature_still_works(self) -> None:
        # scoring.py calls detect_domains(tokens) with one argument.
        self.assertEqual(detect_domains([]), set())
        self.assertIn("networks", detect_domains(["routing", "protocols"]))

    def test_no_spurious_match(self) -> None:
        self.assertEqual(detect_domains(token_set("Paint a landscape in watercolour")), set())


class JaccardTest(unittest.TestCase):
    def test_both_empty(self) -> None:
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_one_empty(self) -> None:
        self.assertEqual(jaccard({"a"}, set()), 0.0)

    def test_identical(self) -> None:
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint(self) -> None:
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_stemmed_token_sets_overlap(self) -> None:
        # End-to-end: gerund CO vs base-form PO now overlap on tokens.
        co = token_set("Developing understanding of managerial practices")
        po = token_set("Develop and understand management practice")
        self.assertGreater(jaccard(co, po), 0.0)
        self.assertIn("develop", co & po)
        self.assertIn("understand", co & po)


if __name__ == "__main__":
    unittest.main()
