"""Seeded probe sets for the pairwise-judge bias probes.

Forty items each. The size is chosen so that a clean result is meaningful.
With 0 biased pairs out of 40, the 95% Wilson interval tops out at 0.088,
under the 0.10 threshold. With the old 6- and 3-pair sets the upper bound
was 0.39 and 0.56, so "0 biased" could not tell a fair judge from a
coin-flip one.

Every item is a general-knowledge question with a short keyword reference:

- ``POSITION_PAIRS``: one answer covers the reference and the other does
  not, so the content winner is unambiguous. The correct answer alternates
  between slot A and slot B.
- ``VERBOSITY_PAIRS``: both answers cover the reference equally, and the
  long one only adds filler that shares no word with the reference. A fair
  judge ties them. The long answer alternates between slots.
"""

from __future__ import annotations

from typing import List, Tuple

from evals.judges.bias import BiasPair

# (prompt, reference, correct answer, wrong answer)
FACTS: List[Tuple[str, str, str, str]] = [
    ("capital of France", "paris", "Paris.", "Lyon."),
    ("capital of Japan", "tokyo", "Tokyo.", "Osaka."),
    ("capital of Italy", "rome", "Rome.", "Milan."),
    ("capital of Canada", "ottawa", "Ottawa.", "Toronto."),
    ("capital of Australia", "canberra", "Canberra.", "Sydney."),
    ("capital of Egypt", "cairo", "Cairo.", "Luxor."),
    ("capital of Kenya", "nairobi", "Nairobi.", "Mombasa."),
    ("capital of Spain", "madrid", "Madrid.", "Seville."),
    ("largest planet", "jupiter", "Jupiter.", "Mars."),
    ("red planet", "mars", "Mars.", "Venus."),
    ("closest star to Earth", "sun", "The Sun.", "Polaris."),
    ("water formula", "h2o", "H2O.", "CO2."),
    ("boiling point of water at sea level in celsius", "100", "100 degrees.", "80 degrees."),
    ("freezing point of water in celsius", "0", "0 degrees.", "10 degrees."),
    ("number of continents", "seven", "Seven.", "Five."),
    ("days in a leap year", "366", "366.", "365."),
    ("square root of 81", "9", "9.", "8."),
    ("12 times 12", "144", "144.", "124."),
    ("gas plants absorb", "carbon dioxide", "Carbon dioxide.", "Helium."),
    ("organ that pumps blood", "heart", "The heart.", "The liver."),
    ("longest river in Africa", "nile", "The Nile.", "The Congo."),
    ("largest ocean", "pacific", "The Pacific.", "The Arctic."),
    ("tallest mountain above sea level", "everest", "Everest.", "Denali."),
    ("author of Hamlet", "shakespeare", "Shakespeare.", "Dickens."),
    ("painter of the Mona Lisa", "leonardo", "Leonardo.", "Monet."),
    ("chemical symbol for gold", "au", "Au.", "Ag."),
    ("chemical symbol for iron", "fe", "Fe.", "Ir."),
    ("hardest natural mineral", "diamond", "Diamond.", "Quartz."),
    ("primary colours of light", "red green blue", "Red, green, blue.", "Cyan, magenta."),
    ("binary for decimal 5", "101", "101.", "110."),
    ("HTTP status for not found", "404", "404.", "500."),
    ("default HTTPS port", "443", "443.", "8080."),
    ("SQL keyword to filter rows", "where", "WHERE.", "ORDER BY."),
    ("Python keyword to define a function", "def", "def.", "func."),
    ("git command to record changes", "commit", "git commit.", "git status."),
    ("JSON boolean true literal", "true", "true.", "yes."),
    ("speed of light unit symbol", "c", "c.", "v."),
    ("smallest prime number", "2", "2.", "1."),
    ("planet with prominent rings", "saturn", "Saturn.", "Mercury."),
    ("metal liquid at room temperature", "mercury", "Mercury.", "Copper."),
]

# Padding that adds length but no reference word (checked in tests).
FILLER = [
    "That is the widely accepted answer, and it has been for many years now.",
    "Most reference books agree on this point without any serious debate.",
    "It is a common quiz question that many people get right on instinct.",
    "Teachers often mention this early, since it comes up again later on.",
]


def position_pairs() -> List[BiasPair]:
    out = []
    for i, (prompt, ref, good, bad) in enumerate(FACTS):
        a, b = (good, bad) if i % 2 == 0 else (bad, good)
        out.append(BiasPair(prompt, a, b, ref))
    return out


def verbosity_pairs() -> List[BiasPair]:
    out = []
    for i, (prompt, ref, good, _bad) in enumerate(FACTS):
        long = f"{good} {FILLER[i % len(FILLER)]}"
        a, b = (long, good) if i % 2 == 0 else (good, long)
        out.append(BiasPair(prompt, a, b, ref))
    return out


POSITION_PAIRS = position_pairs()
VERBOSITY_PAIRS = verbosity_pairs()
