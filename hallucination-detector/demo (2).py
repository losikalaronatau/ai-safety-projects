"""
Demo: Hallucination Detector
==============================
Tests three responses against the same source document:
  1. A fully grounded response
  2. A heavily hallucinated response
  3. A mixed response (some facts right, some fabricated)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.detector import HallucinationDetector

SOURCE = """
The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars
in Paris, France. It was designed by Gustave Eiffel and built between 1887 and 1889
as the entrance arch for the 1889 World's Fair. The tower is 330 metres tall and
was the tallest man-made structure in the world for 41 years until the Chrysler
Building was completed in New York in 1930. Approximately 7 million people visit
the Eiffel Tower every year, making it the most visited paid monument in the world.
The tower has three levels accessible to visitors, with restaurants on the first
and second levels.
"""

GROUNDED_RESPONSE = """
The Eiffel Tower is a wrought-iron lattice tower located in Paris, France on the
Champ de Mars. It was designed by Gustave Eiffel and constructed between 1887 and
1889 for the 1889 World's Fair. Standing 330 metres tall, it held the record as
the world's tallest man-made structure for 41 years. The tower attracts approximately
7 million visitors annually and has three levels accessible to the public, including
restaurants on the first and second levels.
"""

HALLUCINATED_RESPONSE = """
The Eiffel Tower is a steel and concrete structure located in London, England.
It was designed by Louis Renault and built in 1750 as a military watchtower.
The tower stands 500 metres tall and cost $2 billion to construct. Over 20 million
tourists visit each year, making it the most visited monument in Europe. The tower
has seven floors and was painted blue when it was first built. It was nearly
demolished in 1920 by Napoleon Bonaparte to make room for a new railway station.
"""

MIXED_RESPONSE = """
The Eiffel Tower is located in Paris, France and was designed by Gustave Eiffel.
It was built for the 1889 World's Fair and stands 330 metres tall. The tower was
originally intended to be dismantled after 20 years but was saved because it served
as a useful radio transmission tower. It receives around 7 million visitors per year.
The tower was built in just 6 months by a team of 50 engineers, and each of its
four legs points toward a cardinal direction on the compass.
"""


def main():
    os.makedirs("reports", exist_ok=True)
    detector = HallucinationDetector(grounding_threshold=0.25)

    print("\n" + "█" * 68)
    print("  HALLUCINATION DETECTOR DEMO")
    print("█" * 68)

    tests = [
        ("Grounded Response",     GROUNDED_RESPONSE,     "reports/grounded.json"),
        ("Hallucinated Response", HALLUCINATED_RESPONSE, "reports/hallucinated.json"),
        ("Mixed Response",        MIXED_RESPONSE,        "reports/mixed.json"),
    ]

    for name, response, out_path in tests:
        print(f"\n\n  ── TEST: {name} ──")
        report = detector.check(response=response, source=SOURCE)
        detector.print_report(report)
        detector.to_json(report, out_path)


if __name__ == "__main__":
    main()
