"""Demo: Toxicity Classifier"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.classifier import ToxicityClassifier

TEXTS = [
    # Safe
    "I really enjoyed reading your article, it was very informative.",
    "The weather today is beautiful, perfect for a walk in the park.",
    "Can you help me understand how neural networks work?",
    # Profanity
    "This is such a shit idea, I can't believe anyone thought it would work.",
    # Harassment
    "Nobody likes you, you are worthless and everyone knows it.",
    # Threat
    "I will find you and make you pay for what you did.",
    # Hate speech
    "All people from that religion are dangerous and should be removed.",
    # Self harm
    "I want to end my life, I have no reason to live anymore.",
    # Misinformation
    "Vaccines cause autism, the government is hiding the truth from us.",
    # Mixed
    "You are garbage and I hope you get what's coming to you, nobody cares.",
]

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    clf = ToxicityClassifier(threshold=0.2)

    print("\n" + "█" * 60)
    print("  TOXICITY CLASSIFIER DEMO")
    print("█" * 60)

    print("\n── Single Text Analysis ──")
    for text in TEXTS[:3]:
        result = clf.classify(text)
        clf.print_result(result)

    print("\n── Batch Analysis ──")
    report = clf.classify_batch(TEXTS)
    clf.print_batch_report(report)
    clf.to_json(report, "reports/toxicity_report.json")
