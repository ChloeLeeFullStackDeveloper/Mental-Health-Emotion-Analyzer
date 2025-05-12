from predict import predict_emotions

sample_inputs = [
    "I feel overwhelmed and anxious.",
    "I'm excited and joyful!",
    "Everything is frustrating me today.",
    "I feel nothing.",
]

for text in sample_inputs:
    print(f"\n📝 Input: {text}")
    result = predict_emotions(text, threshold=0.3)
    if result:
        for label, prob in result.items():
            print(f"  {label}: {(prob * 100):.1f}%")
    else:
        print("  No strong emotions detected.")
