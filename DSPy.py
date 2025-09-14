import dspy
import random

# 1. Configure your language model
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# 2. Define your signature (task specification)
class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of text as positive, negative, or neutral."""
    text = dspy.InputField(desc="text to analyze")
    sentiment = dspy.OutputField(desc="sentiment: positive, negative, or neutral")

# 3. Create your program/module
class SentimentProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(SentimentAnalysis)
    
    def forward(self, text):
        return self.predictor(text=text)

# 4. Create training data (normally you'd load this from a file)
training_data = [
    dspy.Example(text="I love this product! It's amazing!", sentiment="positive").with_inputs('text'),
    dspy.Example(text="This is terrible. Waste of money.", sentiment="negative").with_inputs('text'),
    dspy.Example(text="It's okay, nothing special.", sentiment="neutral").with_inputs('text'),
    dspy.Example(text="Best purchase I've ever made!", sentiment="positive").with_inputs('text'),
    dspy.Example(text="Completely disappointed with quality.", sentiment="negative").with_inputs('text'),
    dspy.Example(text="Average product, meets expectations.", sentiment="neutral").with_inputs('text'),
    dspy.Example(text="Absolutely fantastic experience!", sentiment="positive").with_inputs('text'),
    dspy.Example(text="Would not recommend to anyone.", sentiment="negative").with_inputs('text'),
]

# 5. Create validation data (for evaluation)
validation_data = [
    dspy.Example(text="Great service and fast delivery!", sentiment="positive").with_inputs('text'),
    dspy.Example(text="Poor customer service experience.", sentiment="negative").with_inputs('text'),
    dspy.Example(text="The product works as described.", sentiment="neutral").with_inputs('text'),
]

# 6. Define your evaluation metric
def sentiment_accuracy_metric(example, pred, trace=None):
    """
    Metric function that evaluates if the predicted sentiment matches the expected sentiment.
    
    Args:
        example: The ground truth example with expected output
        pred: The prediction from your program
        trace: Optional execution trace (not used here)
    
    Returns:
        bool: True if prediction is correct, False otherwise
    """
    # Extract the predicted sentiment
    predicted_sentiment = pred.sentiment.lower().strip()
    expected_sentiment = example.sentiment.lower().strip()
    
    # Check if they match
    return predicted_sentiment == expected_sentiment

# Alternative: More sophisticated metric with partial credit
def enhanced_sentiment_metric(example, pred, trace=None):
    """Enhanced metric that gives partial credit for reasonable predictions."""
    predicted = pred.sentiment.lower().strip()
    expected = example.sentiment.lower().strip()
    
    # Exact match gets full score
    if predicted == expected:
        return 1.0
    
    # Partial credit for reasonable confusion (e.g., neutral vs positive)
    reasonable_confusion = {
        ('neutral', 'positive'): 0.3,
        ('positive', 'neutral'): 0.3,
        ('neutral', 'negative'): 0.3,
        ('negative', 'neutral'): 0.3,
    }
    
    return reasonable_confusion.get((predicted, expected), 0.0)

# 7. Initialize and evaluate your program BEFORE optimization
program = SentimentProgram()

print("=== BEFORE OPTIMIZATION ===")
# Test the program before optimization
for example in validation_data[:2]:  # Test on first 2 examples
    pred = program(text=example.text)
    correct = sentiment_accuracy_metric(example, pred)
    print(f"Text: {example.text}")
    print(f"Expected: {example.sentiment}, Predicted: {pred.sentiment}")
    print(f"Correct: {correct}\n")

# 8. Set up the optimizer
from dspy.teleprompt import BootstrapFewShot

# BootstrapFewShot optimizer configuration
optimizer = BootstrapFewShot(
    metric=sentiment_accuracy_metric,  # Your evaluation function
    max_bootstrapped_demos=4,          # How many examples to use in few-shot
    max_labeled_demos=8,               # Maximum examples to consider from training set
    max_rounds=2,                      # How many optimization rounds
    max_errors=3                       # Allow some errors during bootstrapping
)

# 9. Optimize your program!
print("=== OPTIMIZING... ===")
optimized_program = optimizer.compile(
    student=program,           # Your program to optimize
    trainset=training_data     # Training examples
)

print("=== AFTER OPTIMIZATION ===")
# Test the optimized program
for example in validation_data:
    pred = optimized_program(text=example.text)
    correct = sentiment_accuracy_metric(example, pred)
    print(f"Text: {example.text}")
    print(f"Expected: {example.sentiment}, Predicted: {pred.sentiment}")
    print(f"Correct: {correct}\n")

# 10. Evaluate performance systematically
def evaluate_program(program, dataset, metric_func):
    """Evaluate a program on a dataset using a metric."""
    total_score = 0
    total_examples = len(dataset)
    
    for example in dataset:
        pred = program(text=example.text)
        score = metric_func(example, pred)
        total_score += score
    
    return total_score / total_examples

# Compare before and after
original_accuracy = evaluate_program(SentimentProgram(), validation_data, sentiment_accuracy_metric)
optimized_accuracy = evaluate_program(optimized_program, validation_data, sentiment_accuracy_metric)

print("=== PERFORMANCE COMPARISON ===")
print(f"Original Program Accuracy: {original_accuracy:.2f}")
print(f"Optimized Program Accuracy: {optimized_accuracy:.2f}")
print(f"Improvement: {optimized_accuracy - original_accuracy:.2f}")

# 11. Other optimizers you can try:

# COPRO (Coordinate Ascent Prompt Optimization)
# from dspy.teleprompt import COPRO
# copro_optimizer = COPRO(
#     metric=sentiment_accuracy_metric,
#     breadth=10,
#     depth=3,
#     init_temperature=1.4,
# )

# MIPRO (Multi-prompt Instruction Proposal Optimizer)
# from dspy.teleprompt import MIPRO
# mipro_optimizer = MIPRO(
#     metric=sentiment_accuracy_metric,
#     num_candidates=10,
#     init_temperature=1.0,
# )

print("\n=== ADVANCED TRACE USAGE TIPS ===")
print("1. Use trace to debug why predictions are wrong")
print("2. Create metrics that reward good reasoning processes")
print("3. Analyze prompt patterns that lead to better performance")
print("4. Use trace info to detect when the model is uncertain")
print("5. Build metrics that penalize confident wrong answers")
print("6. Trace helps understand how few-shot examples influence reasoning")

# Example: Confidence-aware metric using trace
def confidence_aware_metric(example, pred, trace=None):
    """
    Metric that penalizes confident wrong answers more than uncertain wrong answers.
    """
    predicted = pred.sentiment.lower().strip()
    expected = example.sentiment.lower().strip()
    
    if predicted == expected:
        return 1.0
    
    # If wrong, check confidence level from trace
    confidence_penalty = 0.0
    if trace:
        for step in trace:
            if hasattr(step, 'rationale') and hasattr(step.rationale, 'rationale'):
                reasoning = step.rationale.rationale.lower()
                # Look for confidence indicators
                high_confidence_words = ['definitely', 'clearly', 'obviously', 'certainly', 'sure']
                if any(word in reasoning for word in high_confidence_words):
                    confidence_penalty = 0.3  # Extra penalty for being confidently wrong
    
    return 0.0 - confidence_penalty  # Wrong answer gets 0, confidently wrong gets negative

# Demonstrate the confidence-aware metric in action
print("\n=== CONFIDENCE-AWARE METRIC DEMONSTRATION ===")

# Test data with examples that might produce confident wrong answers
confidence_test_data = [
    dspy.Example(text="This product is okay, nothing extraordinary.", sentiment="neutral").with_inputs('text'),
    dspy.Example(text="I'm not sure how I feel about this purchase.", sentiment="neutral").with_inputs('text'),
]

print("Testing different metrics on the same predictions:")
for example in confidence_test_data[:1]:  # Test on one example
    pred = program(text=example.text)
    
    # Compare different metric results
    basic_score = sentiment_accuracy_metric(example, pred)
    trace_aware_score = trace_aware_sentiment_metric(example, pred, getattr(pred, '_trace', None))
    confidence_score = confidence_aware_metric(example, pred, getattr(pred, '_trace', None))
    
    print(f"\nExample: '{example.text}'")
    print(f"Expected: {example.sentiment}, Predicted: {pred.sentiment}")
    print(f"Basic accuracy metric: {basic_score}")
    print(f"Trace-aware metric: {trace_aware_score}")
    print(f"Confidence-aware metric: {confidence_score}")

# You can also use the confidence-aware metric with optimizers
print("\n=== USING CONFIDENCE-AWARE METRIC WITH OPTIMIZER ===")
confidence_optimizer = BootstrapFewShot(
    metric=confidence_aware_metric,  # Using our confidence-aware metric
    max_bootstrapped_demos=3,
    max_labeled_demos=6,
    max_rounds=1,
    max_errors=2
)

print("Optimizing with confidence-aware metric...")
confidence_optimized_program = confidence_optimizer.compile(
    student=SentimentProgram(),
    trainset=training_data[:4]  # Use subset for faster demo
)

print("Testing confidence-optimized program:")
for example in validation_data[:1]:
    pred = confidence_optimized_program(text=example.text)
    score = confidence_aware_metric(example, pred, getattr(pred, '_trace', None))
    print(f"Text: {example.text}")
    print(f"Prediction: {pred.sentiment}")
    print(f"Confidence-aware score: {score}")
