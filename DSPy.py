# Standard evaluation function (kept for comparison)
def evaluate_program(program, dataset, metric_func):
    """Evaluate a program on a dataset using a metric."""
    total_score = 0
    total_examples = len(dataset)
    
    for example in dataset:
        pred = program(text=example.text)
        score = metric_func(example, pred)
        total_score += score
    
    return total_score / total_examples

# Run detailed trace evaluations
print("=== TRACE DEMONSTRATIONS ===")

# Test trace evaluation on original program
print("\n1. Testing trace-aware evaluation:")
detailed_trace_evaluation(program, validation_data[:1])

# Test built-in trace functionality
print("\n2. Testing built-in trace access:")
evaluate_with_built_in_trace(program, validation_data[:1])import dspy
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

# 6. Define your evaluation metric WITH trace usage
def sentiment_accuracy_metric(example, pred, trace=None):
    """
    Metric function that evaluates if the predicted sentiment matches the expected sentiment.
    
    Args:
        example: The ground truth example with expected output
        pred: The prediction from your program
        trace: Optional execution trace containing the program's reasoning steps
    
    Returns:
        bool: True if prediction is correct, False otherwise
    """
    # Extract the predicted sentiment
    predicted_sentiment = pred.sentiment.lower().strip()
    expected_sentiment = example.sentiment.lower().strip()
    
    # Use trace for debugging and additional scoring
    if trace is not None:
        print(f"\n--- TRACE ANALYSIS ---")
        print(f"Input text: {example.text}")
        print(f"Expected: {expected_sentiment}, Predicted: {predicted_sentiment}")
        
        # The trace contains the execution history
        for i, step in enumerate(trace):
            print(f"\nStep {i+1}: {step}")
            
            # If this step has a completion (LM output), show it
            if hasattr(step, 'completion') and step.completion:
                print(f"  Raw LM Output: {step.completion}")
            
            # Show any intermediate reasoning
            if hasattr(step, 'rationale') and hasattr(step.rationale, 'rationale'):
                print(f"  Reasoning: {step.rationale.rationale}")
    
    # Check if they match
    return predicted_sentiment == expected_sentiment

# Alternative: Advanced metric using trace for quality assessment
def trace_aware_sentiment_metric(example, pred, trace=None):
    """
    Enhanced metric that uses trace information to assess reasoning quality.
    """
    predicted = pred.sentiment.lower().strip()
    expected = example.sentiment.lower().strip()
    
    base_score = 1.0 if predicted == expected else 0.0
    
    if trace is None:
        return base_score
    
    # Bonus points for good reasoning (if using ChainOfThought)
    reasoning_bonus = 0.0
    
    for step in trace:
        if hasattr(step, 'rationale') and hasattr(step.rationale, 'rationale'):
            reasoning = step.rationale.rationale.lower()
            
            # Check if reasoning mentions relevant sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointing', 'poor']
            neutral_words = ['okay', 'average', 'normal', 'fine', 'adequate']
            
            reasoning_quality = 0
            if expected == 'positive' and any(word in reasoning for word in positive_words):
                reasoning_quality += 0.1
            elif expected == 'negative' and any(word in reasoning for word in negative_words):
                reasoning_quality += 0.1
            elif expected == 'neutral' and any(word in reasoning for word in neutral_words):
                reasoning_quality += 0.1
            
            # Bonus for longer, more detailed reasoning
            if len(reasoning.split()) > 10:
                reasoning_quality += 0.05
                
            reasoning_bonus = min(0.2, reasoning_quality)  # Cap bonus at 0.2
    
    return base_score + reasoning_bonus

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

# 10. Demonstrate trace usage with manual evaluation
def detailed_trace_evaluation(program, examples, show_traces=True):
    """
    Evaluate program with detailed trace information.
    """
    print("=== DETAILED TRACE EVALUATION ===")
    
    for i, example in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        
        # Enable trace collection
        with dspy.context(show_trace=show_traces):
            pred = program(text=example.text)
            
            # Get the trace from the context
            if show_traces:
                # Access the trace through DSPy's context
                import dspy
                trace = dspy.settings.trace if hasattr(dspy.settings, 'trace') else None
                
                print(f"Input: {example.text}")
                print(f"Expected: {example.sentiment}")
                print(f"Predicted: {pred.sentiment}")
                
                # Manual trace analysis
                if hasattr(pred, '_trace') and pred._trace:
                    print("Program execution steps:")
                    for step_idx, step in enumerate(pred._trace):
                        print(f"  Step {step_idx + 1}: {type(step).__name__}")
                        
                        if hasattr(step, 'rationale'):
                            print(f"    Reasoning: {step.rationale}")
                        
                        if hasattr(step, 'completion'):
                            print(f"    Raw output: {step.completion}")
                
                # Evaluate with trace-aware metric
                score = trace_aware_sentiment_metric(example, pred, getattr(pred, '_trace', None))
                print(f"Score (with reasoning bonus): {score:.2f}")

# Alternative way to capture traces using DSPy's built-in tracing
def evaluate_with_built_in_trace(program, examples):
    """
    Use DSPy's built-in trace functionality.
    """
    print("=== BUILT-IN TRACE EVALUATION ===")
    
    for i, example in enumerate(examples[:2]):  # Limit to 2 examples for clarity
        print(f"\n--- Example {i+1} ---")
        
        # Method 1: Use inspect_history after prediction
        pred = program(text=example.text)
        
        print(f"Input: {example.text}")
        print(f"Expected: {example.sentiment}")
        print(f"Predicted: {pred.sentiment}")
        
        # Try to access DSPy's internal history
        try:
            if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
                recent_history = dspy.settings.lm.history[-1:]  # Get most recent call
                for entry in recent_history:
                    print(f"Prompt sent to LM:")
                    print(f"  {entry.get('prompt', 'No prompt found')}")
                    print(f"Response from LM:")
                    print(f"  {entry.get('response', 'No response found')}")
        except Exception as e:
            print(f"Could not access LM history: {e}")
        
        print("-" * 50)

# Compare before and after
original_accuracy = evaluate_program(SentimentProgram(), validation_data, sentiment_accuracy_metric)
optimized_accuracy = evaluate_program(optimized_program, validation_data, sentiment_accuracy_metric)

print("=== PERFORMANCE COMPARISON ===")
print(f"Original Program Accuracy: {original_accuracy:.2f}")
print(f"Optimized Program Accuracy: {optimized_accuracy:.2f}")
print(f"Improvement: {optimized_accuracy - original_accuracy:.2f}")

# =====================================================
# EXTRACTING THE OPTIMIZED PROMPTS AND DEMONSTRATIONS
# =====================================================

print("\n" + "="*60)
print("EXTRACTING OPTIMIZED PROMPTS AND DEMONSTRATIONS")
print("="*60)

def extract_optimized_prompts(optimized_program):
    """
    Extract the optimized prompts and few-shot examples from the optimized program.
    """
    print("\n=== OPTIMIZED PROGRAM STRUCTURE ===")
    
    # Method 1: Access the predictor directly
    if hasattr(optimized_program, 'predictor'):
        predictor = optimized_program.predictor
        print(f"Predictor type: {type(predictor).__name__}")
        
        # Check if it has demonstrations (few-shot examples)
        if hasattr(predictor, 'demos'):
            print(f"\nNumber of demonstrations: {len(predictor.demos)}")
            print("\n--- OPTIMIZED FEW-SHOT EXAMPLES ---")
            
            for i, demo in enumerate(predictor.demos):
                print(f"\nDemo {i+1}:")
                if hasattr(demo, 'text'):
                    print(f"  Input: {demo.text}")
                if hasattr(demo, 'sentiment'):
                    print(f"  Output: {demo.sentiment}")
                if hasattr(demo, 'rationale'):
                    print(f"  Reasoning: {demo.rationale}")
        
        # Check for optimized instructions/prompts
        if hasattr(predictor, 'signature'):
            print(f"\n--- OPTIMIZED SIGNATURE ---")
            print(f"Signature: {predictor.signature}")
            
            # Try to access the actual prompt template
            if hasattr(predictor.signature, 'instructions'):
                print(f"Instructions: {predictor.signature.instructions}")
        
        # Method 2: Try to access the prompt template directly
        if hasattr(predictor, 'prompt_template'):
            print(f"\n--- PROMPT TEMPLATE ---")
            print(predictor.prompt_template)
    
    return predictor if hasattr(optimized_program, 'predictor') else None

def save_optimized_program(optimized_program, filename="optimized_program.json"):
    """
    Save the optimized program to a file for later use.
    """
    try:
        # Try to save the program
        optimized_program.save(filename)
        print(f"\n=== PROGRAM SAVED ===")
        print(f"Optimized program saved to: {filename}")
        print("You can load it later with: program.load('optimized_program.json')")
    except Exception as e:
        print(f"Could not save program: {e}")
        print("Note: Some DSPy versions may not support saving all program types")

def inspect_actual_prompts(program, example_input):
    """
    Run a prediction and capture the actual prompt sent to the language model.
    """
    print(f"\n=== ACTUAL PROMPT INSPECTION ===")
    print("Running prediction to capture the actual prompt...")
    
    # Clear any existing history
    if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
        dspy.settings.lm.history.clear()
    
    # Make a prediction
    result = program(text=example_input)
    
    # Try to capture the prompt from LM history
    try:
        if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
            if dspy.settings.lm.history:
                latest_call = dspy.settings.lm.history[-1]
                
                print(f"\n--- ACTUAL PROMPT SENT TO LM ---")
                if 'prompt' in latest_call:
                    print(latest_call['prompt'])
                elif 'messages' in latest_call:
                    for msg in latest_call['messages']:
                        print(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
                
                print(f"\n--- LM RESPONSE ---")
                if 'response' in latest_call:
                    print(latest_call['response'])
                    
                print(f"\n--- PARSED RESULT ---")
                print(f"Sentiment: {result.sentiment}")
                
                return latest_call
    except Exception as e:
        print(f"Could not capture prompt: {e}")
    
    return None

def compare_prompts(original_program, optimized_program, example_input):
    """
    Compare prompts between original and optimized programs.
    """
    print(f"\n=== PROMPT COMPARISON ===")
    
    print("\n--- ORIGINAL PROGRAM PROMPT ---")
    original_prompt = inspect_actual_prompts(original_program, example_input)
    
    print("\n" + "="*50)
    print("--- OPTIMIZED PROGRAM PROMPT ---")
    optimized_prompt = inspect_actual_prompts(optimized_program, example_input)
    
    return original_prompt, optimized_prompt

# Execute the prompt extraction
print("Extracting optimized prompts and demonstrations...")
predictor = extract_optimized_prompts(optimized_program)

# Save the optimized program
save_optimized_program(optimized_program)

# Inspect actual prompts being sent to the LM
example_text = "This product is really great and I love using it!"
print(f"\nUsing example: '{example_text}'")

original_prompt, optimized_prompt = compare_prompts(
    SentimentProgram(), 
    optimized_program, 
    example_text
)

# Additional method: Manual inspection of program components
print(f"\n=== MANUAL PROGRAM INSPECTION ===")
print("Optimized program attributes:")
for attr in dir(optimized_program):
    if not attr.startswith('_'):
        try:
            value = getattr(optimized_program, attr)
            print(f"  {attr}: {type(value).__name__}")
        except:
            print(f"  {attr}: <could not access>")

# Inspect the predictor more deeply
if predictor:
    print(f"\nPredictor attributes:")
    for attr in dir(predictor):
        if not attr.startswith('_') and not callable(getattr(predictor, attr)):
            try:
                value = getattr(predictor, attr)
                if isinstance(value, (str, int, float, bool, list)) and len(str(value)) < 200:
                    print(f"  {attr}: {value}")
                else:
                    print(f"  {attr}: {type(value).__name__} (length: {len(str(value)) if hasattr(value, '__len__') else 'unknown'})")
            except:
                print(f"  {attr}: <could not access>")

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
