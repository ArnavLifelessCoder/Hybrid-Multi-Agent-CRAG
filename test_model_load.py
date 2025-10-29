print("Testing model download...")
import sys
print(f"Python version: {sys.version}")

print("\n1. Testing transformers import...")
try:
    import transformers
    print(f"   ✓ transformers version: {transformers.__version__}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n2. Testing torch import...")
try:
    import torch
    print(f"   ✓ torch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n3. Testing tokenizer download...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    print(f"   ✓ Tokenizer loaded successfully")
    print(f"   Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing model download...")
try:
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-small", 
        num_labels=3
    )
    print(f"   ✓ Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
