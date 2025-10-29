import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./models/retrieval_evaluator"

def train_evaluator():
    print("="*70)
    print("CRAG RETRIEVAL EVALUATOR TRAINING")
    print("="*70)
    
    print("\nStep 1: Preparing enhanced training data...")
    
    # EXPANDED DATASET - More examples for better training
    data = {
        "query": [
            # Correct examples (about CRAG/RAG) - label 0
            "What is Self-RAG?",
            "Explain Corrective RAG (CRAG).",
            "What are the key components of a multi-agent system?",
            "How does the retrieval evaluator work in CRAG?",
            "What is the decompose-then-recompose algorithm?",
            "What is CRAG in machine learning?",
            "Explain the CRAG methodology",
            "How does CRAG reduce hallucinations?",
            "What is the confidence degree in CRAG?",
            "Describe the knowledge refinement process",
            "How does CRAG work?",
            "What are the actions in CRAG?",
            
            # Ambiguous examples - label 1
            "What is retrieval augmentation?",
            "How do language models handle knowledge?",
            "What are some AI techniques?",
            "What is machine learning?",
            "How do neural networks work?",
            "What is natural language processing?",
            
            # Incorrect examples - label 2
            "What is the capital of Japan?",
            "Who won the 2023 World Cup?",
            "What is the stock price of Tesla?",
            "How many planets are in our solar system?",
            "What is CRAG in geology?",
            "Who is the president of France?",
        ],
        "retrieved_docs": [
            # Correct matches
            "Self-RAG is a framework for LLMs that learns to retrieve, generate, and critique its own output through reflection tokens.",
            "The CRAG methodology involves a lightweight retrieval evaluator and corrective actions including web search and knowledge refinement.",
            "A multi-agent system has a planner, specialized agents, and a synthesizer for coordinating tasks.",
            "The retrieval evaluator in CRAG assesses document quality and triggers corrective actions based on confidence scores.",
            "Decompose-then-recompose splits documents into knowledge strips, filters irrelevant information, and recomposes relevant segments.",
            "CRAG stands for Corrective Retrieval Augmented Generation, designed to improve robustness of RAG systems.",
            "CRAG uses a retrieval evaluator to assess retrieved documents and trigger different actions: Correct, Incorrect, or Ambiguous.",
            "CRAG reduces hallucinations through knowledge refinement, web search supplementation, and document quality evaluation.",
            "The confidence degree in CRAG determines which action to trigger based on retrieval quality assessment.",
            "Knowledge refinement decomposes documents into strips, scores each strip for relevance, and recomposes high-scoring strips.",
            "CRAG improves generation robustness by evaluating retrieval quality and triggering corrective actions when documents are irrelevant.",
            "Three actions in CRAG: Correct (use internal knowledge), Incorrect (use web search), Ambiguous (combine both sources).",
            
            # Ambiguous matches
            "Retrieval augmentation is a technique used in machine learning. It involves retrieving external information to enhance model outputs.",
            "Language models use various techniques to process and generate text. Some models incorporate external knowledge sources.",
            "There are many AI techniques including neural networks, decision trees, and reinforcement learning approaches.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "Natural language processing involves computational techniques for analyzing and generating human language.",
            
            # Incorrect matches
            "Paris is the capital of France, known for its iconic Eiffel Tower and rich cultural heritage.",
            "The tournament was held in various countries with multiple teams competing for the championship title.",
            "Stock markets fluctuate based on various economic factors and investor sentiment. Technology companies have seen growth.",
            "The solar system contains various celestial bodies including planets, asteroids, and comets orbiting the sun.",
            "A crag is a steep or rugged cliff or rock face, commonly found in mountainous terrain.",
            "The president of France is elected for a five-year term and serves as head of state.",
        ],
        "label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    }
    
    df = pd.DataFrame(data)
    print(f"   ✓ Dataset size: {len(df)} examples")
    print(f"   ✓ Label distribution:")
    for label, count in sorted(df['label'].value_counts().items()):
        label_name = ["Correct", "Ambiguous", "Incorrect"][label]
        print(f"      - {label_name} (label {label}): {count} examples")
    
    dataset = Dataset.from_pandas(df)
    
    print(f"\nStep 2: Loading tokenizer and model...")
    print(f"   Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("   ✓ Tokenizer loaded")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    print("   ✓ Model loaded")
    print(f"   ✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    print("\nStep 3: Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["query"], 
            examples["retrieved_docs"],
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(f"   ✓ Tokenized {len(tokenized_dataset)} examples")
    
    print("\nStep 4: Setting up training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}_trainer", exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}_trainer",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=5,
        disable_tqdm=False,
        report_to="none",
    )
    
    print("   ✓ Training configuration:")
    print(f"      - Epochs: {training_args.num_train_epochs}")
    print(f"      - Batch size: {training_args.per_device_train_batch_size}")
    
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    
    print("\nStep 5: Training model...")
    print("   Expected time: 2-3 minutes")
    print("-" * 70)
    
    trainer.train()
    
    print("-" * 70)
    print("   ✓ Training completed!")
    
    print(f"\nStep 6: Saving model to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("   ✓ Model saved")
    
    print(f"\n{'='*70}")
    print("✓ SUCCESS: RETRIEVAL EVALUATOR TRAINED AND SAVED")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        train_evaluator()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
