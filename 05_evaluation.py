from dotenv import load_dotenv
load_dotenv()

from crag_module import execute_crag_workflow
import time

# Test cases with expected keywords to verify answer quality
test_queries = [
    {
        "query": "What is CRAG?",
        "expected_keywords": ["corrective", "retrieval", "generation"],
        "category": "Definition"
    },
    {
        "query": "How does decompose-then-recompose work?",
        "expected_keywords": ["knowledge strips", "decompose", "recompose"],
        "category": "Algorithm"
    },
    {
        "query": "What are the three actions in CRAG?",
        "expected_keywords": ["correct", "incorrect", "ambiguous"],
        "category": "Actions"
    },
    {
        "query": "Explain the retrieval evaluator",
        "expected_keywords": ["evaluator", "confidence", "quality"],
        "category": "Component"
    },
]

def evaluate_system():
    """Run comprehensive evaluation of CRAG system."""
    
    print("\n" + "="*70)
    print("CRAG SYSTEM EVALUATION")
    print("="*70 + "\n")
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected_keywords"]
        category = test["category"]
        
        print(f"\n[Test {i}/{len(test_queries)}] Category: {category}")
        print(f"Query: {query}")
        print("-" * 70)
        
        # Time the execution
        start_time = time.time()
        
        try:
            result = execute_crag_workflow(query, verbose=False)
            
            execution_time = time.time() - start_time
            answer = result["answer"]
            
            # Check keyword presence
            keywords_found = []
            keywords_missing = []
            
            for keyword in expected:
                if keyword.lower() in answer.lower():
                    keywords_found.append(keyword)
                else:
                    keywords_missing.append(keyword)
            
            keyword_match_rate = len(keywords_found) / len(expected) * 100
            
            # Collect metrics
            metrics = {
                "query": query,
                "category": category,
                "confidence": result["confidence"],
                "action": result["action"],
                "verified": result["verification"]["is_verified"],
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "keyword_match_rate": keyword_match_rate,
                "answer_length": len(answer.split()),
                "execution_time": execution_time,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer
            }
            
            results.append(metrics)
            
            # Print results
            print(f"\n  Confidence: {metrics['confidence']:.3f}")
            print(f"  Action: {metrics['action']}")
            print(f"  Verified: {'✓' if metrics['verified'] else '✗'}")
            print(f"  Keywords Found: {len(keywords_found)}/{len(expected)} ({keyword_match_rate:.0f}%)")
            print(f"    - Found: {', '.join(keywords_found) if keywords_found else 'None'}")
            if keywords_missing:
                print(f"    - Missing: {', '.join(keywords_missing)}")
            print(f"  Answer Length: {metrics['answer_length']} words")
            print(f"  Execution Time: {execution_time:.2f}s")
            print(f"\n  Answer Preview: {metrics['answer']}")
            
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            metrics = {
                "query": query,
                "category": category,
                "error": str(e),
                "confidence": 0,
                "verified": False,
                "keyword_match_rate": 0
            }
            results.append(metrics)
    
    # Print summary
    print("\n\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    successful_tests = [r for r in results if "error" not in r]
    
    if not successful_tests:
        print("\n✗ All tests failed")
        return
    
    # Calculate aggregate metrics
    avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
    verification_rate = sum(r["verified"] for r in successful_tests) / len(successful_tests) * 100
    avg_keyword_match = sum(r["keyword_match_rate"] for r in successful_tests) / len(successful_tests)
    avg_answer_length = sum(r["answer_length"] for r in successful_tests) / len(successful_tests)
    avg_execution_time = sum(r["execution_time"] for r in successful_tests) / len(successful_tests)
    
    # Action distribution
    actions = [r["action"] for r in successful_tests]
    action_counts = {action: actions.count(action) for action in set(actions)}
    
    print(f"\nTests Run: {len(test_queries)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(test_queries) - len(successful_tests)}")
    
    print(f"\n--- Performance Metrics ---")
    print(f"Average Confidence: {avg_confidence:.3f} (out of 1.0)")
    print(f"Verification Pass Rate: {verification_rate:.0f}%")
    print(f"Keyword Match Rate: {avg_keyword_match:.0f}%")
    print(f"Average Answer Length: {avg_answer_length:.0f} words")
    print(f"Average Execution Time: {avg_execution_time:.2f}s")
    
    print(f"\n--- Action Distribution ---")
    for action, count in action_counts.items():
        percentage = count / len(successful_tests) * 100
        print(f"{action}: {count} ({percentage:.0f}%)")
    
    print(f"\n--- Quality Assessment ---")
    if avg_confidence > 0.6:
        print("✓ Excellent: High retrieval confidence")
    elif avg_confidence > 0.4:
        print("○ Good: Moderate retrieval confidence")
    else:
        print("✗ Poor: Low retrieval confidence")
    
    if verification_rate == 100:
        print("✓ Excellent: No hallucinations detected")
    elif verification_rate > 80:
        print("○ Good: Minimal hallucinations")
    else:
        print("✗ Attention: High hallucination rate")
    
    if avg_keyword_match > 66:
        print("✓ Excellent: High answer relevance")
    elif avg_keyword_match > 33:
        print("○ Good: Moderate answer relevance")
    else:
        print("✗ Poor: Low answer relevance")
    
    print("\n" + "="*70 + "\n")
    
    return results


if __name__ == "__main__":
    try:
        results = evaluate_system()
        
        # Optionally save results to file
        save = input("Save results to file? (y/n): ").strip().lower()
        if save == 'y':
            import json
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("\n✓ Results saved to evaluation_results.json\n")
            
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.\n")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
