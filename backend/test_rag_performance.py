import sys
import os
import time

# [FIX 3] Robust Pathing
# Gets the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes 'rag_engine.py' is in the same folder as this script (backend/)
# If you run from root, this ensures 'backend' is in path
sys.path.append(current_dir)

try:
    from rag_engine import chat_with_data
except ImportError:
    # Fallback if running from root and 'backend' is a package
    sys.path.append(os.path.join(current_dir, '..'))
    from backend.rag_engine import chat_with_data

# The 15 Edge Questions
TEST_QUESTIONS = [
    "1. Describe the relationship between the 'Master', 'Zombie', and 'Server' as depicted in the 'Generic DDOS Attack Diagram'.",
    "2. According to the comparison table, which is faster: a virus or a worm? Provide the specific example mentioned.",
    "3. What is the exact monetary penalty for 'Failure to furnish information' under the Penalties and Adjudication section of the IT Act?",
    "4. List the exact steps to find the 'Number of columns' in a SQL Injection attack using the 'order by' command.",
    "5. In the provided C code example for buffer overflow, how does the memcpy() function contribute to the vulnerability?",
    "6. Explain the distinction between 'Digital Signature' (IT Act 2000) and 'Electronic Signature' (IT Act 2008).",
    "7. List four specific types of documents or transactions to which the IT Act does not apply.",
    "8. What is a 'NOP-sled' and how does it help an attacker exploit a stack buffer overflow?",
    "9. What specific offense is covered under Section 66E of the amended IT Act?",
    "10. Explain how a SYN Attack uses 'half-open connections' to saturate a server.",
    "11. Describe the difference between the 'Vessel Image' and the 'Stego Image' based on the RGB values provided in the Steganography example.",
    "12. What is 'Back Orifice' and what is its primary function as a Backdoor Trojan?",
    "13. How does Blind SQL Injection differ from standard SQL Injection in terms of the feedback the attacker receives?",
    "14. What specific change did the 2008 Amendment bring regarding the definition of 'Banker's books' in the Third Schedule?",
    "15. If a user receives an email that leads to a fake website designed to steal credit card info, is this classified as Phishing or Spoofing in the text?"
]

def run_tests():
    output_file = "test_results.txt"
    print(f"Starting RAG Performance Test with {len(TEST_QUESTIONS)} questions...")
    print(f"Results will be saved to: {output_file}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"Processing Q{i}/{len(TEST_QUESTIONS)}...")
            
            start_time = time.time()
            try:
                # Query the RAG engine
                response = chat_with_data(question)
            except Exception as e:
                response = f"[ERROR] Failed to generate response: {str(e)}"
            
            end_time = time.time()
            duration = end_time - start_time

            # Write to file
            f.write(f"Question {i}: {question}\n")
            f.write(f"Time Taken: {duration:.2f}s\n")
            f.write("-" * 20 + " RESPONSE " + "-" * 20 + "\n")
            f.write(f"{response}\n")
            f.write("=" * 60 + "\n\n")

    print(f"\nTest Complete! Please upload '{output_file}' for analysis.")

if __name__ == "__main__":
    run_tests()