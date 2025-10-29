import os
from dotenv import load_dotenv

# 1. Attempt to load variables from the .env file
#    This function looks for a file named '.env' in the same directory.
load_dotenv()

# 2. Check if the OPENAI_API_KEY was successfully loaded into the environment
api_key = os.environ.get("OPENAI_API_KEY")

# 3. Print a clear status message
print("\n--- API Key Loading Status ---")
if api_key:
    print("✅ Success! Your OpenAI API Key was found.")
    print(f"   The key starts with: '{api_key[:5]}...'")
else:
    print("❌ Failure! The OpenAI API Key was NOT found.")
    print("\n   >>> Please carefully check the following:")
    print("   1. Is your file named EXACTLY `.env` (with the dot at the beginning)?")
    print("   2. Is the `.env` file in the SAME FOLDER as this `check_key.py` script?")
    print("   3. Is the text inside your `.env` file formatted correctly? It should look like this:")
    print("      OPENAI_API_KEY=\"sk-...\"")

print("----------------------------\n")