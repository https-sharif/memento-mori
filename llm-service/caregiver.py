import os
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 1. Define the local data payload structure
class BehavioralAnalysis(BaseModel):
    category: str = Field(description="Categorization of behavior: e.g., Sundowning, Wandering, Aggression, Confusion")
    observed_triggers: List[str] = Field(description="Possible environmental, physical, or temporal triggers extracted from text.")
    clinical_rationale: str = Field(description="Brief neuro-clinical context explaining why the patient is exhibiting this specific behavior.")
    actionable_intervention: str = Field(description="Direct, actionable, non-pharmacological step for the caregiver to de-escalate.")
    is_crisis: bool = Field(description="Set to True ONLY if immediate physical danger or medical emergency is indicated.")

# 2. Local Test Runner
def run_local_gemini_test():
    # The modern SDK automatically looks for GEMINI_API_KEY
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: Please set your GEMINI_API_KEY environment variable.")
        return

    # Initialize the modern unified client
    client = genai.Client()
    
    # Mock context normally sourced from your app's state tracking
    patient_profile = (
        "Patient: Robert (80yo). Diagnosed with Moderate Stage Alzheimer's. "
        "Tends to exhibit high anxiety late afternoon (sundowning). Has a background in carpentry."
    )
    
    # Raw unstructured text representing a live caregiver application update
    sample_caregiver_input = (
        "It's 5 PM and Robert is getting really frantic. He's trying to tear up the baseboards "
        "in the hallway with a flathead screwdriver saying he 'needs to finish the trim before dark'. "
        "He swiped at my hand when I tried to take the screwdriver away. I don't know how to calm him down."
    )
    
    print("--- Sending Prompt to Gemini Locally ---")
    
    try:
        # Generate content with structured schemas forced on the engine
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Context: {patient_profile}\nInput: {sample_caregiver_input}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a clinical expert system in dementia caregiving. Analyze the user's input. "
                    "Prioritize de-escalation, behavioral redirection, and validating caregiver stress. "
                    "Never suggest medical prescriptions or changing drug dosages."
                ),
                # Enforce JSON formatting targeting our Pydantic class
                response_mime_type="application/json",
                response_schema=BehavioralAnalysis,
                temperature=0.1,
            ),
        )
        
        # Extract the native, fully validated Python object directly from the response
        result: BehavioralAnalysis = response.parsed
        
        print("\n--- Structured Gemini Output Received ---")
        print(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_local_gemini_test()