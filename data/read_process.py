import re
import pandas as pd
import os

# Function to extract content from the provided text
def extract_content(text):
    """
    Extracts <think1>, <query>, <think2>, and <summary> sections from the input text.
    :param text: Input text containing <think1>, <query>, <think2>, and <summary>.
    :return: A dictionary containing the extracted sections.
    """
    # Define regex patterns for each section
    think1_pattern = r"<think1>(.*?)</think1>"
    query_pattern = r"<query>(.*?)</query>"
    think2_pattern = r"<think2>(.*?)</think2>"
    summary_pattern = r"<summary>(.*?)</summary>"
    
    # Extract content using regex
    think1_match = re.search(think1_pattern, text, re.DOTALL)
    query_match = re.search(query_pattern, text, re.DOTALL)
    think2_match = re.search(think2_pattern, text, re.DOTALL)
    summary_match = re.search(summary_pattern, text, re.DOTALL)
    
    # Store extracted content in a dictionary
    extracted_data = {
        "think1": think1_match.group(1).strip() if think1_match else None,
        "query": query_match.group(1).strip() if query_match else None,
        "think2": think2_match.group(1).strip() if think2_match else None,
        "summary": summary_match.group(1).strip() if summary_match else None
    }
    
    return extracted_data

# Function to save or append data to an Excel file
def save_or_append_to_excel(file_path, new_data):
    """
    Save or append data to an Excel file.
    :param file_path: Path to the Excel file.
    :param new_data: List of dictionaries containing new data to be appended.
    """
    # Check if the file already exists
    if os.path.exists(file_path):
        # Read existing data from the file
        existing_df = pd.read_excel(file_path)
        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        # Append new data to existing data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        combined_df = pd.DataFrame(new_data)
    
    # Save the combined DataFrame to the Excel file
    combined_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"Data saved/updated in {file_path}")

# Example usage
if __name__ == "__main__":
    # Simulated list of input texts (you can loop through multiple inputs)
    input_texts = [
        """
        <think1>Source Image Description 1: The dress features a vibrant peacock feather print with shades of blue and green on a black background.</think1>
        <query>Transformation Instruction 1: Convert the dress to a solid black color scheme.</query>
        <think2>Target Image Description 1: The transformed dress is sleek and all-black.</think2>
        <summary>Summary 1: The transformation involved systematic changes.</summary>
        """,
        """
        <think1>Source Image Description 2: The dress has bell sleeves and a round neckline.</think1>
        <query>Transformation Instruction 2: Remove sleeves in favor of slim shoulder straps.</query>
        <think2>Target Image Description 2: The dress now has delicate shoulder straps.</think2>
        <summary>Summary 2: Structural changes enhanced modern appeal.</summary>
        """
    ]
    
    # File path for the Excel file
    excel_file_path = "extracted_data.xlsx"
    
    # Loop through input texts and extract content
    for input_text in input_texts:
        extracted_data = extract_content(input_text)
        # Append the extracted data to the Excel file
        save_or_append_to_excel(excel_file_path, [extracted_data])