import requests
import logging
import time
import PyPDF2
import re
import pandas as pd
import streamlit as st
from azure.cosmos import CosmosClient
import os
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

api_key1 = os.getenv("API_KEY1")
endpoint1 = os.getenv("ENDPOINT1")
model1 = os.getenv("MODEL1")
api_version1 = os.getenv("API_VERSION1")

api_key2 = os.getenv("API_KEY2")
azure_endpoint2 = os.getenv("AZURE_ENDPOINT2")
model2 = os.getenv("MODEL2")
api_version2 = os.getenv("API_VERSION2")
# Azure OpenAI Configuration

retries = 3
initial_delay = 2  # Seconds

HEADERS1 = {"Content-Type": "application/json", "api-key": api_key1}
HEADERS2 = {"Content-Type": "application/json", "api-key": api_key2}

# Fetch Cosmos DB configuration from environment
cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
cosmos_key = os.getenv("COSMOS_KEY")
database_name = os.getenv("DATABASE_NAME")
container_name = os.getenv("CONTAINER_NAME")


# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdfReader = PyPDF2.PdfReader(pdf_file)
        numPages = len(pdfReader.pages)
        full_text = ""
        for page_num in range(numPages):
            page = pdfReader.pages[page_num]
            full_text += page.extract_text()
        logging.info("Extracted text from PDF successfully.")
        return full_text, numPages
    except Exception as e:
        logging.error(f"Error while extracting text from PDF: {e}")
        return "", 0


# Function to clean unnecessary words from extracted fields
def clean_data_field(field_value, unwanted_suffixes):
    try:
        if field_value:
            for suffix in unwanted_suffixes:
                if field_value.endswith(suffix):
                    field_value = field_value[: -len(suffix)].strip()
        return field_value
    except Exception as e:
        logging.error(f"Error in clean_data_field: {e}")
        return field_value


# Function to extract all details from text
def extract_all_details(text):
    try:
        logging.debug("Starting to extract details from text.")
        application_numbers = re.findall(r"Application#: (.+?)\s", text)
        goods_details = re.findall(
            r"Goods & Services translation\s*(.*?)(?=Register Mark|Applicant|Owner|Class|\Z)",
            text,
            re.DOTALL,
        )
        goods_details_cleaned = [
            re.sub(r"\s+", " ", gd).strip() for gd in goods_details
        ]
        owner_details = re.findall(r"Owner:\s*(.+?)\s*Owner Address:", text, re.DOTALL)
        applicant_details = re.findall(r"Applicant:\s*(.+?)\s*Owner:", text, re.DOTALL)
        applicant_details_cleaned = [
            re.sub(r"\s+", " ", applicant).strip() for applicant in applicant_details
        ]
        register_mark_details = re.findall(
            r"Register Mark:\s*(.+?)\s*Application#", text, re.DOTALL
        )
        class_details = re.findall(r"Nice Classes:\s*(\S+)", text)
        application_dates = re.findall(r"Application Date:\s*(\S+)", text)
        statuses = re.findall(
            r"Status:\s*(.+?)\s*Publication for Opposition:", text, re.DOTALL
        )
        statuses_cleaned = [re.sub(r"\s+", " ", status).strip() for status in statuses]
        publication_dates = re.findall(r"Publication for Opposition:\s*(\S+)", text)

        extracted_data = []
        max_length = max(
            len(application_numbers),
            len(goods_details_cleaned),
            len(owner_details),
            len(applicant_details_cleaned),
            len(register_mark_details),
            len(class_details),
            len(application_dates),
            len(statuses_cleaned),
            len(publication_dates),
        )
        for i in range(max_length):
            data = {
                "Application_Number": (
                    application_numbers[i] if i < len(application_numbers) else None
                ),
                "Goods & Services": (
                    goods_details_cleaned[i] if i < len(goods_details_cleaned) else None
                ),
                "Owner": owner_details[i] if i < len(owner_details) else None,
                "Applicant": (
                    applicant_details_cleaned[i]
                    if i < len(applicant_details_cleaned)
                    else None
                ),
                "Register Mark": (
                    register_mark_details[i] if i < len(register_mark_details) else None
                ),
                "Class": clean_data_field(
                    class_details[i] if i < len(class_details) else None,
                    ["Application#:"],
                ),
                "Application Date": (
                    application_dates[i] if i < len(application_dates) else None
                ),
                "Status": statuses_cleaned[i] if i < len(statuses_cleaned) else None,
                "Publication Date": clean_data_field(
                    publication_dates[i] if i < len(publication_dates) else None,
                    ["Applicant:"],
                ),
            }
            extracted_data.append(data)
        logging.info("Successfully extracted all details from text.")
        return extracted_data
    except Exception as e:
        logging.error(f"Error while extracting details: {e}")
        return []


# Function to upload nested JSON data to Cosmos DB
def upload_nested_to_cosmos(nested_data):
    try:
        client = CosmosClient(cosmos_endpoint, cosmos_key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        for record in nested_data:
            if isinstance(record, dict):
                container.upsert_item(record)
        logging.info("Nested JSON data successfully uploaded to Azure Cosmos DB.")
    except Exception as e:
        logging.error(f"Error while uploading nested data to Cosmos DB: {e}")


# Function to assess trademark conflict using Azure LLM
def assess_conflict_with_llm(record):
    if "9" in record["Class"].strip():  # Send to LLM 1 if class matches target
        azure_endpoint = azure_endpoint2
        model = model2
        headers = HEADERS2
        api_version = api_version2
    else:  # Send to LLM 2 otherwise
        azure_endpoint = endpoint1
        model = model1
        headers = HEADERS1
        api_version = api_version1
    url = f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in trademark conflict assessment.",
            },
            {
                "role": "user",
                "content": f"""
You are tasked to assess the conflict level between trademarks record and target trademark that is newly going to be filed based on the condition rules provided.

**Trademark Record**:
Trademark registered: SHIELD
Application Number: {record['Application_Number']}
Goods and Services: {record['Goods & Services']}
Owner: {record['Owner']}
Applicant: {record['Applicant']}
Register Mark: {record['Register Mark']}
Class: {record['Class']}
Application Date: {record['Application Date']}
Status: {record['Status']}
Publication Date: {record['Publication Date']}

Target trademark that is going to be filed:
Trademark registered: SHIELD
Application Number: 983489723
Goods and Services: Software security over cloud based applications
Owner: AWS
Applicant: AWS
Register Mark: SHIELD
Class: 9
Application Date: 11-11-2024
Status: Under creation
Publication Date: 12-12-2024

**Rules**:
    Condition 1: Trademark Name Comparison\n 
    - Condition 1A: The existing trademark's name is a character-for-character match with the proposed trademark name.\n 
    - Condition 1B: The existing trademark's name is semantically equivalent to the proposed trademark name.\n 
    - Condition 1C: The existing trademark's name is phonetically equivalent to the proposed trademark name.\n 
    - Condition 1D: If both the existing trademark's name and the proposed trademark name consist of multiple words, then the first two or more words of the existing trademark's name must be phonetically equivalent to the proposed trademark name.\n
    - Condition 1E: Primary Position Requirement- In the context of trademark conflicts, the primary position of a trademark refers to the first word or phrase element in a multi-word or phrase trademark. For a conflict to exist between an existing trademark and a proposed trademark based on Condition 1E, the proposed trademark name must be in the primary position of the existing trademark. This means that the proposed trademark name should be the first word of the existing trademark.\n
                    Example:\n Existing Trademark: "STORIES AND JOURNEYS"\n Proposed Trademark: "JOURNEY"\n Analysis:\n The existing trademark "STORIES AND JOURNEYS" consists of multiple words/phrases.\n For the proposed trademark "JOURNEY" to be in conflict under Condition 1E, it must appear as the first word/phrase in the existing trademark.\n In this case, the first word/phrase in "STORIES AND JOURNEYS" is "STORIES", not "JOURNEY".\n Therefore, "JOURNEY" does not meet Condition 1E because it is not in the primary position of the existing trademark.\n
                    Example:\n Existing Trademark: "JOURNEY BY COMPANION"\n Proposed Trademark: "JOURNEY"\n Analysis:\n The existing trademark "JOURNEY BY COMPANION" consists of multiple words/phrases.\n For the proposed trademark "JOURNEY" to be in conflict under Condition 1E, it must appear as the first word/phrase in the existing trademark.\n In this case, the first word/phrase in "JOURNEY BY COMPANION" is "JOURNEY".\n Therefore, "JOURNEY" does meet Condition 1E because it is in the primary position of the existing trademark.\n

    Condition 3: Target Market and Products\n 
    - Condition 3A: The existing trademark's goods/services target the exact same products as the proposed trademark.\n 
    - Condition 3B: The existing trademark's goods/services target an exact market as the proposed trademark.\n

    If existing trademark in user given input satisfies:\n\n
    - Special case: If existing Trademark Status is Cancelled or Abandoned, they will automatically be considered as conflict grade low but still give the reasoning for the potential conflicts.\n\n
    - If the existing trademark satisfies Condition 1A, 1B, 1C, or 1D, and also satisfies the revised Condition 1E (when applicable), along with Condition 2, and both Condition 3A (Needed to be fully satisfied) and 3B (Needed to be fully satisfied), then the conflict grade should be High.\n
    - If the existing trademark satisfies any two of the following: Condition 1A, 1B, 1C, or 1D (with the revised Condition 1E being a necessary component for these to be considered satisfied when applicable), Condition 2, Condition 3A (Needed to be fully satisfied) and 3B (Needed to be fully satisfied), then the conflict grade should be Moderate.\n
    - If the existing trademark satisfies only one (or none) of the conditions: Condition 1A, 1B, 1C, 1D and (only if the revised Condition 1E is also satisfied when applicable), Condition 2, Condition 3A and 3B, then the conflict grade should be Low.\n\n

Determine:
1. Conflict Level: Low, Moderate, or High.
2. Rule Codes: Give the codes of the rule conditions that are satisfied.
3. Reasoning: Provide a detailed reasoning for your assessment for why the condition you gave above is satisfied (Give reasoning for only those which are satisfied, don't speak about unsatisfied conditions).
                """,
            },
        ],
        "temperature": 0.0,
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No explanation provided.")
            )
            return content

        except requests.exceptions.Timeout as e:
            if attempt < retries - 1:
                wait_time = initial_delay * (2**attempt)
                logging.warning(
                    f"Timeout error. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Request failed after {retries} attempts due to timeout: {e}"
                )
                return f"Error: Request timed out after {retries} retries."

        except requests.exceptions.RequestException as e:
            logging.error(f"Error requesting trademark conflict assessment: {e}")
            return (
                "Error: Unable to fetch assessment due to network issues or API error."
            )

    return "Error: Max retries reached without success."

st.title("Trademark Data Extractor and Conflict Assessor")
st.write("Upload a PDF to extract details and assess trademark conflicts.")

# File uploader
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    # Store PDF directly in memory
    pdf_stream = BytesIO(pdf_file.getvalue())
    
    # Extract text and number of pages
    text, num_pages = extract_text_from_pdf(pdf_stream)
    st.write(f"Number of Pages: {num_pages}")

    if not text:
        st.error("Failed to extract text from the uploaded PDF. Please check the file.")
    else:
        # Store the extracted text in a variable for further use
        extracted_text = text  # Store text for use in further processing

        # Extract trademark details
        extracted_data = extract_all_details(extracted_text)

        st.subheader("Extracted Details:")
        if extracted_data:
            st.dataframe(pd.DataFrame(extracted_data))

            # Button to assess conflicts
            if st.button("Assess Conflict with Azure LLM"):
                results = assess_conflict_parallel(extracted_data)
                st.write(f"Total API calls made: {api_call_counter}")
                for record in extracted_data:
                    app_num = record["Application_Number"]
                    st.subheader(f"Assessment for Application {app_num}")
                    st.markdown(results.get(app_num, "No result available"))
        else:
            st.write("No details found in the uploaded document.")
