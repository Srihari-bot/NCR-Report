


import streamlit as st
import requests
import json
import urllib.parse
import urllib3
import certifi
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os
from dotenv import load_dotenv
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# WatsonX configuration
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")  # Load API key from .env

# Check environment variables
if not API_KEY or not WATSONX_API_URL:
    st.error("❌ Required environment variables (API_KEY or WATSONX_API_URL) missing!")
    logger.error("Missing API_KEY or WATSONX_API_URL")
    st.stop()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Function to generate access token
def get_access_token(api_key):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            token_info = response.json()
            logger.info("Access token generated successfully")
            return token_info['access_token']
        else:
            logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
            st.error(f"❌ Failed to get access token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

# Login Function
def login_to_asite(email, password):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
    if response.status_code == 200:
        try:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful, Session ID: {session_id}")
            return session_id
        except json.JSONDecodeError:
            logger.error("JSONDecodeError during login")
            st.error("❌ Failed to parse login response")
            return None
    logger.error(f"Login failed: {response.status_code}")
    st.error(f"❌ Login failed: {response.status_code}")
    return None

# Fetch Data Function
def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": f"ASessionID={session_id}",
    }
    all_data = []
    start_record = 1
    total_records = None

    with st.spinner("Fetching data from Asite..."):
        while True:
            search_criteria = {
                "criteria": [
                    {"field": "ProjectName", "operator": 1, "values": [project_name]},
                    {"field": "FormName", "operator": 1, "values": [form_name]}
                ],
                "recordStart": start_record,
                "recordLimit": record_limit
            }
            search_criteria_str = json.dumps(search_criteria)
            encoded_payload = f"searchCriteria={urllib.parse.quote(search_criteria_str)}"
            response = requests.post(SEARCH_URL, headers=headers, data=encoded_payload, verify=certifi.where(), timeout=50)

            try:
                response_json = response.json()
                if total_records is None:
                    total_records = response_json.get("responseHeader", {}).get("results-total", 0)
                all_data.extend(response_json.get("FormList", {}).get("Form", []))
                st.info(f"🔄 Fetched {len(all_data)} / {total_records} records")
                if start_record + record_limit - 1 >= total_records:
                    break
                start_record += record_limit
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                st.error(f"❌ Error fetching data: {str(e)}")
                break

    return {
        "responseHeader": {"results": len(all_data), "total_results": total_records}
    }, all_data, encoded_payload

# Process JSON Data
def process_json_data(json_data):
    data = []
    for item in json_data:
        form_details = item.get('FormDetails', {})
        close_due = form_details.get('FormCreationDate', None)
        update_date = form_details.get('UpdateDate', None)
        form_status = form_details.get('FormStatus', None)
        
        discipline = None
        description = None
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', [])
        for field in custom_fields:
            if field.get('FieldName') == 'CFID_DD_DISC':
                discipline = field.get('FieldValue', None)
            elif field.get('FieldName') == 'CFID_RTA_DES':
                description = BeautifulSoup(field.get('FieldValue', None) or '', "html.parser").get_text()

        days_diff = None
        if update_date and close_due:
            try:
                update_date_obj = datetime.strptime(update_date.split('#')[0], "%d-%b-%Y")
                close_due_obj = datetime.strptime(close_due.split('#')[0], "%d-%b-%Y")
                days_diff = abs((close_due_obj - update_date_obj).days)  # Use absolute value
            except:
                days_diff = None

        data.append([days_diff, update_date, close_due, description, form_status, discipline])

    df = pd.DataFrame(data, columns=['Days', 'UpdateDate', 'CloseDue', 'Description', 'Status', 'Discipline'])
    df['UpdateDate'] = pd.to_datetime(df['UpdateDate'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    df['CloseDue'] = pd.to_datetime(df['CloseDue'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    return df

# Generate NCR Report

def generate_ncr_report(df, report_type, start_date=None, end_date=None):
    with st.spinner(f"Generating {report_type} NCR Report..."):
        if report_type == "Closed":
            filtered_df = df[
                (df['Status'] == 'Closed') &
                (df['UpdateDate'] >= pd.to_datetime(start_date)) &
                (df['UpdateDate'] <= pd.to_datetime(end_date)) &
                (abs(df['Days']) > 21)  # Filter for absolute Days > 21
            ]
        else:  # Open
            # Filter for records that are open as of the end_date (Open Until Date)
            filtered_df = df[
                (df['Status'] == 'Open') &
                (df['UpdateDate'] <= pd.to_datetime(end_date))  # Ensure the record was opened on or before the Open Until Date
            ]
            # Calculate how many days the record has been open (from UpdateDate to end_date)
            filtered_df['DaysOpen'] = (pd.to_datetime(end_date) - filtered_df['UpdateDate']).dt.days
            # Filter for records that have been open for more than 21 days
            filtered_df = filtered_df[filtered_df['DaysOpen'] > 21]

        if filtered_df.empty:
            return {"error": f"No {report_type} records found"}, ""

        # Convert Timestamp columns to strings before creating dictionary
        filtered_df['UpdateDate'] = filtered_df['UpdateDate'].astype(str) if 'UpdateDate' in filtered_df else filtered_df['UpdateDate']
        filtered_df['CloseDue'] = filtered_df['CloseDue'].astype(str) if 'CloseDue' in filtered_df else filtered_df['CloseDue']

        processed_data = filtered_df.to_dict(orient="records")
        
        # Loop through the data to check and format it
        cleaned_data = []
        for record in processed_data:
            # Ensure required fields exist and handle missing or malformed data
            cleaned_record = {
                "Description": str(record.get("Description", "")),
                "Discipline": str(record.get("Discipline", "")),
                "UpdateDate": str(record.get("UpdateDate", "")),
                "CloseDue": str(record.get("CloseDue", "")),
                "Status": str(record.get("Status", ""))
            }

            # Extract Tower from Description and normalize it
            description = cleaned_record["Description"]
            if description:
                tower_match = re.search(r"(Tower|T)\s*-?\s*(\d+)", description, re.IGNORECASE)
                if tower_match:
                    tower_prefix = "Tower"  # Standardize prefix
                    tower_number = tower_match.group(2)  # Extract the number
                    # Pad the number with leading zeros to make it two digits (e.g., "01", "07")
                    tower_number_padded = tower_number.zfill(2)
                    cleaned_record["Tower"] = f"{tower_prefix}-{tower_number_padded}"
                else:
                    cleaned_record["Tower"] = "External Development"  # Replace Unknown with External Development
            else:
                cleaned_record["Tower"] = "External Development"  # Replace Unknown with External Development

            # Categorize discipline into SW, FW, or MEP
            discipline = cleaned_record["Discipline"].strip().lower()
            if "structure" in discipline or "sw" in discipline:
                cleaned_record["Discipline_Category"] = "SW"
            elif "civil" in discipline or "finishing" in discipline or "fw" in discipline:
                cleaned_record["Discipline_Category"] = "FW"
            else:
                cleaned_record["Discipline_Category"] = "MEP"

            cleaned_data.append(cleaned_record)

        # Debug: Log the total number of records being processed
        st.write(f"Debug - Total {report_type} records to process: {len(cleaned_data)}")

        # Get access token
        access_token = get_access_token(API_KEY)
        if not access_token:
            return {"error": "Failed to obtain access token"}, ""

        # WatsonX has token limits, so we'll process the data in chunks if necessary
        chunk_size = 200  # Adjust this based on WatsonX token limits (experiment to find the optimal size)
        all_results = {
            report_type: {
                "Sites": {},
                "Grand_Total": 0
            }
        }

        for i in range(0, len(cleaned_data), chunk_size):
            chunk = cleaned_data[i:i + chunk_size]
            st.write(f"Processing chunk {i // chunk_size + 1}: Records {i} to {min(i + chunk_size, len(cleaned_data))}")

            # Debug: Log the chunk data to inspect Tower values
            logger.debug(f"Chunk data: {json.dumps(chunk, indent=2)}")

            # Create the prompt for this chunk
            prompt = (
                "IMPORTANT: YOU MUST RETURN ONLY VALID JSON. DO NOT INCLUDE ANY TEXT, EXPLANATION, OR NOTES.\n\n"
                f"Task: For {report_type} NCRs, count records by site ('Tower' field) and discipline category ('Discipline_Category' field).\n"
                f"Condition for {report_type} NCRs: {'absolute value of Days > 21' if report_type == 'Closed' else 'open more than 21 days as of the specified end_date'}.\n"
                "Use exact 'Tower' values from data; keep 'External Development' or 'Veridia-Commercial' as is for non-matching cases.\n\n"
                "REQUIRED OUTPUT FORMAT (exactly this structure with actual counts):\n"
                "{\n"
                f'  "{report_type}": {{\n'
                '    "Sites": {\n'
                '      "Site_Name1": {\n'
                '        "SW": number,\n'
                '        "FW": number,\n'
                '        "MEP": number,\n'
                '        "Total": number\n'
                '      },\n'
                '      "Site_Name2": {\n'
                '        "SW": number,\n'
                '        "FW": number,\n'
                '        "MEP": number,\n'
                '        "Total": number\n'
                '      }\n'
                '    },\n'
                '    "Grand_Total": number\n'
                '  }\n'
                '}\n\n'
                f"Data: {json.dumps(chunk)}\n"  # Send the current chunk of data
                f"Return the result strictly as a JSON object—no code, no explanations, no string literal like this ```, only the JSON."
            )

            payload = {
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 8100,
                    "min_new_tokens": 0,
                    "temperature": 0.01
                },
                "model_id": MODEL_ID,
                "project_id": PROJECT_ID
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }

            try:
                response = requests.post(WATSONX_API_URL, headers=headers, json=payload, timeout=500)
                logger.info(f"WatsonX API response status code: {response.status_code}")
                st.write(f"Debug - Response status code: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("results", [{}])[0].get("generated_text", "").strip()
                    st.write(f"Debug - Raw response: {generated_text}")

                    if generated_text:
                        # Extract JSON using regex to handle any potential non-JSON content
                        json_match = re.search(r'({[\s\S]*})', generated_text)

                        if json_match:
                            json_str = json_match.group(1)
                            try:
                                parsed_json = json.loads(json_str)
                                # Merge the results from this chunk into the overall results
                                chunk_result = parsed_json.get(report_type, {})
                                chunk_sites = chunk_result.get("Sites", {})
                                chunk_grand_total = chunk_result.get("Grand_Total", 0)

                                # Merge Sites data
                                for site, counts in chunk_sites.items():
                                    if site not in all_results[report_type]["Sites"]:
                                        all_results[report_type]["Sites"][site] = {
                                            "SW": 0,
                                            "FW": 0,
                                            "MEP": 0,
                                            "Total": 0
                                        }
                                    all_results[report_type]["Sites"][site]["SW"] += counts.get("SW", 0)
                                    all_results[report_type]["Sites"][site]["FW"] += counts.get("FW", 0)
                                    all_results[report_type]["Sites"][site]["MEP"] += counts.get("MEP", 0)
                                    all_results[report_type]["Sites"][site]["Total"] += counts.get("Total", 0)

                                # Add to Grand Total
                                all_results[report_type]["Grand_Total"] += chunk_grand_total

                            except json.JSONDecodeError as e:
                                logger.error(f"JSONDecodeError: {str(e)} - Raw response: {generated_text}")
                                st.error(f"❌ Failed to parse JSON: {str(e)}")
                                default_json = {report_type: {"Sites": {}, "Grand_Total": 0}}
                                return default_json, generated_text
                        else:
                            logger.error("No JSON found in response")
                            st.error("❌ No JSON found in response")
                            default_json = {report_type: {"Sites": {}, "Grand_Total": 0}}
                            return default_json, generated_text
                    else:
                        logger.error("Empty generated_text from WatsonX")
                        st.error("❌ Empty response from WatsonX")
                        return {"error": "Empty response from WatsonX"}, ""
                else:
                    error_msg = f"❌ WatsonX API error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    return {"error": error_msg}, response.text
            except Exception as e:
                error_msg = f"❌ Exception during WatsonX call: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                return {"error": error_msg}, ""

        # Return the aggregated results
        return all_results, json.dumps(all_results)



def generate_consolidated_ncr_excel(combined_result, report_title="NCR"):
    # Create a new Excel writer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        title_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow',
            'border': 1,
            'font_size': 12
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'text_wrap': True
        })
        
        subheader_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        site_format = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter',
            'border': 1
        })
        
        # Create worksheet
        worksheet = workbook.add_worksheet('NCR Report')
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # Site column
        worksheet.set_column('B:H', 12)  # Data columns
        
        # Get all unique sites from both sections
        all_sites = set()
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})
        
        if not isinstance(resolved_data, dict) or "error" in resolved_data:
            resolved_data = {"Sites": {}}
        if not isinstance(open_data, dict) or "error" in open_data:
            open_data = {"Sites": {}}
            
        resolved_sites = resolved_data.get("Sites", {})
        open_sites = open_data.get("Sites", {})
        
        all_sites.update(resolved_sites.keys())
        all_sites.update(open_sites.keys())
        
        # Normalize site names (some might be Tower-X, others T-X)
        # Inside generate_consolidated_ncr_excel, update the normalization loop
        normalized_sites = {}
        for site in all_sites:
            match = re.search(r'(?:tower|t)[- ]?(\d+)', site, re.IGNORECASE)
            if match:
                num = match.group(1)
                normalized_name = f"Veridia- Tower {num.zfill(2)}"  # Consistent with "Tower-XX" format
                normalized_sites[site] = normalized_name
            else:
                normalized_sites[site] = site

        # Add standard sites with the same format
        standard_sites = [
            "Veridia-Club",
            "Veridia- Tower 01",
            "Veridia- Tower 02",
            "Veridia- Tower 03",
            "Veridia- Tower 04",
            "Veridia- Tower 05",
            "Veridia- Tower 06",
            "Veridia- Tower 07",
            "Veridia-Commercial",
            "External Development"
        ]
        for site in standard_sites:
            if site not in normalized_sites.values():
                normalized_sites[site] = site
                
        # Sort the normalized sites
        sorted_sites = sorted(list(set(normalized_sites.values())))
        
        # Title row
        worksheet.merge_range('A1:H1', report_title, title_format)
        
        # Header row
        row = 1
        worksheet.write(row, 0, 'Site', header_format)
        worksheet.merge_range(row, 1, row, 3, 'NCR resolved beyond 21 days', header_format)
        worksheet.merge_range(row, 4, row, 6, 'NCR open beyond 21 days', header_format)
        worksheet.write(row, 7, 'Total', header_format)
        
        # Subheaders
        row = 2
        categories = ['Civil Finishing', 'MEP', 'Structure']
        worksheet.write(row, 0, '', header_format)
        
        # Resolved subheaders
        for i, cat in enumerate(categories):
            worksheet.write(row, i+1, cat, subheader_format)
            
        # Open subheaders
        for i, cat in enumerate(categories):
            worksheet.write(row, i+4, cat, subheader_format)
            
        worksheet.write(row, 7, '', header_format)
        
        # Map our categories to the JSON data categories
        category_map = {
            'Civil Finishing': 'FW',
            'MEP': 'MEP',
            'Structure': 'SW'
        }
        
        # Data rows
        row = 3
        site_totals = {}
        
        for site in sorted_sites:
            worksheet.write(row, 0, site, site_format)
            
            # Find original site key
            original_resolved_key = None
            original_open_key = None
            
            for orig_site, norm_site in normalized_sites.items():
                if norm_site == site:
                    # Check if this original key exists in either dataset
                    if orig_site in resolved_sites:
                        original_resolved_key = orig_site
                    if orig_site in open_sites:
                        original_open_key = orig_site
            
            site_total = 0
            
            # Resolved data
            for i, (display_cat, json_cat) in enumerate(category_map.items()):
                value = 0
                if original_resolved_key:
                    value = resolved_sites.get(original_resolved_key, {}).get(json_cat, 0)
                worksheet.write(row, i+1, value, cell_format)
                site_total += value
                
            # Open data
            for i, (display_cat, json_cat) in enumerate(category_map.items()):
                value = 0
                if original_open_key:
                    value = open_sites.get(original_open_key, {}).get(json_cat, 0)
                worksheet.write(row, i+4, value, cell_format)
                site_total += value
                
            # Total for this site
            worksheet.write(row, 7, site_total, cell_format)
            site_totals[site] = site_total
            row += 1
        
        # Return the Excel file
        output.seek(0)
        return output

# Streamlit UI
st.title("Asite NCR Reporter")

# Initialize session state
if "df" not in st.session_state:
    st.session_state["df"] = None

# Login Section
st.sidebar.title("🔒 Asite Login")
email = st.sidebar.text_input("Email", "impwatson@gadieltechnologies.com")
password = st.sidebar.text_input("Password", "Srihari@790$", type="password")
if st.sidebar.button("Login"):
    session_id = login_to_asite(email, password)
    if session_id:
        st.session_state["session_id"] = session_id
        st.sidebar.success("✅ Login Successful")

# Data Fetch Section
st.sidebar.title("📂 Project Data")
project_name = st.sidebar.text_input("Project Name", "Wave Oakwood, Wave City")
form_name = st.sidebar.text_input("Form Name", "Non Conformity Report")
if "session_id" in st.session_state and st.sidebar.button("Fetch Data"):
    header, data, payload = fetch_project_data(st.session_state["session_id"], project_name, form_name)
    st.json(header)
    if data:
        df = process_json_data(data)
        st.session_state["df"] = df  # Store DataFrame in session state
        st.dataframe(df.head(50))
        st.success("✅ Data fetched and processed successfully!")

# Report Generation Section
if st.session_state["df"] is not None:
    df = st.session_state["df"]
    
    # Combined NCR Report
    st.sidebar.title("📋 Combined NCR Report")
    closed_start = st.sidebar.date_input("Closed Start Date", df['UpdateDate'].min())
    closed_end = st.sidebar.date_input("Closed End Date", df['UpdateDate'].max(), key="closed_end")
    open_end = st.sidebar.date_input("Open Until Date", df['UpdateDate'].max(), key="open_end")
    
    # Fixed by adding a unique key to this button
    if st.sidebar.button("Generate Combined NCR Report", key="generate_report_button"):
        # Get the month name for the title - using the closed end date
        month_name = closed_end.strftime("%B")
        report_title = f"NCR: {month_name}"
        
        # Generate Closed Report
        closed_result, closed_raw = generate_ncr_report(df, "Closed", closed_start, closed_end)
        # Generate Open Report
        open_result, open_raw = generate_ncr_report(df, "Open", end_date=open_end)

        # Combine the results
        combined_result = {}
        if "error" not in closed_result:
            combined_result["NCR resolved beyond 21 days"] = closed_result["Closed"]
        else:
            combined_result["NCR resolved beyond 21 days"] = {"error": closed_result["error"]}
        
        if "error" not in open_result:
            combined_result["NCR open beyond 21 days"] = open_result["Open"]
        else:
            combined_result["NCR open beyond 21 days"] = {"error": open_result["error"]}

        # Display combined result as JSON
        st.subheader("Combined NCR Report (JSON)")
        st.json(combined_result)
        
        # Generate and offer Excel download
        excel_file = generate_consolidated_ncr_excel(combined_result, report_title)
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_file,
            file_name=f"NCR_Report_{month_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


