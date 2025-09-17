# Standard Library
import os
import json
import uuid 
from urllib.parse import quote_plus, urlencode
from datetime import datetime, timezone 

# Flask & Extensions
from flask import Flask, request, jsonify, redirect, url_for, session, send_from_directory, abort
from flask_cors import CORS
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename 

# Environment
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 

# Azure SDK
from azure.cosmos import CosmosClient, PartitionKey, exceptions as CosmosExceptions 
from azure.storage.blob import BlobServiceClient, ContentSettings 
from azure.core.exceptions import ResourceExistsError as BlobResourceExistsError

# HTTP Requests Library
import requests # Import the requests library

# --- Application Configuration ---
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
if not APP_SECRET_KEY:
    raise ValueError("No APP_SECRET_KEY set for Flask application.")

# PromptFlow Endpoint URL and API Key (using your .env names)
PROMPTFLOW_ENDPOINT_URL = os.getenv("URL") 
PROMPTFLOW_API_KEY = os.getenv("API_KEY")
if not PROMPTFLOW_ENDPOINT_URL or not PROMPTFLOW_API_KEY:
    print("Warning: PROMPTFLOW_ENDPOINT_URL or PROMPTFLOW_API_KEY is not set in .env. AI query will use placeholder.")

#--- Azure Storage Configuration ---
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
CASES_CONTAINER_NAME = os.getenv("CASES_CONTAINER_NAME") 
if not all([COSMOS_ENDPOINT, COSMOS_KEY, DATABASE_NAME, CASES_CONTAINER_NAME]):
    raise ValueError("One or more critical environment variables (CosmosDB or Blob Storage) are missing.")

#-- Azure Blob Storage Configuration ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CASE_DOCS_CONTAINER_NAME = os.getenv("AZURE_CASE_DOCS_CONTAINER_NAME")
AZURE_CHAT_DOCS_CONTAINER_NAME = os.getenv("AZURE_CHAT_DOCS_CONTAINER_NAME")
if not all([AZURE_CONNECTION_STRING, AZURE_CASE_DOCS_CONTAINER_NAME, AZURE_CHAT_DOCS_CONTAINER_NAME]):
    raise ValueError("One or more critical environment variables (...) are missing.")

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static')
app.secret_key = APP_SECRET_KEY
CORS(app) 

# --- Initialize Cosmos DB Client ---
cosmos_client = None
database_client = None
cases_container_client = None
try:
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database_client = cosmos_client.get_database_client(DATABASE_NAME)
    cases_container_client = database_client.get_container_client(CASES_CONTAINER_NAME)
    print(f"Successfully connected to Cosmos DB Cases container: {DATABASE_NAME}/{CASES_CONTAINER_NAME}")
except Exception as e:
    print(f"CRITICAL: Error initializing Cosmos DB client: {e}. Cases functionality will be impaired.")
    # Depending on your app's needs, you might want to exit or handle this more gracefully.


# --- Initialize Azure Blob Service Client ---
blob_service_client = None
case_docs_blob_container_client = None
chat_docs_blob_container_client = None

def initialize_blob_container(bs_client, container_name_env_var, container_description):
    container_name = os.getenv(container_name_env_var)
    if not container_name:
        print(f"Warning: {container_name_env_var} is not set. {container_description} functionality will be disabled.")
        return None
    try:
        container_client = bs_client.get_container_client(container_name)
        container_client.create_container()
        print(f"Blob container '{container_name}' for {container_description} created or already exists.")
        return container_client
    except BlobResourceExistsError:
        print(f"Blob container '{container_name}' for {container_description} already exists.")
        return container_client # Already exists, so return the client
    except Exception as e_create_container:
        print(f"Could not create/verify blob container '{container_name}' for {container_description}: {e_create_container}")
        return None

try:
    if AZURE_CONNECTION_STRING: # Check if connection string exists
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        case_docs_blob_container_client = initialize_blob_container(blob_service_client, "AZURE_CASE_DOCS_CONTAINER_NAME", "case documents")
        chat_docs_blob_container_client = initialize_blob_container(blob_service_client, "AZURE_CHAT_DOCS_CONTAINER_NAME", "chat documents")
    else:
        print("Warning: AZURE_CONNECTION_STRING is not set. Blob storage functionality will be disabled.")

except Exception as e:
    print(f"Error initializing Azure Blob Service client or its containers: {e}")
    # blob_service_client will remain None if the initial connection fails
    # case_docs_blob_container_client and chat_docs_blob_container_client will also remain None

# --- Auth0 OAuth Setup ---
oauth = OAuth(app)
oauth.register(
    "auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration'
)

# --- Authentication Routes (/login, /callback, /logout) ---
@app.route("/login")
def login():
    redirect_uri_for_auth0 = url_for("callback", _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri=redirect_uri_for_auth0)

@app.route("/callback", methods=["GET", "POST"])
def callback():
    try:
        token = oauth.auth0.authorize_access_token()
        session["user"] = token 
        userinfo = token.get("userinfo")
        if userinfo:
            print(f"User logged in: {userinfo.get('name')} ({userinfo.get('sub')})")
    except Exception as e:
        print(f"Error during Auth0 callback: {e}")
        return redirect(url_for("login")) 
    vue_app_url = "http://localhost:5173" 
    next_url_path_from_session = session.pop('next_url', None) 
    final_redirect_url = vue_app_url 
    if next_url_path_from_session:
        if next_url_path_from_session.startswith('/'): final_redirect_url = f"{vue_app_url}{next_url_path_from_session}"
        else: final_redirect_url = f"{vue_app_url}/{next_url_path_from_session}"
    return redirect(final_redirect_url)

@app.route("/logout")
def logout():
    session.clear() 
    vue_app_url = "http://localhost:5173" 
    return redirect(
        f"https://{AUTH0_DOMAIN}/v2/logout?" +
        urlencode({"returnTo": vue_app_url, "client_id": AUTH0_CLIENT_ID,}, quote_via=quote_plus,)
    )

# --- API Endpoints ---
@app.route("/api/chat/stage_attachment", methods=["POST"])
def stage_chat_attachment(): # Endpoint to stage a chat attachment 
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user_session_data["userinfo"]["sub"]

    if not chat_docs_blob_container_client:
        print("[API StageChatAttachment] Chat documents blob service not available.")
        return jsonify({"error": "Chat document storage service not available"}), 503

    if 'document' not in request.files: # Expecting a single file with the key 'document'
        return jsonify({"error": "No file part in the request"}), 400

    file_to_upload = request.files['document']

    if file_to_upload.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file_to_upload:
        original_filename = secure_filename(file_to_upload.filename)
        # Create a unique blob name for chat attachments, perhaps in a user-specific staging area
        # This path can be refined later, e.g., to include a session ID or case ID if relevant at staging time
        blob_name = f"{user_id}/chat_staging/{str(uuid.uuid4())}-{original_filename}"
        
        print(f"[API StageChatAttachment] Processing file: {original_filename} for blob: {blob_name}")
        blob_client_instance = chat_docs_blob_container_client.get_blob_client(blob_name)
            
        try:
            file_to_upload.seek(0) 
            blob_content_settings = ContentSettings(content_type=file_to_upload.mimetype)
            blob_client_instance.upload_blob(
                file_to_upload.read(), 
                overwrite=True, # Or generate truly unique names to prevent overwrite
                content_settings=blob_content_settings
            )
            print(f"[API StageChatAttachment] Successfully uploaded '{original_filename}' to blob: {blob_name}")

            file_to_upload.seek(0, os.SEEK_END)
            file_size_bytes = file_to_upload.tell()
            
            # This metadata will be sent back to the frontend to be stored temporarily
            # until the user sends the actual chat message.
            staged_doc_metadata = {
                "documentId": str(uuid.uuid4()), # A unique ID for this staged metadata instance
                "fileName": original_filename,
                "blobName": blob_name,
                "blobContainer": AZURE_CHAT_DOCS_CONTAINER_NAME,
                "url": blob_client_instance.url,
                "fileType": file_to_upload.mimetype,
                "sizeBytes": file_size_bytes,
                "uploadedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "uploaderUserId": user_id,
                "status": "staged" # Indicate it's staged, not yet part of a message
            }
            
            return jsonify({
                "message": f"File '{original_filename}' staged successfully.",
                "stagedDocument": staged_doc_metadata 
            }), 200

        except Exception as e:
            print(f"[API StageChatAttachment] Error uploading file '{original_filename}' to blob: {str(e)}")
            return jsonify({"error": f"Failed to upload file '{original_filename}'.", "details": str(e)}), 500
    
    return jsonify({"error": "Unknown error during file staging."}), 500

@app.route("/api/me", methods=["GET"])
def get_current_user_profile():
    user_session_data = session.get("user")
    if user_session_data and "userinfo" in user_session_data:
        return jsonify(user_session_data.get("userinfo")), 200
    else:
        return jsonify({"error": "User not authenticated"}), 401

@app.route("/api/user/cases", methods=["GET"])
def get_user_cases():
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    if not cases_container_client: 
        return jsonify({"error": "Database service not available"}), 503
    user_id = user_session_data["userinfo"]["sub"]
    query = "SELECT * FROM c WHERE c.userId = @userId AND c.type = 'case'" 
    parameters = [{"name": "@userId", "value": user_id}]
    user_cases = list(cases_container_client.query_items(query=query, parameters=parameters, partition_key=user_id ))
    return jsonify(user_cases), 200

# --- Document Upload Endpoint for Cases ---
@app.route("/api/cases/<string:case_id>/documents/upload", methods=["POST"])
def upload_case_document_to_case(case_id):
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user_session_data["userinfo"]["sub"]

    if not cases_container_client or not case_docs_blob_container_client:
        return jsonify({"error": "Backend storage service not available"}), 503

    uploaded_files = request.files.getlist("documents") 
    if not uploaded_files or not uploaded_files[0].filename : 
        return jsonify({"error": "No files selected for upload"}), 400

    try:
        case_doc = cases_container_client.read_item(item=case_id, partition_key=user_id)
    except CosmosExceptions.CosmosResourceNotFoundError: # Use aliased exception
        return jsonify({"error": f"Case {case_id} not found or access denied."}), 404
    except Exception as e:
        print(f"Error reading case {case_id}: {str(e)}")
        return jsonify({"error": "Could not retrieve case details"}), 500

    if "caseDocuments" not in case_doc:
        case_doc["caseDocuments"] = []

    newly_uploaded_metadata_list = []

    for file_storage_item in uploaded_files:
        if file_storage_item and file_storage_item.filename:
            original_filename = secure_filename(file_storage_item.filename)
            blob_name = f"{case_id}/{original_filename}" 
            blob_client_instance = case_docs_blob_container_client.get_blob_client(blob_name)
            
            try:
                file_storage_item.seek(0) 
                # MODIFIED: Use ContentSettings object
                blob_content_settings = ContentSettings(content_type=file_storage_item.mimetype)
                blob_client_instance.upload_blob(
                    file_storage_item.read(), 
                    overwrite=True, 
                    content_settings=blob_content_settings # Pass ContentSettings object
                )
                print(f"Uploaded '{original_filename}' to Azure Blob Storage as '{blob_name}'")

                file_storage_item.seek(0, os.SEEK_END)
                file_size_bytes = file_storage_item.tell()
                
                doc_metadata = {
                    "documentId": str(uuid.uuid4()),
                    "fileName": original_filename,
                    "blobName": blob_name,
                    "blobContainer": AZURE_CASE_DOCS_CONTAINER_NAME,
                    "url": blob_client_instance.url,
                    "fileType": file_storage_item.mimetype,
                    "sizeBytes": file_size_bytes,
                    "uploadedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "uploaderUserId": user_id
                }
                case_doc["caseDocuments"].append(doc_metadata)
                newly_uploaded_metadata_list.append(doc_metadata)

            except Exception as e:
                print(f"Error uploading file '{original_filename}' to blob: {str(e)}")
                continue 
    
    if not newly_uploaded_metadata_list: 
        return jsonify({"error": "No files were successfully processed for upload."}), 400

    try:
        case_doc["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        cases_container_client.replace_item(item=case_id, body=case_doc) # item should be case_id (document id)
        print(f"Updated case '{case_id}' in Cosmos DB with new document metadata.")
        return jsonify({
            "message": f"{len(newly_uploaded_metadata_list)} file(s) uploaded successfully to case {case_id}.",
            "uploadedDocuments": newly_uploaded_metadata_list 
        }), 200
    except Exception as e:
        print(f"Error updating case '{case_id}' in Cosmos DB: {str(e)}")
        return jsonify({"error": "Failed to update case metadata in database after file uploads."}), 500

# --- Document Deletion Endpoint for Case Documents ---
@app.route("/api/cases/documents/delete", methods=["POST"])
def delete_case_document_route():
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user_session_data["userinfo"]["sub"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    case_id = data.get("caseId")
    # This 'documentId' is the unique ID of the document metadata object within the case's caseDocuments array
    document_metadata_id_to_delete = data.get("documentId") 

    if not case_id or not document_metadata_id_to_delete:
        return jsonify({"error": "Missing caseId or documentId in request"}), 400

    print(f"[API DeleteCaseDocument] User: {user_id} attempting to delete document with metadata ID: {document_metadata_id_to_delete} from case: {case_id}")

    if not cases_container_client or not case_docs_blob_container_client:
        return jsonify({"error": "Backend storage or database service not available"}), 503

    try:
        case_doc = cases_container_client.read_item(item=case_id, partition_key=user_id)
    except CosmosExceptions.CosmosResourceNotFoundError:
        return jsonify({"error": f"Case {case_id} not found or access denied."}), 404
    except Exception as e:
        print(f"Error reading case {case_id} for document deletion: {str(e)}")
        return jsonify({"error": "Could not retrieve case details"}), 500

    doc_to_delete_metadata = None
    original_case_documents = case_doc.get("caseDocuments", [])
    updated_case_documents = []

    for doc_meta in original_case_documents:
        if doc_meta.get("documentId") == document_metadata_id_to_delete:
            doc_to_delete_metadata = doc_meta
        else:
            updated_case_documents.append(doc_meta)

    if not doc_to_delete_metadata:
        print(f"[API DeleteCaseDocument] Document metadata ID {document_metadata_id_to_delete} not found in case {case_id}.")
        return jsonify({"error": "Document not found within the case."}), 404

    # Delete from Azure Blob Storage
    blob_name_to_delete = doc_to_delete_metadata.get("blobName")
    if blob_name_to_delete:
        try:
            blob_client_instance = case_docs_blob_container_client.get_blob_client(blob_name_to_delete)
            blob_client_instance.delete_blob()
            print(f"[API DeleteCaseDocument] Successfully deleted blob: {blob_name_to_delete} from container: {AZURE_CASE_DOCS_CONTAINER_NAME}")
        except BlobResourceNotFoundError:
            print(f"[API DeleteCaseDocument] Blob not found in storage (already deleted?): {blob_name_to_delete}")
            # Continue to remove from DB metadata even if blob not found
        except Exception as e_blob:
            print(f"[API DeleteCaseDocument] Error deleting blob '{blob_name_to_delete}': {str(e_blob)}")
            # Decide if this is a hard failure or if DB metadata removal should still proceed.
            # For now, we'll let it proceed but log the error.
            # return jsonify({"error": "Failed to delete file from storage.", "details": str(e_blob)}), 500

    # Update Cosmos DB
    case_doc["caseDocuments"] = updated_case_documents
    case_doc["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cases_container_client.replace_item(item=case_id, body=case_doc)
    print(f"[API DeleteCaseDocument] Successfully removed document metadata ID {document_metadata_id_to_delete} from case {case_id} in Cosmos DB.")
    return jsonify({"message": f"Document '{doc_to_delete_metadata.get('fileName', 'Unknown')}' deleted successfully from case {case_id}."}), 200

# --- Document Deletion Endpoint for Chat Attachments ---
@app.route("/api/chat/attachments/delete", methods=["POST"])
def delete_chat_attachment():
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user_session_data["userinfo"]["sub"] # For logging or ownership verification if needed

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    blob_name = data.get("blobName")
    blob_container_name = data.get("blobContainer") # e.g., AZURE_CHAT_DOCS_CONTAINER_NAME

    if not blob_name or not blob_container_name:
        return jsonify({"error": "Missing blobName or blobContainer in request"}), 400

    print(f"[API DeleteChatAttachment] User: {user_id} attempting to delete blob: {blob_name} from container: {blob_container_name}")

    # Ensure we are only allowing deletion from expected chat attachment containers for security
    if blob_container_name != AZURE_CHAT_DOCS_CONTAINER_NAME:
        # If you have multiple valid chat attachment containers, check against a list
        print(f"[API DeleteChatAttachment] Attempt to delete from non-chat container: {blob_container_name}. Denied.")
        return jsonify({"error": "Invalid target container for deletion"}), 403


    if not blob_service_client: # Check if the main blob service client is available
        print("[API DeleteChatAttachment] Blob service client not available.")
        return jsonify({"error": "Blob storage service not available"}), 503
    
    target_blob_client = None
    if blob_container_name == AZURE_CHAT_DOCS_CONTAINER_NAME:
        if chat_docs_blob_container_client:
            target_blob_client = chat_docs_blob_container_client.get_blob_client(blob_name)
        else:
            print(f"[API DeleteChatAttachment] Mismatch or uninitialized client for container: {blob_container_name}")
            return jsonify({"error": "Specified blob container client not configured or mismatch"}), 500

    if not target_blob_client: # Should be caught by above, but as a safeguard
        return jsonify({"error": "Could not obtain blob client for deletion."}), 500

    try:
        target_blob_client.delete_blob()
        print(f"[API DeleteChatAttachment] Successfully deleted blob: {blob_name} from container: {blob_container_name}")
        return jsonify({"message": f"File '{blob_name}' deleted successfully from chat context."}), 200

    except BlobResourceNotFoundError:
        print(f"[API DeleteChatAttachment] Blob not found: {blob_name} in container: {blob_container_name}")
        return jsonify({"error": "File not found in storage."}), 404 # Or 200 if "deleted" means "it's gone"
    except Exception as e:
        print(f"[API DeleteChatAttachment] Error deleting blob '{blob_name}': {str(e)}")
        return jsonify({"error": "Failed to delete file from storage.", "details": str(e)}), 500

# --- AI Assistant Query Endpoint ---
# This endpoint is where the AI model will be queried with user input and chat history.
@app.route("/api/query", methods=["POST"])
def query_ai_assistant():
    user_session_data = session.get("user")
    if not (user_session_data and "userinfo" in user_session_data and "sub" in user_session_data["userinfo"]):
        return jsonify({"error": "User not authenticated"}), 401
    
    user_id = user_session_data["userinfo"]["sub"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        user_input = data.get("question", "")
        chat_history = data.get("chat_history", []) 
        # This now comes from the chatStore, representing files already in blob storage for the session context
        staged_chat_documents_metadata = data.get("staged_chat_documents", []) 

        print(f"[API Query] User: {user_id}, Input: '{user_input}'")
        print(f"[API Query] Chat History items: {len(chat_history)}")
        print(f"[API Query] Staged Chat Documents Metadata (for context): {len(staged_chat_documents_metadata)}")
        # for doc_meta in staged_chat_documents_metadata:
        #     print(f"  Context Doc: {doc_meta.get('fileName')}, URL: {doc_meta.get('url')}")

        # Get Prompt Flow credentials from .env
        promptflow_url = os.getenv("URL")
        promptflow_api_key = os.getenv("API_KEY")

        if not promptflow_url or not promptflow_api_key:
            print("[API Query] CRITICAL: Promptflow URL or API_KEY not configured in .env. Cannot call AI.")
            # Return a clear error if PF is not configured
            return jsonify({"error": "AI service not configured on the server."}), 503

        # Prepare data for Prompt Flow
        # The exact structure depends on your Prompt Flow's defined inputs.
        # This matches the structure you provided earlier.
        # You might need to pass document context differently if your PF expects it.
        promptflow_request_data = {
            "question": user_input,
            "chat_history": chat_history,
            # How you pass document context to Prompt Flow depends on your flow's design.
            # Option 1: Pass metadata (URLs, names)
            "documents_context": staged_chat_documents_metadata, 
            # Option 2: If PF expects full text, you'd need a step here or in PF
            # to fetch content from blob URLs and include it.
            # For now, sending metadata.
        }
        
        pf_body = json.dumps(promptflow_request_data).encode('utf-8')
        pf_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json', # Often good to specify what you expect back
            'Authorization': f'Bearer {promptflow_api_key}'
            # Add any other custom headers your Prompt Flow endpoint requires
        }

        print(f"[API Query] Calling Prompt Flow: {promptflow_url}")
        # print(f"[API Query] Sending Payload: {json.dumps(promptflow_request_data, indent=2)}") # For debugging payload

        pf_response = requests.post(promptflow_url, data=pf_body, headers=pf_headers, timeout=120) # Added timeout
        pf_response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)

        result_json = pf_response.json()
        print(f"[API Query] Prompt Flow Raw Response: {json.dumps(result_json, indent=2)}")
        
        # Extract the answer from the Prompt Flow response.
        # This depends on the output schema of your specific Prompt Flow.
        ai_answer = "Could not extract answer from AI response." # Default
        if isinstance(result_json, dict):
            ai_answer = result_json.get("answer", 
                                result_json.get("reply", 
                                result_json.get("output", 
                                "No 'answer', 'reply', or 'output' field found in Prompt Flow response.")))
        else:
            print(f"[API Query] Prompt Flow response was not a dictionary: {type(result_json)}")
            ai_answer = "AI response format was unexpected."

        # If Prompt Flow indicates it used/generated files and returns metadata for *new* files, include it.
        # For now, we assume the main output is the text answer.
        ai_generated_attachments = result_json.get("generated_documents_metadata", [])

        return jsonify({
            "response": {"answer": ai_answer},
            "uploadedChatDocuments": ai_generated_attachments # For files AI might "return"
        })

    except requests.exceptions.Timeout:
        print(f"[API Query] Timeout calling Prompt Flow: {PROMPTFLOW_ENDPOINT_URL}")
        return jsonify({"error": "AI service timed out.", "details": "The request to the AI service took too long to respond."}), 504
    except requests.exceptions.RequestException as e_req:
        print(f"[API Query] HTTP Request Error calling Prompt Flow: {str(e_req)}")
        return jsonify({"error": "Failed to connect to AI service.", "details": str(e_req)}), 503
    except Exception as e:
        print(f"[API Query] General Error in /api/query: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/')
def serve_main_app():
    if not session.get("user"): session['next_url'] = request.path; return redirect(url_for("login"))
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_vue_paths(path):
    if os.path.exists(os.path.join(app.static_folder, path)): return send_from_directory(app.static_folder, path)
    else: 
        if not session.get("user"): session['next_url'] = request.path; return redirect(url_for("login"))
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
