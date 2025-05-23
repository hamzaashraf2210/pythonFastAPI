from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from jsonschema import Draft7Validator, RefResolver, ValidationError
from urllib.parse import urlparse, urljoin
from datetime import datetime
import json
import os
import requests
import traceback
import inspect
import pandas as pd
import io
import re

app = FastAPI()

EXPECTED_FIELDS = {
    "Article": ["headline", "author", "datePublished", "mainEntityOfPage"],
    "NewsArticle": ["headline", "author", "datePublished", "mainEntityOfPage"],
    "BlogPosting": ["headline", "author", "datePublished", "articleBody"],
    "WebPage": ["name", "url"],
    "Organization": ["name", "url", "logo"],
    "Person": ["name"],
    "BreadcrumbList": ["itemListElement"],
    "Product": ["name", "offers", "description", "image"],
    "Offer": ["price", "priceCurrency", "availability", "url"],
    "Review": ["reviewRating", "author", "reviewBody"],
    "Event": ["name", "startDate", "location"],
    "LocalBusiness": ["name", "address", "telephone"],
    "FAQPage": ["mainEntity"],
    "HowTo": ["name", "step"],
    "Recipe": ["name", "recipeIngredient", "recipeInstructions"]
}

SCHEMA_OUTPUT_DIR = os.path.join(os.getcwd(), "validated_schemas")

os.makedirs(SCHEMA_OUTPUT_DIR, exist_ok=True)

class SchemaRequest(BaseModel):
    schema: dict 

class ScriptRequest(BaseModel):
    code: str
    inputs: dict = {}
 
def validate_url(url, base_url=None):
    if not url:
        return False
    if not isinstance(url, str):
        return False
    if base_url:
        # If base_url is provided, resolve relative URLs
        url = urljoin(base_url, url)
    result = urlparse(url)
    return all([result.scheme, result.netloc])    

def validate_logo_field(value):
    if isinstance(value, str) and is_valid_url(value):
        return True
    if isinstance(value, dict):
        url = value.get("url")
        if url and is_valid_url(url):
            return True
    return False

def validate_date(date):
    date_regex = r"^\d{4}-\d{2}-\d{2}([T\s]\d{2}:\d{2}:\d{2}(\.\d{1,3})?)?$"
    return bool(re.match(date_regex, date))
    
def validate_common_schema(schema_data, url, schema_type):
    # Validate URL if missing or invalid
    if not validate_url(schema_data.get('url', ''), url):
        schema_data['url'] = url  # Default to page URL if invalid or missing

    # Article type schema validation
    if schema_type == 'Article':
        if 'headline' not in schema_data:
            schema_data['headline'] = "Untitled Article"
        if 'author' not in schema_data:
            schema_data['author'] = {"@type": "Person", "name": "Anonymous"}
        if 'datePublished' not in schema_data or not validate_date(schema_data['datePublished']):
            schema_data['datePublished'] = "2023-01-01"

    # WebPage type schema validation
    elif schema_type == 'WebPage':
        if 'publisher' not in schema_data:
            schema_data['publisher'] = {"@type": "Organization", "name": "Default Publisher"}
        if 'mainEntityOfPage' not in schema_data:
            schema_data['mainEntityOfPage'] = {"@type": "WebPage", "url": url}

    # Product type schema validation
    elif schema_type == 'Product':
        if 'name' not in schema_data:
            schema_data['name'] = "Default Product Name"
        if 'offers' not in schema_data:
            schema_data['offers'] = {"@type": "Offer", "priceCurrency": "USD", "price": "0.00"}
        if 'sku' not in schema_data:
            schema_data['sku'] = "00000000"

    # Person type schema validation
    elif schema_type == 'Person':
        if 'name' not in schema_data:
            schema_data['name'] = "Unknown Person"

    # Event type schema validation
    elif schema_type == 'Event':
        if 'startDate' not in schema_data or not validate_date(schema_data['startDate']):
            schema_data['startDate'] = "2023-01-01T00:00:00"
        if 'location' not in schema_data:
            schema_data['location'] = {"@type": "Place", "name": "Event Location"}
        if 'name' not in schema_data:
            schema_data['name'] = "Untitled Event"

    # Review type schema validation
    elif schema_type == 'Review':
        if 'reviewBody' not in schema_data:
            schema_data['reviewBody'] = "No review content."
        if 'author' not in schema_data:
            schema_data['author'] = {"@type": "Person", "name": "Anonymous"}

    # Recipe type schema validation
    elif schema_type == 'Recipe':
        if 'name' not in schema_data:
            schema_data['name'] = "Untitled Recipe"
        if 'ingredients' not in schema_data:
            schema_data['ingredients'] = []
        if 'recipeInstructions' not in schema_data:
            schema_data['recipeInstructions'] = "Step 1: Prepare the ingredients."

    # Service type schema validation
    elif schema_type == 'Service':
        if 'serviceType' not in schema_data:
            schema_data['serviceType'] = "General Service"
        if 'provider' not in schema_data:
            schema_data['provider'] = {"@type": "Organization", "name": "Service Provider"}
        if 'areaServed' not in schema_data:
            schema_data['areaServed'] = {"@type": "Place", "name": "Worldwide"}
    
    # Place type schema validation
    elif schema_type == 'Place':
        if 'name' not in schema_data:
            schema_data['name'] = "Unnamed Place"
        if 'address' not in schema_data:
            schema_data['address'] = {"@type": "PostalAddress", "streetAddress": "Unknown Street", "addressLocality": "Unknown City"}
        if 'geo' not in schema_data:
            schema_data['geo'] = {"@type": "GeoCoordinates", "latitude": "0.0", "longitude": "0.0"}

    # Organization type schema validation
    elif schema_type == 'Organization':
        if 'name' not in schema_data:
            schema_data['name'] = "Unnamed Organization"
        if 'address' not in schema_data:
            schema_data['address'] = {"@type": "PostalAddress", "streetAddress": "Unknown Street", "addressLocality": "Unknown City"}
        if 'url' not in schema_data:
            schema_data['url'] = url

    return schema_data
    

def validate_nested_schemas(schema_data, url, schema_type):
    if schema_type == 'Person' and 'address' in schema_data:
        address = schema_data['address']
        if 'streetAddress' not in address:
            address['streetAddress'] = "Unknown Street"
        if 'addressLocality' not in address:
            address['addressLocality'] = "Unknown City"
        if 'addressRegion' not in address:
            address['addressRegion'] = "Unknown Region"
        if 'postalCode' not in address:
            address['postalCode'] = "00000"
    elif schema_type == 'Product' and 'offers' in schema_data:
        offer = schema_data['offers']
        if 'priceCurrency' not in offer:
            offer['priceCurrency'] = "USD"
        if 'price' not in offer:
            offer['price'] = "0.00"
    elif schema_type == 'Service' and 'provider' in schema_data:
        provider = schema_data['provider']
        if 'name' not in provider:
            provider['name'] = "Unknown Provider"
    elif schema_type == 'Place' and 'address' in schema_data:
        address = schema_data['address']
        if 'streetAddress' not in address:
            address['streetAddress'] = "Unknown Street"
        if 'addressLocality' not in address:
            address['addressLocality'] = "Unknown City"
    elif schema_type == 'Organization' and 'contactPoint' in schema_data:
        contact_point = schema_data['contactPoint']
        if 'telephone' not in contact_point:
            contact_point['telephone'] = "+1-800-000-0000"
    return schema_data
    
def fetch_and_update_schema(url):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {e}")

    # Parse the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Look for JSON-LD schema.org data
    schema_tags = soup.find_all('script', {'type': 'application/ld+json'})

    if not schema_tags:
        raise HTTPException(status_code=404, detail="No schema.org JSON-LD schema found on this page.")

    updated_schemas = []
    for tag in schema_tags:
        try:
            schema_data = json.loads(tag.string)
            if isinstance(schema_data, dict):
                schema_type = schema_data.get('@type', '')
                schema_data = validate_common_schema(schema_data, url, schema_type)
                schema_data = validate_nested_schemas(schema_data, url, schema_type)
                updated_schemas.append(schema_data)
            elif isinstance(schema_data, list):
                for item in schema_data:
                    schema_type = item.get('@type', '')
                    item = validate_common_schema(item, url, schema_type)
                    item = validate_nested_schemas(item, url, schema_type)
                    updated_schemas.append(item)
        except json.JSONDecodeError as e:
            continue

    # Wrap in @graph if needed
    if updated_schemas:
        if len(updated_schemas) > 1:
            final_schema = {
                "@context": "https://schema.org",
                "@graph": updated_schemas
            }
        else:
            final_schema = updated_schemas[0]

        return final_schema
    else:
        raise HTTPException(status_code=404, detail="No valid schema found to update.")

def is_valid_url(url):
    if not isinstance(url, str) or not url.strip():
        return False
    # Accept typical URLs validated by validators.url
    if validators.url(url):
        return True
    # Accept protocol-relative URLs (starting with //)
    if url.startswith("//"):
        return True
    # Accept data URLs (e.g., base64-encoded images)
    if url.startswith("data:"):
        return True
    return False

def is_iso_date(value):
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False

def validate_schema_item(schema):
    required_fields = ["@context", "@type"]
    missing_fields = [field for field in required_fields if field not in schema]

    return {
        "valid": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "type": schema.get("@type", "Unknown")
    }


def validate_schema_data(schema, line_number=None):
    errors = []

    if isinstance(schema, list):
        for item in schema:
            errors.extend(validate_schema_data(item))
        return errors

    schema_type = schema.get("@type")
    if not schema_type:
        errors.append({
            "message": "Missing @type in schema.",
            "line": line_number or "unknown",
        })
        return errors

    schema_type = schema_type[0] if isinstance(schema_type, list) else schema_type
    expected_fields = EXPECTED_FIELDS.get(schema_type)

    if expected_fields:
        for field in expected_fields:
            if field not in schema:
                errors.append({
                    "type": schema_type,
                    "missing_field": field,
                    "message": f"Missing expected field '{field}' for type '{schema_type}'",
                    "line": line_number or "unknown"
                })

    if schema_type == "Article" and "datePublished" in schema:
        if not is_iso_date(schema["datePublished"]):
            errors.append({
                "type": schema_type,
                "field": "datePublished",
                "message": "Field 'datePublished' should be an ISO 8601 date (e.g., 2023-01-01T00:00:00)",
                "line": line_number or "unknown"
            })

    if schema_type == "Product" and "offers" in schema:
        offers = schema["offers"]
        if isinstance(offers, dict):
            offer_type = offers.get("@type", "")
            if offer_type != "Offer":
                errors.append({
                    "type": schema_type,
                    "field": "offers",
                    "message": "Field 'offers' should contain an object with '@type': 'Offer'",
                    "line": line_number or "unknown"
                })

    if schema_type == "Organization" and "logo" in schema:
        if not is_valid_url(schema["logo"]):
            errors.append({
                "type": schema_type,
                "field": "logo",
                "message": "Field 'logo' should be a valid URL",
                "line": line_number or "unknown"
            })

    return errors

def fetch_schema_from_url(url: str):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    soup = BeautifulSoup(resp.text, 'html.parser')
    scripts = soup.find_all('script', type='application/ld+json')
    schemas = []

    for script in scripts:
        try:
            data = json.loads(script.string)
            # Support for @graph
            if isinstance(data, dict) and '@graph' in data:
                schemas.extend(data['@graph'])
            elif isinstance(data, list):
                schemas.extend(data)
            else:
                schemas.append(data)
        except Exception:
            # Skip if JSON parsing fails
            continue

    if not schemas:
        raise HTTPException(status_code=404, detail="No JSON-LD schema found on the page.")

    return schemas

def validate_schema(schema):
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(schema), key=lambda e: e.path)

    error_details = []
    for err in errors:
        # Attempt to find line number roughly by JSON path (best effort)
        line_number = None
        if hasattr(err, 'context') and err.context:
            line_number = err.context[0].schema_path if err.context else None

        error_details.append({
            "message": err.message,
            "path": list(err.path),
            "schema_path": list(err.schema_path),
            "line_number": line_number,
        })

    return error_details

def correct_schema(schema):
    """
    Tries to apply common-sense corrections to the given JSON-LD schema.
    Returns the corrected schema.
    """
    import copy

    corrected = copy.deepcopy(schema)

    # Define required fields for common types
    required_fields = {
        "Organization": ["name"],
        "WebSite": ["name", "url"],
        "WebPage": ["name"],
        "Article": ["headline", "author", "datePublished"],
        "Person": ["name"],
        "BreadcrumbList": ["itemListElement"],
        "Product": ["name", "offers"],
        "Review": ["author", "reviewBody"]
    }

    def ensure_fields(obj):
        """
        Ensure required fields are present based on @type
        """
        obj_type = obj.get("@type")
        if isinstance(obj_type, list):
            obj_type = obj_type[0]
        if not obj_type:
            return obj

        required = required_fields.get(obj_type, [])
        for field in required:
            if field not in obj:
                obj[field] = f"[MISSING_{field}]"
        return obj

    def clean_nulls(obj):
        """
        Remove keys with null or empty values
        """
        return {k: v for k, v in obj.items() if v not in [None, ""]}

    def normalize(obj):
        """
        Normalize field formats or apply basic transformations
        """
        if "@type" in obj and isinstance(obj["@type"], str):
            obj["@type"] = obj["@type"].strip()

        if "url" in obj and not isinstance(obj["url"], str):
            obj["url"] = str(obj["url"])

        return obj

    if "@graph" in corrected:
        corrected["@graph"] = [
            normalize(clean_nulls(ensure_fields(item)))
            for item in corrected["@graph"]
        ]
    else:
        corrected = normalize(clean_nulls(ensure_fields(corrected)))

    return corrected
 

@app.get("/true-validate")
async def true_validate(url: str = Query(..., title="URL to validate")):
    try:
        validated_schema = fetch_and_update_schema(url)
        
        # Create a filename based on the URL and current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"schema_{timestamp}.txt"
        file_path = os.path.join(SCHEMA_OUTPUT_DIR, filename)
        
        # Write the validated schema to a txt file
        with open(file_path, 'w') as file:
            json.dump(validated_schema, file, indent=4)
        
        # Return the filename or a download link
        return FileResponse(file_path, media_type="text/plain", filename=filename)
        
    except HTTPException as e:
        raise e


@app.get("/validate-schema")
def validate_schema(url: str = Query(..., description="URL of the webpage to validate")):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    validation_results = []

    for index, tag in enumerate(scripts):
        try:
            content = tag.string
            data = json.loads(content)

            # Support @graph
            schemas = data.get("@graph") if isinstance(data, dict) and "@graph" in data else [data]

            if not isinstance(schemas, list):
                schemas = [schemas]

            errors = validate_schema_data(schemas, line_number=index + 1)

            if errors:  # Only include results with errors
                validation_results.append({
                    "line": index + 1,
                    "errors": errors
                })

        except Exception as e:
            validation_results.append({
                "line": index + 1,
                "errors": [{"message": f"JSON parse error: {str(e)}"}]
            })

    return JSONResponse(content={
        "url": url,
        "results": validation_results
    })

@app.post("/run-script")
async def run_script(request: ScriptRequest):
    local_vars = {}

    try:
        exec(request.code, {}, local_vars)

        user_functions = {
            name: obj for name, obj in local_vars.items()
            if callable(obj) and inspect.isfunction(obj)
        }

        if len(user_functions) == 0:
            raise HTTPException(status_code=400, detail="No function found in script.")
        elif len(user_functions) > 1:
            raise HTTPException(status_code=400, detail="Multiple functions found. Please specify which one to run.")

        function_name, fn = next(iter(user_functions.items()))
        result = fn(**request.inputs)

        return {
            "success": True,
            "function": function_name,
            "inputs": request.inputs,
            "output": result
        }

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{str(e)}\nTraceback:\n{tb}")
        
        

@app.get("/", include_in_schema=False)
async def health_check():
    return JSONResponse(content={"status": "ok"})

# ==== New Endpoint for Parquet File Upload ====

@app.post("/upload-parquet")
async def upload_parquet(
    request: Request,
    file: UploadFile = File(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(1000, ge=1, le=1000)
):
    try:
        if not file.filename.endswith(".parquet"):
            raise HTTPException(status_code=400, detail="Only .parquet files are supported.")

        contents = await file.read()
        buffer = io.BytesIO(contents)

        df = pd.read_parquet(buffer)
        data = df.to_dict(orient="records")

        def clean_nan(obj):
            import math
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(i) for i in obj]
            else:
                return obj

        cleaned_data = clean_nan(data)

        total_rows = len(cleaned_data)
        total_pages = (total_rows + page_size - 1) // page_size  # ceil division

        if page > total_pages and total_pages != 0:
            raise HTTPException(status_code=400, detail=f"Page {page} out of range. Max page is {total_pages}.")

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = cleaned_data[start_idx:end_idx]

        # Build next page URL if there is a next page
        next_page_url = None
        if page < total_pages:
            url = request.url.include_query_params(page=page + 1, page_size=page_size)
            next_page_url = str(url)

        response = {
            "filename": file.filename,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size,
            "rows_returned": len(page_data),
            "data": page_data,
        }

        if next_page_url:
            response["next_page_url"] = next_page_url

        return response

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read parquet file: {str(e)}\nTraceback:\n{tb}"
        )