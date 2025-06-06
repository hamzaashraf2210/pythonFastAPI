from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request, Depends, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import pyarrow.parquet as pq
import pyarrow as pa
from bs4 import BeautifulSoup
from jsonschema import Draft7Validator, RefResolver, ValidationError
from urllib.parse import urlparse, urljoin
from datetime import datetime
import duckdb
import math
import uuid
import json
import os
import requests
import traceback
import inspect
import pandas as pd
import io
import re

app = FastAPI()

security = HTTPBearer()

API_TOKEN = "pythonFastAPI"

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

class JSONDataSet(BaseModel):
    data: List[Dict[str, Any]]

class SchemaRequest(BaseModel):
    schema: dict 

class ScriptRequest(BaseModel):
    code: str
    inputs: dict = {}


class ErrorDetail(BaseModel):
    type: str
    missing_field: str
    message: str
    line: str
    schema_snippet: Dict[str, Any]
    corrected_example: Dict[str, Any]

class Issue(BaseModel):
    line: int
    errors: List[ErrorDetail]

def json_to_parquet(json_data: List[Dict[str, Any]], output_file: str) -> None:
    if not json_data:
        raise ValueError("Input JSON data is empty")

    df = pd.DataFrame(json_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
 
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )

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

def is_valid_url(url):
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"] and bool(parsed.netloc)

def validate_logo_field(value):
    if isinstance(value, str):
        return is_valid_url(value)
    if isinstance(value, dict):
        if value.get("@type") == "ImageObject":
            return is_valid_url(value.get("url"))
    if isinstance(value, list):
        return all(validate_logo_field(v) for v in value)
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

    soup = BeautifulSoup(response.text, 'html.parser')
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
        except json.JSONDecodeError:
            continue

    if not updated_schemas:
        raise HTTPException(status_code=404, detail="No valid schema found to update.")

    # When there's more than one schema object, use @graph
    if len(updated_schemas) > 1:
        # Remove @context from each item
        for item in updated_schemas:
            item.pop("@context", None)
        final_schema = {
            "@context": "https://schema.org",
            "@graph": updated_schemas
        }
    else:
        # For a single schema item, ensure @context is present at top level
        single_schema = updated_schemas[0]
        if "@context" not in single_schema:
            single_schema["@context"] = "https://schema.org"
        final_schema = single_schema

    return final_schema

def is_valid_url(url):
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

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

def generate_corrected_example(schema_type, field):
    # Provide simple corrected example snippets per (type, field)
    examples = {
        ("Organization", "logo"): "https://example.com/logo.png",
        ("Article", "datePublished"): "2023-01-01T00:00:00Z",
        ("Product", "offers"): {
            "@type": "Offer",
            "price": "19.99",
            "priceCurrency": "USD"
        }
    }
    return examples.get((schema_type, field), "UPDATED LINE AND VALUE")

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
            "schema_snippet": schema,
            "corrected_example": {"@type": "TypeName"}
        })
        return errors

    schema_type = schema_type[0] if isinstance(schema_type, list) else schema_type
    expected_fields = EXPECTED_FIELDS.get(schema_type, [])

    # Check for missing expected fields
    for field in expected_fields:
        if field not in schema:
            # Build corrected example: add the missing field with a valid example value
            corrected_example = schema.copy() if isinstance(schema, dict) else {}
            corrected_example[field] = generate_corrected_example(schema_type, field)

            errors.append({
                "type": schema_type,
                "missing_field": field,
                "message": f"Missing expected field '{field}' for type '{schema_type}'",
                "line": line_number or "unknown",
                "schema_snippet": schema,
                "corrected_example": corrected_example
            })

    # Validate Article.datePublished format
    if schema_type == "Article" and "datePublished" in schema:
        date_val = schema["datePublished"]
        if not is_iso_date(date_val):
            corrected_example = schema.copy()
            corrected_example["datePublished"] = generate_corrected_example(schema_type, "datePublished")
            errors.append({
                "type": schema_type,
                "field": "datePublished",
                "message": "Field 'datePublished' should be an ISO 8601 date (e.g., 2023-01-01T00:00:00Z)",
                "line": line_number or "unknown",
                "schema_snippet": {"datePublished": date_val},
                "corrected_example": {"datePublished": corrected_example["datePublished"]}
            })

    # Validate Product.offers @type
    if schema_type == "Product" and "offers" in schema:
        offers = schema["offers"]
        if isinstance(offers, dict):
            offer_type = offers.get("@type", "")
            if offer_type != "Offer":
                corrected_offer = offers.copy()
                corrected_offer["@type"] = "Offer"
                corrected_example = schema.copy()
                corrected_example["offers"] = corrected_offer

                errors.append({
                    "type": schema_type,
                    "field": "offers",
                    "message": "Field 'offers' should contain an object with '@type': 'Offer'",
                    "line": line_number or "unknown",
                    "schema_snippet": {"offers": offers},
                    "corrected_example": {"offers": corrected_offer}
                })

    # Validate Organization.logo
    if schema_type == "Organization" and "logo" in schema:
        logo = schema["logo"]
        if not validate_logo_field(logo):
            if isinstance(logo, dict) and logo.get("@type") == "ImageObject":
                corrected_logo = logo.copy()
                corrected_logo["url"] = generate_corrected_example(schema_type, "logo")
            else:
                corrected_logo = generate_corrected_example(schema_type, "logo")

            corrected_example = schema.copy()
            corrected_example["logo"] = corrected_logo

            errors.append({
                "type": schema_type,
                "field": "logo",
                "message": "Field 'logo' should be a valid URL or a valid ImageObject with a 'url'",
                "line": line_number or "unknown",
                "schema_snippet": {"logo": logo},
                "corrected_example": {"logo": corrected_logo}
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
    Returns a string of the corrected schema with '// update' annotations.
    """
    corrected = copy.deepcopy(schema)
    updated_fields = set()

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
        obj_type = obj.get("@type")
        if isinstance(obj_type, list):
            obj_type = obj_type[0]
        if not obj_type:
            return obj

        required = required_fields.get(obj_type, [])
        for field in required:
            if field not in obj:
                obj[field] = f"[MISSING_{field}]"
                updated_fields.add(field)
        return obj

    def clean_nulls(obj):
        cleaned = {}
        for k, v in obj.items():
            if v not in [None, ""]:
                cleaned[k] = v
            elif v in [None, ""]:
                updated_fields.add(k)
        return cleaned

    def normalize(obj):
        if "@type" in obj and isinstance(obj["@type"], str):
            obj["@type"] = obj["@type"].strip()

        if "url" in obj and not isinstance(obj["url"], str):
            obj["url"] = str(obj["url"])
            updated_fields.add("url")

        return obj

    if "@graph" in corrected:
        corrected["@graph"] = [
            normalize(clean_nulls(ensure_fields(item)))
            for item in corrected["@graph"]
        ]
    else:
        corrected = normalize(clean_nulls(ensure_fields(corrected)))

    # Annotate lines with // update
    json_lines = json.dumps(corrected, indent=4).splitlines()
    annotated = []

    for line in json_lines:
        if any(f'"{field}"' in line for field in updated_fields):
            annotated.append(f"{line}  // update")
        else:
            annotated.append(line)

    return "\n".join(annotated)


def json_pretty(obj: Dict[str, Any]) -> str:
    import json
    return json.dumps(obj, indent=4)


def get_null_counts(df: pd.DataFrame) -> dict:
    return df.isnull().sum().to_dict()
 

@app.get("/true-validate")
async def true_validate(url: str = Query(..., title="URL to validate")):
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
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
def validate_schema(
    url: str = Query(..., description="URL of the webpage to validate"),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)  # Moved here!
):
    headers = {
        "Cache-Control": "no-cache",
        "User-Agent": "SchemaValidator/1.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    validation_results = []
    error_count = 0

    for index, tag in enumerate(scripts):
        try:
            content = tag.string
            data = json.loads(content)
            schemas = data.get("@graph") if isinstance(data, dict) and "@graph" in data else [data]
            if not isinstance(schemas, list):
                schemas = [schemas]
            errors = validate_schema_data(schemas, line_number=index + 1)
            if errors:
                validation_results.append({
                    "line": index + 1,
                    "errors": errors
                })
                error_count += len(errors)
        except Exception as e:
            validation_results.append({
                "line": index + 1,
                "errors": [{"message": f"JSON parse error: {str(e)}"}]
            })

    return JSONResponse(content={
        "requested_url": url,
        "fetched_url": response.url,
        "validation_count": error_count,
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
    page_size: int = Query(1000, ge=1, le=1000),
    sql_query: str = Form(None)  #
):
    try:
        if not file.filename.endswith(".parquet"):
            raise HTTPException(status_code=400, detail="Only .parquet files are supported.")

        contents = await file.read()
        file_size_bytes = len(contents)
        buffer = io.BytesIO(contents)

        df = pd.read_parquet(buffer)
        buffer.seek(0)
        parquet_file = pq.ParquetFile(buffer)

        # Metadata
        metadata = {
            "num_rows": parquet_file.metadata.num_rows,
            "num_columns": parquet_file.metadata.num_columns,
            "row_group_count": parquet_file.num_row_groups,
            "created_by": parquet_file.metadata.created_by,
            "column_names": df.columns.tolist(),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "schema": str(parquet_file.schema),
            "file_size_bytes": file_size_bytes,
            "file_size_kb": round(file_size_bytes / 1024, 2),
            "null_counts": df.isnull().sum().to_dict()
        }

        # SQL query execution
        if sql_query:
            try:
                queried_df = duckdb.query_df(df, "df", sql_query).to_df()
                data = queried_df.to_dict(orient="records")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid SQL query: {e}")
        else:
            # Default pagination on full DataFrame
            data = df.to_dict(orient="records")

        # Clean NaNs
        def clean_nan(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(i) for i in obj]
            return obj

        cleaned_data = clean_nan(data)

        total_rows = len(cleaned_data)
        total_pages = (total_rows + page_size - 1) // page_size
        if page > total_pages and total_pages != 0:
            raise HTTPException(status_code=400, detail=f"Page {page} out of range. Max page is {total_pages}.")

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = cleaned_data[start_idx:end_idx]

        response = {
            "filename": file.filename,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size,
            "rows_returned": len(page_data),
            "metadata": metadata,
            "data": page_data
        }

        if page < total_pages:
            url = request.url.include_query_params(page=page + 1, page_size=page_size)
            response["next_page_url"] = str(url)

        return response

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read parquet file: {str(e)}\nTraceback:\n{tb}"
        )

@app.get("/healthz")
def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})
    

@app.post("/save-parquet")
async def save_parquet(dataset: JSONDataSet):
    try:
        # Generate a unique file name
        filename = f"dataset_{uuid.uuid4().hex}.parquet"
        output_path = os.path.join("parquet_files", filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Parquet
        json_to_parquet(dataset.data, output_path)

        # Return file as a downloadable response
        return FileResponse(
            path=output_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/beautify", response_class=PlainTextResponse)
async def beautify_json(issues: List[Issue]):
    output_lines = []

    for entry in issues:
        line_num = entry.line
        output_lines.append(f"Line Number: {line_num}")
        output_lines.append("=" * 80)

        for i, error in enumerate(entry.errors, 1):
            output_lines.append(f"Issue {i}:")
            output_lines.append(f"  Type: {error.type}")
            output_lines.append(f"  Missing Field: {error.missing_field}")
            output_lines.append(f"  Message: {error.message}")
            output_lines.append(f"  Reported Line: {error.line}\n")

            output_lines.append("  Schema Snippet:")
            schema_str = json_pretty(error.schema_snippet)
            output_lines.append(schema_str + "\n")

            output_lines.append("  Corrected Example:")
            corrected_str = json_pretty(error.corrected_example)
            output_lines.append(corrected_str)
            output_lines.append("-" * 80)

    return "\n".join(output_lines)