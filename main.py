from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
import json
import requests
import traceback
import inspect
import pandas as pd
import io

app = FastAPI()

REQUIRED_FIELDS = {
    "Article": ["headline", "datePublished", "author"],
    "Product": ["name", "offers"],
    "Event": ["name", "startDate", "location"],
    "Recipe": ["name", "recipeIngredient", "recipeInstructions"],
    "FAQPage": ["mainEntity"]
}

def get_json_ld_schema(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    json_ld_tags = soup.find_all("script", {"type": "application/ld+json"})
    schemas = []

    for tag in json_ld_tags:
        try:
            data = json.loads(tag.string)
            if isinstance(data, list):
                schemas.extend(data)
            elif isinstance(data, dict) and "@graph" in data:
                schemas.extend(data["@graph"])
            else:
                schemas.append(data)
        except (json.JSONDecodeError, TypeError):
            continue
    return schemas

def validate_fields(schema):
    errors = []

    if "@type" not in schema:
        errors.append("Missing @type.")
        return False, errors

    schema_type = schema["@type"]
    required = REQUIRED_FIELDS.get(schema_type, [])

    for field in required:
        if field not in schema:
            errors.append(f"Missing required field: {field}")

    if schema_type == "Article" and "author" in schema:
        author = schema["author"]
        if isinstance(author, dict):
            if "@type" not in author or "name" not in author:
                errors.append("Author must have @type and name.")
        elif isinstance(author, list):
            for person in author:
                if "@type" not in person or "name" not in person:
                    errors.append("Each author must have @type and name.")
        else:
            errors.append("Author must be an object or list of objects.")

    if schema_type == "Product" and "offers" in schema:
        offers = schema["offers"]
        if isinstance(offers, dict):
            if "price" not in offers or "priceCurrency" not in offers:
                errors.append("Offers must include price and priceCurrency.")
        else:
            errors.append("Offers should be an object.")

    return len(errors) == 0, errors

# ==== Existing Script Execution Support ====

class ScriptRequest(BaseModel):
    code: str
    inputs: dict = {}
 

@app.get("/validate-schema")
def validate_schema_endpoint(url: str = Query(..., description="URL to extract and validate schema.org data from")):
    try:
        schemas = get_json_ld_schema(url)
        if not schemas:
            raise HTTPException(status_code=404, detail="No JSON-LD schema found.")

        results = []
        for idx, schema in enumerate(schemas):
            valid, messages = validate_fields(schema)
            results.append({
                "index": idx,
                "type": schema.get("@type", "Unknown"),
                "valid": valid,
                "errors" if not valid else "notes": messages,
                "snippet": schema
            })

        return JSONResponse({
            "success": True,
            "url": url,
            "total_blocks": len(schemas),
            "validation_results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-script")
async def run_script(request: ScriptRequest):
    local_vars = {}

    try:
        # Execute the provided script
        exec(request.code, {}, local_vars)

        # Extract all functions defined in local_vars
        user_functions = {
            name: obj for name, obj in local_vars.items()
            if callable(obj) and inspect.isfunction(obj)
        }

        if len(user_functions) == 0:
            raise HTTPException(status_code=400, detail="No function found in script.")
        elif len(user_functions) > 1:
            raise HTTPException(status_code=400, detail="Multiple functions found. Please specify which one to run.")

        # Auto-select the only function
        function_name, fn = next(iter(user_functions.items()))

        # Execute with inputs
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