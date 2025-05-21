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

class ScriptRequest(BaseModel):
    code: str
    inputs: dict = {}

def validate_schema_item(schema):
    required_fields = ["@context", "@type"]
    missing_fields = [field for field in required_fields if field not in schema]

    return {
        "valid": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "type": schema.get("@type", "Unknown")
    }


def extract_json_ld(html):
    soup = BeautifulSoup(html, "html.parser")
    script_tags = soup.find_all("script", type="application/ld+json")

    extracted_schemas = []

    for tag in script_tags:
        try:
            data = json.loads(tag.string)
            if isinstance(data, dict) and "@graph" in data:
                extracted_schemas.extend(data["@graph"])
            else:
                extracted_schemas.append(data)
        except Exception as e:
            continue  # skip broken/invalid JSON

    return extracted_schemas


@app.get("/validate-schema")
def validate_schema(url: str = Query(..., description="URL of the website to validate schema from")):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to fetch the page.")

        html = response.text
        schemas = extract_json_ld(html)

        if not schemas:
            return JSONResponse(status_code=404, content={"message": "No JSON-LD schema found on the page."})

        results = {}

        for idx, schema in enumerate(schemas):
            results[f"schema_{idx}"] = validate_schema_item(schema)

        return {
            "url": url,
            "total_schemas_found": len(schemas),
            "validation_results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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