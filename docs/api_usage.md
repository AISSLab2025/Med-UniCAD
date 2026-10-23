# Med-UniCAD local-network API

## Files

The API is part of the installable package:

```text
Med-UniCAD/
├── configs/inference.example.json
├── scripts/serve_api.py
└── src/medunicad/inference/
```

## Install API dependencies

Activate the same environment used for inference and install the package:

```bat
python -m pip install -e ".[api]"
```

## Start the server

```bat
python scripts/serve_api.py ^
  --config configs/inference.example.json ^
  --host 0.0.0.0 ^
  --port 8100
```

Keep `--workers 1`, because every additional worker loads another full copy of
the model.

Do not use `--reload` for normal inference. Reloading can initialize the model
more than once and consume additional GPU memory.

## Open API documentation

On the server computer:

```text
http://127.0.0.1:8100/docs
```

From another computer on the same network:

```text
http://<SERVER_LOCAL_IP>:8100/docs
```

To find the Windows server address:

```bat
ipconfig
```

Use the IPv4 address of the active network adapter.

## Endpoint 1: infer from a server-side image path

```text
POST /infer/path
Content-Type: application/json
```

Example JSON:

```json
{
  "image_path": "C:/medical-images/case.png",
  "modality": "ultrasound",
  "tasks": ["segmentation", "birads", "pathology", "report"],
  "request_id": "US_API_001",
  "save_input_copy": true
}
```

`tasks` may also be a comma-separated string:

```json
{
  "image_path": "A:/images/case.png",
  "modality": "ct",
  "tasks": "classification,report",
  "request_id": "CT_API_001"
}
```

### cURL

```bat
curl -X POST "http://127.0.0.1:8100/infer/path" ^
  -H "Content-Type: application/json" ^
  -d "{\"image_path\":\"A:/images/case.png\",\"modality\":\"ct\",\"tasks\":\"classification,report\",\"request_id\":\"CT_API_001\"}"
```

### Python

```python
import requests

payload = {
    "image_path": "A:/images/case.png",
    "modality": "mri",
    "tasks": ["detection", "severity", "report"],
    "request_id": "MRI_API_001",
}

response = requests.post(
    "http://127.0.0.1:8100/infer/path",
    json=payload,
    timeout=None,
)
response.raise_for_status()
result = response.json()
print(result)
```

## Endpoint 2: upload an image

```text
POST /infer/upload
Content-Type: multipart/form-data
```

### cURL

```bat
curl -X POST "http://127.0.0.1:8100/infer/upload" ^
  -F "image=@A:\images\breast_us.png" ^
  -F "modality=ultrasound" ^
  -F "tasks=segmentation,birads,pathology,report" ^
  -F "request_id=US_UPLOAD_001"
```

### Python

```python
import requests

with open(r"A:\images\breast_us.png", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8100/infer/upload",
        files={"image": ("breast_us.png", image_file, "image/png")},
        data={
            "modality": "ultrasound",
            "tasks": "segmentation,birads,pathology,report",
            "request_id": "US_UPLOAD_001",
        },
        timeout=None,
    )

response.raise_for_status()
result = response.json()
print(result)
```

## Expected response path fields

Every generated path is absolute on the API server:

```json
{
  "request_id": "US_UPLOAD_001",
  "request_root": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001",
  "output_root": "C:\\med-unicad\\outputs\\inference",
  "input": {
    "source_image": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\input\\breast_us.png",
    "saved_image": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\input\\breast_us.png"
  },
  "outputs": {
    "segmentation": {
      "mask_path": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\segmentation\\mask.png",
      "overlay_path": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\segmentation\\overlay.png"
    },
    "report": {
      "report_path": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\report\\report.txt",
      "prompt_path": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\report\\prompt.txt"
    }
  },
  "result_json": "C:\\med-unicad\\outputs\\inference\\US_UPLOAD_001\\result.json"
}
```

These are server-local paths. A remote client cannot open `A:\...` unless that
folder is also shared or mounted on the client.

## Retrieve a previous JSON result

```text
GET /requests/{request_id}
```

Example:

```text
http://127.0.0.1:8100/requests/US_UPLOAD_001
```

## Retrieve an artifact through HTTP

```text
GET /artifacts/{request_id}/{relative_artifact_path}
```

Example:

```text
http://127.0.0.1:8100/artifacts/US_UPLOAD_001/segmentation/overlay.png
```

This lets another computer access the saved file even though the absolute
Windows path belongs to the inference server.

## Health check

```text
GET /health
```

Example:

```text
http://127.0.0.1:8100/health
```

## Windows Firewall

The first time the server starts, allow Python on private networks. If the
prompt does not appear, create an inbound Windows Firewall rule for TCP port
`8100`.

Only expose this service on a trusted private network. The API does not include
authentication by default.
