from http.server import BaseHTTPRequestHandler
import json
import os
import re
import tempfile
import base64
from gradio_client import Client, handle_file


def parse_summary(summary_text: str) -> dict:
    """
    Extract structured fields from the markdown table returned by the
    PlantSeg decision-support pipeline (see app.py's `diagnose()` summary
    string). The table has this shape:

        ### Predicted disease: **{pred_class}**

        | Metric | Value |
        |---|---|
        | Classification confidence | {..}% |
        | Predicted lesion severity | {..}% of leaf/fruit area |
        | CAM-predicted-mask IoU | {..} |
        | **Confidence flag** | **{HIGH|LOW}** |

        {flag_note}

        *Ground-truth-free at inference time ...*

    Parsed with a table-row splitter (not a single fragile regex) so it
    survives minor formatting/whitespace/unicode-dash changes upstream.
    """
    fields = {}
    for line in summary_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.lower().startswith("| metric"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 2:
            continue
        key = parts[0].replace("*", "").strip()
        value = parts[1].replace("*", "").strip()
        fields[key] = value

    disease_match = re.search(r"Predicted disease:\s*\*\*([^*]+)\*\*", summary_text)
    disease_name = disease_match.group(1).strip() if disease_match else None

    # "CAM-predicted-mask IoU" may use a plain hyphen or an en-dash depending on
    # how the source string was authored, so match on substring rather than the
    # exact character.
    cam_iou_key = next((k for k in fields if "IoU" in k), None)

    flag_note = ""
    for line in summary_text.splitlines():
        line = line.strip()
        if line.startswith("High confidence:") or line.startswith("Low confidence:"):
            flag_note = line
            break

    confidence_flag = fields.get("Confidence flag", "").upper().strip()
    if confidence_flag not in ("HIGH", "LOW"):
        confidence_flag = None  # be explicit that parsing failed rather than guessing

    return {
        "disease_name": disease_name,
        "classification_confidence": fields.get("Classification confidence"),
        "severity": fields.get("Predicted lesion severity"),
        "cam_mask_iou": fields.get(cam_iou_key) if cam_iou_key else None,
        "confidence_flag": confidence_flag,
        "flag_note": flag_note or None,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "running", "message": "API is active. Please use POST to submit images for prediction."}).encode())

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))

            image_b64 = body.get('image')
            if not image_b64:
                self.send_error(400, "No image provided")
                return

            # Decode base64 and save to temp file
            import uuid
            img_data = base64.b64decode(image_b64.split(',')[1])
            temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
            with open(temp_path, "wb") as f:
                f.write(img_data)

            # Connect to Hugging Face API securely using Vercel Environment Variable
            hf_token = os.environ.get("HF_TOKEN")
            client = Client("nomandiu9/diseases_prediction", token=hf_token)

            # Make the prediction
            result = client.predict(
                image=handle_file(temp_path),
                api_name="/diagnose"
            )

            # Helper to convert output images back to base64 for the frontend
            def encode_file(filepath):
                if not filepath or not os.path.exists(filepath): return None
                with open(filepath, "rb") as f:
                    return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

            # The API returns a tuple of 4 elements. result[1] and result[2] are dicts containing the 'path'
            lesion_path = result[1].get('path') if isinstance(result[1], dict) else result[1]
            cam_path = result[2].get('path') if isinstance(result[2], dict) else result[2]

            summary_text = result[3] or ""
            parsed = parse_summary(summary_text)

            response = {
                "summary": summary_text,       # kept for backward compatibility / debugging
                "lesion_overlay": encode_file(lesion_path),
                "cam_overlay": encode_file(cam_path),
                # Structured fields — this is what makes the faithfulness-gated
                # confidence flag (the paper's central contribution) actually
                # reach the UI instead of being silently dropped.
                **parsed,
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
