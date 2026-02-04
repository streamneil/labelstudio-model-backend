import os
import logging
import json
import base64
import asyncio
from typing import List, Dict, Optional, Union

from label_studio_ml.model import LabelStudioMLBase
import openai
import httpx

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class KimiBackend(LabelStudioMLBase):
    """
    Kimi (Moonshot) Backend for Label Studio
    """
    # Class-level defaults to prevent AttributeError
    api_key = None
    base_url = None
    system_prompt = "你是一个智能标注助手。请详细描述这张图片的内容。"
    label_studio_host = None
    label_studio_api_key = None
    client = None

    def __init__(self, **kwargs):
        # Call base class init first
        super(KimiBackend, self).__init__(**kwargs)
        # Ensure setup is called
        self.setup()

    def setup(self):
        """
        Setup model, load config, initialize clients
        """
        logger.info("Initializing KimiBackend configuration...")
        
        # Use direct attribute assignment instead of self.set()
        self.model_version = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k-vision-preview")
        
        # Configuration
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "你是一个智能标注助手。请详细描述这张图片的内容。")
        self.label_studio_host = os.getenv("LABEL_STUDIO_HOST", "")
        self.label_studio_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        
        if self.api_key:
            # OpenAI Client
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=2
            )
            logger.info("OpenAI client initialized.")
        else:
            logger.warning("MOONSHOT_API_KEY not found. Client not initialized.")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        Predict logic
        """
        # Manually parse config to find control tags (SDK v1 compatibility)
        from_name, to_name, value_key = None, None, None
        
        # Debug: Log the parsed config to help diagnose matching issues
        logger.info(f"Parsed Label Config: {json.dumps(self.parsed_label_config, indent=2)}")
        
        # Iterate through parsed config to find the TextArea connected to an Image
        for name, info in self.parsed_label_config.items():
            if info['type'].lower() == 'textarea':
                # Check if it connects to an Image
                # Check both snake_case (SDK standard) and camelCase (raw XML attribute) just in case
                target_names = info.get('to_name') or info.get('toName') or []
                if target_names:
                    target_name = target_names[0]
                    target_info = self.parsed_label_config.get(target_name)
                    if target_info and target_info['type'].lower() == 'image':
                        from_name = name
                        to_name = target_name
                        # Get the variable name from the Image tag (e.g., $captioning -> captioning)
                        if target_info.get('inputs'):
                            value_key = target_info['inputs'][0]['value']
                        break
        
        # Fallback to Environment Variables if auto-detection fails
        if not from_name or not to_name:
            logger.warning("Auto-detection of label config failed. Trying environment variables...")
            from_name = os.getenv("LABEL_STUDIO_FROM_NAME")
            to_name = os.getenv("LABEL_STUDIO_TO_NAME")
            value_key = os.getenv("LABEL_STUDIO_DATA_KEY")
            
        if not from_name or not to_name:
            logger.error("Could not find TextArea connected to Image in Label config, and environment variables are missing.")
            return []

        predictions = []
        
        for task in tasks:
            task_id = task.get('id')
            # Get data path (e.g. image url or path)
            # value_key is usually the '$name' from <Image value="$name">
            data_field = value_key
            image_path = task['data'].get(data_field)
            
            if not image_path:
                logger.warning(f"Task {task_id}: No image path found in field '{data_field}'")
                continue

            try:
                # Process Image (Sync wrapper around async logic)
                generated_text = asyncio.run(self._generate_caption(image_path))
                
                # Construct Result
                # Using the standard "Region" format to force UI rendering
                # x,y,width,height = 0,0,100,100 covers the whole image
                result_item = {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "textarea",
                    "value": {
                        "text": [generated_text],
                        "x": 0,
                        "y": 0,
                        "width": 100,
                        "height": 100,
                        "rotation": 0
                    }
                }
                
                predictions.append({
                    "result": [result_item],
                    "score": 1.0,
                    "model_version": self.model_version
                })
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                # Return empty prediction on failure to keep batch size matching? 
                # SDK handles exceptions, but best to log.
        
        return predictions

    async def _generate_caption(self, path: str) -> str:
        """
        Fetch image and call Kimi API
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 1. Handle Local File (Direct Read - Best for Intranet)
        # Check both raw path and /data prefixed path (container mapping)
        candidates = [path]
        if path.startswith("/data/"):
             candidates.append(path.replace("/data/", "/data/media/", 1))
        
        image_base64 = None
        
        for p in candidates:
            if os.path.exists(p):
                logger.info(f"Reading local image: {p}")
                with open(p, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
                break
        
        # 2. Handle HTTP Fetch (Fallback)
        if not image_base64:
            target_url = path
            
            # Handle local paths that need to be full URLs
            if path.startswith(("/data/", "/files/")):
                if self.label_studio_host:
                    target_url = f"{self.label_studio_host.rstrip('/')}{path}"
            
            # Handle "localhost" URLs that need to be rewritten for Docker
            elif path.startswith(("http://localhost", "http://127.0.0.1")):
                # Replace localhost with host.docker.internal or configured host
                replacement_host = "host.docker.internal"
                if self.label_studio_host:
                    # Extract hostname from configured LABEL_STUDIO_HOST
                    from urllib.parse import urlparse
                    replacement_host = urlparse(self.label_studio_host).netloc
                
                target_url = path.replace("localhost", replacement_host).replace("127.0.0.1", replacement_host)
            
            logger.info(f"Downloading image from: {target_url}")
            
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.label_studio_api_key:
                    headers["Authorization"] = f"Token {self.label_studio_api_key}"
                
                try:
                    resp = await client.get(target_url, headers=headers, timeout=30.0)
                    resp.raise_for_status()
                    image_base64 = base64.b64encode(resp.content).decode('utf-8')
                except Exception as e:
                    logger.error(f"Download failed for {target_url}: {e}")
                    raise

        if not image_base64:
            raise ValueError(f"Could not load image data for: {path}")

        # Construct Payload
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": "请描述图片内容。"}
        ]
        
        messages.append({"role": "user", "content": user_content})
        
        # sync via asyncio.run, so it blocks anyway. That's acceptable for now.)
        
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

# App Entrypoint
from label_studio_ml.api import init_app
app = init_app(model_class=KimiBackend)