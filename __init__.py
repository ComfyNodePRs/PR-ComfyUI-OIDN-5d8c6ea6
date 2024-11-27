from .nodes import denoiseOIDN

# ComfyUI: Node class mappings
NODE_CLASS_MAPPINGS = {"OIDN Denoise": denoiseOIDN}

NODE_DISPLAY_NAME_MAPPINGS = {"denoiseOIDN": "OIDN  denoise"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
