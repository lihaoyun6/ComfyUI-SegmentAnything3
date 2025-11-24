from .utils import get_device_list, get_points_and_labels, get_boxes_and_labels
from .patch import cqdm, propagate_video_patched, propagate_video_tracker_patched

from modelscope.hub.snapshot_download import snapshot_download
from transformers import (
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor
)
from accelerate import Accelerator
from types import MethodType
from PIL import Image

import comfy.model_management as mm
import numpy as np
import folder_paths
import torch
import os
import gc

model_folder = os.path.join(folder_paths.models_dir, "sams")
model_list = ["facebook/sam3"]
device_list = get_device_list()

class SAM3Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (model_list, {"default": "facebook/sam3"}),
                "device": (device_list, {
                    "default": device_list[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "segmentor": (["image", "video"], {
                    "default": "image",
                    "tooltip": "Choose between image or video segmentation mode."
                }),
                "prompt": ("STRING", {
                    "default":"",
                    "multiline": True,
                    "placeholder": "Any other conditional input will be ignored when providing text prompts."
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10 ** 5,
                    "step": 1,
                    "tooltip": "Frame where initial prompt is applied."
                }),
                "start_frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10 ** 5,
                    "step": 1,
                    "tooltip": "Frame index to start propagation from."
                }),
                "max_frames_to_track": ("INT", {
                    "default": -1,
                    "min": -1,
                    "step": 1,
                    "tooltip": "Max frames to process (-1 for all)."
                }),
                "object_id": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Unique ID for multi-object tracking (-1 for all)."
                }),
                "score_threshold_detection": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum confidence score for detections (0.0-1.0). Lower = more detections but more false positives."
                }),
                "new_det_thresh": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum confidence for new object tracking (0.0-1.0). Higher = only track high-confidence objects."
                }),
                "reverse_propagation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to propagate in reverse."
                }),
            },
            "optional": {
                "exatr_config": ("exatr_config",),
                "positive_coords": ("STRING", {
                    "tooltip": "Positive click coordinates as JSON: '[{\"x\": 63, \"y\": 782}]'",
                    "forceInput": True
                }),
                "negative_coords": ("STRING", {
                    "tooltip": "Negative click coordinates as JSON: '[{\"x\": 100, \"y\": 200}]'",
                    "forceInput": True
                }),
                "bbox": ("BBOX", {
                    "tooltip": "Bounding box as (x_min, y_min, x_max, y_max) or (x, y, width, height) tuple. Compatible with KJNodes Points Editor bbox output."
                }),
                "mask": ("MASK", {
                    "tooltip": "The mask(s) to add to the frame."
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "main"
    CATEGORY = "SAM3"
    
    def load_model(self, key, model, process, patch, model_path, device, dtype, exatr_config):
        if not hasattr(self, "_model") or getattr(self, "_model_key", None) != key:
            if hasattr(self, "_model"):
                self._model.to("cpu")
                try:
                    del self._model
                    del self.processor
                    del self.inference_session
                except:
                    pass
                gc.collect()
                mm.soft_empty_cache()
                
            self._model_key = key
            self._model = model.from_pretrained(model_path).to(device, dtype=dtype)
            self.processor = process.from_pretrained(model_path)
            if patch is not None:
                self._model.propagate_in_video_iterator = MethodType(patch, self._model)
            
        self._model.to(device)
        if exatr_config is not None:
            for key, value in exatr_config.items():
                if hasattr(self._model.config, key):
                    print(f"[SAM3] Set model.config.{key} to {value}")
                    setattr(self._model.config, key, value)
    
    def main(self, images, model, device, precision, segmentor, prompt, frame_index, start_frame_index, max_frames_to_track, object_id, score_threshold_detection, new_det_thresh, reverse_propagation, exatr_config=None, positive_coords="", negative_coords="", bbox=None, mask=None):
        
        num_frames, height, width, channels = images.shape
        final_masks_tensor = torch.zeros((num_frames, height, width), dtype=torch.float32, device="cpu")
        exatr_config = {} if exatr_config is None else exatr_config
        exatr_config["score_threshold_detection"] = score_threshold_detection
        exatr_config["new_det_thresh"] = new_det_thresh
        
        _device = Accelerator().device if device == "auto" else torch.device(device)
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.float16
            
        model_path = os.path.join(model_folder, model.replace("/", "-"))
        if not os.path.exists(model_path):
            print(f"[SAM3] Local file doesn't exist. Downloading {model}...")
            _save_dir = snapshot_download(model_id=model, local_dir=model_path)
            
        if segmentor == "video":
            # Convert video frames
            video_frames = (np.clip(255.0 * images.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            
            if prompt and prompt.strip():
                # Load model
                k = f"{model}.video_pcs.{str(_device)}.{precision}"
                self.load_model(k, Sam3VideoModel, Sam3VideoProcessor, propagate_video_patched, model_path, _device, dtype, exatr_config)
                
                # Initialize video inference session
                inference_session = self.processor.init_video_session(
                    video=video_frames,
                    inference_device=_device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=dtype,
                )
                
                # Add text prompt to detect and track objects
                inference_session = self.processor.add_text_prompt(inference_session=inference_session, text=prompt.strip())
                
                print(f"[SAM3] Starting video propagation...")
                try:
                    # Process all frames in the video
                    with torch.no_grad(), torch.autocast(device_type=_device.type, dtype=dtype):
                        for model_outputs in self._model.propagate_in_video_iterator(
                            inference_session = inference_session, start_frame_idx = start_frame_index,
                            max_frame_num_to_track = max_frames_to_track if max_frames_to_track != -1 else None,
                            reverse = reverse_propagation
                        ):
                            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
                            frame_masks = processed_outputs.get('masks')
                            object_ids = processed_outputs.get('object_ids')
                    
                            if object_id > -1:
                                matches = (object_ids == object_id).nonzero(as_tuple=True)[0]
                                if matches.numel() > 0:
                                    target_idx = matches[0]
                                    single_frame_mask = frame_masks[target_idx]
                                else:
                                    single_frame_mask = torch.zeros((height, width), dtype=torch.bool, device="cpu")
                            else:
                                single_frame_mask = frame_masks.any(dim=0) 
                    
                            final_masks_tensor[model_outputs.frame_idx] = single_frame_mask.float().cpu()
                        
                        print(f"[SAM3] Processed {final_masks_tensor.shape[0]} frames.")
                finally:
                    del inference_session
                    gc.collect()
                    self._model.to("cpu")
                    mm.soft_empty_cache()
                        
            else:
                # Load model
                k = f"{model}.video_pvs.{str(_device)}.{precision}"
                self.load_model(k, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, propagate_video_tracker_patched, model_path, _device, dtype, exatr_config)
                
                # Get bboxes and points
                boxes, box_labels = get_boxes_and_labels(bbox)
                points, point_labels = get_points_and_labels(positive_coords, negative_coords)
                if boxes is None and points is None:
                    raise ValueError(
                        "[SAM3] No prompt provided!\n"
                        "Please provide at least one of:\n"
                        "  • Text prompt (e.g., 'person', 'car', etc.)\n"
                        "  • Points or Bounding boxes\n"
                        "\n"
                        "Empty prompts cannot be used for tracking!"
                    )
                
                inference_session = self.processor.init_video_session(
                    video=video_frames,
                    inference_device=_device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=dtype,
                )
                
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=frame_index,
                    obj_ids=object_id,
                    input_points=points,
                    input_labels=point_labels,
                    input_boxes=boxes,
                    input_masks=mask,
                )
                
                outputs = self._model( inference_session=inference_session, frame_idx=frame_index)
                
                video_res_masks = self.processor.post_process_masks(
                    [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
                )[0]
                
                print(f"[SAM3] Starting video propagation...")
                try:
                    # Process all frames in the video
                    with torch.no_grad(), torch.autocast(device_type=_device.type, dtype=dtype):
                        for sam3_tracker_video_output in self._model.propagate_in_video_iterator(
                            inference_session = inference_session, start_frame_idx = start_frame_index,
                            max_frame_num_to_track = max_frames_to_track if max_frames_to_track != -1 else None,
                            reverse = reverse_propagation
                        ):
                            video_res_masks = self.processor.post_process_masks(
                                [sam3_tracker_video_output.pred_masks],
                                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                                binarize=False
                            )[0]
                            active_obj_ids = inference_session.obj_ids
                            if object_id > -1 and active_obj_ids and object_id in active_obj_ids:
                                try:
                                    target_idx = active_obj_ids.index(object_id)
                                    single_frame_mask = video_res_masks[target_idx].squeeze()
                                except (ValueError, IndexError):
                                    pass
                            else:
                                single_frame_mask = video_res_masks.any(dim=0).squeeze()
                        final_masks_tensor[sam3_tracker_video_output.frame_idx] = single_frame_mask.float().cpu()
                    
                        print(f"[SAM3] Processed {final_masks_tensor.shape[0]} frames.")
                finally:
                    del inference_session
                    gc.collect()
                    self._model.to("cpu")
                    mm.soft_empty_cache()
        else:
            if prompt and prompt.strip():
                k = f"{model}.image_pcs.{str(_device)}.{precision}"
                self.load_model(k, Sam3Model, Sam3Processor, None, model_path, _device, dtype, exatr_config)
                
                image = Image.fromarray(np.clip(255.0 * images[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(_device)
                
                try:
                    with torch.no_grad(), torch.autocast(device_type=_device.type, dtype=dtype):
                        outputs = self._model(**inputs)
                    
                        results = self.processor.post_process_instance_segmentation(
                            outputs,
                            threshold=score_threshold_detection,
                            mask_threshold=0.5,
                            target_sizes=inputs.get("original_sizes").tolist()
                        )[0]
                    
                        masks = results['masks']
                        print(f"[SAM3] Found {len(masks)} objects")
                        if object_id > -1:
                            if object_id + 1 <= len(masks):
                                return (masks[object_id].cpu(),)
                        else:
                            return (masks.any(dim=0).cpu().squeeze(),)
                        return (torch.zeros((height, width), dtype=torch.float32, device="cpu"),)
                finally:
                    del inputs, outputs, results
                    gc.collect()
                    self._model.to("cpu")
                    mm.soft_empty_cache()
            else:
                k = f"{model}.image_pvs.{str(_device)}.{precision}"
                self.load_model(k, Sam3TrackerModel, Sam3TrackerProcessor, None, model_path, _device, dtype, exatr_config)
                
                image = Image.fromarray(np.clip(255.0 * images[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                boxes, box_labels = get_boxes_and_labels(bbox)
                points, point_labels = get_points_and_labels(positive_coords, negative_coords)
                
                inputs = self.processor(
                    images=image,
                    input_points=points,
                    input_labels=point_labels,
                    input_boxes=boxes,
                    return_tensors="pt"
                ).to(_device)
                
                try:
                    with torch.no_grad(), torch.autocast(device_type=_device.type, dtype=dtype):
                        outputs = self._model(**inputs)
                        
                        results = self.processor.post_process_masks(
                            outputs.pred_masks.cpu(),
                            inputs["original_sizes"]
                        )[0]
                        
                        print(f"[SAM3] Generated {results.shape[1]} masks with shape {results.shape}")
                        if results.shape[1] > 0:
                            all_masks = results.view(-1, results.shape[-2], results.shape[-1])
                            merged_mask = torch.any(all_masks, dim=0)
                            final_mask = merged_mask.unsqueeze(0).float().cpu()
                            return (final_mask, )
                        return (torch.zeros((height, width), dtype=torch.float32, device="cpu"),)
                finally:
                    del inputs, outputs, results
                    gc.collect()
                    self._model.to("cpu")
                    mm.soft_empty_cache()
        
        return (final_masks_tensor,)

class SAM3ExatrConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fill_hole_area": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Maximum area (in pixels) of holes to fill in masks. 0 disables hole filling. Useful for cleaning up mask interiors."
                }),
                "assoc_iou_thresh": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IOU threshold for detection-to-track association (0.0-1.0). Lower = more lenient matching for maintaining track continuity."
                }),
                "det_nms_thresh": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IOU threshold for Non-Maximum Suppression (0.0-1.0). Lower = more aggressive duplicate removal. 0.0 disables NMS."
                }),
                "hotstart_unmatch_thresh": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": "Number of unmatched frames before removing a track (hotstart heuristic). Higher = more tolerant of temporary occlusions. Set to 999 to effectively disable."
                }),
                "hotstart_dup_thresh": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": "Number of overlapping frames before removing duplicate tracks. Higher = more tolerant of temporary overlaps. Set to 999 to effectively disable."
                }),
                "init_trk_keep_alive": ("INT", {
                    "default": 0,
                    "min": -10,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Initial keep-alive counter for new tracks. Higher = tracks survive longer without matching detections. Recommended: 5-20 for robust tracking."
                }),
                "hotstart_delay": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Delay (in frames) before applying hotstart removal heuristics. Useful to let tracks stabilize in early frames. Set to 999 to disable hotstart entirely."
                }),
                "decrease_keep_alive_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether empty masks (zero area predictions) decrease the keep-alive counter. Disable for more lenient tracking."
                }),
                "suppress_unmatched_globally": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to suppress tracks with keep_alive <= 0 globally (True) or only during hotstart period (False). CRITICAL: Set to True to actually remove dead tracks!"
                }),
            }
        }
    
    RETURN_TYPES = ("exatr_config",)
    FUNCTION = "main"
    CATEGORY = "SAM3"
    
    def main(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3ExatrConfig": SAM3ExatrConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3ExatrConfig": "SAM3 ExatrConfig",
}