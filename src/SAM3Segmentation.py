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

class SAM3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_list, {"default": "facebook/sam3"}),
                "device": (device_list, {
                    "default": device_list[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "segmentor": (["image", "image (text prompt)", "video", "video (text prompt)"], {
                    "default": "image (text prompt)",
                    "tooltip": "Choose between image or video segmentation mode.\nSegmentors with (text prompt) suffix only support text prompts, while others don't."
                })
            }
        }
    
    RETURN_TYPES = ("sam3_model",)
    FUNCTION = "main"
    CATEGORY = "SAM3"
    
    def main(self, model, device, precision, segmentor):
        if hasattr(self, "_model"):
            self._model.to("cpu")
            try:
                del self._model
                del self.processor
            except:
                pass
            gc.collect()
            mm.soft_empty_cache()
            
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
            print(f"[SAM3] Local file doesn't exist. Downloading [{model}]...")
            _save_dir = snapshot_download(model_id=model, ignore_patterns='sam3.pt', local_dir=model_path)
            
        patch = None
        if segmentor == "video (text prompt)":
            mod = Sam3VideoModel
            proc = Sam3VideoProcessor
            patch = propagate_video_patched
        elif segmentor == "video":
            mod = Sam3TrackerVideoModel
            proc = Sam3TrackerVideoProcessor
            patch = propagate_video_tracker_patched
        elif segmentor == "image (text prompt)":
            mod = Sam3Model
            proc = Sam3Processor
        else:
            mod = Sam3TrackerModel
            proc = Sam3TrackerProcessor
            
        self._model = mod.from_pretrained(model_path).to(_device, dtype=dtype)
        self.processor = proc.from_pretrained(model_path)
        if patch is not None:
            self._model.propagate_in_video_iterator = MethodType(patch, self._model)
            
        model_dict = {
            "model": self._model,
            "processor": self.processor,
            "dtype": dtype,
            "device": _device,
            "segmentor": segmentor,
        }
        
        return (model_dict,)

class SAM3Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("sam3_model",),
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default":"",
                    "multiline": True,
                    "placeholder": "text prompts"
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
                    "default": -1,
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
    
    def set_exatr_config(self, model, exatr_config):
        if exatr_config is not None:
            for key, value in exatr_config.items():
                if hasattr(model, key):
                    print(f"[SAM3] Set model.{key} = {value}")
                    setattr(model, key, value)
    
    def main(self, sam3_model, images, prompt, frame_index, start_frame_index, max_frames_to_track, object_id, score_threshold_detection, new_det_thresh, reverse_propagation, exatr_config=None, positive_coords="", negative_coords="", bbox=None, mask=None):
        
        model = sam3_model["model"]
        processor = sam3_model["processor"]
        segmentor = sam3_model["segmentor"]
        dtype = sam3_model["dtype"]
        device = sam3_model["device"]
        
        model.to(device)
        self.set_exatr_config(model, exatr_config)
        
        num_frames, height, width, channels = images.shape
        final_masks_tensor = torch.zeros((num_frames, height, width), dtype=torch.float32, device="cpu")
        
        exatr_config = {} if exatr_config is None else exatr_config
        exatr_config["score_threshold_detection"] = score_threshold_detection
        exatr_config["new_det_thresh"] = new_det_thresh
        
        try:
            if "video" in segmentor:
                # Convert video frames
                video_frames = (np.clip(255.0 * images.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                
                if "(text prompt)" in segmentor:
                    if not prompt or not prompt.strip():
                        raise ValueError(
                            "[SAM3] No text prompt provided!\n"
                            "Please provide any text prompts (e.g., 'person', 'car', etc.)\n"
                            "\nEmpty prompts cannot be used for tracking!"
                        )
                        
                    # Initialize video inference session
                    inference_session = processor.init_video_session(
                        video=video_frames,
                        inference_device=device,
                        processing_device="cpu",
                        video_storage_device="cpu",
                        dtype=dtype,
                    )
                    
                    # Add text prompt to detect and track objects
                    inference_session = processor.add_text_prompt(inference_session=inference_session, text=prompt.strip())
                    
                    print(f"[SAM3] Starting video propagation...")
                    # Process all frames in the video
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                        for model_outputs in model.propagate_in_video_iterator(
                            inference_session = inference_session, start_frame_idx = start_frame_index,
                            max_frame_num_to_track = max_frames_to_track if max_frames_to_track != -1 else None,
                            reverse = reverse_propagation
                        ):
                            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
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
                        del inference_session
                else:
                    # Get bboxes and points
                    boxes, box_labels = get_boxes_and_labels(bbox)
                    points, point_labels = get_points_and_labels(positive_coords, negative_coords)
                    
                    if boxes is None and points is None and mask is None:
                        raise ValueError(
                            "[SAM3] No prompt provided!\n"
                            "Please provide at least one of Points or Bounding boxes or Masks\n"
                            "\nEmpty prompts cannot be used for tracking!"
                        )
                        
                    inference_session = processor.init_video_session(
                        video=video_frames,
                        inference_device=device,
                        processing_device="cpu",
                        video_storage_device="cpu",
                        dtype=dtype,
                    )
                    
                    processor.add_inputs_to_inference_session(
                        inference_session=inference_session,
                        frame_idx=frame_index,
                        obj_ids=object_id,
                        input_points=points,
                        input_labels=point_labels,
                        input_boxes=boxes,
                        input_masks=mask,
                    )
                    
                    outputs = model( inference_session=inference_session, frame_idx=frame_index)
                    
                    video_res_masks = processor.post_process_masks(
                        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
                    )[0]
                    
                    print(f"[SAM3] Starting video propagation...")
                    
                    # Process all frames in the video
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                        for sam3_tracker_video_output in model.propagate_in_video_iterator(
                            inference_session = inference_session, start_frame_idx = start_frame_index,
                            max_frame_num_to_track = max_frames_to_track if max_frames_to_track != -1 else None,
                            reverse = reverse_propagation
                        ):
                            video_res_masks = processor.post_process_masks(
                                [sam3_tracker_video_output.pred_masks],
                                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                                binarize=False
                            )[0]
                            active_obj_ids = inference_session.obj_ids
                            #print(f"frame [{sam3_tracker_video_output.frame_idx}], objects [{active_obj_ids}]")
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
                        del inference_session
            else:
                if "(text prompt)" in segmentor:
                    if not prompt or not prompt.strip():
                        raise ValueError(
                            "[SAM3] No text prompt provided!\n"
                            "Please provide any text prompts (e.g., 'person', 'car', etc.)\n"
                        )
                        
                    image = Image.fromarray(np.clip(255.0 * images[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                        outputs = model(**inputs)
                        
                        results = processor.post_process_instance_segmentation(
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
                else:
                    boxes, box_labels = get_boxes_and_labels(bbox)
                    points, point_labels = get_points_and_labels(positive_coords, negative_coords)
                    
                    if boxes is None and points is None:
                        raise ValueError(
                            "[SAM3] No prompt provided!\n"
                            "Please provide at least one of Points or Bounding boxes\n"
                            "\nEmpty prompts cannot be used for tracking!"
                        )
                        
                    image = Image.fromarray(np.clip(255.0 * images[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    
                    inputs = processor(
                        images=image,
                        input_points=points,
                        input_labels=point_labels,
                        input_boxes=boxes,
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                        outputs = model(**inputs)
                        
                        results = processor.post_process_masks(
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
                    
            return (final_masks_tensor,)
        finally:
            model.to("cpu")
            gc.collect()
            mm.soft_empty_cache()

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
    "SAM3ModelLoader": SAM3ModelLoader,
    "SAM3Segmentation": SAM3Segmentation,
    "SAM3ExatrConfig": SAM3ExatrConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3ModelLoader": "SAM3 Model Loader",
    "SAM3Segmentation": "SAM3 Segmentation",
    "SAM3ExatrConfig": "SAM3 ExatrConfig",
}