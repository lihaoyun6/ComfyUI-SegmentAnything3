from transformers import Sam3TrackerVideoInferenceSession
from typing import Optional
import comfy.utils
import torch

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")
                    
        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            val = next(self.iterable)
            if self.pbar:
                self.pbar.update(1)
            return val
        except StopIteration:
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

@torch.inference_mode()
def propagate_video_patched(self, inference_session, start_frame_idx=0, max_frame_num_to_track=None, reverse=False):
    processing_order, end_frame_idx = self._get_processing_order(
        inference_session,
        start_frame_idx,
        max_frame_num_to_track,
        reverse=reverse,
    )
    
    hotstart_buffer = []
    
    for frame_idx in cqdm(processing_order):
        out = self(inference_session=inference_session, frame_idx=frame_idx, reverse=reverse)
        
        if self.config.hotstart_delay > 0:
            hotstart_buffer.append(out)
            inference_session.hotstart_removed_obj_ids.update(out.removed_obj_ids)
            
            if frame_idx == end_frame_idx:
                yield_list = hotstart_buffer
                hotstart_buffer = []
            elif len(hotstart_buffer) >= self.config.hotstart_delay:
                yield_list = hotstart_buffer[:1]
                hotstart_buffer = hotstart_buffer[1:]
            else:
                yield_list = []
        else:
            yield_list = [out]
            
        for yield_out in yield_list:
            yield yield_out
            
@torch.inference_mode()
def propagate_video_tracker_patched(
    self,
    inference_session: Sam3TrackerVideoInferenceSession,
    start_frame_idx: Optional[int] = None,
    max_frame_num_to_track: Optional[int] = None,
    reverse: bool = False,
):
    num_frames = inference_session.num_frames
    
    if start_frame_idx is None:
        frames_with_inputs = [
            frame_idx
            for obj_output_dict in inference_session.output_dict_per_obj.values()
            for frame_idx in obj_output_dict["cond_frame_outputs"]
        ]
        if not frames_with_inputs:
            raise ValueError(
                "Cannot determine the starting frame index; please specify it manually, or run inference on a frame with inputs first."
            )
        start_frame_idx = min(frames_with_inputs)
        
    if max_frame_num_to_track is None:
        max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
            
        for frame_idx in cqdm(processing_order):
            sam3_tracker_video_output = self(inference_session, frame_idx=frame_idx, reverse=reverse)
            yield sam3_tracker_video_output