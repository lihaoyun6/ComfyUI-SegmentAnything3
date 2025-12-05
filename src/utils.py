import torch
import json

def get_device_list():
	devs = ["auto"]
	try:
		if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
			devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
	except Exception:
		pass
	try:
		if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.backends.mps.is_available():
			devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
	except Exception:
		pass
	return devs

def get_points_and_labels(positive_coords, negative_coords):
	combined_points_list = []
	combined_point_labels_list = []
	
	if positive_coords and isinstance(positive_coords, str):
		try:
			points_list = json.loads(positive_coords)
		except (json.JSONDecodeError, TypeError) as e:
			raise ValueError(f"[SAM3] Could not parse positive_coords!\nError: {e}")
			
		p_points_coords = [[p["x"], p["y"]] for p in points_list]
		combined_points_list.extend(p_points_coords)
		combined_point_labels_list.extend([1] * len(p_points_coords))
		
	if negative_coords and isinstance(negative_coords, str):
		try:
			points_list = json.loads(negative_coords)
		except (json.JSONDecodeError, TypeError) as e:
			raise ValueError(f"[SAM3] Could not parse negative_coords!\nError: {e}")
			
		n_points_coords = [[p["x"], p["y"]] for p in points_list]
		combined_points_list.extend(n_points_coords)
		combined_point_labels_list.extend([0] * len(n_points_coords))
		
	final_input_points = [[combined_points_list]] if combined_points_list else None
	final_input_point_labels = [[combined_point_labels_list]] if combined_point_labels_list else None
	
	return (final_input_points, final_input_point_labels)

def get_boxes_and_labels(bbox):
	combined_boxes_list = []
	combined_box_labels_list = []
	
	if bbox is not None:
		clean_boxes = [list(b) if isinstance(b, (tuple, list)) else b for b in bbox]
		combined_boxes_list.extend(clean_boxes)
		combined_box_labels_list.extend([1] * len(clean_boxes))
		
	final_input_boxes = [combined_boxes_list] if combined_boxes_list else None
	final_input_box_labels = [combined_box_labels_list] if combined_box_labels_list else None
	
	return (final_input_boxes, final_input_box_labels)