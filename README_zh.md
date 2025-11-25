# ComfyUI-SegmentAnything3
在 ComfyUI 中轻松运行 Segment Anything  Model 3  
**[[📃English](./README.md)]**      

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-SegmentAnything3.git
python -m pip install -r ComfyUI-SegmentAnything3/requirements.txt
```  

#### 模型下载:
- 所有所需模型将会被自动下载到 `ComfyUI/models/sams/[model_id]` 文件夹中, 请保持网络通畅. 

## 请注意
安装依赖项后请检查 `transformers` 版本是否大于等于 `5.0.0dev`.  
若未安装成功, 可能需要先卸载本地已安装的 `transformers` 并重新安装依赖项.  

## 致谢  
- [SAM3](https://github.com/facebookresearch/sam3) @facebook
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
