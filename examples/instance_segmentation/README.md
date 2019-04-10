# Instance Segmentation Example

## Annotation

- Instance label string has the format of `.*-[0-9]*$`.
- The `instance_label_auto_increment` feature can be turned off in the config file.

```bash
labelme data_annotated --labels labels.txt --nodata
```

![](.readme/annotation.jpg)

## Convert to VOC-format Dataset

```bash
# It generates:
#   - data_dataset_voc/JPEGImages
#   - data_dataset_voc/SegmentationClass
#   - data_dataset_voc/SegmentationClassVisualization
#   - data_dataset_voc/SegmentationObject
#   - data_dataset_voc/SegmentationObjectVisualization
./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
```

<img src="data_dataset_voc/JPEGImages/2011_000003.jpg" width="33%" /> <img src="data_dataset_voc/SegmentationClassVisualization/2011_000003.jpg" width="33%" /> <img src="data_dataset_voc/SegmentationObjectVisualization/2011_000003.jpg" width="33%" />  
Fig 1. JPEG image (left), JPEG class label visualization (center), JPEG instance label visualization (right)


Note that the label file contains only very low label values (ex. `0, 4, 14`), and
`255` indicates the `__ignore__` label value (`-1` in the npy file).  
You can see the label PNG file by following.

```bash
labelme_draw_label_png data_dataset_voc/SegmentationClassPNG/2011_000003.png   # left
labelme_draw_label_png data_dataset_voc/SegmentationObjectPNG/2011_000003.png  # right
```

<img src=".readme/draw_label_png_class.jpg" width="33%" /> <img src=".readme/draw_label_png_object.jpg" width="33%" />


## Convert to COCO-format Dataset

```bash
# It generates:
#   - data_dataset_coco/images
#   - data_dataset_coco/annotations/instances_split.json
./labelme2coco.py data_annotated data_dataset_coco --labels labels.txt --ann_split split.txt
```
