#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

import labelme

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument('--ann_split', help='annotation split file', required=True)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print('Creating dataset:', args.output_dir)

    image_dir = osp.join(args.output_dir, 'images')
    os.makedirs(image_dir)
    annotation_dir = osp.join(args.output_dir, 'annotations')
    os.makedirs(annotation_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    data_split = osp.splitext(osp.basename(args.ann_split))[0]
    out_ann_file = osp.join(annotation_dir, 'instances_%s.json' % (data_split))
    for image_id, in_ann_file in enumerate(open(args.ann_split).readlines()):
        label_file = osp.join(args.input_dir, in_ann_file.strip())
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file))
        out_img_file = osp.splitext(osp.basename(label_file))[0] + '.jpg'
        PIL.Image.fromarray(img).save(osp.join(image_dir, out_img_file))

        data['images'].append(dict(
            license=0,
            url=None,
            file_name=out_img_file,
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if label in masks:
                masks[label] = masks[label] | mask
            else:
                masks[label] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[label].append(points)

        for label, mask in masks.items():
            cls_name = label.split('-')[0]
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[label],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
