annotations.csv: CSV file of annotations. Each row denotes a unique person, identified by person_id. Person_id is shared across coco_masks.json, coco_boxes.json and annotations.csv.  Each row in annotations.csv contains the demographic and additional attributes for a person annotated in FACET. It also contains class1, class2 and a bounding box for the person. Class2 is an additional class that describes the person, or None.

coco_boxes.json: COCO-style JSON file containing bounding boxes for people in FACET. The id field for each annotation is the same as person_id. Category_id corresponds to the primary class for the person.

coco_masks.json:  COCO-style JSON file containing Segment Anything Model (SAM) generated masks for people in FACET. Each annotation has an additional person_id field, corresponding to the person_id in annotations.csv and coco_boxes.json. Categories for masks are one of [person, clothing, hair]. Masks are non-exhaustive.
