from pathlib import Path

import pandas as pd


def explore_dataset(dataset_path: Path) -> pd.DataFrame:
    bbox_path = dataset_path / "bbox"
    images_path = dataset_path / "img"

    images = images_path.rglob("*.png")
    bboxes = bbox_path.rglob("*.csv")

    data = {
        "image_path": [],
        "bbox_path": [],
        "city": [],
        "n_objects": [],
    }

    for im_path, bbox_path in zip(images, bboxes):
        data["image_path"].append( str(im_path) )
        data["bbox_path"].append( str(bbox_path) )
        data["city"].append( im_path.parent.name )
        data["n_objects"].append( len(pd.read_csv( str(bbox_path) )) )

    final_df = pd.DataFrame(data)
    return final_df


if __name__ == "__main__":
    df = explore_dataset(Path("../data_det_public/data_det_public"))
    df.to_csv("metadata_tables/full_data.csv", index=False)
