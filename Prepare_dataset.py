import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


def extract_data_from_xml(root_dir):
    """
    Trích xuất dữ liệu từ file words.xml trong bộ IC03

    Hàm này dùng để trích các thông tin từ file .xml bao gồm:
    image paths, image sizes, image labels và bboxes

    Parameters:
        root_dir (str): Đường dẫn đến thư mục root của dataset

    Returns:
        tuple: Chứa 4 lists lần lượt là: image paths, image sizes, image labels, và bboxes.
    """

    # Tạo path đến file words.xml
    xml_path = os.path.join(root_dir, 'words.xml')
    # Parse file xml
    tree = ET.parse(xml_path)
    # Đọc thẻ root của file
    root = tree.getroot()

    # Khai báo các list rỗng để lưu dữ liệu
    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    # Duyệt qua từng thẻ ảnh <image>
    for img in root:
        # Khai báo các list rỗng chứa bboxes và labels của ảnh đang xét
        bbs_of_img = []
        labels_of_img = []

        # Duyệt qua từng thẻ boundingbox
        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                # Bỏ qua trường hợp label không phải kí tự alphabet hoặc number
                if not bb[0].text.isalnum():
                    continue

                # Bỏ qua trường hợp label là chữ 'é' hoặc ñ'
                if 'é' in bb[0].text.lower() or 'ñ' in bb[0].text.lower():
                    continue

                # Đưa thông tin tọa độ bbox vào list bbs_of_img
                # Format bbox: (xmin, ymin, bbox_width, bbox_height)
                bbs_of_img.append(
                    [
                        float(bb.attrib['x']),
                        float(bb.attrib['y']),
                        float(bb.attrib['width']),
                        float(bb.attrib['height'])
                    ]
                )
                # Đưa label vào list labels_of_img (đã chuyển chữ viết thường)
                labels_of_img.append(bb[0].text.lower())

        # Đưa thông tin path ảnh đang xét vào list img_paths
        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        # Đưa thông tin độ phân giải ảnh vào list img_sizes
        img_sizes.append((int(img[1].attrib['x']), int(img[1].attrib['y'])))
        # Đưa list bbox vào list bboxes
        bboxes.append(bbs_of_img)
        # Đưa list labels vào list img_labels
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes


def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir):
    """
    Xây dựng thư mục chứa dữ liệu cho Text Recognition.

    Hàm sẽ tạo một thư mục save_dir, lưu các ảnh cắt từ tọa độ bbox.
    Label sẽ được lưu riêng vào file labels.txt.

    Parameters:
        img_paths (list): Danh sách các path ảnh.
        img_labels (list): Danh sách chứa danh sách labels của các ảnh.
        bboxes (list): Danh sách chứa danh sách bounding box của các ảnh.
        save_dir (str): Đường dẫn đến thư mục chứa dữ liệu.
    """
    # Tạo tự động thư mục chứa dữ liệu
    os.makedirs(save_dir, exist_ok=True)

    # Khai báo biến đếm và danh sách rỗng chứa labels
    count = 0
    labels = []

    # Duyệt qua từng cặp (đường dẫn ảnh, list label, list bbox)
    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        # Đọc ảnh
        img = Image.open(img_path)

        # Duyệt qua từng cặp label và bbox
        for label, bb in zip(img_label, bbs):
            # Cắt ảnh theo bbox
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

            # Bỏ qua trường hợp 90% nội dung ảnh cắt là màu trắng hoặc đen.
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue

            # Bỏ qua trường hợp ảnh cắt có width < 10 hoặc heigh < 10
            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue

            # Bỏ qua trường hợp số kí tự của label < 3
            if len(label) < 3:
                continue

            # Tạo tên cho file ảnh đã cắt và lưu vào save_dir
            filename = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, filename))

            new_img_path = os.path.join(save_dir, filename)

            # Đưa format label mới vào list labels
            # Format: img_path\tlabel
            label = new_img_path + '\t' + label

            labels.append(label)  # Append label to the list

            count += 1

    print(f"Created {count} images")

    # Đưa list labels vào file labels.txt
    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")


def convert_to_yolov8_format(image_paths, image_sizes, bounding_boxes):
    """
    Thực hiện normalize bounding box

    Parameters:
        image_paths (list): Danh sách chứa các path ảnh.
        image_sizes (list): Danh sách chứa độ phân giải ảnh.
        bounding_boxes (list): Danh sách chứa danh sách bounding box.

    Returns:
        yolov8_data (list): Danh sách gồm (image_path, image_size, bboxes)
    """
    # Khai báo list rỗng để chứa kết quả
    yolov8_data = []

    # Duyệt qua từng bộ path, resolution và bboxes ảnh.
    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size

        # Khai báo list rỗng để chứa label (format mới)
        yolov8_labels = []

        # Duyệt qua từng bbox
        for bbox in bboxes:
            x, y, w, h = bbox

            # Thực hiện normalize bbox
            # Format bbox hiện tại: (x_min, y_min, width, height)
            # Format bbox của yolo: (x_center, y_center, width, height)
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Mã class, mặc định = 0 vì chỉ có 1 class 'text'
            class_id = 0

            # Đổi format label
            # Format: "class_id x_center y_center width height"
            yolov8_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolov8_labels.append(yolov8_label)

        yolov8_data.append((image_path, yolov8_labels))

    return yolov8_data


def save_data(data, save_dir):
    """
    Xây dựng thư mục chứa dữ liệu theo format YOLO

    Parameters:
        data (list): Danh sách chứa thông tin label ảnh.
        src_img_dir (str): Path đến thư mục dữ liệu gốc.
        save_dir (str): Path đến thư mục dữ liệu mới.
    """
    # Tạo thư mục dữ liệu mới nếu chưa có
    os.makedirs(save_dir, exist_ok=True)

    # Tạo thư mục 'images' và 'labels'
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)

    # Duyệt qua từng bộ path, bbox, label ảnh
    for image_path, yolov8_labels in data:
        # Copy ảnh từ thư mục gốc sang thư mục 'images'
        shutil.copy(
            image_path,
            os.path.join(save_dir, 'images')
        )

        # Ghi nội dung label vào file image_name.txt ở thư mục 'labels'
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(os.path.join(save_dir, 'labels', f"{image_name}.txt"), 'w') as f:
            for label in yolov8_labels:
                f.write(f"{label}\n")


dataset_dir = 'datasets/SceneTrialTrain'
img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(dataset_dir)
# Định nghĩa class
class_labels = ['text']

# Thực hiện lời gọi hàm
yolov8_data = convert_to_yolov8_format(
    img_paths,
    img_sizes,
    bboxes
)
seed = 0
val_size = 0.2
test_size = 0.125
is_shuffle = True

train_data, test_data = train_test_split(
    yolov8_data,
    test_size=val_size,
    random_state=seed,
    shuffle=is_shuffle
)
print(train_data)
test_data, val_data = train_test_split(
    test_data,
    test_size=test_size,
    random_state=seed,
    shuffle=is_shuffle
)
# Thực hiện lời gọi hàm cho 3 set train, val, test
save_yolo_data_dir = 'datasets/yolo_data'
os.makedirs(save_yolo_data_dir, exist_ok=True)
save_train_dir = os.path.join(
    save_yolo_data_dir,
    'train'
)
print(save_yolo_data_dir)
save_val_dir = os.path.join(
    save_yolo_data_dir,
    'val'
)
save_test_dir = os.path.join(
    save_yolo_data_dir,
    'test'
)

save_data(
    train_data,
    save_train_dir
)
save_data(
    test_data,
    save_val_dir
)
save_data(
    val_data,
    save_test_dir
)
data_yaml = {
    'path': 'yolo_data',
    'train': 'train/images',
    'test': 'test/images',
    'val': 'val/images',
    'nc': 1,
    'names': class_labels
}

yolo_yaml_path = os.path.join(
    save_yolo_data_dir,
    'data.yml'
)
with open(yolo_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)
