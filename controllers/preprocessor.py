import boto3
from botocore.client import Config
import os
import io

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class Preprocessing:
    data: list = []
    labels: list = []

    def __init__(self, aws_secret: str = None, aws_access: str = None,  end_point: str = None, region_name: str = None):

        if end_point and aws_access and aws_secret != None:
            self.__s3_resource = boto3.resource('s3', endpoint_url=end_point,
                                                aws_access_key_id=aws_access,
                                                aws_secret_access_key=aws_secret,
                                                region_name=region_name,
                                                config=Config(signature_version='s3v4'))
            self.__s3_client = boto3.client('s3', endpoint_url=end_point,
                                            aws_access_key_id=aws_access,
                                            aws_secret_access_key=aws_secret,
                                            region_name=region_name,
                                            config=Config(signature_version='s3v4'))
            self.lb = LabelBinarizer()
        else:
            self.__s3_resource = boto3.resource('s3')
            self.__s3_client = boto3.client('s3')

    def process_data_from_s3(self, src_bucket: str, categories: str) -> list:
        print("[INFO] loading images...")
        try:
            file_stream = io.BytesIO()
            bucket = self.__s3_resource.Bucket(src_bucket)
            CATEGORIES = categories.split(',')
            for category in CATEGORIES:
                objs = self.__s3_client.list_objects_v2(Bucket=src_bucket,
                                                        Prefix=f"{os.environ['SOURCE_PREFIX']}/{category}/")
                for item in objs['Contents']:
                    object = bucket.Object(item['Key'])
                    object.download_fileobj(file_stream)
                    # img = Image.open(file_stream)
                    img = load_img(file_stream, target_size=(224, 224))
                    image = img_to_array(img)
                    image = preprocess_input(image)
                    self.data.append(image)
                    self.labels.append(category)
            return self.data, self.labels
        except Exception as err:
            print(f"{err}")

    @property
    def one_hot_encoding_train_test(self):
        try:
            self.labels = self.lb.fit_transform(self.labels)
            self.labels = to_categorical(self.labels)
            data = np.array(self.data, dtype="float32")
            self.labels = np.array(self.labels)
            (trainX, testX, trainY, testY) = train_test_split(data, self.labels,
                                                              test_size=0.20, stratify=self.labels, random_state=42)
            return trainX, testX, trainY, testY
        except Exception as err:
            print(f"{err}")


